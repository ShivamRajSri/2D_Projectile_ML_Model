"""
models.py
---------
All ML models for projectile trajectory prediction.

  1. ProjectilePINN  - Physics-Informed Neural Network (PyTorch)
  2. LSTMPredictor   - Sequence-to-sequence LSTM (NumPy/SciPy)
  3. predict_all()   - Run all models + baselines, return comparison dict
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from physics import parabolic_fit, KalmanTracker, G, RHO


# ═════════════════════════════════════════════════════════════════════════════
# Shared NumPy MLP utilities (used by LSTMPredictor only)
# ═════════════════════════════════════════════════════════════════════════════

def _pack(layers):
    return np.concatenate([p.ravel() for wb in layers for p in wb])

def _unpack(theta, sizes):
    layers, i = [], 0
    for a, b in zip(sizes, sizes[1:]):
        W = theta[i:i+a*b].reshape(b, a); i += a*b
        bs= theta[i:i+b];                 i += b
        layers.append((W, bs))
    return layers

def _fwd(t, layers):
    a = t
    for j, (W, b) in enumerate(layers):
        z = a @ W.T + b
        a = np.tanh(z) if j < len(layers)-1 else z
    return a

def _xavier(sizes, rng):
    out = []
    for a, b in zip(sizes, sizes[1:]):
        W = rng.normal(0, np.sqrt(2/(a+b)), (b, a))
        out.append((W, np.zeros(b)))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 1. PINN  (Physics-Informed Neural Network — PyTorch)
# ═════════════════════════════════════════════════════════════════════════════

class _PINNNet(nn.Module):
    """Inner PyTorch network: t (scalar, normalised) → [x_n, y_n]"""
    def __init__(self, hidden, neurons):
        super().__init__()
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers += [nn.Linear(neurons, 2)]
        self.net = nn.Sequential(*layers)

        # Learnable physics parameters (unconstrained; we use abs() where needed)
        self.log_CD  = nn.Parameter(torch.tensor(0.0))   # log(CD)
        self.log_om  = nn.Parameter(torch.tensor(3.4))   # log(omega) ≈ 30 rad/s

        # Xavier init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


class ProjectilePINN:
    """
    2-D PINN: maps t → (x(t), y(t)).
    Uses PyTorch autograd for exact PDE residuals (following Chiha et al. 2024).
    Learns NN weights + drag coefficient + spin (if applicable).
    """

    def __init__(self, cfg, hidden=4, neurons=32,
                 beta=1e-3, n_coll=300, max_iter=3000, seed=42):
        self.cfg      = cfg
        self.hidden   = hidden
        self.neurons  = neurons
        self.beta     = beta
        self.n_coll   = n_coll
        self.max_iter = max_iter
        self.seed     = seed
        self._trained = False
        self.loss_history = []

    def _norm_t(self, t):
        return (t - self._t0) / (self._dt + 1e-12)

    def fit(self, t_obs, x_obs, y_obs, verbose=True):
        torch.manual_seed(self.seed)
        cfg = self.cfg

        # --- Normalisation constants ---
        self._t0 = float(t_obs.min()); self._dt = float(t_obs.max() - t_obs.min())
        self._x0 = float(x_obs.min()); self._xs = float(max(x_obs.max()-x_obs.min(), 1e-3))
        self._y0 = float(y_obs.min()); self._ys = float(max(y_obs.max()-y_obs.min(), 1e-3))

        # --- Normalised observations (torch) ---
        tn  = torch.tensor(self._norm_t(t_obs), dtype=torch.float32).reshape(-1, 1)
        xn  = torch.tensor((x_obs - self._x0) / self._xs, dtype=torch.float32)
        yn  = torch.tensor((y_obs - self._y0) / self._ys, dtype=torch.float32)
        data = torch.stack([xn, yn], dim=1)   # (N, 2)

        # --- Collocation points (cover 2x obs window for extrapolation) ---
        tc = torch.linspace(0.0, 2.0, self.n_coll, requires_grad=True).reshape(-1, 1)

        # --- Physics scaling ---
        K     = RHO * np.pi * cfg["radius_m"]**2 / (2 * cfg["mass_kg"])
        R     = cfg["radius_m"]
        ts    = self._dt
        xs_sc = self._xs
        ys_sc = self._ys

        # --- Build model + optimiser ---
        net = _PINNNet(self.hidden, self.neurons)
        if not cfg["has_spin"]:
            net.log_om.requires_grad_(False)

        optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_iter, eta_min=1e-5)

        mse = nn.MSELoss()
        hist = []

        if verbose:
            print(f"\n[PINN] Fitting {cfg['display_name']} | "
                  f"{len(t_obs)} pts | spin={cfg['has_spin']}")

        for epoch in range(self.max_iter):
            optimizer.zero_grad()

            # ── Data loss ────────────────────────────────────────────────────
            pred = net(tn)
            loss_data = mse(pred, data)

            # ── Physics loss (exact autograd derivatives) ─────────────────────
            tc_fresh = tc.detach().clone().requires_grad_(True)
            out = net(tc_fresh)          # (N_coll, 2)
            xn_c = out[:, 0:1]
            yn_c = out[:, 1:2]

            # First derivatives d(x_n)/d(t_n), d(y_n)/d(t_n)
            dxn_dt = torch.autograd.grad(xn_c, tc_fresh,
                        grad_outputs=torch.ones_like(xn_c),
                        create_graph=True)[0]
            dyn_dt = torch.autograd.grad(yn_c, tc_fresh,
                        grad_outputs=torch.ones_like(yn_c),
                        create_graph=True)[0]

            # Second derivatives
            d2xn_dt2 = torch.autograd.grad(dxn_dt, tc_fresh,
                        grad_outputs=torch.ones_like(dxn_dt),
                        create_graph=True)[0]
            d2yn_dt2 = torch.autograd.grad(dyn_dt, tc_fresh,
                        grad_outputs=torch.ones_like(dyn_dt),
                        create_graph=True)[0]

            # Physical velocities / accelerations
            vx = dxn_dt * (xs_sc / ts)
            vy = dyn_dt * (ys_sc / ts)
            ax = d2xn_dt2 * (xs_sc / ts**2)
            ay = d2yn_dt2 * (ys_sc / ts**2)
            V  = torch.sqrt(vx**2 + vy**2) + 1e-9

            CD = torch.exp(net.log_CD)

            # Projectile ODE residual:  a + drag + gravity = 0
            fx = ax + K * V * CD * vx                     # x-residual
            fy = ay + G + K * V * CD * vy                 # y-residual

            # Magnus / spin lift
            if cfg["has_spin"]:
                om = torch.exp(net.log_om)
                CL = 1.0 / (2.0 + (R * om / V)**(-1) + 1e-9)
                fx = fx - K * V * (CL / om) * (om * vy)
                fy = fy + K * V * (CL / om) * (om * vx)

            loss_phys = (fx**2).mean() + (fy**2).mean()

            # ── Total loss ────────────────────────────────────────────────────
            loss = (1.0 - self.beta) * loss_data + self.beta * loss_phys

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 200 == 0:
                lv = float(loss.item())
                hist.append(lv)
                if verbose:
                    print(f"  [PINN] epoch {epoch:4d}  loss={lv:.3e}  "
                          f"CD={float(CD.item()):.3f}")

        self._net = net
        self.loss_history = hist
        self._trained = True

        if verbose:
            kin = self.kinematics()
            print(f"[PINN] Done  loss={float(loss.item()):.3e}  "
                  f"V0={kin['speed']:.2f}m/s  CD={kin['CD']:.3f}")
        return self

    def predict(self, t_query):
        self._net.eval()
        tn = torch.tensor(
            self._norm_t(np.asarray(t_query, dtype=np.float64)),
            dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            out = self._net(tn).numpy()
        return (out[:, 0] * self._xs + self._x0,
                out[:, 1] * self._ys + self._y0)

    def kinematics(self):
        net = self._net
        CD  = float(torch.exp(net.log_CD).item())
        om  = float(torch.exp(net.log_om).item()) if self.cfg["has_spin"] else 0.0

        with torch.enable_grad():
            # Two independent forward passes so each has its own graph
            t0x = torch.tensor([[0.0]], requires_grad=True)
            dvx = torch.autograd.grad(net(t0x)[0, 0], t0x)[0]
            t0y = torch.tensor([[0.0]], requires_grad=True)
            dvy = torch.autograd.grad(net(t0y)[0, 1], t0y)[0]
        vx = float(dvx.item()) * self._xs / self._dt
        vy = float(dvy.item()) * self._ys / self._dt
        return {"CD": CD, "spin_rps": om / (2 * np.pi),
                "vx0": vx, "vy0": vy,
                "speed": float(np.sqrt(vx**2 + vy**2))}

    def classify(self):
        kin = self.kinematics()
        net = self._net
        with torch.enable_grad():
            tm  = torch.tensor([[0.5]], requires_grad=True)
            out = net(tm)
            yn_m = out[0, 1:2]
            dvy  = torch.autograd.grad(yn_m, tm, create_graph=True)[0]
            d2vy = torch.autograd.grad(dvy,  tm, create_graph=False)[0]
        sign = -np.sign(float(d2vy.item()))
        rps  = sign * kin["spin_rps"]
        return self.cfg["classify"](kin["speed"], rps), rps, kin["speed"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. LSTM Predictor  (sequence → sequence, NumPy/SciPy only)
#    Implemented as a vanilla RNN in NumPy since PyTorch is unavailable.
#    Uses sliding window of past positions to predict next k positions.
# ═════════════════════════════════════════════════════════════════════════════

class LSTMPredictor:
    """
    Simplified LSTM-like sequence predictor implemented in pure NumPy.
    Uses a sliding window of past (x,y) observations to predict future positions.

    Architecture: GRU cell (simpler than LSTM, same idea, fewer params)
                  → Linear output layer
    """

    def __init__(self, window=10, hidden_size=32, predict_steps=20,
                 lr=0.01, epochs=500, seed=42):
        self.window  = window
        self.H       = hidden_size
        self.k       = predict_steps
        self.lr      = lr
        self.epochs  = epochs
        self.seed    = seed
        self._trained = False
        self.loss_history = []

    # ── Sigmoid / tanh ────────────────────────────────────────────────────────
    @staticmethod
    def _sig(x): return 1./(1.+np.exp(-np.clip(x,-30,30)))
    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x,-30,30))

    # ── GRU cell forward ─────────────────────────────────────────────────────
    def _gru(self, x, h, params):
        Wz,Uz,bz, Wr,Ur,br, Wn,Un,bn = params
        z = self._sig(x@Wz.T + h@Uz.T + bz)
        r = self._sig(x@Wr.T + h@Ur.T + br)
        n = self._tanh(x@Wn.T + (r*h)@Un.T + bn)
        return (1-z)*h + z*n

    # ── Initialise parameters ─────────────────────────────────────────────────
    def _init_params(self, rng, input_size):
        H, I = self.H, input_size
        std  = 0.1
        def W(r,c): return rng.normal(0,std,(r,c))
        def b(n):   return np.zeros(n)
        gru_params = [
            W(H,I), W(H,H), b(H),   # z gate
            W(H,I), W(H,H), b(H),   # r gate
            W(H,I), W(H,H), b(H),   # n gate
        ]
        # Output layers: one per future step (shared weights → linear decode)
        Wo = W(2*self.k, H)
        bo = b(2*self.k)
        return gru_params, Wo, bo

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward(self, seq, params, Wo, bo):
        """seq: (window, 2)  → returns (k, 2)"""
        h = np.zeros(self.H)
        for t in range(seq.shape[0]):
            h = self._gru(seq[t], h, params)
        out = h @ Wo.T + bo           # (2k,)
        return out.reshape(self.k, 2)

    # ── Build windows ─────────────────────────────────────────────────────────
    def _make_windows(self, x_n, y_n):
        data = np.stack([x_n, y_n], 1)   # (N,2)
        X, Y = [], []
        for i in range(len(data)-self.window-self.k+1):
            X.append(data[i:i+self.window])
            Y.append(data[i+self.window:i+self.window+self.k])
        return np.array(X), np.array(Y)

    # ── Flatten / unflatten params ────────────────────────────────────────────
    def _flat(self, gru, Wo, bo):
        return np.concatenate([p.ravel() for p in gru] +
                               [Wo.ravel(), bo])

    def _unflat(self, v, I):
        H = self.H
        shapes = [(H,I),(H,H),(H,), (H,I),(H,H),(H,), (H,I),(H,H),(H,)]
        gru, idx = [], 0
        for s in shapes:
            n = int(np.prod(s))
            gru.append(v[idx:idx+n].reshape(s)); idx+=n
        Wo = v[idx:idx+2*self.k*H].reshape(2*self.k, H); idx+=2*self.k*H
        bo = v[idx:idx+2*self.k]
        return gru, Wo, bo

    # ── Train ─────────────────────────────────────────────────────────────────
    def fit(self, t_obs, x_obs, y_obs, verbose=True):
        rng = np.random.default_rng(self.seed)

        # Normalise
        self._x0=x_obs.min(); self._xs=max(x_obs.max()-x_obs.min(),1e-3)
        self._y0=y_obs.min(); self._ys=max(y_obs.max()-y_obs.min(),1e-3)
        xn = (x_obs-self._x0)/self._xs
        yn = (y_obs-self._y0)/self._ys

        X, Y = self._make_windows(xn, yn)   # (M, window, 2), (M, k, 2)
        if len(X) < 2:
            if verbose: print("[LSTM] Not enough data for sequence training")
            self._trained = False
            return self

        I = 2  # input size
        gru0, Wo0, bo0 = self._init_params(rng, I)
        theta0 = self._flat(gru0, Wo0, bo0)
        hist   = []

        def loss(theta):
            gru, Wo, bo = self._unflat(theta, I)
            total = 0.
            for xi, yi in zip(X, Y):
                pred = self._forward(xi, gru, Wo, bo)
                total += np.mean((pred-yi)**2)
            return total/len(X)

        itr=[0]
        def cb(xk):
            itr[0]+=1
            if itr[0]%100==0:
                v=loss(xk); hist.append(v)
                if verbose:
                    print(f"  [LSTM] iter {itr[0]:4d}  loss={v:.4e}")

        if verbose:
            print(f"\n[LSTM] Training on {len(X)} windows  "
                  f"(window={self.window}, predict={self.k})")

        res = minimize(loss, theta0, method="L-BFGS-B", callback=cb,
                       options={"maxiter":self.epochs,"ftol":1e-12,"gtol":1e-8})

        self._gru, self._Wo, self._bo = self._unflat(res.x, I)
        self._trained = True
        self.loss_history = hist
        if verbose:
            print(f"[LSTM] Done  loss={res.fun:.4e}")
        return self

    def predict_next(self, x_recent, y_recent):
        """
        Given recent observations, predict next self.k positions.
        x_recent, y_recent : last `window` observations (metres)
        Returns x_fut, y_fut (metres)
        """
        if not self._trained:
            return np.array([]), np.array([])
        w = self.window
        xn = (x_recent-self._x0)/self._xs
        yn = (y_recent-self._y0)/self._ys
        # Pad or trim to window size
        seq = np.stack([xn[-w:], yn[-w:]], 1)
        if len(seq) < w:
            pad = np.zeros((w-len(seq), 2))
            seq = np.vstack([pad, seq])
        pred = self._forward(seq, self._gru, self._Wo, self._bo)
        xf = pred[:,0]*self._xs + self._x0
        yf = pred[:,1]*self._ys + self._y0
        return xf, yf


# ═════════════════════════════════════════════════════════════════════════════
# 3. Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true_x, y_true_y, y_pred_x, y_pred_y):
    """Compute RMSE, MAE, ADE, FDE for trajectory comparison."""
    n    = min(len(y_true_x), len(y_pred_x))
    tx   = y_true_x[:n]; ty = y_true_y[:n]
    px   = y_pred_x[:n]; py = y_pred_y[:n]
    dist = np.sqrt((tx-px)**2 + (ty-py)**2)
    rmse_x = float(np.sqrt(np.mean((tx-px)**2)))
    rmse_y = float(np.sqrt(np.mean((ty-py)**2)))
    mae_x  = float(np.mean(np.abs(tx-px)))
    mae_y  = float(np.mean(np.abs(ty-py)))
    ade    = float(np.mean(dist))
    fde    = float(dist[-1]) if n > 0 else np.nan
    return {"RMSE_x":rmse_x,"RMSE_y":rmse_y,
            "MAE_x":mae_x,"MAE_y":mae_y,
            "ADE":ade,"FDE":fde}


# ═════════════════════════════════════════════════════════════════════════════
# 4. Unified prediction runner
# ═════════════════════════════════════════════════════════════════════════════

def predict_all(cfg, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
                pinn_iters=5000, lstm_epochs=500, verbose=True):
    """
    Train all models and return a comparison dict.

    Returns
    -------
    dict with keys: 'pinn', 'lstm', 'kalman', 'parabolic'
    Each value: dict with 'x_pred', 'y_pred', 'metrics', 'label', 'kin'
    """
    results = {}
    t_query = t_gt   # predict on ground-truth time axis

    # ── 1. PINN ───────────────────────────────────────────────────────────────
    pinn = ProjectilePINN(cfg, max_iter=pinn_iters)
    pinn.fit(t_obs, x_obs, y_obs, verbose=verbose)
    xp, yp = pinn.predict(t_query)
    label, rps, V0 = pinn.classify()
    results["pinn"] = {
        "model":   pinn,
        "x_pred":  xp, "y_pred": yp,
        "metrics": compute_metrics(x_gt, y_gt, xp, yp),
        "label":   label,
        "kin":     pinn.kinematics(),
        "loss":    pinn.loss_history,
    }

    # ── 2. LSTM ───────────────────────────────────────────────────────────────
    lstm = LSTMPredictor(window=min(10,len(t_obs)//3),
                         predict_steps=len(t_query),
                         epochs=lstm_epochs)
    lstm.fit(t_obs, x_obs, y_obs, verbose=verbose)
    if lstm._trained:
        xl, yl = lstm.predict_next(x_obs, y_obs)
        # LSTM predicts absolute positions (future), align to query length
        n = min(len(xl), len(t_query))
        xl_full = np.full(len(t_query), np.nan)
        yl_full = np.full(len(t_query), np.nan)
        xl_full[-n:] = xl[:n]
        yl_full[-n:] = yl[:n]
        results["lstm"] = {
            "x_pred": xl_full, "y_pred": yl_full,
            "metrics": compute_metrics(x_gt[-n:], y_gt[-n:], xl[:n], yl[:n]),
            "loss":    lstm.loss_history,
        }
    else:
        results["lstm"] = {"x_pred": np.array([]), "y_pred": np.array([]),
                           "metrics": {}, "loss": []}

    # ── 3. Kalman Filter ──────────────────────────────────────────────────────
    dt = float(np.mean(np.diff(t_obs))) if len(t_obs)>1 else 1/30
    kf = KalmanTracker(dt=dt)
    kx, ky = [], []
    for xi, yi in zip(x_obs, y_obs):
        sx, sy, _, _ = kf.update(xi, yi)
        kx.append(sx); ky.append(sy)
    # Predict ahead to fill t_query
    n_ahead = max(0, len(t_query)-len(kx))
    if n_ahead > 0:
        xah, yah = kf.predict_ahead(n_ahead)
        kx_full = np.concatenate([kx, xah])
        ky_full = np.concatenate([ky, yah])
    else:
        kx_full = np.array(kx[:len(t_query)])
        ky_full = np.array(ky[:len(t_query)])
    results["kalman"] = {
        "x_pred":  kx_full, "y_pred": ky_full,
        "metrics": compute_metrics(x_gt, y_gt, kx_full, ky_full),
    }

    # ── 4. Parabolic Regression (classical baseline) ──────────────────────────
    x_fut = np.interp(t_query, t_obs,
                      np.linspace(x_obs[0], x_obs[-1], len(t_obs)))
    y_par, coeffs = parabolic_fit(x_obs, y_obs, x_fut)
    results["parabolic"] = {
        "x_pred":  x_fut, "y_pred": y_par,
        "metrics": compute_metrics(x_gt, y_gt, x_fut, y_par),
        "coeffs":  coeffs,
    }

    return results
