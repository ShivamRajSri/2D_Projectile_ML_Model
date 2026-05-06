"""
visualiser.py
-------------
All plots + annotated video output for the CARS project.

  plot_comparison()    - all models on one figure
  plot_metrics_table() - RMSE/MAE/ADE/FDE bar chart
  plot_velocity()      - velocity profile over time
  make_full_report()   - combined report figure
  annotate_video()     - write output MP4 with overlays
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODEL_COLORS = {
    "pinn":      "#E74C3C",
    "lstm":      "#9B59B6",
    "kalman":    "#27AE60",
    "parabolic": "#F39C12",
    "observed":  "#2C3E50",
    "gt":        "#7F8C8D",
}
MODEL_LABELS = {
    "pinn":      "PINN",
    "lstm":      "LSTM/GRU",
    "kalman":    "Kalman Filter",
    "parabolic": "Parabolic Regression",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trajectory comparison  (all models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results, t_obs, x_obs, y_obs,
                    t_gt=None, x_gt=None, y_gt=None,
                    title="", save_path=None):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title or "Model Comparison — 2D Trajectory", fontsize=13,
                 fontweight="bold")

    for ax_idx, (ax, ylabel, obs_y, gt_y, key_t) in enumerate([
        (axes[0], "y [m]",  y_obs, y_gt, "x_pred"),
        (axes[1], "y [m]",  y_obs, y_gt, "x_pred"),
    ]):
        pass   # handled below

    ax1, ax2 = axes

    # ── 2-D path ──────────────────────────────────────────────────────────────
    if x_gt is not None:
        ax1.plot(x_gt, y_gt, color=MODEL_COLORS["gt"],
                 lw=1.5, ls="--", label="Ground Truth", alpha=0.6)
    ax1.scatter(x_obs, y_obs, s=25, color=MODEL_COLORS["observed"],
                zorder=6, label="Observed", alpha=0.8)
    for key, res in results.items():
        xp, yp = res.get("x_pred"), res.get("y_pred")
        if xp is not None and len(xp) > 1:
            mask = ~np.isnan(xp) & ~np.isnan(yp)
            ax1.plot(xp[mask], yp[mask],
                     color=MODEL_COLORS.get(key,"#999"),
                     lw=2, label=MODEL_LABELS.get(key, key))
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_title("2-D Trajectory"); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="datalim")

    # ── y(t) ──────────────────────────────────────────────────────────────────
    t_dense = t_gt if t_gt is not None else t_obs
    if y_gt is not None:
        ax2.plot(t_dense, y_gt, color=MODEL_COLORS["gt"],
                 lw=1.5, ls="--", label="Ground Truth", alpha=0.6)
    ax2.scatter(t_obs, y_obs, s=15, color=MODEL_COLORS["observed"],
                zorder=6, alpha=0.8, label="Observed")
    for key, res in results.items():
        yp = res.get("y_pred")
        if yp is not None and len(yp) > 1:
            mask = ~np.isnan(yp)
            ax2.plot(t_dense[mask], yp[mask],
                     color=MODEL_COLORS.get(key,"#999"),
                     lw=2, label=MODEL_LABELS.get(key, key))
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("y [m]")
    ax2.set_title("Vertical Position vs Time")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics(results, save_path=None):
    metrics_keys = ["RMSE_x", "RMSE_y", "ADE", "FDE"]
    models = [k for k in results if results[k].get("metrics")]
    if not models: return None

    fig, axes = plt.subplots(1, len(metrics_keys),
                             figsize=(4*len(metrics_keys), 4))
    fig.suptitle("Model Performance Metrics", fontsize=13, fontweight="bold")

    for ax, mk in zip(axes, metrics_keys):
        vals = []
        labels = []
        colors = []
        for m in models:
            v = results[m]["metrics"].get(mk, np.nan)
            if not np.isnan(v):
                vals.append(v)
                labels.append(MODEL_LABELS.get(m, m))
                colors.append(MODEL_COLORS.get(m, "#999"))
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", lw=0.8)
        ax.set_title(mk, fontweight="bold")
        ax.set_ylabel("metres")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Velocity profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_velocity(t_obs, x_obs, y_obs, kin=None, pinn_model=None, save_path=None):
    dt  = np.diff(t_obs)+1e-9
    vx  = np.diff(x_obs)/dt
    vy  = np.diff(y_obs)/dt
    spd = np.sqrt(vx**2+vy**2)
    tm  = (t_obs[:-1]+t_obs[1:])/2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Velocity Analysis", fontsize=13, fontweight="bold")

    axes[0].plot(tm, vx, color="#3498DB", lw=2, alpha=0.7, label="Measured vx")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("vx [m/s]")
    axes[0].set_title("Horizontal Velocity"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(tm, vy, color="#E74C3C", lw=2, alpha=0.7, label="Measured vy")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("vy [m/s]")
    axes[1].set_title("Vertical Velocity"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(tm, spd, color="#27AE60", lw=2, alpha=0.7, label="Measured speed")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("speed [m/s]")
    axes[2].set_title("Speed Profile"); axes[2].grid(True, alpha=0.3)

    if pinn_model and getattr(pinn_model, "_trained", False):
        import torch
        # PINN exact derivatives (two independent forward passes to avoid graph destruction)
        tn_x = torch.tensor(pinn_model._norm_t(tm), dtype=torch.float32).reshape(-1, 1)
        tn_x.requires_grad_(True)
        xn_c = pinn_model._net(tn_x)[:, 0:1]
        dxn_dt = torch.autograd.grad(xn_c, tn_x, grad_outputs=torch.ones_like(xn_c))[0]
        
        tn_y = torch.tensor(pinn_model._norm_t(tm), dtype=torch.float32).reshape(-1, 1)
        tn_y.requires_grad_(True)
        yn_c = pinn_model._net(tn_y)[:, 1:2]
        dyn_dt = torch.autograd.grad(yn_c, tn_y, grad_outputs=torch.ones_like(yn_c))[0]
        
        vx_pinn = (dxn_dt.detach().numpy().flatten() * (pinn_model._xs / pinn_model._dt))
        vy_pinn = (dyn_dt.detach().numpy().flatten() * (pinn_model._ys / pinn_model._dt))
        spd_pinn = np.sqrt(vx_pinn**2 + vy_pinn**2)
        
        axes[0].plot(tm, vx_pinn, ls="--", color="#F39C12", lw=2, label="PINN")
        axes[1].plot(tm, vy_pinn, ls="--", color="#F39C12", lw=2, label="PINN")
        axes[2].plot(tm, spd_pinn, ls="--", color="#F39C12", lw=2, label="PINN")

    for ax in axes:
        ax.legend()
        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        
        if y_min >= 0:
            ax.set_ylim(bottom=0, top=y_max + margin)
        elif y_max <= 0:
            ax.set_ylim(bottom=y_min - margin, top=0)
        else:
            ax.set_ylim(bottom=y_min - margin, top=y_max + margin)
            
        # Force strict (0,0) Cartesian origin
        ax.set_xlim(left=0)
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full report
# ─────────────────────────────────────────────────────────────────────────────

def make_full_report(results, t_obs, x_obs, y_obs,
                     t_gt, x_gt, y_gt,
                     cfg, stroke_label, save_path=None):

    kin   = results["pinn"]["kin"] if "pinn" in results else {}
    color = "#E74C3C"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{cfg['display_name']}  ·  Trajectory Analysis  ·  {stroke_label}",
        fontsize=14, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── 2-D trajectory (tall left panel) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[:2, 0])
    if x_gt is not None:
        ax1.plot(x_gt, y_gt, color=MODEL_COLORS["gt"], lw=1.5,
                 ls="--", label="Ground Truth", alpha=0.6)
    ax1.scatter(x_obs, y_obs, s=20, color="black", zorder=6,
                alpha=0.7, label="Observed")
    for key, res in results.items():
        xp, yp = res.get("x_pred"), res.get("y_pred")
        if xp is not None and len(xp)>1:
            mask=~np.isnan(xp)&~np.isnan(yp)
            ax1.plot(xp[mask], yp[mask],
                     color=MODEL_COLORS.get(key,"#999"), lw=2,
                     label=MODEL_LABELS.get(key,key))
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_title("2-D Trajectory"); ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="datalim")

    # ── y(t) ─────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if y_gt is not None:
        ax2.plot(t_gt, y_gt, color=MODEL_COLORS["gt"], lw=1.5, ls="--", alpha=0.5)
    ax2.scatter(t_obs, y_obs, s=10, color="black", alpha=0.7, zorder=6)
    for key, res in results.items():
        yp=res.get("y_pred")
        if yp is not None and len(yp)>1:
            mask=~np.isnan(yp)
            ax2.plot(t_gt[mask], yp[mask],
                     color=MODEL_COLORS.get(key,"#999"), lw=1.5,
                     label=MODEL_LABELS.get(key,key))
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("y [m]")
    ax2.set_title("Vertical vs Time"); ax2.legend(fontsize=6); ax2.grid(True,alpha=0.3)

    # ── x(t) ─────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(t_obs, x_obs, s=10, color="black", alpha=0.7, zorder=6)
    for key, res in results.items():
        xp=res.get("x_pred")
        if xp is not None and len(xp)>1:
            mask=~np.isnan(xp)
            ax3.plot(t_gt[mask], xp[mask],
                     color=MODEL_COLORS.get(key,"#999"), lw=1.5)
    ax3.set_xlabel("t [s]"); ax3.set_ylabel("x [m]")
    ax3.set_title("Horizontal vs Time"); ax3.grid(True, alpha=0.3)

    # ── PINN loss ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
    if results.get("pinn",{}).get("loss"):
        iters=np.arange(1,len(results["pinn"]["loss"])+1)*200
        ax4.semilogy(iters, results["pinn"]["loss"], color="#E74C3C", lw=2)
        ax4.set_xlabel("Iter"); ax4.set_ylabel("Loss")
        ax4.set_title("PINN Training Loss"); ax4.grid(True, which="both", alpha=0.3)

    # ── LSTM loss ─────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if results.get("lstm",{}).get("loss"):
        iters=np.arange(1,len(results["lstm"]["loss"])+1)*100
        ax5.semilogy(iters, results["lstm"]["loss"], color="#9B59B6", lw=2)
        ax5.set_xlabel("Iter"); ax5.set_ylabel("Loss")
        ax5.set_title("LSTM Training Loss"); ax5.grid(True, which="both", alpha=0.3)

    # ── Metrics bar chart ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[0, 3])
    mods  = [k for k in results if results[k].get("metrics")]
    ades  = [results[m]["metrics"].get("ADE", np.nan) for m in mods]
    clrs  = [MODEL_COLORS.get(m,"#999") for m in mods]
    lbls  = [MODEL_LABELS.get(m,m) for m in mods]
    bars  = ax6.bar(lbls, ades, color=clrs, edgecolor="white")
    ax6.set_title("ADE (lower=better)"); ax6.set_ylabel("ADE [m]")
    ax6.tick_params(axis="x", rotation=30)
    ax6.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, ades):
        if not np.isnan(v):
            ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Kinematics table ──────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1:, 3])
    ax7.axis("off")
    rows = [["Parameter", "Value", "Unit"],
            ["Stroke/Phase", stroke_label, ""],
            ["Speed V₀",     f"{kin.get('speed',0):.2f}", "m/s"],
            ["vₓ₀",          f"{kin.get('vx0',0):.2f}",  "m/s"],
            ["v_y₀",         f"{kin.get('vy0',0):.2f}",  "m/s"],
            ["Drag Cᴅ",      f"{kin.get('CD',0):.3f}",   "—"]]
    if cfg["has_spin"]:
        rows.append(["Spin ω", f"{kin.get('spin_rps',0):.2f}", "rps"])

    tbl = ax7.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center",
                    colWidths=[0.44, 0.32, 0.22])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for j in range(3):
        tbl[0,j].set_facecolor("#2C3E50")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    tbl[1,0].set_facecolor(color+"33")
    ax7.set_title("PINN Kinematics", fontsize=10)

    # ── Velocity bottom row ───────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, :3])
    if len(t_obs) > 2:
        dt   = np.diff(t_obs)+1e-9
        spd  = np.sqrt(np.diff(x_obs)**2+np.diff(y_obs)**2)/dt
        tm   = (t_obs[:-1]+t_obs[1:])/2
        ax8.plot(tm, spd, color="#2ECC71", lw=2, label="Estimated speed")
        if kin.get("speed"):
            ax8.axhline(kin["speed"], ls="--", color="#E74C3C", lw=1.5,
                        label=f"PINN V₀={kin['speed']:.1f} m/s")
        ax8.set_xlabel("t [s]"); ax8.set_ylabel("speed [m/s]")
        ax8.set_title("Speed Profile over Time")
        ax8.legend(); ax8.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Annotated video output
# ─────────────────────────────────────────────────────────────────────────────

def annotate_video(input_path, output_path, cfg,
                   det_data, pinn_model, lstm_model, kf,
                   scale_pxm, height_px):
    """
    Write output video with detection + all model prediction overlays.

    det_data : dict from detector.extract_trajectory()
    scale_pxm: px per metre (for converting predictions back to pixels)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[video] Cannot open {input_path}"); return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc= cv2.VideoWriter_fourcc(*"mp4v")
    out   = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    fid_to_det = {int(fi): (x, y)
                  for fi, x, y in zip(det_data["frame_indices"],
                                      det_data["xs"], det_data["ys"])}
    trail   = []
    PREDICT = 20   # frames ahead to predict

    def m2px(xm, ym):
        """Convert metres back to pixel coords."""
        px = int(xm * scale_pxm + det_data["xs"].min())
        py = int(height_px - ym * scale_pxm - (height_px - det_data["ys"].max()))
        return px, py

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t_now = frame_idx / fps

        # Detected position for this frame
        det = fid_to_det.get(frame_idx)
        if det:
            cx, cy = det
            trail.append((int(cx), int(cy)))
            if len(trail) > 40: trail.pop(0)

        # Draw trail
        for i in range(1, len(trail)):
            a = i/len(trail)
            cv2.line(frame, trail[i-1], trail[i],
                     (int(255*a), int(100*(1-a)), 50), 2)

        if det:
            cx, cy = det
            cv2.circle(frame, (int(cx), int(cy)), 8, (0,255,0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 3, (0,0,255), -1)

        # PINN prediction line
        if pinn_model and pinn_model._trained:
            t_fut  = np.linspace(t_now, t_now + PREDICT/fps, PREDICT)
            xp, yp = pinn_model.predict(t_fut)
            pts    = []
            for xm, ym in zip(xp, yp):
                px, py = m2px(float(xm), float(ym))
                if 0 <= px < W and 0 <= py < H:
                    pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0,0,220), 2)
            if pts:
                cv2.putText(frame, "PINN", pts[0],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,220), 1)

        # Kalman prediction line
        if kf:
            kxp, kyp = kf.predict_ahead(PREDICT)
            pts = []
            for xm, ym in zip(kxp, kyp):
                px, py = m2px(float(xm), float(ym))
                if 0 <= px < W and 0 <= py < H:
                    pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0,180,0), 2)

        # HUD: object name + speed
        speed = kf.get_speed() if kf else 0.
        cv2.rectangle(frame, (0,0),(270,55),(0,0,0,100),-1)
        cv2.putText(frame, cfg["display_name"],
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255,255,255), 1)
        cv2.putText(frame, f"Speed: {speed:.2f} m/s  |  t={t_now:.2f}s",
                    (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0,230,230), 1)

        # Legend
        for i, (name, col) in enumerate([
            ("PINN", (0,0,220)),
            ("Kalman", (0,180,0)),
            ("Detected trail", (200,100,50))
        ]):
            y0 = H - 70 + i*22
            cv2.line(frame, (W-140, y0), (W-110, y0), col, 3)
            cv2.putText(frame, name, (W-105, y0+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255,255,255), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[video] Saved annotated video → {output_path}")
