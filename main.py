"""
main.py
-------
CARS Project: Prediction of Projectile Trajectory and Velocity
             in 2D through Machine Learning and Computer Vision

Sponsor : HEMRL, Pune
PI      : Dr. Anubhav Rawat, Dr. Ashutosh Mishra, Dr. A.K. Tiwari

USAGE
-----
  # List all supported objects
  python main.py --list

  # Simulate any object
  python main.py --simulate --object football --preset banana

  # Analyse a real video
  python main.py --video clip.mp4 --object tennis

  # Run all objects (demo/report)
  python main.py --demo
"""

import argparse, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

from config     import OBJECTS, get_object, list_objects
from physics    import simulate, KalmanTracker
from detector   import extract_trajectory, to_meters, clean
from models     import ProjectilePINN, LSTMPredictor, predict_all
from visualiser import (plot_comparison, plot_metrics, plot_velocity,
                        make_full_report, annotate_video)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CARS: 2-D Projectile Trajectory Prediction via ML + CV",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--object",      default="table_tennis",
                   help="Object type (use --list to see all)")
    p.add_argument("--preset",      default=None,
                   help="Simulation preset (e.g. topspin, banana, fastball)")
    p.add_argument("--video",       default=None,
                   help="Path to input video file")
    p.add_argument("--color",       default=None,
                   help="Override detection colour: white/orange/yellow/green/red")
    p.add_argument("--output",      default="results",
                   help="Output folder  (default: results/)")
    p.add_argument("--simulate",    action="store_true",
                   help="Use synthetic simulation instead of real video")
    p.add_argument("--demo",        action="store_true",
                   help="Run all objects and produce comparison report")
    p.add_argument("--interactive", action="store_true",
                   help="Use manual interactive point selection instead of RANSAC")
    p.add_argument("--list",        action="store_true",
                   help="List all available objects and exit")
    p.add_argument("--no-bg-sub",   action="store_true")
    p.add_argument("--max-frames",  type=int, default=None)
    p.add_argument("--pinn-iters",  type=int, default=5000)
    p.add_argument("--lstm-epochs", type=int, default=500)
    p.add_argument("--noise",       type=float, default=0.005,
                   help="Gaussian noise std (metres) for simulation")
    p.add_argument("--annotate-video", action="store_true",
                   help="Write annotated output video (real video mode only)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sim(cfg, preset_name, noise=0.005):
    key    = preset_name or list(cfg["presets"].keys())[0]
    params = cfg["presets"][key]
    print(f"\n[data] Simulating {cfg['display_name']}  ·  preset='{key}'")
    t, x, y, vx, vy = simulate(cfg, noise_m=noise, **params)

    # Sparse subsample (mimics tracking → sparse detections)
    rng = np.random.default_rng(42)
    n   = max(10, min(30, len(t)//4))
    idx = np.sort(rng.choice(len(t), n, replace=False))
    print(f"[data] Full traj: {len(t)} pts  |  sparse obs: {len(idx)}")
    return (t[idx], x[idx], y[idx],
            {"full": (t, x, y), "preset": key, "simulated": True})


def load_video(cfg, args):
    if not os.path.isfile(args.video):
        print(f"ERROR: not found: {args.video}"); sys.exit(1)

    colors = [args.color] if args.color else cfg["colors"]
    det    = extract_trajectory(
        args.video, cfg,
        color_override=colors,
        use_bg_sub=not args.no_bg_sub,
        max_frames=args.max_frames,
        annotate_output=None,
    )
    if len(det["times"]) < 5:
        print("ERROR: <5 detections. Try --color or --no-bg-sub"); sys.exit(1)

    # Interactive Scale Calibration
    import cv2
    import math
    scale_override = None
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()
    if ret:
        pts = []
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
                pts.append((x, y))
                
        cv2.namedWindow("Calibrate Scale")
        cv2.setMouseCallback("Calibrate Scale", on_click)
        print("\n[+] Click two points to define a known distance. Press 'q' to skip.")
        
        while len(pts) < 2:
            disp = frame.copy()
            cv2.putText(disp, "Click two points for scale calibration. 'q' to skip.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for p in pts:
                cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.imshow("Calibrate Scale", disp)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        if len(pts) == 2:
            disp = frame.copy()
            cv2.circle(disp, pts[0], 5, (0, 0, 255), -1)
            cv2.circle(disp, pts[1], 5, (0, 0, 255), -1)
            cv2.line(disp, pts[0], pts[1], (0, 255, 0), 2)
            cv2.imshow("Calibrate Scale", disp)
            cv2.waitKey(500)
            
            px_dist = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            try:
                val = float(input(f"\nEnter real-world distance between points (pixels={px_dist:.1f}): "))
                if val > 0:
                    scale_override = px_dist / val
                    print(f"[data] Scale set manually to {scale_override:.2f} px/m")
            except ValueError:
                pass
        cv2.destroyWindow("Calibrate Scale")

    x_m, y_m = to_meters(det["xs"], det["ys"], det["height"],
                          cfg["scene_width_m"], scale_override=scale_override)
                          
    # ── Save Raw Data ───────────────────────────────────────────────────────
    if args.video:
        import csv
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        out_dir = os.path.join(args.output, base_name)
        os.makedirs(out_dir, exist_ok=True)
        
        csv_path = os.path.join(out_dir, "raw_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x_meters", "y_meters"])
            for ti, xi, yi in zip(det["times"], x_m, y_m):
                writer.writerow([f"{ti:.4f}", f"{xi:.4f}", f"{yi:.4f}"])
        print(f"[data] Raw trajectory (unfiltered) saved to: {csv_path}")

    if args.interactive:
        t, x, y  = clean(det["times"], x_m, y_m)

        # Interactive Point Selection
        import numpy as np
        print("\n[+] Launching interactive point selector...")
        
        W_plot, H_plot = 800, 600
        margin = 50
        active = np.ones(len(t), dtype=bool)
        
        min_t, max_t = t.min(), t.max()
        min_y, max_y = y.min(), y.max()
        span_t = max(max_t - min_t, 1e-3)
        span_y = max(max_y - min_y, 1e-3)
        
        def to_px(ti, yi):
            px = int(margin + (ti - min_t) / span_t * (W_plot - 2*margin))
            py = int(H_plot - margin - (yi - min_y) / span_y * (H_plot - 2*margin))
            return px, py
            
        pts_px = [to_px(t[i], y[i]) for i in range(len(t))]
        
        lasso_pts = []
        
        def on_mouse(event, mx, my, flags, param):
            nonlocal lasso_pts
            
            if event == cv2.EVENT_LBUTTONDOWN:
                lasso_pts = [(mx, my)]
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if (flags & cv2.EVENT_FLAG_LBUTTON) and lasso_pts:
                    if (mx - lasso_pts[-1][0])**2 + (my - lasso_pts[-1][1])**2 > 9:
                        lasso_pts.append((mx, my))
                        draw()
                        
            elif event == cv2.EVENT_LBUTTONUP:
                if not lasso_pts:
                    return
                
                is_drag = False
                if len(lasso_pts) > 5:
                    pts_arr = np.array(lasso_pts)
                    if pts_arr[:, 0].max() - pts_arr[:, 0].min() > 10 or pts_arr[:, 1].max() - pts_arr[:, 1].min() > 10:
                        is_drag = True
                        
                if is_drag:
                    cnt = np.array(lasso_pts, dtype=np.int32)
                    for i, p in enumerate(pts_px):
                        if active[i]:
                            if cv2.pointPolygonTest(cnt, (float(p[0]), float(p[1])), False) >= 0:
                                active[i] = False
                else:
                    best_idx = -1
                    best_dist = float('inf')
                    for i, (px, py) in enumerate(pts_px):
                        dist = (px - mx)**2 + (py - my)**2
                        if dist < 400: 
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = i
                    if best_idx != -1:
                        active[best_idx] = not active[best_idx]
                
                lasso_pts = []
                draw()

        win_name = "Select Points (Blue=Keep, Red=Discard)"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, on_mouse)
        
        def draw():
            img = np.ones((H_plot, W_plot, 3), dtype=np.uint8) * 40
            cv2.putText(img, "L-Click: Toggle point. L-Click & Drag: Lasso remove.", 
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, "Press 'q', 'Space', or 'Enter' when done.", 
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(img, "Plot: X-Axis = Time (t)  |  Y-Axis = Vertical Pos (y)", 
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            for i in range(len(t)):
                color = (255, 100, 50) if active[i] else (0, 0, 255)
                cv2.circle(img, pts_px[i], 6, color, -1)
                cv2.circle(img, pts_px[i], 6, (255,255,255), 1)
                
            if len(lasso_pts) > 1:
                pts = np.array(lasso_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                
            cv2.imshow(win_name, img)
            
        draw()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 13, 27, ord(' ')):
                break
                
        cv2.destroyWindow(win_name)

        t = t[active]
        x = x[active]
        y = y[active]

        print(f"[data] {len(t)} clean positions kept")
    else:
        from filterer import filter_projectile_ransac
        import numpy as np
        
        t_raw = np.array(det["times"])
        x_raw = np.array(x_m)
        y_raw = np.array(y_m)
        
        print("\n[+] Running RANSAC trajectory filter...")
        model_coeffs, inliers = filter_projectile_ransac(t_raw, y_raw)
        
        if model_coeffs is not None:
            t = t_raw[inliers]
            x = x_raw[inliers]
            y = y_raw[inliers]
            a, b, c = model_coeffs
            print(f"[data] RANSAC Found: y = {a:.2f}t^2 + {b:.2f}t + {c:.2f}")
            print(f"[data] {len(t)} true points kept (removed {len(t_raw)-len(t)} noise points)")
        else:
            print("[!] RANSAC failed to find valid trajectory, falling back to basic clean()")
            t, x, y = clean(det["times"], x_m, y_m)

    return t, x, y, {"det": det, "simulated": False}


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg, t_obs, x_obs, y_obs, meta, args, out_dir=None):
    out = out_dir or args.output
    os.makedirs(out, exist_ok=True)

    # Ground truth (simulation only)
    if meta.get("simulated") and "full" in meta:
        t_gt, x_gt, y_gt = meta["full"]
    else:
        t_gt, x_gt, y_gt = t_obs, x_obs, y_obs

    print(f"\n{'='*60}")
    print(f"  CARS PIPELINE  —  {cfg['display_name']}")
    print(f"{'='*60}")

    # ── Train all models ──────────────────────────────────────────────────────
    results = predict_all(
        cfg, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
        pinn_iters=args.pinn_iters,
        lstm_epochs=args.lstm_epochs,
        verbose=True,
    )

    # ── Stroke classification ─────────────────────────────────────────────────
    stroke_label = results["pinn"]["label"]
    kin          = results["pinn"]["kin"]
    pinn_model   = results["pinn"]["model"]

    # Calculate equation of motion from PINN
    x0 = x_obs.min()
    y0 = y_obs.min()
    vx0 = kin["vx0"]
    vy0 = kin["vy0"]
    G = 9.81
    eq_x = f"x(t) = {x0:.3f} + {vx0:.3f} * t"
    eq_y = f"y(t) = {y0:.3f} + {vy0:.3f} * t - {0.5*G:.3f} * t^2"

    print("\n" + "="*60)
    print("  PINN EQUATIONS OF MOTION (Approximated)")
    print("="*60)
    print(f"  {eq_x}")
    print(f"  {eq_y}")
    print("="*60)

    # ── Print metrics table ───────────────────────────────────────────────────
    print(f"\n  STROKE / PHASE  : {stroke_label}")
    print(f"  Speed V₀        : {kin['speed']:.2f} m/s")
    if cfg["has_spin"]:
        print(f"  Spin            : {kin['spin_rps']:.2f} rps")
    print(f"  Drag Cᴅ         : {kin['CD']:.3f}")
    print(f"\n  {'Model':<20} {'RMSE_x':>8} {'RMSE_y':>8} "
          f"{'ADE':>8} {'FDE':>8}")
    print(f"  {'-'*55}")
    for key, res in results.items():
        m = res.get("metrics", {})
        if m:
            print(f"  {key.upper():<20} "
                  f"{m.get('RMSE_x',np.nan):>8.4f} "
                  f"{m.get('RMSE_y',np.nan):>8.4f} "
                  f"{m.get('ADE',np.nan):>8.4f} "
                  f"{m.get('FDE',np.nan):>8.4f}")
    print(f"{'='*60}")

    # ── Save figures ──────────────────────────────────────────────────────────
    title = f"{cfg['display_name']}  ·  {stroke_label}"
    make_full_report(results, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
                     cfg, stroke_label,
                     save_path=os.path.join(out, "full_report.png"))

    plot_comparison(results, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
                    title=title,
                    save_path=os.path.join(out, "comparison.png"))

    plot_metrics(results,
                 save_path=os.path.join(out, "metrics.png"))

    plot_velocity(t_obs, x_obs, y_obs, kin=kin, pinn_model=pinn_model,
                  save_path=os.path.join(out, "velocity.png"))

    print(f"\n  Saved to: {os.path.abspath(out)}/")

    # ── Optional annotated video ──────────────────────────────────────────────
    if args.annotate_video and not meta.get("simulated") and "det" in meta:
        det  = meta["det"]
        span = max(det["xs"].max()-det["xs"].min(), 1.)
        scl  = span / cfg["scene_width_m"]
        pinn_model  = results["pinn"]["model"]
        # Re-build KF for video annotation
        dt_v = 1./det["fps"]
        kf_v = KalmanTracker(dt=dt_v)
        for xi, yi in zip(det["xs"], det["ys"]):
            xm = (xi-det["xs"].min())/scl
            ym = (det["height"]-yi)/scl
            kf_v.update(xm, ym)
        annotate_video(
            args.video,
            os.path.join(out, "annotated.mp4"),
            cfg, det, pinn_model, None, kf_v,
            scale_pxm=scl, height_px=det["height"],
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — all objects
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(args):
    print("\n" + "="*60)
    print("  CARS DEMO  —  All Objects Comparison")
    print("="*60)

    summary = []
    for key, cfg in OBJECTS.items():
        out = os.path.join(args.output, key)
        t, x, y, meta = load_sim(cfg, preset_name=None, noise=args.noise)
        quick = argparse.Namespace(
            pinn_iters=2000, lstm_epochs=300,
            annotate_video=False
        )
        res = run(cfg, t, x, y, meta, quick, out_dir=out)
        kin = res["pinn"]["kin"]
        summary.append({
            "name":  cfg["display_name"],
            "V0":    kin["speed"],
            "CD":    kin["CD"],
            "ADE":   res["pinn"]["metrics"]["ADE"],
            "label": res["pinn"]["label"],
        })

    print("\n" + "="*60)
    print("  SUMMARY — All Objects")
    print(f"  {'Object':<25} {'V0 m/s':>8} {'CD':>6} {'ADE m':>8}  Label")
    print("  " + "-"*60)
    for s in summary:
        print(f"  {s['name']:<25} {s['V0']:>8.2f} {s['CD']:>6.3f} "
              f"{s['ADE']:>8.4f}  {s['label']}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list:
        list_objects(); return

    print("\n" + "="*60)
    print("  CARS — Projectile Trajectory Prediction")
    print("  ML + Computer Vision  |  HEMRL / MNNIT Allahabad")
    print("="*60)

    if args.demo:
        run_demo(args); return

    cfg = get_object(args.object)

    if args.simulate or args.video is None:
        t, x, y, meta = load_sim(cfg, args.preset, noise=args.noise)
        if args.simulate and args.preset:
            args.output = os.path.join(args.output, f"sim_{args.preset}")
    else:
        t, x, y, meta = load_video(cfg, args)
        if args.video:
            base_name = os.path.splitext(os.path.basename(args.video))[0]
            args.output = os.path.join(args.output, base_name)

    run(cfg, t, x, y, meta, args)
    print("\nDone! All outputs saved to:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
