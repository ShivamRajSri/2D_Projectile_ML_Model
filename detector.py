"""
detector.py
-----------
OpenCV-based detector for ANY round object in video.

Supports:
  - HSV colour thresholding (multiple colours tried simultaneously)
  - MOG2 background subtraction
  - Circularity filtering
  - Velocity-based outlier removal
  - Annotated video output
"""

import cv2
import numpy as np
from config import COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_frame(frame, color_keys, min_r=3, max_r=80, bg_sub=None):
    """
    Detect the most circular blob matching any of the given colours.
    Returns (cx, cy, radius) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Combine masks for all candidate colours
    if "any" in color_keys or not color_keys:
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    else:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for ck in color_keys:
            if ck in COLORS:
                cr   = COLORS[ck]
                m    = cv2.inRange(hsv, cr["lower"], cr["upper"])
                mask = cv2.bitwise_or(mask, m)

    # Foreground gating
    if bg_sub is not None:
        fg   = bg_sub.apply(frame)
        fg   = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(mask, fg)

    # 1. Aggressive Morphology
    # A larger kernel for MORPH_OPEN will erase scattered 1-2 pixel noise.
    # A very large kernel for MORPH_CLOSE will fuse the ball into a solid mass.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best, best_score = None, -1
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        
        # 2. Strict Area Filtering
        # A tiny ball is usually 5 to 50 pixels. A human body/arm is 500+ pixels.
        if area < 1 or area > 300: 
            continue
            
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if not (min_r <= r <= max_r):
            continue
        perim = cv2.arcLength(cnt, True)
        circ  = (4 * np.pi * area) / (perim**2 + 1e-6)
        if circ > best_score:
            best_score = circ
            best = (float(cx), float(cy), float(r))

    return best if (best and best_score > 0.6) else None


# ─────────────────────────────────────────────────────────────────────────────
# Full video trajectory extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectory(video_path, cfg,
                       color_override=None,
                       use_bg_sub=True,
                       max_frames=None,
                       annotate_output=None):
    """
    Process a video and return the detected object trajectory.

    Parameters
    ----------
    video_path      : str   – input video path
    cfg             : dict  – object config from config.py
    color_override  : list  – override colour keys
    use_bg_sub      : bool  – MOG2 background subtraction
    max_frames      : int   – max frames to process
    annotate_output : str   – if given, write annotated video here

    Returns
    -------
    dict:
      times, xs, ys : detected positions (pixel)
      fps, width, height
      frame_indices
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames: total = min(total, max_frames)

    colors  = color_override or cfg["colors"]
    min_r   = cfg.get("min_radius_px", 3)
    max_r   = cfg.get("max_radius_px", 80)
    bg_sub  = cv2.createBackgroundSubtractorMOG2(
                  history=500, varThreshold=16, detectShadows=False
              ) if use_bg_sub else None

    writer = None
    if annotate_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotate_output, fourcc, fps, (W, H))

    times, xs, ys, fids = [], [], [], []
    trail = []     # last N positions for trail drawing
    idx = 0

    print(f"[detector] {cfg['display_name']}  |  "
          f"{W}x{H}@{fps:.0f}fps  |  colours: {colors}")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and idx >= max_frames):
            break

        det = detect_frame(frame, colors, min_r, max_r, bg_sub)
        if det:
            cx, cy, r = det
            times.append(idx / fps)
            xs.append(cx); ys.append(cy)
            fids.append(idx)
            trail.append((int(cx), int(cy)))
            if len(trail) > 40: trail.pop(0)

            if writer:
                # Draw trail
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    cv2.line(frame, trail[i-1], trail[i],
                             (int(255*alpha), int(100*(1-alpha)), 50), 2)
                # Detection circle
                cv2.circle(frame, (int(cx), int(cy)), int(r), (0,255,0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 3,      (0,0,255), -1)
                cv2.putText(frame, cfg["display_name"],
                            (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if writer:
            cv2.putText(frame, f"Frame {idx}  |  Det: {len(times)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,255), 1)
            writer.write(frame)

        idx += 1
        if idx % 200 == 0:
            pct = 100*len(times)/max(idx,1)
            print(f"  {idx}/{total}  |  detections: {len(times)} ({pct:.0f}%)")

    cap.release()
    if writer: writer.release()

    rate = 100*len(times)/max(idx,1)
    print(f"[detector] Done  →  {len(times)} positions  "
          f"({rate:.1f}% detection rate)")
    return {
        "times": np.array(times), "xs": np.array(xs), "ys": np.array(ys),
        "fps": fps, "width": W, "height": H,
        "frame_indices": np.array(fids, dtype=int),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pixel → metre conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_meters(xs, ys, height_px, scene_width_m, scale_override=None):
    """Convert pixel coordinates to metres."""
    if scale_override:
        scale = scale_override
    else:
        span  = max(xs.max() - xs.min(), 1.0)
        scale = span / scene_width_m          # px/m

    x_m = (xs - xs.min()) / scale
    y_m = (height_px - ys) / scale            # flip y
    y_m = y_m - y_m.min()
    return x_m, y_m


# ─────────────────────────────────────────────────────────────────────────────
# Outlier removal
# ─────────────────────────────────────────────────────────────────────────────

def clean(t, x, y, iqr_factor=2.5):
    """IQR + velocity-based outlier filter."""
    def iqr_ok(arr):
        q1, q3 = np.percentile(arr, [10, 90])
        d = iqr_factor * (q3 - q1)
        return (arr >= q1 - d) & (arr <= q3 + d)

    mask = iqr_ok(x) & iqr_ok(y)

    if mask.sum() > 3:
        xt, yt, tt = x[mask], y[mask], t[mask]
        dt  = np.diff(tt) + 1e-9
        spd = np.sqrt(np.diff(xt)**2 + np.diff(yt)**2) / dt
        med = np.median(spd)
        ok  = np.concatenate([[True], spd < med * 10])
        idx = np.where(mask)[0][ok]
        m2  = np.zeros(len(t), bool)
        m2[idx] = True
        mask = m2

    return t[mask], x[mask], y[mask]
