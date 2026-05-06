import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def filter_projectile_ransac(t, y, iterations=1500, threshold=0.5):
    """
    Isolates parabolic projectile data (y vs t) from heavy noise using RANSAC.
    Enforces a physics constraint: the parabola must open downwards (a < 0).
    """
    best_inliers = []
    best_model = None
    max_inlier_count = 0
    n_points = len(t)

    for _ in range(iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        t_sample, y_sample = t[idx], y[idx]

        with np.errstate(all='ignore'):
            coeffs = np.polyfit(t_sample, y_sample, 2)
        
        a, b, c = coeffs

        if a >= 0:
            continue

        y_calc = a * (t**2) + b * t + c
        errors = np.abs(y - y_calc)
        current_inliers = np.where(errors <= threshold)[0]

        if len(current_inliers) > max_inlier_count:
            max_inlier_count = len(current_inliers)
            best_inliers = current_inliers
            best_model = coeffs

    return best_model, best_inliers

if __name__ == "__main__":
    # ==========================================
    # 1. COMMAND LINE ARGUMENT PARSING
    # ==========================================
    parser = argparse.ArgumentParser(description="Filter projectile trajectory from noisy CSV data using RANSAC.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Inlier distance tolerance in meters. Default: 0.5")
    parser.add_argument("--iterations", type=int, default=1500, help="Number of RANSAC iterations. Default: 1500")
    
    args = parser.parse_args()

    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    try:
        df = pd.read_csv(args.csv_path)
        t = df['t'].values
        y = df['y_meters'].values
    except FileNotFoundError:
        print(f"Error: The file '{args.csv_path}' was not found.")
        sys.exit(1)
    except KeyError:
        print("Error: The CSV must contain columns named exactly 't' and 'y_meters'.")
        sys.exit(1)

    # ==========================================
    # 3. EXECUTE RANSAC
    # ==========================================
    print(f"Running RANSAC on {args.csv_path}...")
    print(f"Parameters: {args.iterations} iterations, {args.threshold}m threshold")
    
    model_coeffs, inlier_indices = filter_projectile_ransac(t, y, iterations=args.iterations, threshold=args.threshold)

    if model_coeffs is None:
        print("Failed to find a valid downward parabolic trajectory.")
        sys.exit(1)

    a, b, c = model_coeffs
    print(f"\nBest Model Found: y = {a:.2f}t² + {b:.2f}t + {c:.2f}")
    print(f"Filtered out {len(t) - len(inlier_indices)} noise points.")
    print(f"Retained {len(inlier_indices)} true trajectory points.")

    # ==========================================
    # 4. VISUALIZE BEFORE AND AFTER
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.scatter(t, y, s=10, color='gray', alpha=0.7)
    ax1.set_title("Before: Raw Data (y vs t)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("y_meters")
    ax1.grid(True, linestyle='--', alpha=0.5)

    noise_mask = np.ones(len(t), dtype=bool)
    noise_mask[inlier_indices] = False
    
    ax2.scatter(t[noise_mask], y[noise_mask], s=10, color='lightgray', alpha=0.3, label='Noise')
    ax2.scatter(t[inlier_indices], y[inlier_indices], s=15, color='blue', label='Extracted Trajectory')

    t_line = np.linspace(min(t), max(t), 100)
    y_line = a * (t_line**2) + b * t_line + c
    ax2.plot(t_line, y_line, color='red', linewidth=2, label='RANSAC Fitted Parabola')

    ax2.set_title("After: Isolated Trajectory")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()