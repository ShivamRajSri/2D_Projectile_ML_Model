import cv2
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize MOG2 background subtraction.")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--history", type=int, default=500, help="MOG2 history (default: 500)")
    parser.add_argument("--varThreshold", type=int, default=16, help="MOG2 varThreshold (default: 16)")
    parser.add_argument("--detectShadows", action="store_true", help="Enable shadow detection")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open {args.video}")
        return

    # Initialize MOG2 with the exact parameters from your detector
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=args.history, 
        varThreshold=args.varThreshold, 
        detectShadows=args.detectShadows
    )

    print(f"[*] Playing video: {args.video}")
    print(f"[*] MOG2 Params -> History: {args.history}, VarThreshold: {args.varThreshold}")
    print("[*] Press 'SPACE' to pause/play")
    print("[*] Press 'q' or 'ESC' to quit")

    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            # 1. Apply MOG2 to get the raw foreground mask
            fg_mask = bg_sub.apply(frame)

            # 2. Replicate the exact thresholding and morphology from detector.py
            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=2)

            # 3. Create side-by-side view (Original | MOG2 Mask | Cleaned Mask)
            # Convert masks to BGR so we can concatenate them with the original color frame
            fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

            # Add labels
            cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(fg_bgr, "Raw MOG2 Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(cleaned_bgr, "After Morphology", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Combine horizontally
            top_row = np.hstack((frame, fg_bgr))
            bottom_row = np.hstack((cleaned_bgr, np.zeros_like(frame))) # padding
            
            # Since videos can be 1080p, shrink it so it fits on screen
            h, w = frame.shape[:2]
            scale = min(1200 / (w * 2), 800 / h)
            
            combined = np.hstack((frame, fg_bgr, cleaned_bgr))
            display = cv2.resize(combined, (0,0), fx=scale, fy=scale)
            
            cv2.imshow("MOG2 Debugger", display)

        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key in (ord('q'), 27): # q or ESC
            break
        elif key == ord(' '): # Space
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
