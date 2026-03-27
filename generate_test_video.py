import cv2
import numpy as np
import os

width, height = 1280, 720
fps = 30
duration = 1.5
num_frames = int(fps * duration)

# create test video
output_path = 'test_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

x0, y0 = 100, 100
v_x = (width - 200) / duration
v_y = 900
g = 1200

# Moving background pattern or static
bg = np.zeros((height, width, 3), dtype=np.uint8)
# add some static noise or gradient to background for realism? Not strictly needed.

for i in range(num_frames):
    t = i / fps
    x = int(x0 + v_x * t)
    y = int(y0 + v_y * t - 0.5 * g * t**2)
    
    y_draw = height - y
    
    frame = bg.copy()
    
    # Draw a yellow tennis ball (BGR = 0, 255, 255)
    cv2.circle(frame, (x, y_draw), 20, (0, 255, 255), -1)
    
    out.write(frame)

out.release()
print(f"Generated {output_path}")
