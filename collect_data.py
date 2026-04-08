"""
collect_data.py — Run this on your Pi5 to collect training photos.

Usage:
    python3 collect_data.py

Controls:
    S         = save current frame to the current class folder
    N         = next class
    P         = previous class
    Q / ESC   = quit

Photos are saved to dataset/<class_name>/ automatically.
Aim for 80-150 photos per class with varied angles, distances, lighting.
"""

from picamera2 import Picamera2
import cv2
import os
import time

CLASS_NAMES = [
    '0_arrow_right', '1_arrow_left', '2_arrow_forwards', '3_arrow_backwards',
    '4_octagon', '5_star', '6_circular_segment', '7_pacman',
    '8_trapezoid', '9_diamond', '10_cross', '11_recycle',
    '12_warning', '13_press_button', '14_qr_code', '15_fingerprint'
]

DATASET_DIR = 'dataset'
os.makedirs(DATASET_DIR, exist_ok=True)
for name in CLASS_NAMES:
    os.makedirs(os.path.join(DATASET_DIR, name), exist_ok=True)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)

cv2.namedWindow("Collect Data")
cv2.moveWindow("Collect Data", 0, 0)

current_class = 0

def count_images(class_name):
    path = os.path.join(DATASET_DIR, class_name)
    return len([f for f in os.listdir(path) if f.endswith('.jpg')])

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        class_name = CLASS_NAMES[current_class]
        n_imgs     = count_images(class_name)

        # Overlay
        cv2.putText(frame, f"Class {current_class}: {class_name}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Images saved: {n_imgs}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "S=save  N=next  P=prev  Q=quit",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        # Draw guide box for where to hold the sign
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 3
        bw, bh = 200, 200
        cv2.rectangle(frame,
                      (cx - bw//2, cy - bh//2),
                      (cx + bw//2, cy + bh//2),
                      (0, 200, 255), 2)
        cv2.putText(frame, "Hold sign here",
                    (cx - 70, cy - bh//2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            timestamp = int(time.time() * 1000)
            fname = os.path.join(DATASET_DIR, class_name, f"{timestamp}.jpg")
            cv2.imwrite(fname, frame)
            print(f"Saved: {fname}  (total: {n_imgs + 1})")
        elif key == ord('n'):
            current_class = (current_class + 1) % len(CLASS_NAMES)
            print(f"→ Class: {CLASS_NAMES[current_class]}")
        elif key == ord('p'):
            current_class = (current_class - 1) % len(CLASS_NAMES)
            print(f"→ Class: {CLASS_NAMES[current_class]}")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("\nCollection complete. Copy the dataset/ folder to your laptop for training.")
    for name in CLASS_NAMES:
        print(f"  {name}: {count_images(name)} images")
