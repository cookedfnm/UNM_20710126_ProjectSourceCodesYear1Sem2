from picamera2 import Picamera2
import cv2
import time
import numpy as np

# --------------------------------------------------
# HSV Value Checker for Raspberry Pi Camera
# - Shows BGR camera feed
# - Converts to HSV
# - Reads HSV at centre crosshair
# - Press 's' to print current centre HSV
# - Press 'q' or ESC to quit
# --------------------------------------------------

def main():
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (640, 480)},
        controls={"FrameRate": 30}
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    cv2.namedWindow("HSV Checker")

    try:
        while True:
            frame = picam2.capture_array()

            # Picamera2 often gives BGRA
            if frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame.copy()

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            h, w = frame_bgr.shape[:2]
            cx, cy = w // 2, h // 2

            hsv_pixel = hsv[cy, cx]
            h_val, s_val, v_val = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

            # Optional conversions for easier comparison with colour pickers
            hue_deg = h_val * 2
            sat_pct = (s_val / 255.0) * 100.0
            val_pct = (v_val / 255.0) * 100.0

            display = frame_bgr.copy()

            # Crosshair
            cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
            cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

            # Text
            cv2.putText(
                display,
                f"OpenCV HSV: H={h_val} S={s_val} V={v_val}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Picker style: H={hue_deg}deg S={sat_pct:.1f}% V={val_pct:.1f}%",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                "Center pixel only | Press S to print | Q or ESC to quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow("HSV Checker", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("---- SAMPLE ----")
                print(f"OpenCV HSV : [{h_val}, {s_val}, {v_val}]")
                print(f"Picker HSV : {hue_deg} deg, {sat_pct:.1f}%, {val_pct:.1f}%")
                print("----------------")

            if key in (27, ord('q')):
                break

    finally:
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
