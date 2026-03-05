from picamera2 import Picamera2
import cv2
import time

picam2 = None

def init_camera():
    global picam2

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480)},
        controls={"FrameRate": 60}
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(1)

    cv2.namedWindow("Line Follow PID")
    cv2.namedWindow("Mask (ROI)")


def get_frame_with_overlay(running):

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    H, W = frame.shape[:2]
    frame_center = W // 2

    roi = frame[H//2:H, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    _, mask = cv2.threshold(blur,100,255,cv2.THRESH_BINARY_INV)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    found_line = False
    error = 0

    if contours:

        largest = max(contours,key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 1000:

            x,y,w_box,h_box = cv2.boundingRect(largest)

            line_center = x + w_box//2
            error = frame_center - line_center
            found_line = True

            y_full = y + H//2

            cv2.rectangle(frame,(x,y_full),
                          (x+w_box,y_full+h_box),
                          (0,255,0),2)

            cv2.circle(frame,
                       (line_center,y_full+h_box//2),
                       6,(0,0,255),-1)

    # overlays
    cv2.line(frame,(frame_center,0),(frame_center,H),(255,255,0),2)

    status = "RUNNING" if running else "PAUSED"
    line_state = "LINE FOUND" if found_line else "SEARCHING"

    cv2.putText(frame,f"{status} - {line_state}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,(0,255,0),2)

    cv2.imshow("Line Follow PID",frame)
    cv2.imshow("Mask (ROI)",mask)

    return found_line, error


def close_camera():
    picam2.stop()
    cv2.destroyAllWindows()
