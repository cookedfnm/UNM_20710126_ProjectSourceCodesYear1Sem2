import cv2
from picamera2 import Picamera2
import time
import numpy as np
import movement
import sys

def clamp(value, mn, mx):
    if value > mx:
        return mx
    elif value < mn:
        return mn
    return value

def getSign(n):
    return (n > 0) - (n < 0)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])  # expects ~0.4 to 0.7
        kp = float(sys.argv[2])
        ki = float(sys.argv[3])
        kd = float(sys.argv[4])
    else:
        raise Exception("Usage: python3 line_follow_friend_style.py <base_speed> <kp> <ki> <kd>")

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

error = 0.0
total_error = 0.0
last_error = 0.0
diff_error = 0.0
first = True

# anti-windup clamp (helps a lot)
I_CLAMP = 2.0

while True:
    try:
        time_marker = time.perf_counter()

        frame = picam2.capture_array()

        roi = frame[240:480, :]  # bottom half

        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(
            imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)

        if len(contours) > 0:
            contour_areas = [cv2.contourArea(cnt) for cnt in contours]
            filtered_contours = []
            filtered_contour_areas = []

            for i, a in enumerate(contour_areas):
                if a >= 7500 and a <= 40000:
                    filtered_contours.append(contours[i])
                    filtered_contour_areas.append(a)

            if len(filtered_contours) > 0 and ret < 180:
                if len(filtered_contours) > 1:
                    sorted_pairs = sorted(zip(filtered_contour_areas, filtered_contours), reverse=True)
                    _, sorted_contours = zip(*sorted_pairs)
                    line_contour = sorted_contours[0]
                    cv2.drawContours(im2, sorted_contours[1:], -1, (255, 255, 255), thickness=cv2.FILLED)
                else:
                    line_contour = filtered_contours[0]

                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

                M = cv2.moments(line_contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                elapsed_time = time.perf_counter() - time_marker
                if elapsed_time <= 0:
                    elapsed_time = 0.0001

                # normalized error [-1,1]
                error = (320 - cx) / 320.0

                total_error += error * elapsed_time
                total_error = clamp(total_error, -I_CLAMP, I_CLAMP)

                if not first:
                    diff_error = (error - last_error) / elapsed_time
                else:
                    first = False

                pid = kp * error + ki * total_error + kd * diff_error
                last_error = error

            else:
                # line lost -> turn toward last seen direction
                pid = getSign(last_error) * 2.0
                total_error *= 0.8  # bleed integral when lost (helps recover)

            left_pwm = base_speed + pid
            right_pwm = base_speed - pid

            clamped_left_pwm = clamp(left_pwm, -1, 1)
            clamped_right_pwm = clamp(right_pwm, -1, 1)

            movement.move(clamped_left_pwm, clamped_right_pwm)

        else:
            # no contours at all -> just keep turning toward last direction
            pid = getSign(last_error) * 2.0
            movement.move(clamp(base_speed + pid, -1, 1), clamp(base_speed - pid, -1, 1))

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error occurred - {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()