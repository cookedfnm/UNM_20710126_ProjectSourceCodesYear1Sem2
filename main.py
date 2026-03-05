import time
import new_movement
import feed

def clamp(v, mn, mx):
    return mn if v < mn else mx if v > mx else v

def sign(n):
    return (n > 0) - (n < 0)

# ---------------- ROBOT SETTINGS ----------------

LEFT_BASE_SPEED = 0.70
RIGHT_BASE_SPEED = 0.70

# PID constants
KP = 1.2
KI = 0.001
KD = 0.2

# Anti-windup clamp
I_CLAMP = 2.0

# When line is lost
SEARCH_TURN = 0.75

# Small error zone
DEADBAND = 0.02

print("SPACE = start/pause")
print("Q or ESC = quit (emergency stop)")

# ---------------- INIT ----------------

feed.init_camera()

running = False

# PID state
total_error = 0.0
last_error = 0.0
first = True

try:
    while True:
        loop_start = time.perf_counter()

        found_line, raw_error_px = feed.get_frame_with_overlay(running)

        key = feed.cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break

        if key == 32:
            running = not running
            total_error = 0.0
            last_error = 0.0
            first = True
            print("Running:", running)

        if not running:
            new_movement.move(0, 0)
            continue

        dt = time.perf_counter() - loop_start
        if dt <= 0:
            dt = 1e-3

        # ---------------- PID ----------------
        if found_line:

            error = raw_error_px / 320.0

            # smoothing filter
            error = 0.7 * last_error + 0.3 * error

            if abs(error) < DEADBAND:
                error = 0.0

            total_error += error * dt
            total_error = clamp(total_error, -I_CLAMP, I_CLAMP)

            if first:
                derivative = 0.0
                first = False
            else:
                derivative = (error - last_error) / dt

            pid = KP * error + KI * total_error + KD * derivative
            last_error = error

        else:
            pid = sign(last_error) * SEARCH_TURN
            total_error *= 0.8

        # ---------------- MOTOR MIXING ----------------

        left = LEFT_BASE_SPEED - pid
        right = RIGHT_BASE_SPEED + pid

        left = clamp(left, -1.0, 1.0)
        right = clamp(right, -1.0, 1.0)

        new_movement.move(left, right)

except KeyboardInterrupt:
    pass

finally:
    new_movement.move(0, 0)

    try:
        new_movement.pi.stop()
    except:
        pass

    try:
        feed.close_camera()
    except:
        pass
