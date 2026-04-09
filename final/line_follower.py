"""
Line-following process with PID control and colour state machine.
Reads frames from shared memory, writes motor commands to shared state.
"""

import time
import numpy as np
import cv2
from shared_state import (
    FRAME_W, FRAME_H, open_frame_buffer,
    MODE_SEARCH, MODE_BLACK, MODE_RED, MODE_YELLOW, MODE_LOOK_RED, MODE_LOOK_YELLOW,
)

# ── HSV ranges (camera-specific) ──
RED_LO = np.array([108, 140, 120], np.uint8)
RED_HI = np.array([134, 255, 255], np.uint8)
YEL_LO = np.array([80, 180, 140], np.uint8)
YEL_HI = np.array([106, 255, 255], np.uint8)
BLACK_THRESH = 100

# ── Area thresholds ──
MIN_COLOUR = 500
MIN_BLACK = 1000
MIN_LOOKAHEAD = 200

# ── PID tuning ──
BASE_SPEED = 0.3
APPROACH_SPEED = 0.25
COLOUR_SPEED = 0.25
EXIT_SPEED = 0.2

KP = 0.10
KI = 0.0001
KD = 0.10
I_CLAMP = 1.0
D_FILTER = 0.7
DEADBAND = 0.03
SEARCH_TURN = 0.35
EXIT_BIAS_STR = 0.6
MIN_COLOUR_SECS = 2.0
COLOUR_LOST_MAX = 20

# ── ROI slicing ──
ROI_MAIN_TOP = FRAME_H // 3
ROI_LOOK_TOP = FRAME_H // 4
ROI_LOOK_BOT = FRAME_H // 3
FRAME_CX = FRAME_W // 2

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _largest_contour(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    return (c, a) if a >= min_area else (None, 0)


def _colour_detect(hsv, min_area):
    """Returns (mode_int, contour, area) for best colour or (None, None, 0)."""
    mr = cv2.morphologyEx(cv2.inRange(hsv, RED_LO, RED_HI), cv2.MORPH_OPEN, KERNEL)
    my = cv2.morphologyEx(cv2.inRange(hsv, YEL_LO, YEL_HI), cv2.MORPH_OPEN, KERNEL)
    cr, ar = _largest_contour(mr, min_area)
    cy, ay = _largest_contour(my, min_area)
    if cr is None and cy is None:
        return None, None, 0
    if ar >= ay:
        return MODE_RED, cr, ar
    return MODE_YELLOW, cy, ay


def _centroid_error(contour):
    x, _, w, _ = cv2.boundingRect(contour)
    cx = x + w // 2
    return (FRAME_CX - cx) / float(FRAME_CX)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _sign(v):
    return (v > 0) - (v < 0)


def line_follower_process(
    frame_seq, frame_lock, running, quit_flag,
    motor_left, motor_right, line_found, line_error, line_mode,
):
    shm, frame_buf = open_frame_buffer(writable=False)
    print("[LINE] started")

    # PID state
    integral = 0.0
    prev_err = 0.0
    prev_deriv = 0.0
    first_tick = True

    # State machine
    ST_BLACK, ST_LOOK, ST_COLOUR, ST_EXIT = range(4)
    state = ST_BLACK
    colour_lost = 0
    entry_dir = 0.0
    exit_bias = 0.0
    colour_start = 0.0
    last_seq = 0
    t_prev = time.perf_counter()

    try:
        while not quit_flag.value:
            # Wait for new frame
            seq = frame_seq.value
            if seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq

            # Snapshot the frame under lock
            with frame_lock:
                frame = frame_buf.copy()

            if not running.value:
                motor_left.value = 0.0
                motor_right.value = 0.0
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            dt = max(now - t_prev, 1e-3)
            t_prev = now

            # ── Image processing ──
            roi_main = frame[ROI_MAIN_TOP:, :]
            roi_look = frame[ROI_LOOK_TOP:ROI_LOOK_BOT, :]

            hsv_main = cv2.cvtColor(cv2.GaussianBlur(roi_main, (7, 7), 0), cv2.COLOR_BGR2HSV)
            hsv_look = cv2.cvtColor(cv2.GaussianBlur(roi_look, (5, 5), 0), cv2.COLOR_BGR2HSV)

            # Colour in main ROI
            cm, cc, _ = _colour_detect(hsv_main, MIN_COLOUR)
            # Black in main ROI
            gray = cv2.cvtColor(cv2.GaussianBlur(roi_main, (7, 7), 0), cv2.COLOR_BGR2GRAY)
            bm = cv2.threshold(gray, BLACK_THRESH, 255, cv2.THRESH_BINARY_INV)[1]
            bc, _ = _largest_contour(bm, MIN_BLACK)
            # Colour in lookahead
            lm, lc, _ = _colour_detect(hsv_look, MIN_LOOKAHEAD)

            # ── Determine detected mode and error ──
            det_mode = MODE_SEARCH
            found = False
            err = 0.0

            if cm is not None:
                det_mode = cm
                err = _centroid_error(cc)
                found = True
            elif lm is not None:
                det_mode = MODE_LOOK_RED if lm == MODE_RED else MODE_LOOK_YELLOW
                err = _centroid_error(lc)
                found = True
            elif bc is not None:
                det_mode = MODE_BLACK
                err = _centroid_error(bc)
                found = True

            colour_visible = det_mode in (MODE_RED, MODE_YELLOW, MODE_LOOK_RED, MODE_LOOK_YELLOW)

            # ── State transitions ──
            if state == ST_BLACK:
                if det_mode in (MODE_RED, MODE_YELLOW):
                    state = ST_COLOUR
                    colour_lost = 0
                    entry_dir = _sign(err) if abs(err) > DEADBAND else 0.0
                    colour_start = now
                    integral = prev_err = prev_deriv = 0.0
                    first_tick = True
                elif det_mode in (MODE_LOOK_RED, MODE_LOOK_YELLOW):
                    state = ST_LOOK

            elif state == ST_LOOK:
                if det_mode in (MODE_RED, MODE_YELLOW):
                    state = ST_COLOUR
                    colour_lost = 0
                    entry_dir = _sign(err) if abs(err) > DEADBAND else 0.0
                    colour_start = now
                    integral = prev_err = prev_deriv = 0.0
                    first_tick = True
                elif det_mode == MODE_BLACK:
                    state = ST_BLACK

            elif state == ST_COLOUR:
                in_colour_long = (now - colour_start) >= MIN_COLOUR_SECS
                if in_colour_long and det_mode == MODE_BLACK:
                    exit_bias = entry_dir * EXIT_BIAS_STR
                    state = ST_EXIT
                    entry_dir = 0.0
                elif colour_visible:
                    colour_lost = 0
                else:
                    colour_lost += 1

            elif state == ST_EXIT:
                if det_mode == MODE_BLACK:
                    state = ST_BLACK
                    exit_bias = 0.0
                    integral = prev_err = prev_deriv = 0.0
                    first_tick = True

            # ── Speed ──
            speed = {ST_BLACK: BASE_SPEED, ST_LOOK: APPROACH_SPEED,
                     ST_COLOUR: COLOUR_SPEED, ST_EXIT: EXIT_SPEED}[state]

            # ── PID / motor calc ──
            if state == ST_EXIT:
                pid = exit_bias
            elif state == ST_COLOUR and not colour_visible:
                pid = entry_dir * SEARCH_TURN
            elif found:
                smoothed = 0.7 * prev_err + 0.3 * err
                if abs(smoothed) < DEADBAND:
                    smoothed = 0.0
                integral = _clamp(integral + smoothed * dt, -I_CLAMP, I_CLAMP)
                raw_d = 0.0 if first_tick else (smoothed - prev_err) / dt
                deriv = D_FILTER * prev_deriv + (1 - D_FILTER) * raw_d
                pid = KP * smoothed + KI * integral + KD * deriv
                prev_err = smoothed
                prev_deriv = deriv
                first_tick = False
            else:
                pid = _sign(prev_err) * SEARCH_TURN
                integral *= 0.8
                prev_deriv = 0.0

            left = _clamp(speed - pid, -1.0, 1.0)
            right = _clamp(speed + pid, -1.0, 1.0)

            motor_left.value = left
            motor_right.value = right
            line_found.value = found
            line_error.value = err
            line_mode.value = det_mode

    except KeyboardInterrupt:
        pass
    finally:
        motor_left.value = 0.0
        motor_right.value = 0.0
        shm.close()
        print("[LINE] stopped")
