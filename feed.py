from picamera2 import Picamera2
import cv2
import numpy as np
import time

picam2 = None

# ─────────────────────────────────────────────────────────────────
#  HSV COLOUR RANGES
#  Your camera shifts real colours: real-red → H≈121, real-yellow → H≈93
# ─────────────────────────────────────────────────────────────────

RED_LOWER    = np.array([108, 140, 120], dtype=np.uint8)
RED_UPPER    = np.array([134, 255, 255], dtype=np.uint8)

YELLOW_LOWER = np.array([80,  180, 140], dtype=np.uint8)
YELLOW_UPPER = np.array([106, 255, 255], dtype=np.uint8)

BLACK_THRESHOLD = 100

# ── Area thresholds ───────────────────────────────────────────────
# Middle-ground starting values for the full-sensor FOV with the camera
# physically lowered to compensate. Tune up if you get false positives,
# tune down if line/colour detection drops out.
#
# Reference points:
#   Original cropped FOV:  COLOUR=500  BLACK=1000  LOOKAHEAD=200
#   Full FOV (camera not lowered): COLOUR=250  BLACK=500   LOOKAHEAD=100
MIN_COLOUR_AREA    = 350
MIN_BLACK_AREA     = 750
MIN_LOOKAHEAD_AREA = 150


def init_camera():
    global picam2
    picam2 = Picamera2()

    # ── FOV config: must match collect_data.py ─────────────────
    # Include a `raw` stream at the FULL sensor resolution so Picamera2
    # uses the entire sensor area as the source for the downscaled `main`
    # stream — instead of the default behaviour of cropping to a smaller
    # sensor sub-region.
    #
    # This MUST match collect_data.py so the trained model sees the
    # same FOV at inference time as it did during training.
    full_w, full_h = picam2.sensor_resolution
    print(f"[feed] sensor resolution: {full_w}x{full_h}")

    config = picam2.create_preview_configuration(
        main={"size": (640, 480)},
        raw={"size": (full_w, full_h)},
        controls={"FrameRate": 60},
    )
    picam2.configure(config)
    print(f"[feed] full sensor area → downscaled to 640x480")

    picam2.start()
    time.sleep(1)

    # ── Windows — tiled left to right ───────────────────────────
    cv2.namedWindow("Line Follow PID")
    cv2.moveWindow("Line Follow PID", 0, 0)

    cv2.namedWindow("Mask (ROI)")
    cv2.moveWindow("Mask (ROI)", 660, 0)

    cv2.namedWindow("Lookahead")
    cv2.moveWindow("Lookahead", 660, 340)


def _colour_contour(hsv_region, min_area):
    """
    Run red and yellow detection on an HSV region.
    Returns (mode, contour, area) for whichever colour wins, or ('none', None, 0).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_r = cv2.inRange(hsv_region, RED_LOWER,    RED_UPPER)
    mask_y = cv2.inRange(hsv_region, YELLOW_LOWER, YELLOW_UPPER)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel)

    def best(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        return (c, a) if a >= min_area else (None, 0)

    c_r, a_r = best(mask_r)
    c_y, a_y = best(mask_y)

    if c_r is None and c_y is None:
        return 'none', None, 0, mask_r, mask_y
    if c_r is not None and c_y is not None:
        if a_r >= a_y:
            return 'red',    c_r, a_r, mask_r, mask_y
        else:
            return 'yellow', c_y, a_y, mask_r, mask_y
    if c_r is not None:
        return 'red',    c_r, a_r, mask_r, mask_y
    return     'yellow', c_y, a_y, mask_r, mask_y


def get_frame_with_overlay(running):
    """
    Returns
    -------
    found_line  : bool
    error       : float  normalised [-1, +1], positive = line left of centre
    active_mode : str    'red' | 'yellow' | 'black' | 'lookahead_red' |
                         'lookahead_yellow' | 'search'
    """
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    H, W = frame.shape[:2]
    frame_center = W // 2

    # ── ROI layout ───────────────────────────────────────────────
    #
    #   0          ┌──────────────────────┐
    #              │   (ignored)          │
    #   H//4       ├──────────────────────┤  ← lookahead top
    #              │  LOOKAHEAD STRIP     │  colour-only, early warning
    #   H//3       ├──────────────────────┤  ← lookahead bottom / main ROI top
    #              │                      │
    #              │  MAIN ROI            │  colour + black PID tracking
    #              │                      │
    #   H          └──────────────────────┘
    #
    roi_main      = frame[H // 3 : H,       :]
    roi_lookahead = frame[H // 4 : H // 3,  :]

    # ── Pre-process main ROI ──────────────────────────────────────
    blurred_main = cv2.GaussianBlur(roi_main, (7, 7), 0)
    hsv_main     = cv2.cvtColor(blurred_main, cv2.COLOR_BGR2HSV)

    # ── Pre-process lookahead strip ───────────────────────────────
    blurred_look = cv2.GaussianBlur(roi_lookahead, (5, 5), 0)
    hsv_look     = cv2.cvtColor(blurred_look, cv2.COLOR_BGR2HSV)

    # ── Colour detection — main ROI ───────────────────────────────
    colour_mode, colour_cont, colour_area, mr, my = _colour_contour(hsv_main, MIN_COLOUR_AREA)

    # ── Black line — main ROI ─────────────────────────────────────
    gray = cv2.cvtColor(blurred_main, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    def best_black(mask, min_area):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        return (c, a) if a >= min_area else (None, 0)

    c_black, a_black = best_black(mask_black, MIN_BLACK_AREA)

    # ── Lookahead colour detection ────────────────────────────────
    look_mode, look_cont, look_area, lr, ly = _colour_contour(hsv_look, MIN_LOOKAHEAD_AREA)

    # ── Priority decision ─────────────────────────────────────────
    # 1. Colour in main ROI  → full tracking
    # 2. Colour in lookahead → early warning (use lookahead centroid for error)
    # 3. Black in main ROI   → normal black tracking
    # 4. Nothing             → search

    COLOUR_MAP = {
        'red':              (0,   0,   255),
        'yellow':           (0,   215, 255),
        'lookahead_red':    (0,   80,  200),
        'lookahead_yellow': (0,   160, 200),
        'black':            (0,   255, 0),
        'search':           (128, 128, 128),
    }

    active_mode = 'search'
    active_cont = None
    roi_offset  = H // 3
    look_offset = H // 4

    found_line = False
    error      = 0.0

    if colour_mode != 'none':
        active_mode = colour_mode
        active_cont = colour_cont
        x, y, w_box, h_box = cv2.boundingRect(active_cont)
        line_center = x + w_box // 2
        error       = (frame_center - line_center) / float(frame_center)
        found_line  = True

        y_full = y + roi_offset
        cv2.rectangle(frame, (x, y_full), (x + w_box, y_full + h_box), COLOUR_MAP[active_mode], 2)
        cv2.circle(frame, (line_center, y_full + h_box // 2), 6, COLOUR_MAP[active_mode], -1)

    elif look_mode != 'none':
        active_mode = 'lookahead_' + look_mode
        x, y, w_box, h_box = cv2.boundingRect(look_cont)
        line_center = x + w_box // 2
        error       = (frame_center - line_center) / float(frame_center)
        found_line  = True

        y_full = y + look_offset
        cv2.rectangle(frame, (x, y_full), (x + w_box, y_full + h_box), COLOUR_MAP[active_mode], 2)
        cv2.circle(frame, (line_center, y_full + h_box // 2), 6, COLOUR_MAP[active_mode], -1)

    elif c_black is not None:
        active_mode = 'black'
        active_cont = c_black
        x, y, w_box, h_box = cv2.boundingRect(active_cont)
        line_center = x + w_box // 2
        error       = (frame_center - line_center) / float(frame_center)
        found_line  = True

        y_full = y + roi_offset
        cv2.rectangle(frame, (x, y_full), (x + w_box, y_full + h_box), COLOUR_MAP['black'], 2)
        cv2.circle(frame, (line_center, y_full + h_box // 2), 6, COLOUR_MAP['black'], -1)

    # ── Draw ROI boundary lines on frame for debug ────────────────
    cv2.line(frame, (0, H // 3), (W, H // 3), (200, 200, 0),  1)
    cv2.line(frame, (0, H // 4), (W, H // 4), (200, 100, 0),  1)
    cv2.line(frame, (frame_center, 0), (frame_center, H), (255, 255, 0), 2)

    status = "RUNNING" if running else "PAUSED"
    cv2.putText(
        frame,
        f"{status} | MODE: {active_mode.upper()} | err:{error:+.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        COLOUR_MAP.get(active_mode, (255, 255, 255)), 2
    )

    # ── Debug windows ─────────────────────────────────────────────
    debug_mask = cv2.bitwise_or(mr, my)
    lookahead_debug = cv2.bitwise_or(lr, ly)

    cv2.imshow("Line Follow PID", frame)
    cv2.imshow("Mask (ROI)",      debug_mask)
    cv2.imshow("Lookahead",       lookahead_debug)

    return found_line, error, active_mode


def close_camera():
    picam2.stop()
    cv2.destroyAllWindows()
