SYMBOL_MODEL_PATH = "my_model_ncnn_model"

BASESPEED  = 23
HARD_TURN  = 50
DEADZONE   = 160
KP_BLACK   = 0.40
KP_COLOR   = 0.70
KI         = 0
KD         = 0.15

FRAME_W, FRAME_H = 640, 480
FRAME_BYTES      = FRAME_W * FRAME_H * 3
ROI_START        = 0.6
BLUR_K           = 5
THRESH_BLOCK     = 157
THRESH_C         = 40
MIN_AREA         = 500

SYMBOL_CONF     = 0.2
SYMBOL_COOLDOWN = 4.0

ARROW_HARD_TURN_TIME = 0.3
ARROW_SEARCH_SPEED   = 85

TURN_BIAS_PX         = 160
ARROW_BIAS_TIMEOUT   = 5.0

COLOR_PRIORITIES = ["black", "red", "yellow"]

HSV_RANGES = {
    "red":    [((0, 90, 60), (15, 255, 255)),
               ((165, 90, 60), (180, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}

LINE_DRAW_COLOR = {
    "black":  (0, 165, 60),
    "red":    (55, 55, 210),
    "yellow": (0, 200, 230),
}

ENB, IN3, IN4 = 24, 22, 23
ENA, IN1, IN2 = 18, 17, 27
PWM_FREQ      = 100

DBG_CONSOLE  = False
DBG_LOG      = False
DBG_OVERLAY  = True
LOG_INTERVAL = 0.5

LABEL_MAP = {
    0: None,
    1: "Arrow Left",    2: "Arrow Right",
    3: "Cross",         4: "Semi Circle",    5: "Hazard",     6: "Green Hand",
    7: "Star",          8: "Diamond",         9: "QR Code",  10: "Quarter Circle",
   11: "Trapezoid",    12: "Octagon",        13: "Fingerprint", 14: "Recycle",
}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items() if v is not None}

ARROW_SYMBOLS = {"Arrow Left", "Arrow Right"}

NON_ARROW_SYMBOLS = {
    "Cross", "Semi Circle", "Hazard", "Green Hand",
    "Star", "Diamond", "QR Code", "Quarter Circle",
    "Trapezoid", "Octagon", "Fingerprint", "Recycle",
}
