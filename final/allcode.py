#!/usr/bin/env python3
"""
Auto-generated packer for folder: linefollower2

Run:
    python linefollower2-comb.py
"""

import pathlib, textwrap

PKG = pathlib.Path("linefollower2")
PKG.mkdir(exist_ok=True)

def W(name, content):
    path = PKG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    print(f"  wrote  {path.as_posix()}")

# ────────────────────────────────────────────────────────────────────────────
W("__init__.py", r'''
"""Line-following robot — multiprocessing package."""
''')

# ────────────────────────────────────────────────────────────────────────────
W("__main__.py", r'''
from .main import main
main()
''')

# ────────────────────────────────────────────────────────────────────────────
W("ai.py", r'''
"""
AI subprocess — YOLO inference only.
The model directly classifies Arrow Left and Arrow Right (no contour analysis).
Publishes results via shared multiprocessing.Value objects.
"""

import time
import signal
from ultralytics import YOLO
from . import config as cfg
from .shm import open_frame_shm, read_frame


def ai_proc(shm_name, frame_lock, stop_evt,
            ai_label_id, ai_conf_val, ai_new_flag, ai_lock):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shm, buf = open_frame_shm(shm_name)
    sym_model = YOLO(cfg.SYMBOL_MODEL_PATH)
    last_symbol_t = {}

    try:
        while not stop_evt.is_set():
            frame = read_frame(buf, frame_lock)
            now = time.time()
            best_label = None
            best_conf = cfg.SYMBOL_CONF

            res = sym_model(frame, verbose=False)
            for box in (res[0].boxes or []):
                conf = float(box.conf[0])
                label = res[0].names[int(box.cls[0])]

                if label in cfg.ARROW_SYMBOLS or label in cfg.NON_ARROW_SYMBOLS:
                    if conf > best_conf:
                        elapsed = now - last_symbol_t.get(label, 0)
                        if elapsed >= cfg.SYMBOL_COOLDOWN:
                            best_conf = conf
                            best_label = label

            if best_label is not None:
                last_symbol_t[best_label] = now
                lid = cfg.LABEL_TO_ID.get(best_label, 0)
                with ai_lock:
                    ai_label_id.value = lid
                    ai_conf_val.value = best_conf
                    ai_new_flag.value = 1

            time.sleep(0.005)
    finally:
        shm.close()
''')

# ────────────────────────────────────────────────────────────────────────────
W("calibration.py", r'''
"""
Colour calibration — click on the camera preview to sample HSV.
  K  toggle calibration mode
  1  target red
  2  target yellow
  A  apply collected samples as new HSV range
"""

import cv2
import numpy as np
from . import config as cfg

# ── State (main process only) ─────────────────────────────────────────────
active = False
target = "red"
clicks = []        # list of (H, S, V) tuples


def on_mouse(event, x, y, flags, frame_rgb):
    """cv2 mouse callback — samples a 10x10 HSV patch."""
    if event != cv2.EVENT_LBUTTONDOWN or not active:
        return
    if frame_rgb is None:
        return
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    h_img, w_img = hsv.shape[:2]
    x1 = max(0, x - 5)
    y1 = max(0, y - 5)
    x2 = min(w_img, x + 5)
    y2 = min(h_img, y + 5)
    patch = hsv[y1:y2, x1:x2]
    lo = patch.min(axis=(0, 1))
    hi = patch.max(axis=(0, 1))
    avg = patch.mean(axis=(0, 1)).astype(int)
    clicks.append(tuple(avg))
    print("[CALIB] ({},{}) HSV avg={}  range=[{} - {}]  target={}".format(
        x, y, tuple(avg), tuple(lo), tuple(hi), target))


def apply():
    """Compute HSV range from collected clicks and write to config."""
    global clicks
    if not clicks:
        print("[CALIB] No samples collected.")
        return
    arr = np.array(clicks)
    h_min, s_min, v_min = arr.min(axis=0)
    h_max, s_max, v_max = arr.max(axis=0)
    MH, MS, MV = 10, 40, 40
    lo = (max(0, int(h_min - MH)),
          max(0, int(s_min - MS)),
          max(0, int(v_min - MV)))
    hi = (min(180, int(h_max + MH)),
          min(255, int(s_max + MS)),
          min(255, int(v_max + MV)))

    # Red can wrap around hue 0/180
    if target == "red" and (h_min < 15 or h_max > 165):
        cfg.HSV_RANGES["red"] = [
            ((max(0, int(h_min - MH)), lo[1], lo[2]),
             (min(15, int(h_max + MH)), hi[1], hi[2])),
            ((max(165, int(h_min - MH)), lo[1], lo[2]),
             (180, hi[1], hi[2])),
        ]
    else:
        cfg.HSV_RANGES[target] = [(lo, hi)]

    print("[CALIB] Applied {}: {}".format(target, cfg.HSV_RANGES[target]))
    clicks = []
''')

# ────────────────────────────────────────────────────────────────────────────
W("camera.py", r'''
"""Camera capture subprocess — writes frames into SHM continuously."""

import time
import signal
from picamera2 import Picamera2
from . import config as cfg
from .shm import open_frame_shm, write_frame


def camera_proc(shm_name, frame_lock, stop_evt):
    """Target for multiprocessing.Process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shm, buf = open_frame_shm(shm_name)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (cfg.FRAME_W, cfg.FRAME_H)}
    ))
    picam2.start()
    picam2.set_controls({"AwbEnable": False, "ColourGains": (1.5, 1.5)})
    time.sleep(1.0)

    try:
        while not stop_evt.is_set():
            frame = picam2.capture_array()
            write_frame(buf, frame, frame_lock)
            time.sleep(0.001)
    finally:
        picam2.stop()
        shm.close()
''')

# ────────────────────────────────────────────────────────────────────────────
W("config.py", r'''
"""
All tunables, thresholds, and lookup tables.
Mutable dicts (HSV_RANGES) are modified in-place by calibration —
safe because calibration only runs in the main process.
"""

# ── Models ────────────────────────────────────────────────────────────────
SYMBOL_MODEL_PATH = "my_model_ncnn_model"
# Arrow direction is now resolved directly by the model (Arrow Left / Arrow Right).

# ── Motor tuning ──────────────────────────────────────────────────────────
BASESPEED  = 26
HARD_TURN  = 85
DEADZONE   = 80
KP_BLACK   = 0.35
KP_COLOR   = 0.85
KI         = 0
KD         = 0.05

# ── Camera / frame ────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 640
FRAME_BYTES      = FRAME_W * FRAME_H * 3
ROI_START        = 0.6
BLUR_K           = 5
THRESH_BLOCK     = 157
THRESH_C         = 40
MIN_AREA         = 500

# ── Symbol detection ──────────────────────────────────────────────────────
SYMBOL_CONF     = 0.6
SYMBOL_COOLDOWN = 4.0

# ── Arrow manoeuvre ───────────────────────────────────────────────────────
ARROW_HARD_TURN_TIME = 0.2    # seconds to hard-turn in arrow direction
ARROW_SEARCH_SPEED   = 85     # motor speed while spinning to find the line

# ── Arrow error bias (applied after the manoeuvre) ────────────────────────
TURN_BIAS_PX         = 45     # px bias added to the PID error
ARROW_BIAS_TIMEOUT   = 15.0   # seconds the bias stays active

# ── Colour line detection (red & yellow only) ─────────────────────────────
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

# ── GPIO pins ─────────────────────────────────────────────────────────────
ENB, IN3, IN4 = 24, 22, 23
ENA, IN1, IN2 = 18, 27, 17
PWM_FREQ      = 1000

# ── Debug defaults ────────────────────────────────────────────────────────
DBG_CONSOLE  = False
DBG_LOG      = False
DBG_OVERLAY  = True
LOG_INTERVAL = 0.5

# ── Label maps ────────────────────────────────────────────────────────────
# The model directly classifies Arrow Left and Arrow Right.
# On detection the robot hard-turns in that direction for 0.5 s,
# then spins the opposite way until the line is reacquired,
# then applies an error bias for 15 s.
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
''')

# ────────────────────────────────────────────────────────────────────────────
W("debug.py", r'''
"""Console telemetry printing and CSV logging."""

import time
from . import config as cfg
from . import vision

_log_file = None


def start_log():
    global _log_file
    fname = "log_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))
    _log_file = open(fname, "w")
    _log_file.write(
        "timestamp,status,mode,error_px,correction,"
        "left_pct,right_pct,fps,cx,cy,"
        "cnt_area,cnt_count,pid_p,pid_i,pid_d,bias_dir\n"
    )
    print("[DBG] Logging -> " + fname)


def stop_log():
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None
        print("[DBG] Log closed.")


def console_print(status, mode, error, correction, ls, rs, fps, cx, cy,
                  pid, bd):
    p, i, d = pid.components
    ts = time.strftime("%H:%M:%S")
    bd_s = "L" if bd < 0 else ("R" if bd > 0 else "-")
    print(
        "[{}] {:<8} {:<9} | "
        "err={:+4.0f}px corr={:+6.2f} | "
        "L={:+6.1f}% R={:+6.1f}% | FPS={:4.1f} | "
        "cnts={} area={} | "
        "P={:+.2f} I={:+.3f} D={:+.2f} | "
        "bias={}".format(
            ts, status, mode,
            error, correction,
            ls, rs, fps,
            vision.dbg_cnt_count, vision.dbg_cnt_area,
            p, i, d,
            bd_s,
        )
    )


def log_row(status, mode, error, correction, ls, rs, fps, cx, cy, pid, bd):
    if not _log_file:
        return
    p, i, d = pid.components
    cx_s = str(cx) if cx is not None else ""
    cy_s = str(cy) if cy is not None else ""
    _log_file.write(
        "{:.3f},{},{},{:.1f},{:.4f},"
        "{:.1f},{:.1f},{:.1f},"
        "{},{},"
        "{},{},{:.4f},{:.4f},{:.4f},{}\n".format(
            time.time(), status, mode, error, correction,
            ls, rs, fps,
            cx_s, cy_s,
            vision.dbg_cnt_area, vision.dbg_cnt_count,
            p, i, d, bd,
        )
    )
''')

# ────────────────────────────────────────────────────────────────────────────
W("main.py", r'''
"""Main control loop — primary process."""

import time
import os
import cv2
import numpy as np
from multiprocessing import Process, Value, Lock, Event
from simple_pid import PID

from . import config as cfg
from . import shm as shm_mod
from . import vision
from .camera import camera_proc
from .ai import ai_proc
from .motors import gpio_setup, set_motors, stop, cleanup as gpio_cleanup
from .symbols import handle_symbol
from . import calibration as calib
from . import overlay
from . import debug as dbg


def main():
    # ── Shared memory ─────────────────────────────────────────────────────
    shm_name = "lf_frame_{}".format(os.getpid())
    frame_shm, frame_buf = shm_mod.create_frame_shm(shm_name)
    frame_lock = Lock()
    stop_evt = Event()

    # ── AI shared values ──────────────────────────────────────────────────
    ai_label_id = Value("i", 0)
    ai_conf_val = Value("d", 0.0)
    ai_new_flag = Value("i", 0)
    ai_lock = Lock()

    # ── Start subprocesses ────────────────────────────────────────────────
    cam_p = Process(
        target=camera_proc,
        args=(shm_name, frame_lock, stop_evt),
        daemon=True,
    )
    ai_p = Process(
        target=ai_proc,
        args=(shm_name, frame_lock, stop_evt,
              ai_label_id, ai_conf_val, ai_new_flag, ai_lock),
        daemon=True,
    )
    cam_p.start()
    time.sleep(1.5)
    ai_p.start()
    print("[MAIN] Camera + AI processes started.")

    # ── GPIO + PID ────────────────────────────────────────────────────────
    gpio_setup()
    pid = PID(cfg.KP_BLACK, cfg.KI, cfg.KD, setpoint=0)
    pid.output_limits = (-cfg.BASESPEED, cfg.BASESPEED)
    pid.sample_time = None

    # ── Error bias state ──────────────────────────────────────────────────
    # bias_dir: -1 = bias left (Arrow Left), +1 = bias right (Arrow Right), 0 = none
    bias_dir   = 0
    bias_start = 0.0

    # ── Runtime state ─────────────────────────────────────────────────────
    color_priority = cfg.COLOR_PRIORITIES[1]   # "red"
    show_bw = True
    last_error = 0
    ls = rs = correction = fps = 0.0
    _fps_count = 0
    _fps_t = time.time()
    _last_log_t = 0.0
    _cur_frame = None

    # ── OpenCV window + mouse callback ────────────────────────────────────
    cv2.namedWindow("Line Follower")

    def _mouse(ev, x, y, flags, param):
        calib.on_mouse(ev, x, y, flags, _cur_frame)

    cv2.setMouseCallback("Line Follower", _mouse)

    try:
        while True:
            frame = shm_mod.read_frame(frame_buf, frame_lock)
            _cur_frame = frame
            centre = frame.shape[1] // 2
            now = time.time()

            # ── Poll AI ───────────────────────────────────────────────────
            det_label = None
            with ai_lock:
                if ai_new_flag.value == 1:
                    det_label = cfg.LABEL_MAP.get(ai_label_id.value)
                    det_conf = ai_conf_val.value
                    ai_new_flag.value = 0

            if det_label is not None:
                print("[AI] {}  conf={:.2f}".format(det_label, det_conf))

                if det_label in cfg.ARROW_SYMBOLS:
                    # 1) Execute blocking manoeuvre (hard turn + find line)
                    stop()
                    handle_symbol(det_label, pid, frame_buf,
                                  frame_lock, color_priority)
                    # 2) After line is reacquired, activate error bias
                    if det_label == "Arrow Left":
                        bias_dir = -1.8
                    else:
                        bias_dir = 1.8
                    bias_start = time.time()
                    print("[BIAS] {} active for {:.0f}s".format(
                        det_label, cfg.ARROW_BIAS_TIMEOUT))
                else:
                    stop()
                    handle_symbol(det_label, pid, frame_buf,
                                  frame_lock, color_priority)

            # ── Expire / compute bias ─────────────────────────────────────
            bias_on = (bias_dir != 0
                       and (now - bias_start) < cfg.ARROW_BIAS_TIMEOUT)
            if bias_dir != 0 and not bias_on:
                print("[BIAS] Expired.")
                bias_dir = 0

            bias_rem = (cfg.ARROW_BIAS_TIMEOUT - (now - bias_start)
                        ) if bias_on else 0.0
            cur_bias = int(cfg.TURN_BIAS_PX * bias_dir) if bias_on else 0

            # ── Line detection ────────────────────────────────────────────
            pid.Kp = vision.active_kp
            cx, cy, bw, line_cnt = vision.find_line(
                frame, color_priority, cur_bias)

            if cx is not None:
                error = cx - centre
                last_error = error
                correction = float(pid(error))
                if abs(error) > cfg.DEADZONE:
                    mode = "HARD TURN"
                    if error > 0:
                        ls, rs = cfg.HARD_TURN, -cfg.HARD_TURN
                    else:
                        ls, rs = -cfg.HARD_TURN, cfg.HARD_TURN
                else:
                    mode = "PID"
                    ls = float(np.clip(cfg.BASESPEED - correction, -100, 100))
                    rs = float(np.clip(cfg.BASESPEED + correction, -100, 100))
                set_motors(ls, rs)
                status = "TRACKING"
            else:
                mode = "SEARCH"
                error = last_error
                if last_error > 0:
                    ls, rs = cfg.HARD_TURN, -cfg.HARD_TURN
                    set_motors(ls, rs)
                elif last_error < 0:
                    ls, rs = -cfg.HARD_TURN, cfg.HARD_TURN
                    set_motors(ls, rs)
                else:
                    ls = rs = 0.0
                    stop()
                status = "SEARCH" if last_error else "STOPPED"

            # ── FPS ───────────────────────────────────────────────────────
            _fps_count += 1
            if (now - _fps_t) >= 0.5:
                fps = _fps_count / (now - _fps_t)
                _fps_count = 0
                _fps_t = now

            # ── Debug I/O ─────────────────────────────────────────────────
            if ((cfg.DBG_CONSOLE or cfg.DBG_LOG)
                    and (now - _last_log_t >= cfg.LOG_INTERVAL)):
                _last_log_t = now
                if cfg.DBG_CONSOLE:
                    dbg.console_print(status, mode, error, correction,
                                      ls, rs, fps, cx, cy, pid, bias_dir)
                if cfg.DBG_LOG:
                    dbg.log_row(status, mode, error, correction,
                                ls, rs, fps, cx, cy, pid, bias_dir)

            # ── Draw ──────────────────────────────────────────────────────
            overlay.draw_hud(frame, cx, cy, line_cnt, error, ls, rs,
                             status, mode, fps, color_priority,
                             bias_dir, bias_rem, pid)
            if cfg.DBG_OVERLAY:
                overlay.draw_debug(frame, cx, cy, error, correction,
                                   pid, bias_dir, bias_rem)

            disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Line Follower", disp)

            # Threshold view
            if show_bw and bw is not None:
                bw_d = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
                overlay.txt_c(bw_d, "THRESHOLD VIEW",
                              bw_d.shape[1] // 2, 18, 0.45, (55, 200, 55))
                cv2.imshow("Threshold", bw_d)

            # ── Keyboard ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fn = "snap_{}.png".format(time.strftime("%H%M%S"))
                cv2.imwrite(fn, disp)
                print("Saved " + fn)
            elif key == ord("t"):
                show_bw = not show_bw
                if not show_bw:
                    cv2.destroyWindow("Threshold")
            elif key == ord("d"):
                cfg.DBG_CONSOLE = not cfg.DBG_CONSOLE
                state = "ON" if cfg.DBG_CONSOLE else "OFF"
                print("[DBG] Console " + state)
            elif key == ord("l"):
                cfg.DBG_LOG = not cfg.DBG_LOG
                if cfg.DBG_LOG:
                    dbg.start_log()
                else:
                    dbg.stop_log()
            elif key == ord("v"):
                cfg.DBG_OVERLAY = not cfg.DBG_OVERLAY
            elif key == ord("c"):
                idx = cfg.COLOR_PRIORITIES.index(color_priority)
                idx = (idx + 1) % len(cfg.COLOR_PRIORITIES)
                color_priority = cfg.COLOR_PRIORITIES[idx]
                print("[COLOR] -> " + color_priority.upper())
            elif key == ord("k"):
                calib.active = not calib.active
                if calib.active:
                    calib.clicks = []
                    print("[CALIB] ON  target=" + calib.target
                          + "  (1=red 2=yellow A=apply K=exit)")
                else:
                    print("[CALIB] OFF")
            elif key == ord("1") and calib.active:
                calib.target = "red"
                calib.clicks = []
                print("[CALIB] -> RED")
            elif key == ord("2") and calib.active:
                calib.target = "yellow"
                calib.clicks = []
                print("[CALIB] -> YELLOW")
            elif key == ord("a") and calib.active:
                calib.apply()

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted.")
    finally:
        print("[MAIN] Shutting down...")
        stop_evt.set()
        stop()
        cam_p.join(timeout=3)
        ai_p.join(timeout=3)
        if cam_p.is_alive():
            cam_p.terminate()
        if ai_p.is_alive():
            ai_p.terminate()
        gpio_cleanup()
        cv2.destroyAllWindows()
        dbg.stop_log()
        frame_shm.close()
        frame_shm.unlink()
        print("[MAIN] Done.")
''')

# ────────────────────────────────────────────────────────────────────────────
W("motors.py", r'''
"""GPIO setup and motor helpers — main process only."""

import RPi.GPIO as GPIO
from . import config as cfg

pwm_l = None
pwm_r = None


def gpio_setup():
    global pwm_l, pwm_r
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (cfg.ENA, cfg.IN1, cfg.IN2, cfg.ENB, cfg.IN3, cfg.IN4):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)
    pwm_l = GPIO.PWM(cfg.ENA, cfg.PWM_FREQ)
    pwm_l.start(0)
    pwm_r = GPIO.PWM(cfg.ENB, cfg.PWM_FREQ)
    pwm_r.start(0)


def set_motors(left, right):
    left = max(-100.0, min(100.0, float(left)))
    right = max(-100.0, min(100.0, float(right)))

    GPIO.output(cfg.IN1, left >= 0)
    GPIO.output(cfg.IN2, left < 0)
    GPIO.output(cfg.IN3, right >= 0)
    GPIO.output(cfg.IN4, right < 0)

    pwm_l.ChangeDutyCycle(abs(left))
    pwm_r.ChangeDutyCycle(abs(right))


def stop():
    for pin in (cfg.IN1, cfg.IN2, cfg.IN3, cfg.IN4):
        GPIO.output(pin, False)
    pwm_l.ChangeDutyCycle(0)
    pwm_r.ChangeDutyCycle(0)


def cleanup():
    stop()
    GPIO.cleanup()
''')

# ────────────────────────────────────────────────────────────────────────────
W("overlay.py", r'''
"""HUD, debug panel, and drawing primitives."""

import cv2
import numpy as np
from . import config as cfg
from . import vision
from . import calibration as calib

# ── Palette (BGR) ─────────────────────────────────────────────────────────
C_GREEN  = (55, 200, 55)
C_ORANGE = (0, 150, 255)
C_RED    = (55, 55, 210)
C_CYAN   = (210, 200, 0)
C_PANEL  = (18, 18, 18)
C_BORDER = (65, 65, 65)
C_DIM    = (55, 55, 55)
C_TXT    = (185, 185, 185)
C_TXT2   = (90, 90, 90)
C_BARFWD = (55, 180, 55)
C_BARBWD = (55, 55, 195)
HUD_H    = 138

# ── Text helpers ──────────────────────────────────────────────────────────

def txt(img, text, x, y, sc=0.42, col=C_TXT, bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                sc, col, 2 if bold else 1, cv2.LINE_AA)

def txt_c(img, text, cx, y, sc=0.42, col=C_TXT):
    w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
    txt(img, text, cx - w // 2, y, sc, col)

def txt_r(img, text, rx, y, sc=0.42, col=C_TXT):
    w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
    txt(img, text, rx - w, y, sc, col)

# ── Graphic primitives ────────────────────────────────────────────────────

def dashed(img, pt1, pt2, col, thick=1, dash=9, gap=7):
    x1, y1 = pt1
    x2, y2 = pt2
    total = np.hypot(x2 - x1, y2 - y1)
    if total == 0:
        return
    step = dash + gap
    for i in range(int(total / step) + 1):
        a = i * step / total
        b = min((i * step + dash) / total, 1.0)
        p1 = (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a))
        p2 = (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b))
        cv2.line(img, p1, p2, col, thick, cv2.LINE_AA)

def crosshair(img, x, y, col=C_CYAN, arm=16, gap=5, th=2):
    cv2.line(img, (x - arm, y), (x - gap, y), col, th, cv2.LINE_AA)
    cv2.line(img, (x + gap, y), (x + arm, y), col, th, cv2.LINE_AA)
    cv2.line(img, (x, y - arm), (x, y - gap), col, th, cv2.LINE_AA)
    cv2.line(img, (x, y + gap), (x, y + arm), col, th, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, col, -1, cv2.LINE_AA)

def hbar(img, x, y, w, h, val, vmax):
    val = float(np.clip(val, -vmax, vmax))
    cv2.rectangle(img, (x, y), (x + w, y + h), (38, 38, 38), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (68, 68, 68), 1)
    f = int(abs(val) / vmax * (w - 2))
    if f > 0:
        c = C_BARFWD if val >= 0 else C_BARBWD
        cv2.rectangle(img, (x + 1, y + 2), (x + 1 + f, y + h - 2), c, -1)

def center_bar(img, cx, y, hw, h, val, vmax):
    val = float(np.clip(val, -vmax, vmax))
    cv2.rectangle(img, (cx - hw, y), (cx + hw, y + h), (38, 38, 38), -1)
    cv2.rectangle(img, (cx - hw, y), (cx + hw, y + h), (68, 68, 68), 1)
    f = int(abs(val) / vmax * (hw - 1))
    if f > 0:
        c = C_ORANGE if abs(val) > cfg.DEADZONE else C_GREEN
        if val > 0:
            cv2.rectangle(img, (cx + 1, y + 2), (cx + f, y + h - 2), c, -1)
        else:
            cv2.rectangle(img, (cx - f, y + 2), (cx - 1, y + h - 2), c, -1)
    cv2.line(img, (cx, y - 2), (cx, y + h + 2), (165, 165, 165), 2)

# ── HUD ───────────────────────────────────────────────────────────────────

def draw_hud(frame, cx, cy, line_cnt, error, ls, rs, status, mode, fps,
             color_priority, bias_dir, bias_rem, pid_obj):
    fh, fw = frame.shape[:2]
    centre = fw // 2
    roi_y = int(fh * cfg.ROI_START)
    hud_y = fh - HUD_H

    dashed(frame, (0, roi_y), (fw, roi_y), (40, 80, 40), 1, 14, 7)
    txt(frame, "ROI", 6, roi_y - 6, 0.30, (40, 80, 40))
    dashed(frame, (centre, 0), (centre, hud_y), (50, 50, 50), 1, 6, 9)

    draw_col = cfg.LINE_DRAW_COLOR.get(color_priority, (200, 200, 200))
    if line_cnt is not None:
        cv2.drawContours(frame, [line_cnt], -1, draw_col, 2, cv2.LINE_AA)
    if cx is not None and cy is not None:
        crosshair(frame, cx, cy)
        if abs(error) > 5:
            tip = int(cx + np.clip(-error * 0.45, -70, 70))
            cv2.arrowedLine(frame, (cx, cy), (tip, cy),
                            (0, 165, 255), 1, cv2.LINE_AA, tipLength=0.35)

    # Bias banner
    if bias_dir != 0:
        arrow_tag = "<< LEFT BIAS" if bias_dir < 0 else "RIGHT BIAS >>"
        intent_txt = "{}  {:.0f}s".format(arrow_tag, bias_rem)
        txt_c(frame, intent_txt, centre, 18, 0.40, C_ORANGE)

    # Calibration banner
    if calib.active:
        cal_txt = "CALIB: {} - click to sample".format(calib.target.upper())
        txt_c(frame, cal_txt, centre, 35, 0.38, (0, 255, 255))

    # HUD background
    ov = frame.copy()
    cv2.rectangle(ov, (0, hud_y), (fw, fh), C_PANEL, -1)
    cv2.addWeighted(ov, 0.86, frame, 0.14, 0, frame)
    cv2.line(frame, (0, hud_y), (fw, hud_y), C_BORDER, 1)

    CL, CM, CR = 10, 182, 458
    cv2.line(frame, (CM - 6, hud_y + 6), (CM - 6, fh - 6), C_DIM, 1)
    cv2.line(frame, (CR - 6, hud_y + 6), (CR - 6, fh - 6), C_DIM, 1)

    # Left column — status
    Y1 = hud_y + 22
    sc = C_GREEN if status == "TRACKING" else (
         C_ORANGE if status == "SEARCH" else C_RED)
    cv2.circle(frame, (CL + 6, Y1 - 5), 5, sc, -1, cv2.LINE_AA)
    txt(frame, status, CL + 17, Y1, 0.48, sc, bold=True)
    txt(frame, mode, CL + 6, Y1 + 18, 0.36, C_TXT2)
    txt(frame, "FPS  {:.1f}".format(fps), CL + 6, Y1 + 35, 0.36, C_TXT2)
    dot = cfg.LINE_DRAW_COLOR.get(color_priority, (200, 200, 200))
    cv2.circle(frame, (CL + 6, Y1 + 47), 4, dot, -1, cv2.LINE_AA)
    txt(frame, "LINE " + color_priority.upper(),
        CL + 17, Y1 + 52, 0.33, C_TXT2)

    # Centre column — steering
    cc = (CM + CR) // 2
    bh = (CR - CM - 20) // 2
    BY = hud_y + 25
    BH = 22
    txt_c(frame, "STEERING ERROR", cc, hud_y + 18, 0.33, C_TXT2)
    center_bar(frame, cc, BY, bh, BH, error, fw // 2)
    ecol = C_ORANGE if abs(error) > cfg.DEADZONE else C_TXT
    txt_c(frame, "{:+.0f} px".format(error), cc, BY + BH + 16, 0.40, ecol)
    pid_txt = "Kp {:.3f}  Ki {:.4f}  Kd {:.3f}  DZ +/-{}px".format(
        vision.active_kp, cfg.KI, cfg.KD, cfg.DEADZONE)
    txt_c(frame, pid_txt, cc, fh - 7, 0.28, (52, 52, 52))

    # Right column — motors
    MW = fw - CR - 22
    MH = 14
    YL = hud_y + 20
    YLB = YL + 13
    YR = YLB + MH + 20
    YRB = YR + 13
    txt(frame, "L MOTOR", CR, YL, 0.33, C_TXT2)
    hbar(frame, CR, YLB, MW, MH, ls, 100)
    txt_r(frame, "{:+.0f}%".format(ls), fw - 8, YLB + MH - 1, 0.34,
          C_BARFWD if ls >= 0 else C_BARBWD)
    txt(frame, "R MOTOR", CR, YR, 0.33, C_TXT2)
    hbar(frame, CR, YRB, MW, MH, rs, 100)
    txt_r(frame, "{:+.0f}%".format(rs), fw - 8, YRB + MH - 1, 0.34,
          C_BARFWD if rs >= 0 else C_BARBWD)

    txt_r(frame,
          "Q quit  S snap  T thresh  D con  L log  V ovl  C col  K calib",
          fw - 8, fh - 7, 0.28, (45, 45, 45))

# ── Debug panel ───────────────────────────────────────────────────────────

def draw_debug(frame, cx, cy, error, correction, pid_obj,
               bias_dir, bias_rem):
    fh, fw = frame.shape[:2]
    for c in vision.dbg_all_cnts:
        cv2.drawContours(frame, [c], -1, (60, 60, 60), 1, cv2.LINE_AA)
    PW, PH = 210, 130
    px, py = fw - PW - 6, 6
    pnl = frame.copy()
    cv2.rectangle(pnl, (px, py), (px + PW, py + PH), (12, 12, 12), -1)
    cv2.addWeighted(pnl, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (px, py), (px + PW, py + PH), (90, 30, 30), 1)
    cv2.rectangle(frame, (px, py), (px + 34, py + 14), (60, 30, 30), -1)
    txt(frame, "DBG", px + 3, py + 11, 0.32, (80, 80, 220), bold=True)

    p, i, d = pid_obj.components
    bd_str = "L" if bias_dir < 0 else ("R" if bias_dir > 0 else "-")
    cx_str = str(cx) if cx is not None else "--"
    cy_str = str(cy) if cy is not None else "--"

    rows = [
        ("cnts  {}".format(vision.dbg_cnt_count), C_TXT2),
        ("area  {}".format(vision.dbg_cnt_area), C_TXT2),
        ("cx={}  cy={}".format(cx_str, cy_str), C_TXT2),
        ("P {:+7.2f}".format(p), (100, 200, 100)),
        ("I {:+7.3f}".format(i), (100, 160, 220)),
        ("D {:+7.2f}".format(d), (220, 160, 100)),
        ("Kp {:.3f}".format(vision.active_kp), C_TXT2),
        ("BIAS {} {:.1f}s".format(bd_str, bias_rem),
         C_ORANGE if bias_dir else C_TXT2),
    ]
    for idx, (row_txt, col) in enumerate(rows):
        txt(frame, row_txt, px + 6, py + 26 + idx * 13, 0.32, col)
''')

# ────────────────────────────────────────────────────────────────────────────
W("shm.py", r'''
"""
POSIX SharedMemory helpers for the frame buffer.
One segment is created by the main process; camera and AI attach by name.
"""

import numpy as np
from multiprocessing import shared_memory
from . import config as cfg


def create_frame_shm(name):
    """Create a new SHM segment for one RGB frame.  Returns (shm, np_array)."""
    shape = (cfg.FRAME_H, cfg.FRAME_W, 3)
    nbytes = int(np.prod(shape))
    try:
        stale = shared_memory.SharedMemory(name=name, create=False)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass
    shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    arr[:] = 0
    return shm, arr


def open_frame_shm(name):
    """Attach to an existing SHM segment.  Returns (shm, np_array)."""
    shape = (cfg.FRAME_H, cfg.FRAME_W, 3)
    shm = shared_memory.SharedMemory(name=name, create=False)
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    return shm, arr


def read_frame(arr, lock):
    """Copy the frame out of SHM under a lock."""
    with lock:
        return arr.copy()


def write_frame(arr, frame, lock):
    """Write a captured frame into SHM under a lock."""
    with lock:
        np.copyto(arr, frame)
''')

# ────────────────────────────────────────────────────────────────────────────
W("symbols.py", r'''
"""Symbol action handlers — executed in the main process."""

import time
import os
import cv2
import numpy as np
import face_recognition
from . import config as cfg
from .motors import set_motors, stop
from .shm import read_frame
from .vision import find_line


def handle_symbol(label, pid, shm_arr, frame_lock, color_priority):
    """Execute the behaviour for *label*, then return."""
    label = label.strip()
    print("[SYMBOL] Executing: " + label)

    if label == "Arrow Right":
        # Hard turn right for 0.5 s, then spin left until line found.
        spd = cfg.ARROW_SEARCH_SPEED
        set_motors(spd, -spd)
        time.sleep(cfg.ARROW_HARD_TURN_TIME)
        # Now spin opposite (left) until line reacquired
        _spin_until_line(shm_arr, frame_lock, color_priority,
                         direction="left")

    elif label == "Arrow Left":
        # Hard turn left for 0.5 s, then spin right until line found.
        spd = cfg.ARROW_SEARCH_SPEED
        set_motors(-spd, spd)
        time.sleep(cfg.ARROW_HARD_TURN_TIME)
        # Now spin opposite (right) until line reacquired
        _spin_until_line(shm_arr, frame_lock, color_priority,
                         direction="right")

    elif label in ("Hazard", "Green Hand"):
        stop()
        time.sleep(1.0)

    elif label in ("QR Code", "Fingerprint"):
        set_motors(0, 0)
        _face_recognition(shm_arr, frame_lock)

    elif label == "Recycle":
        set_motors(100, -100)
        time.sleep(1.5)
        stop()
        time.sleep(0.2)

    elif label in ("Cross", "Octagon", "Star", "Diamond",
                    "Trapezoid", "Quarter Circle", "Semi Circle"):
        stop()
        time.sleep(0.1)
        print("  " + label + " detected")

    else:
        stop()
        time.sleep(0.2)

    pid.reset()


def _spin_until_line(shm_arr, lock, color_priority, direction="left"):
    """Spin in place until the line is reacquired.

    direction: "left"  → robot spins left  (right motor fwd, left motor bwd)
               "right" → robot spins right (left motor fwd, right motor bwd)
    """
    spd = cfg.ARROW_SEARCH_SPEED
    if direction == "left":
        set_motors(-spd, spd)
    else:
        set_motors(spd, -spd)

    while True:
        frame = read_frame(shm_arr, lock)
        cx, _, _, _ = find_line(frame, color_priority)
        if cx is not None:
            break
        time.sleep(0.03)
    stop()
    time.sleep(0.1)


def _face_recognition(shm_arr, lock):
    """Face recognition routine — blocks up to 10 s."""
    known_files = [
        ("Colin.jpg",        "Colin Yap"),
        ("Dr. Hermawan.jpg", "Dr. Hermawan"),
        ("Edmund.jpg",       "Edmund"),
        ("Radhita.jpg",      "Radhita"),
        ("Austin.jpg",       "Austin"),
    ]
    encs, names = [], []
    for path, name in known_files:
        if not os.path.exists(path):
            continue
        img = face_recognition.load_image_file(path)
        e = face_recognition.face_encodings(img)
        if e:
            encs.append(e[0])
            names.append(name)

    if not encs:
        print("[FACE] No known-face encodings loaded.")
        return

    start = time.time()
    while (time.time() - start) < 10:
        frame = read_frame(shm_arr, lock)
        locs = face_recognition.face_locations(frame)
        fencs = face_recognition.face_encodings(frame, locs)
        for (top, right, bottom, left), fe in zip(locs, fencs):
            matches = face_recognition.compare_faces(encs, fe)
            dists = face_recognition.face_distance(encs, fe)
            idx = np.argmin(dists)
            name = names[idx] if matches[idx] else "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                          (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Recognition", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Face Recognition")
''')

# ────────────────────────────────────────────────────────────────────────────
W("vision.py", r'''
"""
Line detection — colour priority (red/yellow) with black fallback.
Supports turn-intent via error-bias pixels.
All mutable state lives at module level (main process only).
"""

import cv2
import numpy as np
from . import config as cfg

# ── Module-level state (read by overlay.py for debug display) ─────────────
last_cx       = cfg.FRAME_W // 2
active_kp     = cfg.KP_BLACK
dbg_cnt_count = 0
dbg_cnt_area  = 0
dbg_all_cnts  = []


def find_line(frame, color_priority, turn_bias_px=0):
    """
    Detect the line in *frame*.

    Parameters
    ----------
    color_priority : str   "black", "red", or "yellow"
    turn_bias_px   : int   added to detected cx (positive = bias right)

    Returns
    -------
    (cx, cy, bw_or_mask, contour_fullframe)   or  (None, None, bw, None)
    """
    global last_cx, active_kp, dbg_cnt_count, dbg_cnt_area, dbg_all_cnts

    h, w = frame.shape[:2]
    roi_y = int(h * cfg.ROI_START)
    roi = frame[roi_y:, :].copy()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pick_best(valid):
        def cx_of(c):
            m = cv2.moments(c)
            return int(m["m10"] / m["m00"]) if m["m00"] else last_cx
        return min(valid, key=lambda c: abs(cx_of(c) - last_cx))

    def _filter(cnts, mask):
        valid, dbg = [], []
        rh, rw = mask.shape[:2]
        for c in cnts:
            if cv2.contourArea(c) < cfg.MIN_AREA:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            touches = (y <= 5 or (y + ch) >= rh - 5
                       or x <= 5 or (x + cw) >= rw - 5)
            if touches:
                valid.append(c)
                cf = c.copy()
                cf[:, :, 1] += roi_y
                dbg.append(cf)
        return valid, dbg

    def _extract(cnt, mask, dbg_list, bias):
        global last_cx, dbg_cnt_count, dbg_cnt_area, dbg_all_cnts
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None, None, mask, None
        cx = int(M["m10"] / M["m00"]) + bias
        cx = max(0, min(w - 1, cx))
        cy = int(M["m01"] / M["m00"]) + roi_y
        last_cx = cx
        dbg_cnt_count = len(dbg_list)
        dbg_cnt_area = int(cv2.contourArea(cnt))
        dbg_all_cnts = dbg_list
        cf = cnt.copy()
        cf[:, :, 1] += roi_y
        return cx, cy, mask, cf

    # ── Colour detection ──────────────────────────────────────────────────
    if color_priority in cfg.HSV_RANGES:
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        cmask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        scan_list = (["red", "yellow"]
                     if color_priority in ("red", "yellow")
                     else [color_priority])
        for col in scan_list:
            for lo, hi in cfg.HSV_RANGES.get(col, []):
                cmask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8))
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))

        cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        valid, dbg = _filter(cnts, cmask)
        if valid:
            active_kp = cfg.KP_COLOR
            cnt = _pick_best(valid)
            return _extract(cnt, cmask, dbg, turn_bias_px)

    # ── Black fallback ────────────────────────────────────────────────────
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (cfg.BLUR_K, cfg.BLUR_K), 0)
    bw = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, cfg.THRESH_BLOCK, cfg.THRESH_C,
    )
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    valid, dbg = _filter(cnts, bw)

    if not valid:
        dbg_cnt_count = 0
        dbg_cnt_area = 0
        dbg_all_cnts = []
        return None, None, bw, None

    active_kp = cfg.KP_BLACK
    cnt = _pick_best(valid)
    return _extract(cnt, bw, dbg, turn_bias_px)
''')

print("\nDone.")
print("Packed files: 13")
print("Skipped files:")
print("  - __pycache__/__init__.cpython-313.pyc")
print("  - __pycache__/__main__.cpython-313.pyc")
print("  - __pycache__/ai.cpython-313.pyc")
print("  - __pycache__/calibration.cpython-313.pyc")
print("  - __pycache__/camera.cpython-313.pyc")
print("  - __pycache__/config.cpython-313.pyc")
print("  - __pycache__/debug.cpython-313.pyc")
print("  - __pycache__/main.cpython-313.pyc")
print("  - __pycache__/motors.cpython-313.pyc")
print("  - __pycache__/overlay.cpython-313.pyc")
print("  - __pycache__/shm.cpython-313.pyc")
print("  - __pycache__/symbols.cpython-313.pyc")
print("  - __pycache__/vision.cpython-313.pyc")
