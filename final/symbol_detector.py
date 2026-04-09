"""
Symbol detection process.
Runs template matching on frames from shared memory at a lower rate (~5-10 Hz)
to avoid hogging CPU. Writes best match to shared state.
"""

import os
import time
import numpy as np
import cv2
from shared_state import open_frame_buffer

DETECT_HZ = 8           # run detection at this rate
MIN_CONFIDENCE = 0.55    # minimum normalized correlation to accept
TEMPLATE_SIZE = (64, 64) # resize templates to this for speed

# Symbol names indexed by ID (stable ordering)
SYMBOL_FILES = [
    ("3_4circle",        "3_4circle.png"),
    ("blue_arrow_right", "blue_arrow_right.jpg"),
    ("circular_segment", "circular_segment.png"),
    ("diamond",          "diamond.png"),
    ("fingerprint",      "fingerprint.png"),
    ("green_arrow_up",   "green_arrow_up.jpg"),
    ("octagon",          "octagon.png"),
    ("orange_arrow_left","orange_arrow_left.png"),
    ("plus",             "plus.png"),
    ("press_button",     "press_button.png"),
    ("qr_code",          "qr_code.png"),
    ("recycle",          "recycle.png"),
    ("red_arrow_down",   "red_arrow_down.png"),
    ("star",             "star.png"),
    ("trapezoid",        "trapezoid.png"),
    ("warning",          "warning.png"),
]

SYMBOL_NAMES = [s[0] for s in SYMBOL_FILES]


def _load_templates(template_dir):
    """Load and preprocess templates. Returns list of (name, gray_template)."""
    templates = []
    for name, fname in SYMBOL_FILES:
        path = os.path.join(template_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[SYMBOL] WARNING: could not load {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, TEMPLATE_SIZE)
        templates.append((name, gray))
    print(f"[SYMBOL] loaded {len(templates)} templates")
    return templates


def _multi_scale_match(gray_frame, template, scales=(0.5, 0.75, 1.0, 1.25, 1.5)):
    """Try template at multiple scales, return best confidence."""
    best = 0.0
    th, tw = template.shape[:2]
    fh, fw = gray_frame.shape[:2]
    for s in scales:
        nh, nw = int(th * s), int(tw * s)
        if nh >= fh or nw >= fw or nh < 10 or nw < 10:
            continue
        resized = cv2.resize(template, (nw, nh))
        result = cv2.matchTemplate(gray_frame, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best:
            best = max_val
    return best


def symbol_detector_process(
    frame_seq, frame_lock, quit_flag, symbol_id, symbol_conf, template_dir,
):
    shm, frame_buf = open_frame_buffer(writable=False)
    templates = _load_templates(template_dir)
    if not templates:
        print("[SYMBOL] no templates — exiting")
        shm.close()
        return

    period = 1.0 / DETECT_HZ
    last_seq = 0
    print("[SYMBOL] started")

    try:
        while not quit_flag.value:
            seq = frame_seq.value
            if seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq

            with frame_lock:
                frame = frame_buf.copy()

            # Use upper portion of frame for symbol detection (symbols are signs above line)
            roi = frame[:frame.shape[0] // 2, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            best_id = -1
            best_conf = 0.0

            for idx, (name, tmpl) in enumerate(templates):
                conf = _multi_scale_match(gray, tmpl)
                if conf > best_conf:
                    best_conf = conf
                    best_id = idx

            if best_conf >= MIN_CONFIDENCE:
                symbol_id.value = best_id
                symbol_conf.value = best_conf
            else:
                symbol_id.value = -1
                symbol_conf.value = 0.0

            time.sleep(period)

    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        print("[SYMBOL] stopped")
