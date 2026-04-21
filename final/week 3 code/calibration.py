import cv2
import numpy as np
from . import config as cfg

active = False
target = "red"
clicks = []

def on_mouse(event, x, y, flags, frame_rgb):
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
