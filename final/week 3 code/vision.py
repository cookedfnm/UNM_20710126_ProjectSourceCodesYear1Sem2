import cv2
import numpy as np
from . import config as cfg

last_cx       = cfg.FRAME_W // 2
active_kp     = cfg.KP_BLACK
dbg_cnt_count = 0
dbg_cnt_area  = 0
dbg_all_cnts  = []

def find_line(frame, color_priority, turn_bias_px=0):
    global last_cx, active_kp, dbg_cnt_count, dbg_cnt_area, dbg_all_cnts

    h, w = frame.shape[:2]
    roi_y = int(h * cfg.ROI_START)
    roi = frame[roi_y:, :].copy()

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
