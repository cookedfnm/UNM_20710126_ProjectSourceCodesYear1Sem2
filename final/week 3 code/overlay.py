import cv2
import numpy as np
from . import config as cfg
from . import vision
from . import calibration as calib

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

def txt(img, text, x, y, sc=0.42, col=C_TXT, bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                sc, col, 2 if bold else 1, cv2.LINE_AA)

def txt_c(img, text, cx, y, sc=0.42, col=C_TXT):
    w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
    txt(img, text, cx - w // 2, y, sc, col)

def txt_r(img, text, rx, y, sc=0.42, col=C_TXT):
    w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
    txt(img, text, rx - w, y, sc, col)

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

    if bias_dir != 0:
        arrow_tag = "<< LEFT BIAS" if bias_dir < 0 else "RIGHT BIAS >>"
        intent_txt = "{}  {:.0f}s".format(arrow_tag, bias_rem)
        txt_c(frame, intent_txt, centre, 18, 0.40, C_ORANGE)

    if calib.active:
        cal_txt = "CALIB: {} - click to sample".format(calib.target.upper())
        txt_c(frame, cal_txt, centre, 35, 0.38, (0, 255, 255))

    ov = frame.copy()
    cv2.rectangle(ov, (0, hud_y), (fw, fh), C_PANEL, -1)
    cv2.addWeighted(ov, 0.86, frame, 0.14, 0, frame)
    cv2.line(frame, (0, hud_y), (fw, hud_y), C_BORDER, 1)

    CL, CM, CR = 10, 182, 458
    cv2.line(frame, (CM - 6, hud_y + 6), (CM - 6, fh - 6), C_DIM, 1)
    cv2.line(frame, (CR - 6, hud_y + 6), (CR - 6, fh - 6), C_DIM, 1)

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
