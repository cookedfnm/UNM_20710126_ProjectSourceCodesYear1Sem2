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
