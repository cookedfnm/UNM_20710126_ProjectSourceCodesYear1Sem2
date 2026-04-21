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
    shm_name = "lf_frame_{}".format(os.getpid())
    frame_shm, frame_buf = shm_mod.create_frame_shm(shm_name)
    frame_lock = Lock()
    stop_evt = Event()

    ai_label_id = Value("i", 0)
    ai_conf_val = Value("d", 0.0)
    ai_new_flag = Value("i", 0)
    ai_lock = Lock()

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

    gpio_setup()
    pid = PID(cfg.KP_BLACK, cfg.KI, cfg.KD, setpoint=0)
    pid.output_limits = (-cfg.BASESPEED, cfg.BASESPEED)
    pid.sample_time = None

    bias_dir   = 0
    bias_start = 0.0

    color_priority = cfg.COLOR_PRIORITIES[1]
    show_bw = True
    last_error = 0
    ls = rs = correction = fps = 0.0
    _fps_count = 0
    _fps_t = time.time()
    _last_log_t = 0.0
    _cur_frame = None

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

            det_label = None
            with ai_lock:
                if ai_new_flag.value == 1:
                    det_label = cfg.LABEL_MAP.get(ai_label_id.value)
                    det_conf = ai_conf_val.value
                    ai_new_flag.value = 0

            if det_label is not None:
                print("[AI] {}  conf={:.2f}".format(det_label, det_conf))

                if det_label in cfg.ARROW_SYMBOLS:
                    stop()
                    handle_symbol(det_label, pid, frame_buf,
                                  frame_lock, color_priority)
                    if det_label == "Arrow Left":
                        bias_dir = 1.8
                    else:
                        bias_dir = -1.8
                    bias_start = time.time()
                    print("[BIAS] {} active for {:.0f}s".format(
                        det_label, cfg.ARROW_BIAS_TIMEOUT))
                else:
                    stop()
                    handle_symbol(det_label, pid, frame_buf,
                                  frame_lock, color_priority)

            bias_on = (bias_dir != 0
                       and (now - bias_start) < cfg.ARROW_BIAS_TIMEOUT)
            if bias_dir != 0 and not bias_on:
                print("[BIAS] Expired.")
                bias_dir = 0

            bias_rem = (cfg.ARROW_BIAS_TIMEOUT - (now - bias_start)
                        ) if bias_on else 0.0
            cur_bias = int(cfg.TURN_BIAS_PX * bias_dir) if bias_on else 0

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

            _fps_count += 1
            if (now - _fps_t) >= 0.5:
                fps = _fps_count / (now - _fps_t)
                _fps_count = 0
                _fps_t = now

            if ((cfg.DBG_CONSOLE or cfg.DBG_LOG)
                    and (now - _last_log_t >= cfg.LOG_INTERVAL)):
                _last_log_t = now
                if cfg.DBG_CONSOLE:
                    dbg.console_print(status, mode, error, correction,
                                      ls, rs, fps, cx, cy, pid, bias_dir)
                if cfg.DBG_LOG:
                    dbg.log_row(status, mode, error, correction,
                                ls, rs, fps, cx, cy, pid, bias_dir)

            overlay.draw_hud(frame, cx, cy, line_cnt, error, ls, rs,
                             status, mode, fps, color_priority,
                             bias_dir, bias_rem, pid)
            if cfg.DBG_OVERLAY:
                overlay.draw_debug(frame, cx, cy, error, correction,
                                   pid, bias_dir, bias_rem)

            disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Line Follower", disp)

            if show_bw and bw is not None:
                bw_d = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
                overlay.txt_c(bw_d, "THRESHOLD VIEW",
                              bw_d.shape[1] // 2, 18, 0.45, (55, 200, 55))
                cv2.imshow("Threshold", bw_d)

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
