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
