"""
symbol_detector.py — Runs on Pi5 alongside main.py.

Loads the TFLite model and runs inference in a background thread
so it never blocks the PID loop.

Requirements (Pi5):
    pip install tflite-runtime numpy opencv-python --break-system-packages
    OR if tflite-runtime unavailable:
    pip install tensorflow --break-system-packages
"""

import threading
import time
import numpy as np
import cv2

# ── Try tflite-runtime first (lighter), fall back to full TF ─────
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ── Settings ──────────────────────────────────────────────────────
MODEL_PATH      = 'symbol_model.tflite'
CLASS_NAMES_PATH = 'class_names.txt'
IMG_SIZE        = 96
CONFIDENCE_THRESHOLD = 0.85   # minimum confidence to accept a detection
INFERENCE_INTERVAL   = 0.15   # seconds between inferences (~6-7 fps detection)

# Symbol indices
SYM_ARROW_RIGHT  = 0
SYM_ARROW_LEFT   = 1
SYM_FORWARDS     = 2
SYM_BACKWARDS    = 3
SYM_OCTAGON      = 4
SYM_STAR         = 5
SYM_CIRC_SEG     = 6
SYM_PACMAN       = 7
SYM_TRAPEZOID    = 8
SYM_DIAMOND      = 9
SYM_CROSS        = 10
SYM_RECYCLE      = 11
SYM_WARNING      = 12
SYM_PRESS_BUTTON = 13
SYM_QR_CODE      = 14
SYM_FINGERPRINT  = 15

# ── Load class names ──────────────────────────────────────────────
try:
    with open(CLASS_NAMES_PATH) as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    CLASS_NAMES = [str(i) for i in range(16)]


class SymbolDetector:
    """
    Runs TFLite inference in a background thread.
    Call get_detection() from your main loop to get the latest result.
    """

    def __init__(self, frame_source_fn):
        """
        frame_source_fn: callable that returns the latest BGR frame (numpy array).
        Typically a lambda that reads from your picamera2 instance.
        """
        self._get_frame   = frame_source_fn
        self._interpreter = self._load_model()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Shared result — updated by background thread
        self._lock            = threading.Lock()
        self._symbol_id       = -1       # -1 = nothing confident
        self._symbol_name     = 'none'
        self._confidence      = 0.0
        self._detection_time  = 0.0

        # Start background thread
        self._running = True
        self._thread  = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print("[SymbolDetector] Started background inference thread")

    def _load_model(self):
        interp = Interpreter(model_path=MODEL_PATH)
        interp.allocate_tensors()
        print(f"[SymbolDetector] Model loaded: {MODEL_PATH}")
        return interp

    def _preprocess(self, frame):
        """Crop top half of frame (where signs will appear), resize, convert."""
        h, w = frame.shape[:2]
        # Signs are in the upper portion of the frame
        sign_roi = frame[0 : h // 2, :]
        img = cv2.resize(sign_roi, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Match quantization type expected by model
        input_type = self._input_details[0]['dtype']
        if input_type == np.uint8:
            return img.astype(np.uint8)[np.newaxis]
        else:
            # Float model — normalise to [-1, 1]
            return ((img.astype(np.float32) / 127.5) - 1.0)[np.newaxis]

    def _inference_loop(self):
        while self._running:
            t0 = time.perf_counter()
            try:
                frame = self._get_frame()
                if frame is None:
                    time.sleep(INFERENCE_INTERVAL)
                    continue

                inp = self._preprocess(frame)
                self._interpreter.set_tensor(self._input_details[0]['index'], inp)
                self._interpreter.invoke()

                output = self._interpreter.get_tensor(self._output_details[0]['index'])

                # Dequantize if INT8 output
                if self._output_details[0]['dtype'] == np.uint8:
                    scale, zero_point = self._output_details[0]['quantization']
                    probs = (output.astype(np.float32) - zero_point) * scale
                else:
                    probs = output.astype(np.float32)

                probs     = probs[0]
                best_idx  = int(np.argmax(probs))
                best_conf = float(probs[best_idx])

                with self._lock:
                    if best_conf >= CONFIDENCE_THRESHOLD:
                        self._symbol_id      = best_idx
                        self._symbol_name    = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else str(best_idx)
                        self._confidence     = best_conf
                        self._detection_time = time.perf_counter()
                    else:
                        self._symbol_id   = -1
                        self._symbol_name = 'none'
                        self._confidence  = best_conf

            except Exception as e:
                print(f"[SymbolDetector] Inference error: {e}")

            # Sleep for remainder of interval
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, INFERENCE_INTERVAL - elapsed)
            time.sleep(sleep_t)

    def get_detection(self):
        """
        Returns (symbol_id, symbol_name, confidence).
        symbol_id = -1 means no confident detection.
        """
        with self._lock:
            return self._symbol_id, self._symbol_name, self._confidence

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


# ── Symbol behaviour mapping ──────────────────────────────────────
# Maps symbol_id → (description, suggested_action)
SYMBOL_BEHAVIOURS = {
    SYM_ARROW_RIGHT:  ('Arrow Right',   'turn_right'),
    SYM_ARROW_LEFT:   ('Arrow Left',    'turn_left'),
    SYM_FORWARDS:     ('Arrow Forward', 'go_forward'),
    SYM_BACKWARDS:    ('Arrow Back',    'go_back'),
    SYM_OCTAGON:      ('Octagon/Stop',  'stop'),
    SYM_STAR:         ('Star',          'none'),
    SYM_CIRC_SEG:     ('Circ Segment',  'none'),
    SYM_PACMAN:       ('Pac-Man',       'none'),
    SYM_TRAPEZOID:    ('Trapezoid',     'none'),
    SYM_DIAMOND:      ('Diamond',       'none'),
    SYM_CROSS:        ('Cross',         'none'),
    SYM_RECYCLE:      ('Recycle',       'none'),
    SYM_WARNING:      ('Warning',       'slow'),
    SYM_PRESS_BUTTON: ('Press Button',  'none'),
    SYM_QR_CODE:      ('QR Code',       'none'),
    SYM_FINGERPRINT:  ('Fingerprint',   'none'),
}

def get_action(symbol_id):
    """Returns the action string for a given symbol_id, or 'none'."""
    return SYMBOL_BEHAVIOURS.get(symbol_id, ('Unknown', 'none'))[1]
