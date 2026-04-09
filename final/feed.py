"""
Camera capture process.
Continuously grabs frames from PiCamera2 and writes them into shared memory.
"""

import time
import numpy as np
from picamera2 import Picamera2
from shared_state import (
    FRAME_W, FRAME_H, FRAME_SHAPE, open_frame_buffer,
)


def camera_process(frame_seq, frame_lock, quit_flag):
    """Target for multiprocessing.Process."""
    shm, frame_buf = open_frame_buffer(writable=True)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H)},
        controls={"FrameRate": 60},
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    print("[CAMERA] started")

    try:
        while not quit_flag.value:
            raw = picam2.capture_array()
            # PiCamera2 often gives BGRA or XRGB; convert to BGR
            if raw.shape[2] == 4:
                bgr = raw[:, :, :3].copy()  # drop alpha, ensure contiguous
            else:
                bgr = raw

            with frame_lock:
                np.copyto(frame_buf, bgr)
                frame_seq.value += 1

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        shm.close()
        print("[CAMERA] stopped")
