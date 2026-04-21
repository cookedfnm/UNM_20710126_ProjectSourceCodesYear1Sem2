import time
import signal
from picamera2 import Picamera2
from . import config as cfg
from .shm import open_frame_shm, write_frame

def camera_proc(shm_name, frame_lock, stop_evt):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shm, buf = open_frame_shm(shm_name)

    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (cfg.FRAME_W, cfg.FRAME_H)},
        sensor={"output_size": (1640, 1232)}
    )
    picam2.configure(cam_config)

    picam2.start()
    picam2.set_controls({
	'AeEnable': False,
	"ColourGains": (1.5, 1.5),
	'ExposureTime': 5000,
	'AnalogueGain': 4.5
    })
    time.sleep(1.0)

    try:
        while not stop_evt.is_set():
            frame = picam2.capture_array()
            write_frame(buf, frame, frame_lock)
            time.sleep(0.001)
    finally:
        picam2.stop()
        shm.close()
