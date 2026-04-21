import numpy as np
from multiprocessing import shared_memory
from . import config as cfg

def create_frame_shm(name):
    shape = (cfg.FRAME_H, cfg.FRAME_W, 3)
    nbytes = int(np.prod(shape))
    try:
        stale = shared_memory.SharedMemory(name=name, create=False)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass
    shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    arr[:] = 0
    return shm, arr

def open_frame_shm(name):
    shape = (cfg.FRAME_H, cfg.FRAME_W, 3)
    shm = shared_memory.SharedMemory(name=name, create=False)
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    return shm, arr

def read_frame(arr, lock):
    with lock:
        return arr.copy()

def write_frame(arr, frame, lock):
    with lock:
        np.copyto(arr, frame)
