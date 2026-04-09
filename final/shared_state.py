"""
Shared memory layout for inter-process communication.

Frame buffer: raw BGR 640x480x3 in shared memory (921600 bytes)
Control/state: multiprocessing.Value / Array for small scalars
"""

import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

FRAME_W = 640
FRAME_H = 480
FRAME_CH = 3
FRAME_BYTES = FRAME_W * FRAME_H * FRAME_CH
FRAME_SHAPE = (FRAME_H, FRAME_W, FRAME_CH)
SHM_NAME = "robot_frame"


def create_shared_state():
    """Call once in main process. Returns dict of all shared objects."""

    # --- Frame buffer in shared memory ---
    try:
        old = shared_memory.SharedMemory(name=SHM_NAME)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=FRAME_BYTES)

    # --- Atomic scalars via mp.Value ---
    frame_seq = mp.Value(ctypes.c_uint64, 0)       # incremented each new frame
    running = mp.Value(ctypes.c_bool, False)        # pause / run toggle
    quit_flag = mp.Value(ctypes.c_bool, False)      # signal all processes to exit

    # Line follower -> movement
    motor_left = mp.Value(ctypes.c_double, 0.0)
    motor_right = mp.Value(ctypes.c_double, 0.0)

    # Line follower outputs (for debug / symbol logic)
    line_found = mp.Value(ctypes.c_bool, False)
    line_error = mp.Value(ctypes.c_double, 0.0)
    # active mode encoded as int: 0=search 1=black 2=red 3=yellow 4=lookahead_red 5=lookahead_yellow
    line_mode = mp.Value(ctypes.c_int, 0)

    # Symbol detector output: detected symbol index (-1 = none)
    # Symbol names stored separately (not shared), index maps to list
    symbol_id = mp.Value(ctypes.c_int, -1)
    symbol_conf = mp.Value(ctypes.c_double, 0.0)

    # Lock for frame buffer writes
    frame_lock = mp.Lock()

    return {
        "shm": shm,
        "frame_seq": frame_seq,
        "frame_lock": frame_lock,
        "running": running,
        "quit_flag": quit_flag,
        "motor_left": motor_left,
        "motor_right": motor_right,
        "line_found": line_found,
        "line_error": line_error,
        "line_mode": line_mode,
        "symbol_id": symbol_id,
        "symbol_conf": symbol_conf,
    }


def open_frame_buffer(writable=False):
    """Attach to the shared frame buffer. Returns (shm, numpy_array)."""
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    arr = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    if not writable:
        arr.flags.writeable = False
    return shm, arr


MODE_SEARCH = 0
MODE_BLACK = 1
MODE_RED = 2
MODE_YELLOW = 3
MODE_LOOK_RED = 4
MODE_LOOK_YELLOW = 5

MODE_NAMES = {
    MODE_SEARCH: "search",
    MODE_BLACK: "black",
    MODE_RED: "red",
    MODE_YELLOW: "yellow",
    MODE_LOOK_RED: "lookahead_red",
    MODE_LOOK_YELLOW: "lookahead_yellow",
}
