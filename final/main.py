#!/usr/bin/env python3
"""
Main orchestrator for the line-following robot.

Processes:
  1. Camera   — captures frames into shared memory
  2. Line     — PID line following, writes motor commands
  3. Motor    — reads motor commands, drives L298N
  4. Symbol   — template matching at lower Hz

Keyboard (in terminal):
  Enter  = toggle running / paused
  q      = quit all
"""

import multiprocessing as mp
import signal
import sys
import os
import time

from shared_state import create_shared_state, MODE_NAMES
from feed import camera_process
from line_follower import line_follower_process
from movement import motor_process
from symbol_detector import symbol_detector_process, SYMBOL_NAMES

TEMPLATE_DIR = os.path.expanduser("~/pw3/final/templates")


def main():
    mp.set_start_method("spawn", force=True)

    state = create_shared_state()
    shm = state["shm"]

    procs = []

    # 1. Camera
    p_cam = mp.Process(
        target=camera_process,
        args=(state["frame_seq"], state["frame_lock"], state["quit_flag"]),
        daemon=True, name="camera",
    )
    procs.append(p_cam)

    # 2. Line follower
    p_line = mp.Process(
        target=line_follower_process,
        args=(
            state["frame_seq"], state["frame_lock"],
            state["running"], state["quit_flag"],
            state["motor_left"], state["motor_right"],
            state["line_found"], state["line_error"], state["line_mode"],
        ),
        daemon=True, name="line",
    )
    procs.append(p_line)

    # 3. Motor
    p_motor = mp.Process(
        target=motor_process,
        args=(state["motor_left"], state["motor_right"], state["quit_flag"]),
        daemon=True, name="motor",
    )
    procs.append(p_motor)

    # 4. Symbol detector
    p_sym = mp.Process(
        target=symbol_detector_process,
        args=(
            state["frame_seq"], state["frame_lock"], state["quit_flag"],
            state["symbol_id"], state["symbol_conf"], TEMPLATE_DIR,
        ),
        daemon=True, name="symbol",
    )
    procs.append(p_sym)

    for p in procs:
        p.start()

    print("=" * 50)
    print("  ROBOT READY")
    print("  Press ENTER to start/pause, 'q' to quit")
    print("=" * 50)

    def shutdown():
        state["quit_flag"].value = True
        state["motor_left"].value = 0.0
        state["motor_right"].value = 0.0
        time.sleep(0.3)
        for p in procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        shm.close()
        shm.unlink()
        print("Shutdown complete.")

    signal.signal(signal.SIGINT, lambda *_: None)  # handle in main loop

    try:
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break

            if cmd == "q":
                print("Quit requested.")
                break

            # Toggle running on Enter or any other key
            state["running"].value = not state["running"].value
            r = state["running"].value
            print(f"{'RUNNING' if r else 'PAUSED'}")

            # Print status
            if r:
                sid = state["symbol_id"].value
                sname = SYMBOL_NAMES[sid] if 0 <= sid < len(SYMBOL_NAMES) else "none"
                sconf = state["symbol_conf"].value
                mode = MODE_NAMES.get(state["line_mode"].value, "?")
                print(f"  mode={mode}  symbol={sname}({sconf:.2f})")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
