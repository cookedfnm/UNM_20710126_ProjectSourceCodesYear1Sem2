import time
import os
import cv2
import numpy as np
import face_recognition
from . import config as cfg
from .motors import set_motors, stop
from .shm import read_frame
from .vision import find_line

def handle_symbol(label, pid, shm_arr, frame_lock, color_priority):
    label = label.strip()
    print("[SYMBOL] Executing: " + label)

    if label == "Arrow Right":
        spd = cfg.ARROW_SEARCH_SPEED
        set_motors(spd, -spd)
        time.sleep(cfg.ARROW_HARD_TURN_TIME)
        _spin_until_line(shm_arr, frame_lock, color_priority,
                         direction="left")

    elif label == "Arrow Left":
        spd = cfg.ARROW_SEARCH_SPEED
        set_motors(-spd, spd)
        time.sleep(cfg.ARROW_HARD_TURN_TIME)
        _spin_until_line(shm_arr, frame_lock, color_priority,
                         direction="right")

    elif label in ("Hazard", "Green Hand"):
        stop()
        time.sleep(1.0)

    elif label in ("QR Code", "Fingerprint"):
        set_motors(0, 0)
        _face_recognition(shm_arr, frame_lock)

    elif label == "Recycle":
        set_motors(100, -100)
        time.sleep(1.5)
        stop()
        time.sleep(0.2)

    elif label in ("Cross", "Octagon", "Star", "Diamond",
                    "Trapezoid", "Quarter Circle", "Semi Circle"):
        stop()
        time.sleep(0.1)
        print("  " + label + " detected")

    else:
        stop()
        time.sleep(0.2)

    pid.reset()

def _spin_until_line(shm_arr, lock, color_priority, direction="left"):
    spd = cfg.ARROW_SEARCH_SPEED
    if direction == "left":
        set_motors(-spd, spd)
    else:
        set_motors(spd, -spd)

    while True:
        frame = read_frame(shm_arr, lock)
        cx, _, _, _ = find_line(frame, color_priority)
        if cx is not None:
            break
        time.sleep(0.03)
    stop()
    time.sleep(0.1)

def _face_recognition(shm_arr, lock):
    known_files = [
        ("Colin.jpg",        "Colin Yap"),
        ("Dr. Hermawan.jpg", "Dr. Hermawan"),
        ("Edmund.jpg",       "Edmund"),
        ("Radhita.jpg",      "Radhita"),
        ("Austin.jpg",       "Austin"),
    ]
    encs, names = [], []
    for path, name in known_files:
        if not os.path.exists(path):
            continue
        img = face_recognition.load_image_file(path)
        e = face_recognition.face_encodings(img)
        if e:
            encs.append(e[0])
            names.append(name)

    if not encs:
        print("[FACE] No known-face encodings loaded.")
        return

    start = time.time()
    while (time.time() - start) < 10:
        frame = read_frame(shm_arr, lock)
        locs = face_recognition.face_locations(frame)
        fencs = face_recognition.face_encodings(frame, locs)
        for (top, right, bottom, left), fe in zip(locs, fencs):
            matches = face_recognition.compare_faces(encs, fe)
            dists = face_recognition.face_distance(encs, fe)
            idx = np.argmin(dists)
            name = names[idx] if matches[idx] else "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                          (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Recognition", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Face Recognition")
