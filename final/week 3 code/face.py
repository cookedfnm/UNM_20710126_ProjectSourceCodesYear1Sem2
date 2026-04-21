import os
import time
import cv2
import numpy as np
import face_recognition

def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    face_files = [
        ("/home/benedictngai/pw3/final/linefollower2/images/ben.jpeg", "Ben"),
        ("/home/benedictngai/pw3/final/linefollower2/images/dr-hermawan.jpeg", "Dr Hermawan"),
    ]

    for path, name in face_files:
        if not os.path.exists(path):
            print(f"[FACE] Missing file: {path}")
            continue

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"[FACE] No face found in: {path}")
            continue

        known_face_encodings.append(encodings[0])
        known_face_names.append(name)

    return known_face_encodings, known_face_names

def run_face_recognition(shm_arr, lock, duration=10):
    known_face_encodings, known_face_names = load_known_faces()

    if not known_face_encodings:
        print("[FACE] No known-face encodings loaded.")
        return

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    start = time.time()

    while (time.time() - start) < duration:
        with lock:
            frame = shm_arr.copy()

        rgb_frame = frame.copy()

        if process_this_frame:
            small_rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

            face_locations = face_recognition.face_locations(small_rgb_frame)
            face_encodings = face_recognition.face_encodings(
                small_rgb_frame, face_locations
            )

            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"

                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            print("[FACE] Names:", face_names)
            print("[FACE] Locations:", face_locations)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(
                rgb_frame,
                (left, bottom - 35),
                (right, bottom),
                (255, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                rgb_frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                1,
            )

        disp = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Recognition", disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Face Recognition")
