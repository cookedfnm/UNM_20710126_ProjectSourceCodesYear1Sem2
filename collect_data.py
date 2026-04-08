"""
collect_data.py — Pi dataset collection for downward-facing camera.

Setup assumed:
    - IMX219 (or similar) mounted above the floor, pointed straight down
    - Symbols printed in TWO physical sizes (mixed into the same class folder)
    - Symbols sit ON the black line (line runs under/through them)
    - Car positioned over the symbol so it's centred in the camera view
    - Classifier will use a 240x240 centre crop, so the symbol MUST fit
      inside the orange guide box

Controls:
    S         = save ONE frame (single shot)
    B         = BURST capture 10 frames over 2s — wiggle the car slightly!
    N / P     = next / previous class
    G         = jump to background class
    Q / ESC   = quit

The saved file is the CLEAN 640x480 frame with NO overlay text baked in.
Filenames are auto-numbered per class:
    dataset/4_octagon/0000.jpg
    dataset/4_octagon/0001.jpg
    ...
"""

from picamera2 import Picamera2
import cv2
import os
import time

# ── Classes ──────────────────────────────────────────────────────
CLASS_NAMES = [
    '0_arrow_right',
    '1_arrow_left',
    '2_arrow_forwards',
    '3_arrow_backwards',
    '4_octagon',
    '5_star',
    '6_circular_segment',
    '7_pacman',
    '8_trapezoid',
    '9_diamond',
    '10_cross',
    '11_recycle',
    '12_warning',
    '13_press_button',
    '14_qr_code',
    '15_fingerprint',
    '16_background',
]

ARROW_CLASSES = {0, 1, 2, 3}
BACKGROUND_CLASS = 16

PER_CLASS_TARGET   = 144
BACKGROUND_TARGET  = 200

# ── Capture / crop config ───────────────────────────────────────
CAPTURE_W, CAPTURE_H = 640, 480
CROP_SIZE = 240

# ── Burst settings ──────────────────────────────────────────────
BURST_COUNT    = 10
BURST_DURATION = 2.0   # seconds

DATASET_DIR = 'dataset'
os.makedirs(DATASET_DIR, exist_ok=True)
for name in CLASS_NAMES:
    os.makedirs(os.path.join(DATASET_DIR, name), exist_ok=True)

# ── Camera ──────────────────────────────────────────────────────
# FOV fix (Option B): include a `raw` stream at the full sensor resolution.
# This forces Picamera2 to use the entire sensor area as the source for the
# downscaled `main` stream — instead of the default behaviour of cropping
# to a smaller sensor sub-region.
picam2 = Picamera2()
full_w, full_h = picam2.sensor_resolution
print(f"[camera] sensor resolution: {full_w}x{full_h}")

config = picam2.create_preview_configuration(
    main={"size": (CAPTURE_W, CAPTURE_H)},
    raw={"size": (full_w, full_h)},
)
picam2.configure(config)
print(f"[camera] full sensor area → downscaled to {CAPTURE_W}x{CAPTURE_H}")

picam2.start()
time.sleep(1)

cv2.namedWindow("Collect Data")
cv2.moveWindow("Collect Data", 0, 0)

current_class = 0


def list_class_files(class_name):
    path = os.path.join(DATASET_DIR, class_name)
    return sorted(f for f in os.listdir(path) if f.endswith('.jpg'))


def count_images(class_name):
    return len(list_class_files(class_name))


def next_index(class_name):
    """Next sequential numeric prefix for this class folder."""
    files = list_class_files(class_name)
    if not files:
        return 0
    nums = []
    for f in files:
        try:
            nums.append(int(f.split('.', 1)[0]))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 0


def target_for(class_idx):
    return BACKGROUND_TARGET if class_idx == BACKGROUND_CLASS else PER_CLASS_TARGET


def build_filename(class_name, idx):
    return os.path.join(DATASET_DIR, class_name, f"{idx:04d}.jpg")


def save_clean(class_name, clean_frame):
    idx = next_index(class_name)
    fname = build_filename(class_name, idx)
    cv2.imwrite(fname, clean_frame)
    return fname


def draw_overlay(display, class_name, class_idx, n_imgs, burst_state=None):
    h, w = display.shape[:2]

    cx, cy = w // 2, h // 2
    half = CROP_SIZE // 2

    # Centre crop guide
    box_colour = (0, 0, 255) if burst_state else (0, 165, 255)
    cv2.rectangle(display, (cx - half, cy - half), (cx + half, cy + half),
                  box_colour, 2)
    cv2.putText(display, "symbol must fit inside",
                (cx - half, cy - half - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_colour, 1)

    # Crosshair
    cv2.line(display, (cx - 10, cy), (cx + 10, cy), box_colour, 1)
    cv2.line(display, (cx, cy - 10), (cx, cy + 10), box_colour, 1)

    # Header
    target = target_for(class_idx)
    pct = min(100, int(100 * n_imgs / target))
    header = f"[{class_idx:02d}] {class_name}   {n_imgs}/{target}  ({pct}%)"
    cv2.putText(display, header, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Progress bar
    bar_w = w - 20
    cv2.rectangle(display, (10, 38), (10 + bar_w, 48), (60, 60, 60), -1)
    cv2.rectangle(display, (10, 38), (10 + int(bar_w * pct / 100), 48),
                  (0, 255, 0), -1)

    # Class-specific reminder
    if class_idx in ARROW_CLASSES:
        reminder = "ARROW: head-on orientation only — never rotate the print"
        col = (0, 200, 255)
    elif class_idx == BACKGROUND_CLASS:
        reminder = "BACKGROUND: no symbol — empty track / line / colour / floor"
        col = (200, 200, 200)
    else:
        reminder = "SHAPE: vary position and small wiggle during burst"
        col = (200, 255, 200)
    cv2.putText(display, reminder, (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # Controls
    cv2.putText(display,
                "S=save  B=burst(10)  N/P=class  G=bg  Q=quit",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Burst indicator
    if burst_state is not None:
        i, total = burst_state
        msg = f"BURST {i}/{total}"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(display, msg,
                    (cx - tw // 2, cy + half + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def print_class_change(class_idx):
    name = CLASS_NAMES[class_idx]
    n = count_images(name)
    t = target_for(class_idx)
    print(f"→ [{class_idx:02d}] {name}   {n}/{t}")


def do_burst(class_name, class_idx):
    """Capture BURST_COUNT frames over BURST_DURATION seconds."""
    interval = BURST_DURATION / BURST_COUNT
    print(f"  burst start ({BURST_COUNT} frames over {BURST_DURATION}s) — wiggle the car!")
    saved = 0
    for i in range(BURST_COUNT):
        t0 = time.perf_counter()

        clean = picam2.capture_array()
        clean = cv2.cvtColor(clean, cv2.COLOR_BGRA2BGR)

        display = clean.copy()
        n_imgs = count_images(class_name)
        draw_overlay(display, class_name, class_idx, n_imgs,
                     burst_state=(i + 1, BURST_COUNT))
        cv2.imshow("Collect Data", display)
        cv2.waitKey(1)

        save_clean(class_name, clean)
        saved += 1

        elapsed = time.perf_counter() - t0
        sleep_t = max(0.0, interval - elapsed)
        time.sleep(sleep_t)

    n_imgs = count_images(class_name)
    print(f"  burst done — saved {saved}, total now {n_imgs}/{target_for(class_idx)}")


print("=" * 60)
print("DATASET COLLECTION")
print("=" * 60)
print(f"Camera: {CAPTURE_W}x{CAPTURE_H}, classifier crop: {CROP_SIZE}x{CROP_SIZE}")
print(f"Per-class target: {PER_CLASS_TARGET}  |  Background target: {BACKGROUND_TARGET}")
print()
print("Workflow:")
print("  1. Print symbols in TWO sizes — mix both into the same class folder")
print("  2. Mostly straight track sections, ~20% on curves")
print("  3. Move to a different track location between bursts")
print("  4. Place symbol on the black line, centred in the orange box")
print("  5. Press B to burst-capture 10 frames over 2s — wiggle slightly")
print("  6. N to advance to the next class, repeat")
print()
print("Arrows specifically:")
print("  At least 30% of arrow photos should be taken at REAL junctions")
print()
print("Background class (G):")
print("  Burst-capture while pushing the car around empty track,")
print("  black line alone, colour sections alone, junctions alone")
print("=" * 60)
print()
print_class_change(current_class)

try:
    while True:
        clean = picam2.capture_array()
        clean = cv2.cvtColor(clean, cv2.COLOR_BGRA2BGR)

        class_name = CLASS_NAMES[current_class]
        n_imgs = count_images(class_name)

        display = clean.copy()
        draw_overlay(display, class_name, current_class, n_imgs)

        cv2.imshow("Collect Data", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            continue

        if key in (ord('q'), 27):
            break

        elif key == ord('s'):
            fname = save_clean(class_name, clean)
            n_imgs = count_images(class_name)
            print(f"  saved  {os.path.basename(fname)}  "
                  f"(total {n_imgs}/{target_for(current_class)})")

        elif key == ord('b'):
            do_burst(class_name, current_class)

        elif key == ord('n'):
            current_class = (current_class + 1) % len(CLASS_NAMES)
            print_class_change(current_class)

        elif key == ord('p'):
            current_class = (current_class - 1) % len(CLASS_NAMES)
            print_class_change(current_class)

        elif key == ord('g'):
            current_class = BACKGROUND_CLASS
            print_class_change(current_class)

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print()
    print("=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    total = 0
    for idx, name in enumerate(CLASS_NAMES):
        n = count_images(name)
        t = target_for(idx)
        bar = '#' * int(20 * min(1.0, n / t))
        bar = bar.ljust(20, '.')
        flag = ' ' if n >= t else '!'
        print(f" {flag} [{idx:02d}] {name:22s} [{bar}] {n}/{t}")
        total += n
    print(f"\n total images: {total}")
    print("\n copy ./dataset/ to your laptop for training.")
