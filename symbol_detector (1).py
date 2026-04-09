"""
symbol_detect.py — Classical CV symbol classifier (no YOLO)

Drop-in replacement for the AI subprocess. Plugs into the same
SharedMemory frame source and multiprocessing.Value interface.

Pipeline:
  Stage 1  Blob detection in upper frame region (< 2 ms)
  Stage 2  Contour feature triage: color, vertices, solidity,
           circularity, Hu moments  (< 3 ms)
  Stage 3  Template-match confirmation against 2-3 candidates (< 5 ms)

Total per-frame: ~5-10 ms on Pi 5 (vs 50-150 ms for YOLO-NCNN).
"""

import cv2
import numpy as np
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

# ─── CONFIGURATION ──────────────────────────────────────────────────
# These mirror your config.py constants. Adjust or import from config.

FRAME_W, FRAME_H = 300, 300

# Symbol detection ROI: upper 70% of frame (line ROI is bottom 40%)
# Overlap is intentional — symbols ON the line can appear in lower-mid frame
SYMBOL_ROI_TOP = 0.0
SYMBOL_ROI_BOTTOM = 0.75  # ignore bottom 25% (pure line zone)

# Blob filtering
MIN_SYMBOL_AREA = 800      # px² — reject tiny noise
MAX_SYMBOL_AREA = 45000    # px² — reject if blob is most of the frame
MIN_SOLIDITY_BLOB = 0.25   # recycle symbol has low solidity
MAX_ASPECT_RATIO = 3.5     # reject very elongated (line segments)

# Template matching
TEMPLATE_SIZE = (64, 64)   # all templates resized to this
MATCH_THRESHOLD = 0.55     # NCC threshold to accept a match

# Confidence required to publish a detection
MIN_CONFIDENCE = 0.60

# Cooldown: don't re-detect same symbol within N seconds
DETECTION_COOLDOWN = 2.0

# Label map (matches your config.LABEL_MAP)
LABEL_MAP = {
    0: None,
    1: "Arrow Left", 2: "Arrow Right", 3: "Arrow Up", 4: "Arrow Down",
    5: "Hazard", 6: "Green Hand", 7: "QR Code", 8: "Fingerprint",
    9: "Recycle", 10: "Cross", 11: "Octagon", 12: "Star",
    13: "Diamond", 14: "Trapezoid", 15: "Quarter Circle", 16: "Semi Circle",
}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items() if v is not None}

# ─── COLOR GROUPS ───────────────────────────────────────────────────
# HSV ranges for each color family present in the symbols.
# These are for SYMBOL ISOLATION (not line following).
# Tuned for the exact designs shown. Adjust if printing differs.

COLOR_GROUPS = {
    "blue": {
        "hsv_ranges": [((95, 80, 80), (130, 255, 255))],
        "labels": ["Arrow Right", "Quarter Circle", "QR Code", "Octagon"],
    },
    "green": {
        "hsv_ranges": [((35, 80, 80), (85, 255, 255))],
        "labels": ["Arrow Up", "Recycle", "Green Hand"],
    },
    "red": {
        "hsv_ranges": [
            ((0, 80, 80), (10, 255, 255)),
            ((170, 80, 80), (180, 255, 255)),
        ],
        "labels": ["Arrow Down", "Semi Circle"],
    },
    "orange": {
        "hsv_ranges": [((10, 100, 100), (25, 255, 255))],
        "labels": ["Arrow Left", "Cross"],
    },
    "purple": {
        "hsv_ranges": [((125, 40, 40), (165, 255, 255))],
        "labels": ["Diamond", "Trapezoid", "Fingerprint"],
    },
    "yellow": {
        "hsv_ranges": [((20, 100, 100), (35, 255, 255))],
        "labels": ["Star", "Hazard"],
    },
}


# ─── SHAPE FEATURES ─────────────────────────────────────────────────

@dataclass
class ShapeFeatures:
    """Geometric features extracted from a contour."""
    vertices: int          # approxPolyDP vertex count
    solidity: float        # area / convex hull area
    circularity: float     # 4π·area / perimeter²
    aspect_ratio: float    # bounding rect w/h
    hu_moments: np.ndarray # 7 Hu moments (log-transformed)
    extent: float          # area / bounding rect area
    area: float
    centroid: tuple        # (cx, cy)
    bbox: tuple            # (x, y, w, h)


def extract_features(contour: np.ndarray) -> Optional[ShapeFeatures]:
    """Extract geometric features from a contour."""
    area = cv2.contourArea(contour)
    if area < MIN_SYMBOL_AREA:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    # Vertices via polygon approximation
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    # Bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0

    # Hu moments
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform for scale invariance
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # Centroid
    cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else x + w // 2
    cy = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else y + h // 2

    return ShapeFeatures(
        vertices=vertices,
        solidity=solidity,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        hu_moments=hu_log,
        extent=extent,
        area=area,
        centroid=(cx, cy),
        bbox=(x, y, w, h),
    )


# ─── FEATURE-BASED TRIAGE ──────────────────────────────────────────
# Rules that narrow candidates before template matching.

# Expected feature ranges per symbol (vertices, solidity, circularity)
# These are approximate — the template match is the final arbiter.
SHAPE_RULES = {
    "Star":           {"vertices": (8, 14),  "solidity": (0.45, 0.70), "circularity": (0.15, 0.50)},
    "Octagon":        {"vertices": (7, 10),  "solidity": (0.90, 1.00), "circularity": (0.75, 0.95)},
    "Diamond":        {"vertices": (3, 6),   "solidity": (0.90, 1.00), "circularity": (0.55, 0.85)},
    "Trapezoid":      {"vertices": (3, 6),   "solidity": (0.90, 1.00), "circularity": (0.55, 0.85)},
    "Cross":          {"vertices": (10, 16), "solidity": (0.75, 0.95), "circularity": (0.20, 0.55)},
    "Quarter Circle": {"vertices": (4, 10),  "solidity": (0.85, 1.00), "circularity": (0.50, 0.85)},
    "Semi Circle":    {"vertices": (5, 14),  "solidity": (0.85, 1.00), "circularity": (0.60, 0.90)},
    "Arrow Left":     {"vertices": (5, 10),  "solidity": (0.55, 0.85), "circularity": (0.15, 0.55)},
    "Arrow Right":    {"vertices": (5, 10),  "solidity": (0.55, 0.85), "circularity": (0.15, 0.55)},
    "Arrow Up":       {"vertices": (5, 10),  "solidity": (0.55, 0.85), "circularity": (0.15, 0.55)},
    "Arrow Down":     {"vertices": (5, 10),  "solidity": (0.55, 0.85), "circularity": (0.15, 0.55)},
    "Recycle":        {"vertices": (8, 20),  "solidity": (0.30, 0.65), "circularity": (0.08, 0.35)},
    "Hazard":         {"vertices": (6, 16),  "solidity": (0.55, 0.90), "circularity": (0.30, 0.75)},
    "Green Hand":     {"vertices": (4, 12),  "solidity": (0.75, 1.00), "circularity": (0.40, 0.85)},
    "QR Code":        {"vertices": (4, 20),  "solidity": (0.60, 1.00), "circularity": (0.30, 0.85)},
    "Fingerprint":    {"vertices": (6, 20),  "solidity": (0.30, 0.70), "circularity": (0.10, 0.45)},
}


def triage_candidates(features: ShapeFeatures, color_labels: list) -> list:
    """
    Given extracted features and the color-group's possible labels,
    return a ranked list of candidate labels (most likely first).
    """
    scored = []
    for label in color_labels:
        rules = SHAPE_RULES.get(label)
        if rules is None:
            continue

        score = 0.0
        total = 0.0

        # Vertex count match
        vmin, vmax = rules["vertices"]
        total += 1.0
        if vmin <= features.vertices <= vmax:
            # Closer to center of range = higher score
            center = (vmin + vmax) / 2
            dist = abs(features.vertices - center) / ((vmax - vmin) / 2 + 1)
            score += max(0, 1.0 - dist * 0.5)

        # Solidity match
        smin, smax = rules["solidity"]
        total += 1.0
        if smin <= features.solidity <= smax:
            score += 1.0
        elif features.solidity < smin:
            score += max(0, 1.0 - (smin - features.solidity) * 5)
        else:
            score += max(0, 1.0 - (features.solidity - smax) * 5)

        # Circularity match
        cmin, cmax = rules["circularity"]
        total += 1.0
        if cmin <= features.circularity <= cmax:
            score += 1.0
        elif features.circularity < cmin:
            score += max(0, 1.0 - (cmin - features.circularity) * 5)
        else:
            score += max(0, 1.0 - (features.circularity - cmax) * 5)

        if total > 0:
            scored.append((label, score / total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ─── ARROW DIRECTION DETECTION ──────────────────────────────────────

def detect_arrow_direction(contour: np.ndarray, features: ShapeFeatures) -> str:
    """
    Determine arrow direction from contour geometry.
    Uses centroid vs bounding-box center (same logic as your existing ai.py).
    """
    cx, cy = features.centroid
    bx = features.bbox[0] + features.bbox[2] // 2
    by = features.bbox[1] + features.bbox[3] // 2

    dx = cx - bx
    dy = cy - by

    if abs(dx) > abs(dy):
        # Horizontal arrow — centroid is toward the TAIL (heavy side)
        # so if centroid is LEFT of center, arrow points RIGHT
        return "Arrow Right" if dx < 0 else "Arrow Left"
    else:
        # Vertical arrow — centroid toward tail
        return "Arrow Down" if dy < 0 else "Arrow Up"


# ─── TEMPLATE STORE ─────────────────────────────────────────────────

class TemplateStore:
    """
    Loads and caches reference templates for each symbol.
    Templates should be placed in a `templates/` directory as:
        templates/arrow_left.png
        templates/arrow_right.png
        templates/star.png
        etc.

    Each template is a clean image of the symbol on white background,
    stored as both grayscale and binary mask.
    """

    def __init__(self, template_dir: str = "templates"):
        self.template_dir = template_dir
        self.templates = {}  # label -> (gray_template, binary_mask)
        self._load_templates()

    def _label_to_filename(self, label: str) -> str:
        return label.lower().replace(" ", "_") + ".png"

    def _load_templates(self):
        if not os.path.isdir(self.template_dir):
            logging.warning(
                f"Template directory '{self.template_dir}' not found. "
                f"Template matching will be disabled. Run generate_templates() first."
            )
            return

        for label_id, label in LABEL_MAP.items():
            if label is None:
                continue
            fname = self._label_to_filename(label)
            path = os.path.join(self.template_dir, fname)
            if not os.path.isfile(path):
                logging.warning(f"Missing template: {path}")
                continue

            img = cv2.imread(path)
            if img is None:
                logging.warning(f"Failed to read template: {path}")
                continue

            # Resize to standard size
            img = cv2.resize(img, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Binary mask (symbol pixels)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

            self.templates[label] = (gray, mask)

        logging.info(f"Loaded {len(self.templates)} symbol templates")

    def match(self, crop_gray: np.ndarray, candidates: list) -> tuple:
        """
        Match a grayscale crop against candidate templates.
        Returns (best_label, confidence) or (None, 0.0).

        candidates: list of (label, triage_score) from triage step.
        """
        if not self.templates:
            return None, 0.0

        crop_resized = cv2.resize(crop_gray, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)

        best_label = None
        best_score = 0.0

        for label, triage_score in candidates[:4]:  # check top 4 at most
            if label not in self.templates:
                continue

            tmpl_gray, tmpl_mask = self.templates[label]

            # Normalized cross-correlation
            result = cv2.matchTemplate(
                crop_resized, tmpl_gray, cv2.TM_CCOEFF_NORMED
            )
            ncc = result[0][0]  # single-pixel result since same size

            # Combined score: template match weighted more than triage
            combined = 0.65 * ncc + 0.35 * triage_score

            if combined > best_score:
                best_score = combined
                best_label = label

        if best_score < MATCH_THRESHOLD:
            return None, 0.0

        return best_label, best_score


# ─── MAIN DETECTOR CLASS ───────────────────────────────────────────

class SymbolDetector:
    """
    Main symbol detection pipeline.
    Call detect(frame_rgb) each frame — returns (label, confidence) or (None, 0).
    """

    def __init__(self, template_dir: str = "templates"):
        self.templates = TemplateStore(template_dir)
        self.last_detection_time = 0.0
        self.last_label = None

        # Pre-compute HSV masks for each color group
        self._color_groups = []
        for color_name, group in COLOR_GROUPS.items():
            self._color_groups.append({
                "name": color_name,
                "ranges": group["hsv_ranges"],
                "labels": group["labels"],
            })

    def detect(self, frame_rgb: np.ndarray) -> tuple:
        """
        Run the full detection pipeline on one frame.

        Args:
            frame_rgb: 300x300 RGB frame from camera

        Returns:
            (label_str, confidence) or (None, 0.0)
        """
        now = time.monotonic()

        # ── Stage 0: Extract symbol ROI ──
        h, w = frame_rgb.shape[:2]
        y1 = int(h * SYMBOL_ROI_TOP)
        y2 = int(h * SYMBOL_ROI_BOTTOM)
        roi = frame_rgb[y1:y2, :]

        # Convert to HSV once
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # ── Stage 1: Color-based blob detection ──
        best_result = (None, 0.0)

        for group in self._color_groups:
            # Build mask for this color group
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lo, hi) in group["ranges"]:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_SYMBOL_AREA or area > MAX_SYMBOL_AREA:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                ar = max(bw, bh) / (min(bw, bh) + 1)
                if ar > MAX_ASPECT_RATIO:
                    continue  # too elongated — probably the line

                # ── Stage 2: Feature extraction + triage ──
                features = extract_features(cnt)
                if features is None:
                    continue

                candidates = triage_candidates(features, group["labels"])
                if not candidates:
                    continue

                # ── Handle arrows specially ──
                top_label = candidates[0][0]
                is_arrow_candidate = any(
                    "Arrow" in c[0] for c in candidates[:2]
                )

                if is_arrow_candidate:
                    # Determine direction from contour geometry
                    direction = detect_arrow_direction(cnt, features)
                    # Replace generic arrow candidates with specific direction
                    # but keep the triage score
                    arrow_score = max(
                        (s for l, s in candidates if "Arrow" in l), default=0.5
                    )
                    candidates = [(direction, arrow_score)] + [
                        (l, s) for l, s in candidates if "Arrow" not in l
                    ]

                # ── Stage 3: Template match confirmation ──
                # Crop the bounding box region from grayscale
                pad = 5
                cx1 = max(0, x - pad)
                cy1 = max(0, y - pad)
                cx2 = min(gray.shape[1], x + bw + pad)
                cy2 = min(gray.shape[0], y + bh + pad)
                crop = gray[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    continue

                label, conf = self.templates.match(crop, candidates)

                # If no templates loaded, fall back to triage-only
                if label is None and not self.templates.templates:
                    label = candidates[0][0]
                    conf = candidates[0][1] * 0.75  # discount without template

                if label is None:
                    continue

                # Cooldown check
                if (label == self.last_label and
                        now - self.last_detection_time < DETECTION_COOLDOWN):
                    continue

                if conf > best_result[1]:
                    best_result = (label, conf)

        # Update tracking
        if best_result[0] is not None and best_result[1] >= MIN_CONFIDENCE:
            self.last_detection_time = now
            self.last_label = best_result[0]
            return best_result

        return (None, 0.0)


# ─── SUBPROCESS ENTRY POINT ─────────────────────────────────────────
# Drop-in replacement for your ai.py's ai_proc()

def symbol_proc(shm_name, shm_lock, label_val, conf_val, new_flag, ai_lock,
                template_dir="templates"):
    """
    Subprocess that replaces ai_proc().
    Reads frames from shared memory, runs classical CV detection,
    publishes results through the same Value interface.

    Args:
        shm_name:     name of the POSIX shared memory segment
        shm_lock:     multiprocessing.Lock for frame SHM access
        label_val:    multiprocessing.Value('i') — label ID
        conf_val:     multiprocessing.Value('d') — confidence
        new_flag:     multiprocessing.Value('i') — new result flag
        ai_lock:      multiprocessing.Lock for result Values
        template_dir: path to template images
    """
    from multiprocessing.shared_memory import SharedMemory
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logging.basicConfig(level=logging.INFO)
    logging.info("[SymbolDetect] Starting classical CV symbol detector")

    # Open shared memory
    shm = SharedMemory(name=shm_name, create=False)
    frame_buf = np.ndarray((FRAME_H, FRAME_W, 3), dtype=np.uint8, buffer=shm.buf)

    detector = SymbolDetector(template_dir=template_dir)

    try:
        while True:
            # Read frame
            with shm_lock:
                frame = frame_buf.copy()

            # Detect
            label_str, confidence = detector.detect(frame)

            # Publish result
            if label_str is not None:
                label_id = LABEL_TO_ID.get(label_str, 0)
                with ai_lock:
                    label_val.value = label_id
                    conf_val.value = confidence
                    new_flag.value = 1
                logging.info(
                    f"[SymbolDetect] {label_str} ({confidence:.2f})"
                )

            # Throttle to ~30 FPS detection rate (frame capture is separate)
            time.sleep(0.015)

    except Exception as e:
        logging.error(f"[SymbolDetect] Error: {e}")
    finally:
        shm.close()
        logging.info("[SymbolDetect] Stopped")


# ─── TEMPLATE GENERATION HELPER ────────────────────────────────────
# Run this once with your symbol images to create the templates/ dir.

def generate_templates(image_paths: dict, output_dir: str = "templates"):
    """
    Generate normalized template images from your symbol source images.

    Args:
        image_paths: dict of {"Arrow Left": "path/to/arrow_left.png", ...}
        output_dir:  where to save processed templates
    """
    os.makedirs(output_dir, exist_ok=True)

    for label, path in image_paths.items():
        img = cv2.imread(path)
        if img is None:
            print(f"WARNING: Could not read {path}")
            continue

        # Resize to standard size
        img = cv2.resize(img, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)

        # Save
        fname = label.lower().replace(" ", "_") + ".png"
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img)
        print(f"Saved template: {out_path}")

    print(f"\nGenerated {len(image_paths)} templates in '{output_dir}/'")


# ─── STANDALONE TEST ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python symbol_detect.py <image_path>")
        print("  Tests detection on a single image.")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read {sys.argv[1]}")
        sys.exit(1)

    img = cv2.resize(img, (FRAME_W, FRAME_H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector = SymbolDetector()
    label, conf = detector.detect(rgb)

    if label:
        print(f"Detected: {label} (confidence: {conf:.3f})")
    else:
        print("No symbol detected")
