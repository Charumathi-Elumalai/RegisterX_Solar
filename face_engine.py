"""
RegistraX Solar — face recognition engine.

Real models, not a toy comparison:
  - Detection : MTCNN (Zhang et al., 2016) — bundled weights, no network needed at runtime.
  - Recognition: ArcFace (Deng et al., 2019), 512-d embeddings — a published,
                 widely-used face recognition model (via the `deepface` library).
                 Weights auto-download once from a public GitHub release the
                 first time the server starts.
  - Liveness  : DeepFace's built-in Silent-Face-Anti-Spoofing model (FasNet),
                 a real trained anti-spoof classifier that looks at a single
                 frame for printing/screen-replay artifacts, PLUS a cheap
                 motion/duplicate-frame heuristic as a first-pass filter.

Honest limits (tell your client this — don't oversell it):
  - No face system, including this one, can promise 100% separation between
    very close look-alikes (e.g. identical twins) under all lighting/angles.
    What this pipeline does do: it requires the best match to clearly beat
    the next-closest match by a margin (MATCH_MARGIN below) before accepting
    it, and refuses to guess when two people score too close together —
    which is the main real-world failure mode with similar-looking relatives.
  - Anti-spoofing needs `torch` installed (see requirements.txt). If it's not
    available in a given environment, the system automatically falls back to
    the lighter motion/duplicate-frame heuristic only, and logs a warning —
    it never silently pretends to have full liveness checking.
"""
import base64
import json
import time
import logging

import numpy as np
import cv2

from database import get_db

logger = logging.getLogger("registrax.face_engine")

DETECTOR_BACKEND = "mtcnn"
MODEL_NAME = "ArcFace"

MIN_FACE_CONFIDENCE = 0.90        # MTCNN's own "is this a face" confidence
MIN_FACE_WIDTH_RATIO = 0.12       # face must fill at least this much of the frame width
MATCH_SIMILARITY_THRESHOLD = 0.62  # cosine similarity required to accept any match at all (was 0.55 — tightened)
MATCH_MARGIN = 0.08                # best match must beat the runner-up by this much, else "ambiguous" (was 0.06 — tightened for closer look-alike/twin separation)
# A blink or small head movement between two deliberately-spaced prompted
# frames should visibly change a meaningful fraction of the 8x8 fingerprint
# grid used by _frame_fingerprint(); a static photo/screen held in front of
# the camera won't. See check_prompted_liveness() below.
PROMPTED_LIVENESS_MIN_CHANGE = 3   # out of 64 grid cells — must change at least this many
PROMPTED_LIVENESS_MAX_CHANGE = 40  # ...but not look like a totally different scene/person swap

_ANTI_SPOOFING_AVAILABLE = None  # lazily determined on first use

_KNOWN_CACHE = []          # [{"employee_id": int, "embedding": np.ndarray}, ...]
_LIVENESS_BUFFERS = {}     # tracking_key -> list of recent-frame fingerprints


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def decode_base64_image(image_data):
    """Accepts a data URL (data:image/jpeg;base64,....) or raw base64 string."""
    try:
        if image_data is None:
            return None
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        logger.exception("Failed to decode incoming image")
        return None


def _lazy_deepface():
    from deepface import DeepFace
    return DeepFace


def _anti_spoofing_available():
    global _ANTI_SPOOFING_AVAILABLE
    if _ANTI_SPOOFING_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _ANTI_SPOOFING_AVAILABLE = True
        except ImportError:
            _ANTI_SPOOFING_AVAILABLE = False
            logger.warning(
                "torch is not installed — running WITHOUT the deep anti-spoofing "
                "model. Falling back to the lightweight motion/duplicate-frame "
                "check only. Install torch (see requirements.txt) for full liveness checking."
            )
    return _ANTI_SPOOFING_AVAILABLE


# ---------------------------------------------------------------------------
# Quality gate (runs on every frame before we even attempt recognition)
# ---------------------------------------------------------------------------
def detect_face_quality(img):
    """
    Cheap checks that run on every frame from the kiosk's live-scan loop
    (several times a second), so this needs to be fast and needs to fail
    quietly for the extremely common "nobody in frame yet" case.
    """
    if img is None:
        return {"ok": False, "reason": "no_image"}

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return {"ok": False, "reason": "no_image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 35:
        return {"ok": False, "reason": "too_dark"}
    if brightness > 235:
        return {"ok": False, "reason": "too_bright"}

    DeepFace = _lazy_deepface()
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=False,
        )
    except Exception:
        logger.exception("Face detector error")
        return {"ok": False, "reason": "detector_error"}

    real_faces = [f for f in faces if f.get("confidence", 0) >= MIN_FACE_CONFIDENCE]

    if len(real_faces) == 0:
        return {"ok": False, "reason": "no_face"}
    if len(real_faces) > 1:
        return {"ok": False, "reason": "multiple_faces"}

    area = real_faces[0]["facial_area"]
    if w == 0 or (area["w"] / w) < MIN_FACE_WIDTH_RATIO:
        return {"ok": False, "reason": "face_too_small"}

    if area.get("left_eye") is None or area.get("right_eye") is None:
        return {"ok": False, "reason": "eyes_not_visible"}

    return {"ok": True, "reason": "ok", "facial_area": area}


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------
def save_training_face(employee_id, img, sample_index=0):
    quality = detect_face_quality(img)
    if not quality["ok"]:
        return False

    DeepFace = _lazy_deepface()
    try:
        reps = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
    except Exception:
        logger.exception("Failed to compute face embedding during enrollment")
        return False

    if not reps:
        return False

    embedding = reps[0]["embedding"]

    conn = get_db()
    conn.execute(
        "INSERT INTO face_encodings (employee_id, sample_index, encoding) VALUES (?, ?, ?)",
        (employee_id, sample_index, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()
    return True


def train_model():
    """
    Not a training step in the ML sense (ArcFace's weights are fixed/pretrained) —
    this reloads every enrolled employee's stored embeddings from the DB into an
    in-memory cache so recognize_face() doesn't hit the database on every frame.
    Call this after any enrollment change, and once at server startup.
    """
    global _KNOWN_CACHE
    conn = get_db()
    rows = conn.execute("SELECT employee_id, encoding FROM face_encodings").fetchall()
    conn.close()

    cache = []
    for r in rows:
        try:
            cache.append({
                "employee_id": r["employee_id"],
                "embedding": np.array(json.loads(r["encoding"]), dtype=np.float32),
            })
        except Exception:
            continue
    _KNOWN_CACHE = cache
    logger.info("Face cache reloaded: %d samples across enrolled employees", len(cache))
    return len(cache)


def _cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


# ---------------------------------------------------------------------------
# Liveness (see module docstring for what this does and doesn't cover)
# ---------------------------------------------------------------------------
def _frame_fingerprint(img):
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (8, 8))
    avg = small.mean()
    return (small > avg).flatten()


def _passes_motion_heuristic(tracking_key, img, facial_area):
    now = time.time()
    buf = _LIVENESS_BUFFERS.setdefault(tracking_key, [])
    buf[:] = [b for b in buf if now - b["ts"] < 6]  # keep last ~6 seconds

    fp = _frame_fingerprint(img)
    cx = cy = None
    if facial_area:
        cx = facial_area["x"] + facial_area["w"] / 2
        cy = facial_area["y"] + facial_area["h"] / 2

    if buf:
        hamming = int(np.sum(fp != buf[-1]["fp"]))
        if hamming == 0:
            static_streak = buf[-1].get("static_streak", 0) + 1
            buf.append({"ts": now, "fp": fp, "cx": cx, "cy": cy, "static_streak": static_streak})
            if static_streak >= 2:
                return False, "duplicate_frame"
        else:
            buf.append({"ts": now, "fp": fp, "cx": cx, "cy": cy, "static_streak": 0})
    else:
        buf.append({"ts": now, "fp": fp, "cx": cx, "cy": cy, "static_streak": 0})

    if len(buf) >= 4 and cx is not None:
        xs = [b["cx"] for b in buf if b["cx"] is not None]
        ys = [b["cy"] for b in buf if b["cy"] is not None]
        if len(xs) >= 4 and (max(xs) - min(xs)) + (max(ys) - min(ys)) < 0.5:
            return False, "zero_motion"

    return True, "ok"


def _passes_deep_anti_spoof(img, facial_area):
    if not _anti_spoofing_available() or not facial_area:
        return True, "skipped"
    try:
        DeepFace = _lazy_deepface()
        result = DeepFace.extract_faces(
            img_path=img,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            anti_spoofing=True,
        )
        if not result:
            return True, "no_face_on_recheck"
        is_real = result[0].get("is_real", True)
        score = result[0].get("antispoof_score", None)
        return bool(is_real), f"antispoof_score={score}"
    except Exception:
        logger.exception("Anti-spoof model failed; allowing frame through on heuristic only")
        return True, "antispoof_error"


def check_liveness(tracking_key, img, facial_area):
    ok, reason = _passes_motion_heuristic(tracking_key, img, facial_area)
    if not ok:
        return False, reason
    return _passes_deep_anti_spoof(img, facial_area)


def check_prompted_liveness(img_before, img_after):
    """Active liveness for one-off verifications (the manual 'Scan Now'
    button and the PIN-fallback path) that don't get the benefit of the
    kiosk auto-scan loop's several-seconds-of-natural-micro-movement
    history: the UI prompts the person to blink or turn their head
    slightly between capturing img_before and img_after (see kiosk.js),
    and this checks that something genuinely, physically changed between
    the two frames — a printed photo or a phone held up to the camera
    can't blink or move on cue between two prompted captures.

    This is intentionally a lightweight, dependency-free heuristic (reuses
    the same 8x8 frame-fingerprint diff as the passive motion check) rather
    than true landmark-based eye-aspect-ratio blink detection, which would
    need a facial-landmark model (e.g. mediapipe FaceMesh or dlib) that
    isn't in requirements.txt today. Documenting that honestly rather than
    quietly overselling it — this catches "completely static" spoofing
    attempts (which is the common case), not a sophisticated video replay.
    """
    if img_before is None or img_after is None:
        return False, "missing_frame"
    try:
        fp_before = _frame_fingerprint(img_before)
        fp_after = _frame_fingerprint(img_after)
    except Exception:
        logger.exception("Prompted liveness fingerprinting failed")
        return False, "fingerprint_error"

    changed_cells = int(np.sum(fp_before != fp_after))
    if changed_cells < PROMPTED_LIVENESS_MIN_CHANGE:
        return False, "no_movement_detected"
    if changed_cells > PROMPTED_LIVENESS_MAX_CHANGE:
        return False, "scene_changed_too_much"
    return True, "ok"


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def recognize_face(img, tracking_key=None):
    """
    Returns (employee_id, confidence) or (None, best_score_seen) if no
    confident, unambiguous match was found.

    tracking_key: pass the site/kiosk id when calling this from a live
    camera loop, so the liveness buffer can track motion across frames for
    that specific kiosk. Omit (None) to skip liveness (e.g. for one-off
    manual checkin where there's no frame history to compare against).
    """
    if not _KNOWN_CACHE:
        train_model()
    if not _KNOWN_CACHE:
        return None, 0.0

    quality = detect_face_quality(img)
    if not quality["ok"]:
        return None, 0.0

    DeepFace = _lazy_deepface()
    try:
        reps = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
    except Exception:
        logger.exception("Failed to compute probe embedding")
        return None, 0.0

    if not reps:
        return None, 0.0

    probe = np.array(reps[0]["embedding"], dtype=np.float32)

    best_per_employee = {}
    for entry in _KNOWN_CACHE:
        sim = _cosine_similarity(probe, entry["embedding"])
        eid = entry["employee_id"]
        if eid not in best_per_employee or sim > best_per_employee[eid]:
            best_per_employee[eid] = sim

    if not best_per_employee:
        return None, 0.0

    ranked = sorted(best_per_employee.items(), key=lambda x: x[1], reverse=True)
    top_id, top_sim = ranked[0]
    second_sim = ranked[1][1] if len(ranked) > 1 else -1.0

    if top_sim < MATCH_SIMILARITY_THRESHOLD:
        return None, round(top_sim, 4)

    # Two people who genuinely look alike (siblings, close relatives) can both
    # score high similarity against one probe photo. Refusing to guess when
    # the top two candidates are too close together is the main real defence
    # here — it trades a few "please try again" prompts for not silently
    # marking the wrong person present.
    if (top_sim - second_sim) < MATCH_MARGIN:
        return None, round(top_sim, 4)

    if tracking_key is not None:
        live_ok, live_reason = check_liveness(tracking_key, img, quality.get("facial_area"))
        if not live_ok:
            logger.info("Liveness check failed for tracking_key=%s reason=%s", tracking_key, live_reason)
            return None, round(top_sim, 4)

    return top_id, round(top_sim, 4)
