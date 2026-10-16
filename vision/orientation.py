"""Suggest a video orientation from sampled upright person detections.

The suggestion is only a UI hint. Operators can override it after viewing the
preview, and no event is created during this check.
"""

from pathlib import Path

import cv2


def suggest_video_orientation(source: str, model_path: str | None = None) -> str:
    """Return the best-supported rotation, or ``As shown`` when ambiguous."""
    from ultralytics import YOLO

    model = YOLO(model_path or str(Path(__file__).resolve().parents[1] / "yolov8n.pt"))
    scores = {label: 0.0 for label in (
        "As shown", "Turn left 90°", "Turn right 90°", "Turn upside down",
    )}
    capture = cv2.VideoCapture(source)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or count <= 0:
            return "As shown"
        duration = count / fps
        for fraction in (0.02, 0.15, 0.3, 0.55, 0.7):
            when = min(max(0.1, duration * fraction), duration * 0.9)
            capture.set(cv2.CAP_PROP_POS_MSEC, when * 1000)
            ok, image = capture.read()
            if not ok:
                continue
            rotated = (
                ("As shown", image),
                ("Turn left 90°", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ("Turn right 90°", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                ("Turn upside down", cv2.rotate(image, cv2.ROTATE_180)),
            )
            for label, frame in rotated:
                result = model.predict(frame, classes=[0], conf=0.25, verbose=False)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    width, height = x2 - x1, y2 - y1
                    if height > 0:
                        scores[label] += float(box.conf[0]) * min(1.0, height / max(width, 1))
    finally:
        capture.release()

    best = max(scores, key=scores.get)
    if best != "As shown" and scores[best] > max(1.0, scores["As shown"] * 1.5):
        return best
    return "As shown"
