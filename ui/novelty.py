"""Small, local incident-report and footage-quality helpers for the dashboard."""

import base64
import html
from pathlib import Path

import cv2


def footage_quality(source: Path) -> dict:
    """Sample a few decoded frames; these are advisory image checks, not AI accuracy scores."""
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or frame_count <= 0:
            return {"sampled": 0, "brightness": None, "sharpness": None, "warnings": ["Footage quality could not be checked."]}
        readings = []
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, int(frame_count * fraction)))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 640 / max(width, height))
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            readings.append((float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())))
        if not readings:
            return {"sampled": 0, "brightness": None, "sharpness": None, "warnings": ["Footage quality could not be checked."]}
        brightness = sum(item[0] for item in readings) / len(readings)
        sharpness = sum(item[1] for item in readings) / len(readings)
        warnings = []
        if brightness < 45:
            warnings.append("Footage appears dark; people may be missed.")
        if sharpness < 45:
            warnings.append("Footage may be blurry or low detail; check the preview.")
        return {"sampled": len(readings), "brightness": brightness, "sharpness": sharpness, "warnings": warnings}
    finally:
        capture.release()


def incident_report_html(
    *, label: str, camera_id: str, created_at: str, video_time: str,
    summary: str, reason: str, review_status: str, review_note: str,
    keyframe_path: Path | None, detector_score: float | None = None,
) -> bytes:
    """Build a self-contained report that can be downloaded or printed to PDF."""
    escape = lambda value: html.escape(str(value), quote=True)
    picture = "<p>Keyframe unavailable.</p>"
    if keyframe_path and keyframe_path.is_file():
        encoded = base64.b64encode(keyframe_path.read_bytes()).decode("ascii")
        picture = f'<img alt="Incident keyframe" src="data:image/jpeg;base64,{encoded}">'
    score = (
        f"<p><strong>Detector score:</strong> {detector_score:.0%} (not a calibrated probability)</p>"
        if detector_score is not None else "<p><strong>Detector score:</strong> None — operator flag</p>"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Incident report</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:850px;margin:35px auto;padding:0 20px;color:#241927}}
header{{border-bottom:3px solid #8b5d86;margin-bottom:20px}}h1{{margin-bottom:0}}.tag{{color:#7b426f;font-weight:700}}
img{{max-width:100%;max-height:520px;border-radius:12px}}section{{margin:22px 0}}p{{white-space:pre-wrap}}
@media print{{body{{margin:15mm}}}}</style></head><body>
<header><h1>Incident report</h1><p class="tag">{escape(label)}</p></header>
<section><p><strong>Camera:</strong> {escape(camera_id)}</p>
<p><strong>Saved:</strong> {escape(created_at)}</p>
<p><strong>Video position:</strong> {escape(video_time)}</p>{score}</section>
<section><h2>Keyframe</h2>{picture}</section>
<section><h2>Description</h2><p>{escape(summary)}</p></section>
<section><h2>Why it was flagged</h2><p>{escape(reason)}</p></section>
<section><h2>Human review</h2><p><strong>Status:</strong> {escape(review_status)}</p>
<p><strong>Operator note:</strong> {escape(review_note) if review_note else 'None'}</p></section>
<footer><small>Local prototype report. An AI alert is a review candidate, not proof of a fall.</small></footer>
</body></html>"""
    return document.encode("utf-8")
