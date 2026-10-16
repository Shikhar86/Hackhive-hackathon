"""Exercise rule-to-event-to-SQLite integration with controlled tracks."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.db import get_events
from vision.detect_track import TrackedFrame
from vision.event_rules import DetectedEvent
from vision.run_pipeline import run_pipeline, save_keyframe


def make_frame(index: int, fallen: bool = False, crowded: bool = True) -> TrackedFrame:
    tracks = []
    count = 8 if crowded else 0
    for track_id in range(count):
        if track_id == 7 and fallen:
            box = [0, 80, 100, 110]
        else:
            box = [track_id * 25, 0, track_id * 25 + 20, 100]
        tracks.append({
            "track_id": track_id,
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
            "xyxy": box,
        })
    return TrackedFrame(
        camera_id="camera-test",
        frame_index=index,
        video_time_s=index / 2,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        tracks=tracks,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def main():
    keyframe_parent = Path("data/keyframes").resolve()
    keyframe_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=keyframe_parent) as temp_dir:
        sample = DetectedEvent(
            camera_id="camera-test",
            category="person_down",
            confidence=0.9,
            summary="Synthetic keyframe writer check",
            frame_index=1,
            video_time_s=0.5,
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
            image=np.zeros((64, 64, 3), dtype=np.uint8),
        )
        keyframe_path = save_keyframe(sample, Path(temp_dir))
        assert keyframe_path
        saved_image = cv2.imread(str(Path(__file__).resolve().parents[1] / keyframe_path))
        assert saved_image is not None and saved_image.shape == (64, 64, 3)

    fake_frames = (frame for frame in [
        make_frame(0),
        make_frame(1, fallen=True),
        make_frame(2, fallen=True),
        make_frame(3, fallen=True),
        make_frame(4, crowded=False),
    ])
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "events.db"
        with (
            patch("vision.run_pipeline.detect_and_track", return_value=fake_frames),
            patch("vision.run_pipeline.save_keyframe", return_value="data/keyframes/test.jpg"),
            patch("vision.run_pipeline.summarize_image", return_value="test factual caption"),
        ):
            events = run_pipeline("unused-source", db_path=db_path, crowd_threshold=8)

        rows = get_events(db_path)
        assert len(events) == 2, f"Expected one crowd and one person-down event, got {len(events)}"
        assert {event.category for event in events} == {"crowd", "person_down"}
        assert len(rows) == 2
        assert all(row["camera_id"] == "camera-test" for row in rows)
        assert all(row["keyframe_path"] == "data/keyframes/test.jpg" for row in rows)
        assert all(row["summary"].startswith("Possible ") and "test factual caption" in row["summary"] for row in rows)
        assert all(set(row) == {"id", "timestamp", "camera_id", "category", "confidence", "summary", "keyframe_path"} for row in rows)
        print("PASS: keyframe saved; person-down and crowd detections became contract-shaped SQLite events")


if __name__ == "__main__":
    main()
