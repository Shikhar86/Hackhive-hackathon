"""Focused checks for the stateful event heuristics."""

from datetime import datetime, timezone

from vision.detect_track import TrackedFrame
from vision.event_rules import EventDetector


def frame(index, tracks):
    return TrackedFrame(
        camera_id="camera-test",
        frame_index=index,
        video_time_s=index / 2,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        tracks=tracks,
    )


def person(track_id, x1, y1, x2, y2, confidence=0.9):
    return {
        "track_id": track_id,
        "class_id": 0,
        "class_name": "person",
        "confidence": confidence,
        "xyxy": [x1, y1, x2, y2],
    }


def main():
    detector = EventDetector(person_down_frames=3, crowd_threshold=8)
    crowded = [person(i, 0, 0, 20, 80) for i in range(8)]
    assert len(detector.process_frame(frame(0, crowded))) == 1
    assert len(detector.process_frame(frame(1, crowded))) == 0
    assert len(detector.process_frame(frame(2, []))) == 0
    assert len(detector.process_frame(frame(3, crowded))) == 1

    detector = EventDetector(person_down_frames=3, person_down_aspect=1.0, person_down_height_drop=0.55)
    standing = person(7, 0, 0, 20, 100)
    fallen = person(7, 0, 80, 100, 110)
    assert detector.process_frame(frame(0, [standing])) == []
    assert detector.process_frame(frame(1, [fallen])) == []
    assert detector.process_frame(frame(2, [fallen])) == []
    events = detector.process_frame(frame(3, [fallen]))
    assert len(events) == 1 and events[0].category == "person_down"
    assert detector.process_frame(frame(4, [fallen])) == []
    print("PASS: crowd alerts deduplicate and re-arm after dispersal; person-down fires once after three qualifying frames")


if __name__ == "__main__":
    main()
