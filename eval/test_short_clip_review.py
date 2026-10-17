"""Offline regression checks for the optional short-clip review proposal."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from chatbot.chat import search_events
from core.db import get_event_evidence
from vision.detect_track import TrackedFrame
from vision.short_clip_review import propose_short_clip_review


def make_frames(track_index=0):
    frames = []
    for index in range(6):
        tracks = ([{
            "track_id": 7,
            "class_name": "person",
            "confidence": 0.7,
            "xyxy": [440.0, 12.0, 605.0, 216.0],
        }] if index == track_index else [])
        frames.append(TrackedFrame(
            camera_id="review-test", frame_index=index,
            video_time_s=index * 0.5,
            captured_at_utc="2026-10-17T00:00:00+00:00",
            tracks=tracks,
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
        ))
    return frames


def main():
    with TemporaryDirectory() as temp:
        source = Path(temp) / "sample.mp4"
        source.write_bytes(b"fixture")
        db = Path(temp) / "events.db"
        seen_heights = []

        def caption(crop, prompt, max_new_tokens):
            seen_heights.append(crop.shape[0])
            return "Yes" if crop.shape[0] >= 500 else "No"

        with (
            patch("vision.short_clip_review.caption_image", side_effect=caption),
            patch("vision.short_clip_review.save_keyframe", return_value=None),
        ):
            events = propose_short_clip_review(make_frames(), str(source), "review-test", db)
            assert len(events) == 1 and events[0].confidence == 0.0
            assert max(seen_heights) >= 500, seen_heights
            evidence = get_event_evidence(events[0].id, db)
            assert evidence and evidence["video_time_s"] == 0.5
            assert len(search_events(db_path=db)) == 1
            assert search_events(min_confidence=0.0, db_path=db) == []
            repeated = propose_short_clip_review(make_frames(), str(source), "review-test", db)
            assert repeated[0].id == events[0].id
            assert len(search_events(db_path=db)) == 1
            late = propose_short_clip_review(make_frames(track_index=3), str(source), "other-camera", db)
            assert late == [], "A track lost only one second before the clip ends needs no proposal"

        terminal = make_frames(track_index=4)
        terminal[-1].image[:] = 255
        def changed_posture(crop, prompt, max_new_tokens):
            return "One person is lying down." if crop.mean() > 100 else "One person is standing."
        with (
            patch("vision.short_clip_review.caption_image", side_effect=changed_posture),
            patch("vision.short_clip_review.save_keyframe", return_value=None),
        ):
            end_event = propose_short_clip_review(terminal, str(source), "end-review", db)
            assert len(end_event) == 1
            assert end_event[0].confidence == 0.0
            assert "end of the clip" in end_event[0].summary
            assert get_event_evidence(end_event[0].id, db)["video_time_s"] == 2.5
            unchanged = propose_short_clip_review(
                make_frames(track_index=4), str(source), "end-normal", db,
            )
            assert unchanged == []

    print("PASS: expanded crop, unscored candidate, confidence filter, repeat analysis, and end-frame review")


if __name__ == "__main__":
    main()
