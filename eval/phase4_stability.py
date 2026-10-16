"""Run the 10-minute replay with real captioning and isolated event storage."""

import contextlib
import io
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from vision.detect_track import detect_and_track
from vision.run_pipeline import run_pipeline, save_keyframe


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "videos" / "phase3_stability_10m.mp4"
KEYFRAMES = ROOT / "data" / "keyframes" / "phase4-all-features"


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    sampled_frames = 0

    def counted_frames(*args, **kwargs):
        nonlocal sampled_frames
        source = detect_and_track(*args, **kwargs)
        try:
            for frame in source:
                sampled_frames += 1
                yield frame
        finally:
            source.close()

    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("vision.run_pipeline.detect_and_track", side_effect=counted_frames),
            patch(
                "vision.run_pipeline.save_keyframe",
                side_effect=lambda event: save_keyframe(event, KEYFRAMES),
            ),
        ):
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                events = run_pipeline(
                    str(SOURCE), camera_id="phase4-all-features",
                    sample_fps=2.0, db_path=Path(temp_dir) / "events.db",
                )
            elapsed = time.perf_counter() - started

    captioned = sum(event.summary.startswith("Possible ") for event in events)
    saved = sum(bool(event.keyframe_path and (ROOT / event.keyframe_path).is_file()) for event in events)
    print(f"Video duration: 600.0 s; processing time: {elapsed:.1f} s")
    print(f"Sampled frames: {sampled_frames}; events: {len(events)}")
    print(f"Real VLM captions: {captioned}; saved keyframes: {saved}")
    assert sampled_frames == 1200
    assert events and captioned == saved == len(events)
    assert elapsed <= 600, "Replay did not keep up with the 10-minute video"
    print("PASS: all-feature 10-minute replay remained at or above real-time speed")


if __name__ == "__main__":
    main()
