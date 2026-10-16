"""Count attempted VLM calls on the saved five-minute normal-video tracks."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from vision.detect_track import TrackedFrame
from vision.run_pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "data" / "videos" / "phase3_motion_tracks.json"


def main():
    samples = json.loads(TRACKS.read_text(encoding="utf-8"))["normal"]
    calls = 0

    def frames():
        for sample in samples:
            yield TrackedFrame(
                camera_id="phase4-normal",
                frame_index=sample["frame_index"],
                video_time_s=sample["t"],
                captured_at_utc=datetime.now(timezone.utc).isoformat(),
                tracks=sample["tracks"],
            )

    def count_caption(_image):
        nonlocal calls
        calls += 1
        return "Counted VLM call"

    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("vision.run_pipeline.detect_and_track", return_value=frames()),
            patch("vision.run_pipeline.save_keyframe", return_value=None),
            patch("vision.run_pipeline.summarize_image", side_effect=count_caption),
        ):
            events = run_pipeline("cached-normal-tracks", db_path=Path(temp_dir) / "events.db")

    assert calls == len(events) == 2, (calls, len(events))
    print(f"VLM attempts: {calls}/{len(samples)} sampled frames ({calls / len(samples):.2%})")
    print("Captioning was counted with a stub; this replay does not measure VLM latency or success.")


if __name__ == "__main__":
    main()
