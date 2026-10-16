"""Read a video file, webcam, or RTSP stream at a chosen sample rate."""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class SampledFrame:
    camera_id: str
    frame_index: int
    video_time_s: float
    captured_at_utc: str
    image: np.ndarray


def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    """Return capture and whether it is a finite video file."""
    if source.isdigit():
        target: int | str = int(source)
        is_file = False
    elif source.lower().startswith(("rtsp://", "rtsps://")):
        target = source
        is_file = False
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        target = str(path)
        is_file = True
    capture = cv2.VideoCapture(target)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {source}")
    if is_file and hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        # Phone recordings may keep encoded frames sideways and describe the
        # intended portrait orientation only in container metadata.
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return capture, is_file


def sampled_frames(source: str, camera_id: str, sample_fps: float = 2.0) -> Iterator[SampledFrame]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if not camera_id.strip():
        raise ValueError("camera_id is required")

    capture, is_file = open_source(source)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if is_file and source_fps <= 0:
        capture.release()
        raise RuntimeError("Video file has no valid FPS metadata")

    frame_index = 0
    next_sample_s = 0.0
    start_monotonic = time.monotonic()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            video_time_s = frame_index / source_fps if is_file else time.monotonic() - start_monotonic
            if video_time_s + 1e-9 >= next_sample_s:
                yield SampledFrame(
                    camera_id=camera_id,
                    frame_index=frame_index,
                    video_time_s=video_time_s,
                    captured_at_utc=datetime.now(timezone.utc).isoformat(),
                    image=image,
                )
                next_sample_s += 1.0 / sample_fps
                while next_sample_s <= video_time_s:
                    next_sample_s += 1.0 / sample_fps
            frame_index += 1
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sampled frames from a video source")
    parser.add_argument("source", help="Video file path, camera index such as 0, or RTSP URL")
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    frames = sampled_frames(args.source, args.camera_id, args.sample_fps)
    try:
        for count, frame in enumerate(frames, start=1):
            print(
                f"sample={count} camera={frame.camera_id} frame={frame.frame_index} "
                f"video_time={frame.video_time_s:.3f}s shape={frame.image.shape}"
            )
            if count >= args.max_frames:
                break
    finally:
        frames.close()


if __name__ == "__main__":
    main()
