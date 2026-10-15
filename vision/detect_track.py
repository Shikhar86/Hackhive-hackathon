"""Run YOLO detection and ByteTrack on sampled video frames."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from vision.ingest import SampledFrame, sampled_frames


@dataclass
class TrackedFrame:
    camera_id: str
    frame_index: int
    video_time_s: float
    captured_at_utc: str
    tracks: list[dict]


def detect_and_track(
    source: str,
    camera_id: str = "camera-1",
    sample_fps: float = 2.0,
    model_path: str = "yolov8n.pt",
    tracker: str = "bytetrack.yaml",
    confidence: float = 0.25,
    device: str | None = None,
) -> Iterator[TrackedFrame]:
    """Yield detections with persistent tracker IDs for each sampled frame."""
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    from ultralytics import YOLO

    model = YOLO(model_path)
    frames = sampled_frames(source, camera_id, sample_fps)
    try:
        for frame in frames:
            result = model.track(
                source=frame.image,
                persist=True,
                tracker=tracker,
                conf=confidence,
                device=device,
                verbose=False,
            )[0]
            boxes = result.boxes
            tracks = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for box, class_id, score, track_id in zip(xyxy, classes, scores, ids):
                    class_id = int(class_id)
                    tracks.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": model.names[class_id],
                        "confidence": float(score),
                        "xyxy": [float(value) for value in box],
                    })
            yield TrackedFrame(
                camera_id=frame.camera_id,
                frame_index=frame.frame_index,
                video_time_s=frame.video_time_s,
                captured_at_utc=frame.captured_at_utc,
                tracks=tracks,
            )
    finally:
        frames.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and track objects on sampled frames")
    parser.add_argument("source", help="Video path, camera index, or RTSP URL")
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--jsonl", type=Path, help="Optionally save frame results as JSONL")
    parser.add_argument("--max-frames", type=int, help="Stop after this many sampled frames")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")

    output = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        for count, frame_result in enumerate(
            detect_and_track(
                args.source, args.camera_id, args.sample_fps,
                args.model, args.tracker, args.confidence, args.device,
            ),
            start=1,
        ):
            data = asdict(frame_result)
            encoded = json.dumps(data, ensure_ascii=False)
            if output:
                output.write(encoded + "\n")
            print(encoded)
            if args.max_frames is not None and count >= args.max_frames:
                break
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
