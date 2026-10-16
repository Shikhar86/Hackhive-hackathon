"""End-to-end event pipeline: video → YOLO + ByteTrack → rules → DB.

Usage
-----
Mock event (existing smoke test)::

    python -m vision.run_pipeline --mock

Real video::

    python -m vision.run_pipeline data/videos/clip.avi
    python -m vision.run_pipeline data/videos/clip.avi --camera-id camera-2 --sample-fps 2
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import cv2

from core.db import insert_event
from core.schema import Event
from vision.detect_track import TrackedFrame, detect_and_track
from vision.event_rules import DetectedEvent, EventDetector
from vision.vlm import summarize_image


ROOT = Path(__file__).resolve().parents[1]
KEYFRAME_DIR = ROOT / "data" / "keyframes"


# ---------------------------------------------------------------------------
# Keyframe helper
# ---------------------------------------------------------------------------

def save_keyframe(detected: DetectedEvent, keyframe_dir: Path = KEYFRAME_DIR) -> str | None:
    """Save the frame image as a JPEG and return the relative path."""
    if detected.image is None:
        return None
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{detected.category}_{detected.frame_index}_{detected.captured_at_utc.replace(':', '-')}.jpg"
    path = keyframe_dir / filename
    cv2.imwrite(str(path), detected.image)
    # Store a repo-relative path in the database so it is portable.
    return str(path.relative_to(ROOT))


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------

def run_mock(camera_id: str) -> Event:
    """Insert a single mock event (smoke-test shortcut)."""
    event = Event.create(
        camera_id=camera_id,
        category="person_down",
        confidence=0.92,
        summary="Mock event: possible person-down incident",
    )
    insert_event(event)
    return event


def run_pipeline(
    source: str,
    camera_id: str = "camera-1",
    sample_fps: float = 2.0,
    model_path: str = "yolov8n.pt",
    tracker: str = "bytetrack.yaml",
    confidence: float = 0.25,
    device: str | None = None,
    max_frames: int | None = None,
    person_down_aspect: float = 1.2,
    person_down_height_drop: float = 0.1,
    person_down_frames: int = 3,
    crowd_threshold: int | None = None,
    use_vlm: bool = True,
    db_path=None,
    on_frame: Callable[[TrackedFrame], None] | None = None,
) -> list[Event]:
    """Run the full detection → rules → DB pipeline on a video source.

    Returns every Event that was created and persisted.
    """
    from core.db import insert_event as _insert

    detector = EventDetector(
        person_down_aspect=person_down_aspect,
        person_down_height_drop=person_down_height_drop,
        person_down_frames=person_down_frames,
        crowd_threshold=crowd_threshold,
    )
    created_events: list[Event] = []

    insert = (lambda ev: _insert(ev, db_path)) if db_path else _insert

    frames_gen = detect_and_track(
        source, camera_id, sample_fps, model_path, tracker, confidence, device,
    )
    try:
        for count, tracked_frame in enumerate(frames_gen, start=1):
            if on_frame is not None:
                on_frame(tracked_frame)
            new_events = detector.process_frame(tracked_frame)
            for det in new_events:
                kf_path = save_keyframe(det)
                summary = det.summary
                if use_vlm:
                    try:
                        caption = summarize_image(det.image)
                        summary = f"Possible {det.category.replace('_', ' ')}: {caption}"
                    except Exception as exc:
                        summary = f"VLM summary unavailable ({type(exc).__name__}); {det.summary}"
                        print(f"WARNING: {summary}")
                event = Event.create(
                    camera_id=det.camera_id,
                    category=det.category,
                    confidence=det.confidence,
                    summary=summary,
                    keyframe_path=kf_path,
                )
                insert(event)
                created_events.append(event)
                print(f"EVENT: {json.dumps(event.to_dict(), ensure_ascii=False)}")
            if max_frames is not None and count >= max_frames:
                break
    finally:
        frames_gen.close()

    return created_events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the surveillance event pipeline")
    parser.add_argument("source", nargs="?", default=None, help="Video path, camera index, or RTSP URL")
    parser.add_argument("--mock", action="store_true", help="Insert one sample event and exit")
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--max-frames", type=int, help="Stop after this many sampled frames")
    parser.add_argument("--person-down-aspect", type=float, default=1.2)
    parser.add_argument("--person-down-height-drop", type=float, default=0.1)
    parser.add_argument("--person-down-frames", type=int, default=3)
    parser.add_argument(
        "--crowd-threshold", type=int, default=None,
        help="Enable the experimental count-only crowd rule at this people count (disabled by default)",
    )
    parser.add_argument("--no-vlm", action="store_true", help="Use heuristic summaries without loading SmolVLM")
    args = parser.parse_args()

    if args.mock:
        event = run_mock(args.camera_id)
        print(f"Inserted mock event: {event.to_dict()}")
        return

    if args.source is None:
        parser.error("A video source is required unless --mock is given")

    events = run_pipeline(
        source=args.source,
        camera_id=args.camera_id,
        sample_fps=args.sample_fps,
        model_path=args.model,
        tracker=args.tracker,
        confidence=args.confidence,
        device=args.device,
        max_frames=args.max_frames,
        person_down_aspect=args.person_down_aspect,
        person_down_height_drop=args.person_down_height_drop,
        person_down_frames=args.person_down_frames,
        crowd_threshold=args.crowd_threshold,
        use_vlm=not args.no_vlm,
    )
    print(f"\nPipeline finished: {len(events)} event(s) created.")


if __name__ == "__main__":
    main()
