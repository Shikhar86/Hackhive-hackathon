"""Measure the Phase 5 stages using one real, local video and an isolated DB.

Run from the repository root with its Python environment. The script writes no
events or keyframes into the shared project data directory.
"""

import argparse
import csv
import importlib
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch


def describe(values):
    return {
        "samples": len(values),
        "average_ms": round(statistics.mean(values), 2) if values else None,
        "worst_ms": round(max(values), 2) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    source = (args.source or repo / "data/videos/phase1_user_30s.mp4").resolve()
    if not source.is_file():
        parser.error(f"Source video is missing: {source}")
    sys.path.insert(0, str(repo))

    import cv2
    import torch

    from chatbot.chat import ask
    from core.db import get_events

    db_module = importlib.import_module("core.db")
    pipeline = importlib.import_module("vision.run_pipeline")
    original_detect = pipeline.detect_and_track
    original_caption = pipeline.summarize_image
    original_insert = db_module.insert_event

    timings = {key: [] for key in ("detect_track", "caption", "keyframe", "db_write", "chatbot", "dashboard_db_read")}

    def timed_frames(*positional, **keyword):
        frames = original_detect(*positional, **keyword)
        try:
            while True:
                started = time.perf_counter()
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                timings["detect_track"].append((time.perf_counter() - started) * 1000)
                yield frame
        finally:
            frames.close()

    def timed_caption(image):
        started = time.perf_counter()
        try:
            return original_caption(image)
        finally:
            timings["caption"].append((time.perf_counter() - started) * 1000)

    def timed_insert(event, db_path):
        started = time.perf_counter()
        try:
            return original_insert(event, db_path)
        finally:
            timings["db_write"].append((time.perf_counter() - started) * 1000)

    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)

        def timed_keyframe(event):
            started = time.perf_counter()
            try:
                if event.image is None:
                    return None
                path = temp / f"keyframe_{event.frame_index}.jpg"
                if not cv2.imwrite(str(path), event.image):
                    raise RuntimeError(f"Could not save keyframe: {path}")
                return str(path)
            finally:
                timings["keyframe"].append((time.perf_counter() - started) * 1000)

        event_db = temp / "events.db"
        with (
            patch.object(pipeline, "detect_and_track", side_effect=timed_frames),
            patch.object(pipeline, "summarize_image", side_effect=timed_caption),
            patch.object(pipeline, "save_keyframe", side_effect=timed_keyframe),
            patch.object(db_module, "insert_event", side_effect=timed_insert),
        ):
            started = time.perf_counter()
            events = pipeline.run_pipeline(str(source), camera_id="phase5-latency", db_path=event_db)
            total_s = time.perf_counter() - started

        shared_db = repo / "data/events.db"
        chat_db = temp / "chat.db"
        if shared_db.is_file():
            shutil.copyfile(shared_db, chat_db)
        else:
            shutil.copyfile(event_db, chat_db)
        chat_rows = len(get_events(chat_db, limit=10000))

        questions_path = repo / "eval/phase3_real_chat_questions.csv"
        with questions_path.open(newline="", encoding="utf-8") as handle:
            questions = [row["question"] for row in csv.DictReader(handle)]
        for question in questions:
            started = time.perf_counter()
            ask(question, db_path=chat_db, semantic=False)
            timings["chatbot"].append((time.perf_counter() - started) * 1000)

        for _ in range(25):
            started = time.perf_counter()
            get_events(chat_db, limit=500)
            timings["dashboard_db_read"].append((time.perf_counter() - started) * 1000)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    result = {
        "source": source.name,
        "device": device,
        "sample_fps": 2.0,
        "total_pipeline_s": round(total_s, 2),
        "frames": len(timings["detect_track"]),
        "events": len(events),
        "caption_fallbacks": sum(not event.summary.startswith("Possible ") for event in events),
        "chat_db_rows": chat_rows,
        "stages": {key: describe(values) for key, values in timings.items()},
        "first_detect_track_ms_includes_model_load": round(timings["detect_track"][0], 2),
        "warm_detect_track": describe(timings["detect_track"][1:]),
        "event_to_visible_dashboard_ms": None,
        "dashboard_note": "Database read is measured; browser display delay has not been measured. The UI refresh interval is configured at 5 seconds.",
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
