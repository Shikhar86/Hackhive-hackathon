"""Small, honest gate runner for the 48-hour build plan.

Run: python -m eval.phase_gate check 0
Record a human check: python -m eval.phase_gate record 0 team_handoff "B and C cloned commit abc123"
"""

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.db import get_events, insert_event
from core.schema import CATEGORIES, Event


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "eval" / "gate_evidence.json"

GATES = {
    0: {
        "auto": {"foundation": "Schema, SQLite, and mock event round trip", "opencv": "OpenCV imports"},
        "manual": {
            "gpu_models": "YOLO and SmolVLM load on the main laptop; record GPU and versions",
            "team_handoff": "B and C have the frozen schema, categories, and repo",
            "teammate_smoke": "B's Ollama prompt and C's blank Streamlit page work",
        },
    },
    1: {
        "auto": {
            "ingest": "10-second video yields 20 timestamped samples at 2 FPS",
            "event_rules": "Crowd deduplication and person-down rule behavior",
            "pipeline": "Synthetic detections become contract-shaped SQLite events",
        },
        "manual": {
            "vision": "30-second tracks are stable; person-down fires on 2/3 labeled clips",
            "chat": "At least 15/20 fake-data questions correct; non-SELECT SQL rejected",
            "ui": "Fake events, filters, and chatbot results appear in UI",
            "merge": "All modules merged and run from the main laptop",
        },
    },
    2: {
        "auto": {},
        "manual": {
            "event": "Real-time clip creates correct event, key frame, and VLM summary",
            "dashboard": "Event appears in dashboard without manual reload",
            "chat": "Chatbot retrieves the new person-down event",
        },
    },
    3: {
        "auto": {},
        "manual": {
            "stability": "10-minute run has no crash",
            "categories": "Each kept category fires on 2/3 clips; others marked cut",
            "false_alarms": "Fewer than 3 false alarms on normal 5-minute clip",
            "chat": "At least 20/25 real-event questions correct",
            "cameras": "Two camera sources run and queries filter correctly",
        },
    },
    4: {
        "auto": {},
        "manual": {
            "switches": "Each kept novelty feature works on and off",
            "metrics": "VLM call percentage measured on a 5-minute clip",
            "semantic": "Semantic query finds an event beyond keyword matching",
            "stability": "10-minute all-features run remains real-time and crash-free",
        },
    },
    5: {
        "auto": {},
        "manual": {
            "accuracy": "Precision and recall measured for each kept category",
            "latency": "Average and worst latency measured for every required stage",
            "report": "Two failure cases explained; slides agree with report",
        },
    },
    6: {
        "auto": {},
        "manual": {
            "fresh_clone": "Offline fresh clone runs from README",
            "demo": "Full demo succeeds three times consecutively",
            "backup": "Backup video plays and covers full flow",
            "submission": "README, slides, report submitted with at least 1-hour buffer",
        },
    },
}


def check_foundation():
    assert CATEGORIES == {"person_down", "crowd", "tailgating", "fighting", "suspicious"}
    from vision.run_pipeline import run_mock

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "events.db"
        with patch("vision.run_pipeline.insert_event", side_effect=lambda event: insert_event(event, db_path)):
            event = run_mock("gate-test-camera")
        rows = get_events(db_path)
        assert rows == [event.to_dict()]
        assert set(rows[0]) == {"id", "timestamp", "camera_id", "category", "confidence", "summary", "keyframe_path"}
    try:
        Event.create("camera", "invalid", 0.5, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid category was accepted")


def check_opencv():
    import cv2
    assert cv2.__version__


def check_ingest():
    from eval.test_ingest import main

    with contextlib.redirect_stdout(io.StringIO()):
        main()


def check_event_rules():
    from eval.test_event_rules import main

    with contextlib.redirect_stdout(io.StringIO()):
        main()


def check_pipeline():
    from eval.test_pipeline import main

    with contextlib.redirect_stdout(io.StringIO()):
        main()


AUTO_CHECKS = {
    "foundation": check_foundation,
    "opencv": check_opencv,
    "ingest": check_ingest,
    "event_rules": check_event_rules,
    "pipeline": check_pipeline,
}


def load_evidence():
    if not EVIDENCE_PATH.exists():
        return {}
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def show_gate(phase):
    gate = GATES[phase]
    evidence = load_evidence().get(str(phase), {})
    print(f"Phase {phase} gate")
    passed = True
    for check_id, description in gate["auto"].items():
        try:
            AUTO_CHECKS[check_id]()
            print(f"  PASS {check_id}: {description}")
        except Exception as exc:
            passed = False
            print(f"  FAIL {check_id}: {description} ({type(exc).__name__}: {exc})")
    for check_id, description in gate["manual"].items():
        note = evidence.get(check_id)
        if note:
            print(f"  PASS {check_id}: {description} | Evidence: {note}")
        else:
            passed = False
            print(f"  TODO {check_id}: {description}")
    print("GATE PASS" if passed else "GATE OPEN")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description="Check a 48-hour build phase gate")
    sub = parser.add_subparsers(dest="command")
    check = sub.add_parser("check", help="Run checks and show remaining work")
    check.add_argument("phase", type=int, choices=GATES)
    record = sub.add_parser("record", help="Record evidence for a human check")
    record.add_argument("phase", type=int, choices=GATES)
    record.add_argument("check_id")
    record.add_argument("evidence")
    args = parser.parse_args()

    if args.command == "check":
        return show_gate(args.phase)
    if args.command == "record":
        if args.check_id not in GATES[args.phase]["manual"]:
            parser.error("Unknown human check ID for this phase")
        if not args.evidence.strip():
            parser.error("Evidence must be non-empty")
        data = load_evidence()
        data.setdefault(str(args.phase), {})[args.check_id] = args.evidence.strip()
        EVIDENCE_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Recorded phase {args.phase} / {args.check_id}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
