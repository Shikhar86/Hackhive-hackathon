"""Score fixed natural-language questions against captured video events."""

import csv
import tempfile
from pathlib import Path

from chatbot.chat import ask
from core.db import get_events, insert_event
from core.schema import Event


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "eval" / "phase3_real_chat_questions.csv"


def main():
    captured_events = [
        row for row in get_events(limit=500)
        if not row["summary"].startswith("Mock event:")
    ]
    if not captured_events:
        raise AssertionError("No captured video events found; run the reference clips first")

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "real_events.db"
        for row in captured_events:
            insert_event(Event(**row), db_path)

        with QUESTIONS_PATH.open(encoding="utf-8-sig", newline="") as file:
            cases = list(csv.DictReader(file))
        if len(cases) != 25:
            raise AssertionError(f"Expected 25 saved questions, found {len(cases)}")

        passed = 0
        for case in cases:
            expected = [
                event for event in captured_events
                if (not case["expected_category"] or event["category"] == case["expected_category"])
                and (not case["expected_camera"] or event["camera_id"] == case["expected_camera"])
                and (not case["min_confidence"] or event["confidence"] >= float(case["min_confidence"]))
            ]
            expected = expected[: int(case["limit"] or 100)]
            _, observed = ask(case["question"], db_path)
            expected_ids = [event["id"] for event in expected]
            observed_ids = [event["id"] for event in observed]
            assert observed_ids == expected_ids, (
                f"Question failed: {case['question']!r}; expected {expected_ids}, got {observed_ids}"
            )
            passed += 1

    print(f"PASS: {passed}/25 saved questions matched captured video events")


if __name__ == "__main__":
    main()
