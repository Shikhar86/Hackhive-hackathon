"""Evaluate the read-only chatbot against a fixed synthetic event fixture."""

import csv
import tempfile
from contextlib import closing
from pathlib import Path

from chatbot.chat import ask
from chatbot.seed_fake_events import seed_fake_events
from core.db import connect, get_events


ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "fake_events.db"
        seed_fake_events(db_path, 200)
        all_events = get_events(db_path, limit=200)
        passed = 0
        with (ROOT / "eval" / "chat_questions.csv").open(encoding="utf-8-sig", newline="") as file:
            cases = list(csv.DictReader(file))
        for case in cases:
            expected = [
                event for event in all_events
                if (not case["expected_category"] or event["category"] == case["expected_category"])
                and (not case["expected_camera"] or event["camera_id"] == case["expected_camera"])
                and (not case["min_confidence"] or event["confidence"] >= float(case["min_confidence"]))
            ]
            expected = expected[: int(case["limit"] or 100)]
            message, observed = ask(case["question"], db_path)
            expected_ids = [event["id"] for event in expected]
            observed_ids = [event["id"] for event in observed]
            assert observed_ids == expected_ids, (
                f"Question failed: {case['question']!r}; expected {expected_ids}, got {observed_ids}"
            )
            passed += 1

        for operation in ("DROP", "UPDATE", "DELETE"):
            try:
                ask(f"{operation} events", db_path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{operation} request was not rejected")

        with closing(connect(db_path)) as connection:
            table_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert table_count == 200, f"Expected 200 intact fixture events, found {table_count}"
        print(f"PASS: {passed}/{len(cases)} questions returned the expected rows")
        print("PASS: DROP, UPDATE, and DELETE requests rejected; fixture remains intact")


if __name__ == "__main__":
    main()
