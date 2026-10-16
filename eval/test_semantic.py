"""Phase 4 check of the optional offline semantic search."""

import tempfile
from dataclasses import replace
from pathlib import Path

from chatbot.chat import ask
from core.db import insert_event
from core.schema import Event


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "events.db"
        target = replace(Event.create(
            "lobby", "person_down", 0.9,
            "A woman is kneeling on the floor and a man is grabbing her arm.",
        ), timestamp="2026-10-16T00:00:00+00:00")
        distractor = replace(Event.create(
            "lobby", "crowd", 0.8,
            "Several visitors are waiting in a line by the entrance.",
        ), timestamp="2026-10-16T00:00:01+00:00")
        insert_event(target, db_path)
        insert_event(distractor, db_path)

        question = "top 1 someone receiving assistance while prone"
        _, latest = ask(question, db_path)
        _, ranked = ask(question, db_path, semantic=True)
        assert latest[0]["id"] == distractor.id
        assert ranked[0]["id"] == target.id
        print("PASS: offline semantic ranking found the assisted person from a paraphrased question")
        print("PASS: semantic switch off kept the original latest-event behavior")


if __name__ == "__main__":
    main()
