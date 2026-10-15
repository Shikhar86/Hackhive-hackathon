"""Seed a separate SQLite file with synthetic events for local UI demos."""

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.db import connect, init_db
from core.schema import CATEGORIES, Event


def seed_fake_events(db_path: Path, count: int = 200) -> int:
    if count < 1:
        raise ValueError("count must be positive")
    init_db(db_path)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    categories = sorted(CATEGORIES)
    rows = []
    for index in range(count):
        category = categories[index % len(categories)]
        event = Event(
            id=f"fake-{index:04d}",
            timestamp=(start + timedelta(seconds=index)).isoformat(),
            camera_id=f"camera-{index % 2 + 1}",
            category=category,
            confidence=round(0.25 + (index % 70) / 100, 2),
            summary=f"Synthetic fixture event {index:04d} ({category})",
            keyframe_path=None,
        )
        rows.append(event.to_dict())
    with closing(connect(db_path)) as connection, connection:
        connection.executemany(
            "INSERT OR REPLACE INTO events "
            "(id, timestamp, camera_id, category, confidence, summary, keyframe_path) "
            "VALUES (:id, :timestamp, :camera_id, :category, :confidence, :summary, :keyframe_path)",
            rows,
        )
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/fake_events.db"))
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    print(f"Seeded {seed_fake_events(args.db, args.count)} synthetic events in {args.db}")


if __name__ == "__main__":
    main()
