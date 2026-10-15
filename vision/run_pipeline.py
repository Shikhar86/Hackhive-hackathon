"""Entry point for the event pipeline. Start with a database-backed mock event."""

import argparse

from core.db import insert_event
from core.schema import Event


def run_mock(camera_id: str) -> Event:
    event = Event.create(
        camera_id=camera_id,
        category="person_down",
        confidence=0.92,
        summary="Mock event: possible person-down incident",
    )
    insert_event(event)
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the surveillance event pipeline")
    parser.add_argument("--mock", action="store_true", help="Insert one sample event")
    parser.add_argument("--camera-id", default="camera-1")
    args = parser.parse_args()

    if not args.mock:
        parser.error("Only --mock is available at this stage")

    event = run_mock(args.camera_id)
    print(f"Inserted event: {event.to_dict()}")


if __name__ == "__main__":
    main()
