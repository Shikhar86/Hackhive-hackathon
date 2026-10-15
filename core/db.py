import sqlite3
from contextlib import closing
from pathlib import Path

from core.schema import Event


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "events.db"


def connect(db_path=DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    with closing(connect(db_path)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                summary TEXT NOT NULL,
                keyframe_path TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_camera_category
            ON events(camera_id, category)
        """)


def insert_event(event: Event, db_path=DB_PATH):
    init_db(db_path)
    with closing(connect(db_path)) as conn, conn:
        conn.execute("""
            INSERT INTO events
            (id, timestamp, camera_id, category, confidence, summary, keyframe_path)
            VALUES (:id, :timestamp, :camera_id, :category, :confidence, :summary, :keyframe_path)
        """, event.to_dict())


def get_events(db_path=DB_PATH, limit=100):
    if limit < 1:
        raise ValueError("Limit must be positive")
    init_db(db_path)
    with closing(connect(db_path)) as conn:
        rows = conn.execute("""
            SELECT id, timestamp, camera_id, category, confidence, summary, keyframe_path
            FROM events ORDER BY timestamp DESC, id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]
