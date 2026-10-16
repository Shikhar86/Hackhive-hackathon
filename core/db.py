import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_evidence (
                event_id TEXT PRIMARY KEY,
                source_path TEXT,
                video_time_s REAL,
                frame_index INTEGER,
                track_id INTEGER,
                rule_reason TEXT NOT NULL,
                track_points_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_reviews (
                event_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('Unreviewed', 'Confirmed', 'False alarm')),
                note TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS operator_flags (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                video_time_s REAL NOT NULL,
                note TEXT NOT NULL,
                keyframe_path TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_operator_flags_time ON operator_flags(created_at)")


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


def get_event(event_id: str, db_path=DB_PATH):
    init_db(db_path)
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    return dict(row) if row else None


def insert_event_evidence(
    event_id: str, source_path: str | None, video_time_s: float,
    frame_index: int, track_id: int | None, rule_reason: str,
    track_points: list[dict], db_path=DB_PATH,
) -> None:
    """Store detection evidence separately from the stable seven-field event contract."""
    init_db(db_path)
    with closing(connect(db_path)) as conn, conn:
        conn.execute("""
            INSERT INTO event_evidence
            (event_id, source_path, video_time_s, frame_index, track_id, rule_reason, track_points_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, source_path, video_time_s, frame_index, track_id,
            rule_reason, json.dumps(track_points),
        ))


def get_event_evidence(event_id: str, db_path=DB_PATH):
    init_db(db_path)
    with closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM event_evidence WHERE event_id = ?", (event_id,),
        ).fetchone()
    if row is None:
        return None
    evidence = dict(row)
    evidence["track_points"] = json.loads(evidence.pop("track_points_json"))
    return evidence


def get_review(event_id: str, db_path=DB_PATH) -> dict:
    init_db(db_path)
    with closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT status, note, updated_at FROM event_reviews WHERE event_id = ?",
            (event_id,),
        ).fetchone()
    return dict(row) if row else {"status": "Unreviewed", "note": "", "updated_at": None}


def get_review_map(db_path=DB_PATH) -> dict[str, str]:
    init_db(db_path)
    with closing(connect(db_path)) as conn:
        rows = conn.execute("SELECT event_id, status FROM event_reviews").fetchall()
    return {row["event_id"]: row["status"] for row in rows}


def set_review(event_id: str, status: str, note: str, db_path=DB_PATH) -> None:
    if status not in {"Unreviewed", "Confirmed", "False alarm"}:
        raise ValueError("Unknown review status")
    init_db(db_path)
    with closing(connect(db_path)) as conn, conn:
        if conn.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone() is None:
            raise ValueError("Event does not exist")
        conn.execute("""
            INSERT INTO event_reviews (event_id, status, note, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                status = excluded.status,
                note = excluded.note,
                updated_at = excluded.updated_at
        """, (event_id, status, note.strip(), datetime.now(timezone.utc).isoformat()))


def insert_operator_flag(
    camera_id: str, source_path: str, video_time_s: float,
    note: str, keyframe_path: str | None, db_path=DB_PATH,
) -> dict:
    """Keep operator observations separate from model-generated events."""
    if not camera_id.strip() or not source_path.strip() or not note.strip():
        raise ValueError("Camera, source video, and operator note are required")
    if not 0 <= video_time_s < float("inf"):
        raise ValueError("Video time must be a finite nonnegative number")
    flag = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id.strip(),
        "source_path": source_path,
        "video_time_s": float(video_time_s),
        "note": note.strip(),
        "keyframe_path": keyframe_path,
    }
    init_db(db_path)
    with closing(connect(db_path)) as conn, conn:
        conn.execute("""
            INSERT INTO operator_flags
            (id, created_at, camera_id, source_path, video_time_s, note, keyframe_path)
            VALUES (:id, :created_at, :camera_id, :source_path, :video_time_s, :note, :keyframe_path)
        """, flag)
    return flag


def get_operator_flags(db_path=DB_PATH, limit=50) -> list[dict]:
    if limit < 1:
        raise ValueError("Limit must be positive")
    init_db(db_path)
    with closing(connect(db_path)) as conn:
        rows = conn.execute("""
            SELECT id, created_at, camera_id, source_path, video_time_s, note, keyframe_path
            FROM operator_flags ORDER BY created_at DESC, id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]
