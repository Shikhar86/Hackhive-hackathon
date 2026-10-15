"""Read-only natural-language search over the shared event database."""

import re

from core.db import connect, init_db
from core.schema import CATEGORIES

WRITE_WORDS = ("drop", "delete", "update", "insert", "alter", "replace", "create")
CATEGORY_ALIASES = {
    "person_down": ("person_down", "person down", "someone fell", "person fell", "collapse", "collapsing"),
    "fighting": ("fighting", "fight", "physical fight"),
    "tailgating": ("tailgating",),
    "crowd": ("crowd", "crowd incident", "crowd_incident"),
    "suspicious": ("suspicious", "suspicious activity", "suspicious_activity"),
}


def _filters(question: str):
    text = question.strip().lower()
    category = next((value for value, aliases in CATEGORY_ALIASES.items() if any(alias in text for alias in aliases)), None)
    camera_match = re.search(r"\bcamera\s*[-#]?\s*([a-z0-9]+)\b", text)
    camera_id = f"camera-{camera_match.group(1)}" if camera_match else None
    confidence_match = re.search(r"(?:above|over|at least|minimum|min confidence|confidence of)?\s*(\d+(?:\.\d+)?)\s*%", text)
    min_confidence = float(confidence_match.group(1)) / 100 if confidence_match else None
    limit_match = re.search(r"\b(?:last|latest|top)\s+(\d+)\b", text)
    limit = min(int(limit_match.group(1)), 100) if limit_match else 100
    return category, camera_id, min_confidence, limit


def search_events(category=None, camera_id=None, min_confidence=None, limit=100, db_path=None):
    """Search events using only parameterized, read-only SQL."""
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if db_path:
        init_db(db_path)
        connection = connect(db_path)
    else:
        init_db()
        connection = connect()
    try:
        clauses, parameters = [], []
        if category:
            clauses.append("category = ?")
            parameters.append(category)
        if camera_id:
            clauses.append("camera_id = ?")
            parameters.append(camera_id)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            parameters.append(min_confidence)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = "SELECT id, timestamp, camera_id, category, confidence, summary, keyframe_path " + f"FROM events{where} ORDER BY timestamp DESC, id DESC LIMIT ?"
        parameters.append(limit)
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]
    finally:
        connection.close()


def ask(question: str, db_path=None):
    """Answer a supported event question and return rows for a UI to render."""
    text = question.strip()
    if not text:
        raise ValueError("Question is required")
    if any(word in text.lower() for word in WRITE_WORDS):
        raise ValueError("Only read-only event questions are allowed")
    category, camera_id, min_confidence, limit = _filters(text)
    events = search_events(category, camera_id, min_confidence, limit, db_path)
    return (f"I found {len(events)} matching event(s).", events) if events else ("No matching events found.", [])
