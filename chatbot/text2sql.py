"""Conservative natural-language-to-SQL helpers for stored events."""

import re


ALLOWED_COLUMNS = {"id", "timestamp", "camera_id", "category", "confidence", "summary", "keyframe_path"}


def question_to_sql(question: str) -> tuple[str, tuple]:
    """Return a parameterized read-only query for a small supported question set."""
    text = question.strip().lower()
    if not text:
        raise ValueError("Question is required")
    if any(word in text for word in ("drop", "delete", "update", "insert", "alter", "replace", "create")):
        raise ValueError("Only read-only event questions are allowed")

    category = next((value for value in ("person_down", "fighting", "tailgating", "crowd", "suspicious") if value in text), None)
    camera = re.search(r"camera[- ]?[a-z0-9]+", text)
    limit_match = re.search(r"\b(?:last|latest|top)\s+(\d+)\b", text)
    limit = min(int(limit_match.group(1)), 100) if limit_match else 20

    clauses = []
    params: list[str] = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if camera:
        clauses.append("camera_id = ?")
        params.append(camera.group(0).replace(" ", "-"))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return (
        "SELECT id, timestamp, camera_id, category, confidence, summary, keyframe_path "
        f"FROM events{where} ORDER BY timestamp DESC, id DESC LIMIT ?",
        tuple(params + [limit]),
    )
