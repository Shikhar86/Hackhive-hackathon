from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


CATEGORIES = frozenset({"person_down", "crowd", "tailgating", "fighting", "suspicious"})


@dataclass(frozen=True)
class Event:
    id: str
    timestamp: str
    camera_id: str
    category: str
    confidence: float
    summary: str
    keyframe_path: Optional[str] = None

    @classmethod
    def create(
        cls,
        camera_id: str,
        category: str,
        confidence: float,
        summary: str,
        keyframe_path: Optional[str] = None,
    ) -> "Event":
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if not camera_id.strip() or not summary.strip():
            raise ValueError("Camera ID and summary are required")
        return cls(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            camera_id=camera_id,
            category=category,
            confidence=float(confidence),
            summary=summary,
            keyframe_path=keyframe_path,
        )

    def to_dict(self) -> dict:
        return asdict(self)
