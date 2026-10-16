"""Event detection rules that operate on tracked frames.

Each rule accumulates per-track or per-frame history and fires an event
when its heuristic thresholds are met.  Events are deduplicated by
(category, track_id) so each incident is reported once.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vision.detect_track import TrackedFrame


# ---------------------------------------------------------------------------
# Data container returned by every rule
# ---------------------------------------------------------------------------

@dataclass
class DetectedEvent:
    camera_id: str
    category: str
    confidence: float
    summary: str
    frame_index: int
    video_time_s: float
    captured_at_utc: str
    image: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Stateful detector
# ---------------------------------------------------------------------------

class EventDetector:
    """Accumulates track history and fires events based on heuristic rules.

    Parameters
    ----------
    person_down_aspect : float
        Width / height ratio above which a bounding box is considered
        "lying down".  A standing person typically has aspect < 0.6;
        a person lying flat > 1.0.  Tuned default 1.2.
    person_down_height_drop : float
        Fraction of the maximum observed height below which the current
        height counts as "collapsed". Tuned default 0.1 (height dropped to
        10 % of maximum).
    person_down_frames : int
        Number of consecutive qualifying frames before the rule fires.
    crowd_threshold : int | None
        Minimum number of simultaneously visible people to trigger a
        crowd event. None disables this experimental count-only rule; this
        is the default because it does not distinguish incidents from queues.
    """

    def __init__(
        self,
        person_down_aspect: float = 1.2,
        person_down_height_drop: float = 0.1,
        person_down_frames: int = 3,
        crowd_threshold: int | None = None,
        person_down_requires_transition: bool = False,
    ):
        # Per-track history: track_id -> list of observation dicts
        self._history: dict[int, list[dict]] = defaultdict(list)
        # Maximum bbox height ever observed per track (for height-drop rule)
        self._max_height: dict[int, float] = {}
        # Events that have already been fired — (category, key)
        self._fired: set[tuple[str, int]] = set()
        self._crowd_active: set[str] = set()

        self.person_down_aspect = person_down_aspect
        self.person_down_height_drop = person_down_height_drop
        self.person_down_frames = person_down_frames
        if crowd_threshold is not None and crowd_threshold < 1:
            raise ValueError("crowd_threshold must be positive or None")
        self.crowd_threshold = crowd_threshold
        # Experimental opt-in: require an observed upright-to-low transition.
        # The default keeps the shipped detector behavior unchanged.
        self.person_down_requires_transition = person_down_requires_transition

    # ------------------------------------------------------------------
    def process_frame(self, frame: TrackedFrame) -> list[DetectedEvent]:
        """Process one tracked frame and return any newly detected events."""
        events: list[DetectedEvent] = []

        person_tracks = [t for t in frame.tracks if t["class_name"] == "person"]

        # --- update history -------------------------------------------------
        for track in person_tracks:
            tid = track["track_id"]
            if tid is None:
                continue
            x1, y1, x2, y2 = track["xyxy"]
            w = x2 - x1
            h = y2 - y1
            self._history[tid].append({
                "frame_index": frame.frame_index,
                "video_time_s": frame.video_time_s,
                "confidence": track["confidence"],
                "w": w,
                "h": h,
            })
            self._max_height[tid] = max(self._max_height.get(tid, 0.0), h)

        # --- person-down rule -----------------------------------------------
        for track in person_tracks:
            tid = track["track_id"]
            if tid is None:
                continue
            key = ("person_down", tid)
            if key in self._fired:
                continue
            history = self._history[tid]
            if len(history) < self.person_down_frames:
                continue
            recent = history[-self.person_down_frames:]
            max_h = self._max_height.get(tid, 1.0)

            if self.person_down_requires_transition:
                # Establish a standing baseline, then require a low/wide
                # posture relative to that baseline for the configured run.
                # A person first seen already on the ground is not a fall
                # transition and must not be labeled as one.
                upright_heights = [
                    obs["h"] for obs in history
                    if obs["h"] > 0 and obs["w"] / obs["h"] <= 0.8
                ]
                if not upright_heights:
                    continue
                upright_max_h = max(upright_heights)
                qualifying = 0
                for obs in recent:
                    aspect = obs["w"] / obs["h"] if obs["h"] > 0 else 0
                    low_vs_upright = obs["h"] < upright_max_h * 0.75
                    if aspect > self.person_down_aspect and low_vs_upright:
                        qualifying += 1
            else:
                qualifying = 0
                for obs in recent:
                    aspect = obs["w"] / obs["h"] if obs["h"] > 0 else 0
                    height_ratio = obs["h"] / max_h if max_h > 0 else 1
                    if aspect > self.person_down_aspect or height_ratio < self.person_down_height_drop:
                        qualifying += 1

            if qualifying >= self.person_down_frames:
                avg_conf = sum(o["confidence"] for o in recent) / len(recent)
                self._fired.add(key)
                events.append(DetectedEvent(
                    camera_id=frame.camera_id,
                    category="person_down",
                    confidence=round(min(avg_conf, 1.0), 4),
                    summary=(
                        f"Person (track {tid}) appears to be on the ground — "
                        f"aspect ratio or height-drop threshold exceeded for "
                        f"{self.person_down_frames} consecutive frames at "
                        f"{frame.video_time_s:.1f}s"
                    ),
                    frame_index=frame.frame_index,
                    video_time_s=frame.video_time_s,
                    captured_at_utc=frame.captured_at_utc,
                    image=frame.image,
                ))

        # --- crowd rule -----------------------------------------------------
        # Emit one crowd event per camera per detector run. Using the frame
        # number here would create a duplicate event on every sampled frame.
        is_crowd = (
            self.crowd_threshold is not None
            and len(person_tracks) >= self.crowd_threshold
        )
        if is_crowd and frame.camera_id not in self._crowd_active:
            self._crowd_active.add(frame.camera_id)
            avg_conf = sum(t["confidence"] for t in person_tracks) / len(person_tracks)
            events.append(DetectedEvent(
                camera_id=frame.camera_id,
                category="crowd",
                confidence=round(min(avg_conf, 1.0), 4),
                summary=(
                    f"Crowd detected: {len(person_tracks)} people visible "
                    f"at {frame.video_time_s:.1f}s"
                ),
                frame_index=frame.frame_index,
                video_time_s=frame.video_time_s,
                captured_at_utc=frame.captured_at_utc,
                image=frame.image,
            ))
        elif not is_crowd:
            self._crowd_active.discard(frame.camera_id)

        return events
