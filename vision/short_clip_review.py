"""Conservative VLM review proposal for short clips with missed people.

This does not classify a fall. It proposes one moment for human review only
when YOLO loses an interior person track, or finds no people in a short
clip, and the local VLM proposes a person down. The evaluated standard
pipeline is unchanged.
"""

from collections import defaultdict
from contextlib import closing
from pathlib import Path
import re

from core.db import DB_PATH, connect, init_db, insert_event, insert_event_evidence
from core.schema import Event
from vision.detect_track import TrackedFrame
from vision.event_rules import DetectedEvent
from vision.run_pipeline import save_keyframe
from vision.vlm import caption_image


GROUND_PROMPT = (
    "Is a person visibly lying down, collapsed, or sprawled across chairs "
    "or on the floor? Answer Yes or No only."
)
POSTURE_PROMPT = (
    "Describe the visible people's posture in one factual sentence. "
    "Mention anyone lying down if clearly visible."
)
NO_TRACK_PROMPT = (
    "Describe the visible people and their posture in one factual sentence. "
    "Mention if anyone is lying on the floor."
)


def _describes_person_down(answer: str) -> bool:
    text = answer.lower().strip()
    if re.search(r"\b(no|nobody|not|isn't|aren't|without)\b", text):
        return False
    return bool(re.search(
        r"\b(person|people|someone|man|woman|individual)\b.{0,40}"
        r"\b(lying down|lying on|collapsed|sprawled)\b", text,
    ))


def _nearby_crops(image, box):
    """Inspect nearby floor and furniture at two scales."""
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    windows = (
        (x1 - 2.0 * width, y1 - 0.3 * height, x2 + 2.0 * width, y2 + 1.5 * height),
        (x1 - width, y1 - 0.2 * height, x2 + width, y2 + 0.5 * height),
    )
    for left, top, right, bottom in windows:
        left, top = max(0, int(left)), max(0, int(top))
        right, bottom = min(image_w, int(right)), min(image_h, int(bottom))
        if right - left >= 60 and bottom - top >= 60:
            yield image[top:bottom, left:right]


def _propose_no_track_review(frames, resolved_source, camera_id, db_path):
    """Ask two local image views to agree when YOLO missed every person."""
    # One check per second, starting halfway through the first second. Keep
    # the true last frame so a collapse at the clip boundary is still seen.
    candidate_frames = frames[1::2]
    if all(frame is not frames[-1] for frame in candidate_frames):
        candidate_frames.append(frames[-1])
    for frame in candidate_frames:
        if frame.image is None:
            continue
        image_h, image_w = frame.image.shape[:2]
        floor_view = frame.image[
            int(image_h * 0.24):, int(image_w * 0.2):int(image_w * 0.83)
        ]
        if floor_view.size == 0:
            continue
        whole_answer = caption_image(frame.image, NO_TRACK_PROMPT, max_new_tokens=40)
        if not _describes_person_down(whole_answer):
            continue
        floor_answer = caption_image(floor_view, NO_TRACK_PROMPT, max_new_tokens=40)
        if not _describes_person_down(floor_answer):
            continue
        reason = (
            "VLM-assisted no-track review candidate: YOLO found no tracked "
            "people in this short clip. Two local views of the same frame "
            f"suggest a person down at {frame.video_time_s:.1f}s. This has no "
            "incident confidence score and does not confirm a fall. "
            "Review the original video."
        )
        detected = DetectedEvent(
            camera_id=camera_id, category="person_down", confidence=0.0,
            summary="", frame_index=frame.frame_index,
            video_time_s=frame.video_time_s,
            captured_at_utc=frame.captured_at_utc, image=floor_view,
            track_id=None, rule_reason=reason,
        )
        keyframe_path = save_keyframe(detected)
        event = Event.create(
            camera_id=camera_id, category="person_down", confidence=0.0,
            summary=(
                f"VLM-assisted review candidate at {frame.video_time_s:.1f}s: "
                "two image views suggest a person may be down, but YOLO "
                "found no person track. Check the original video; this is "
                "not a confirmed fall."
            ),
            keyframe_path=keyframe_path,
        )
        if db_path is None:
            insert_event(event)
            insert_event_evidence(
                event.id, resolved_source, frame.video_time_s,
                frame.frame_index, None, reason, [],
            )
        else:
            insert_event(event, db_path)
            insert_event_evidence(
                event.id, resolved_source, frame.video_time_s,
                frame.frame_index, None, reason, [], db_path=db_path,
            )
        return [event]
    return []


def propose_short_clip_review(
    frames: list[TrackedFrame], source_path: str, camera_id: str,
    db_path=None,
) -> list[Event]:
    """Save at most one labeled VLM-assisted review candidate for a short clip."""
    if not 3 <= len(frames) <= 26 or not Path(source_path).is_file():
        return []
    resolved_source = str(Path(source_path).resolve())
    selected_db = DB_PATH if db_path is None else db_path
    init_db(selected_db)
    with closing(connect(selected_db)) as connection:
        existing = connection.execute(
            "SELECT events.* FROM events JOIN event_evidence evidence "
            "ON evidence.event_id = events.id WHERE evidence.source_path = ? "
            "AND events.camera_id = ? "
            "AND events.summary LIKE 'VLM-assisted review candidate%' "
            "ORDER BY events.timestamp DESC LIMIT 1",
            (resolved_source, camera_id),
        ).fetchone()
    if existing is not None:
        return [Event(**dict(existing))]
    history = defaultdict(list)
    for index, frame in enumerate(frames):
        if frame.image is None:
            continue
        image_h, image_w = frame.image.shape[:2]
        for track in frame.tracks:
            if track.get("class_name") != "person" or track.get("track_id") is None:
                continue
            x1, y1, x2, y2 = track["xyxy"]
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            history[track["track_id"]].append((index, frame, track, image_w, image_h))

    if not history:
        return _propose_no_track_review(
            frames, resolved_source, camera_id, db_path,
        )

    candidates = []
    terminal_candidates = []
    for track_id, observations in history.items():
        last_index, last_frame, track, image_w, image_h = observations[-1]
        x1, y1, x2, y2 = track["xyxy"]
        center_x, center_y = (x1 + x2) / 2 / image_w, (y1 + y2) / 2 / image_h
        area_ratio = (x2 - x1) * (y2 - y1) / (image_w * image_h)
        if (
            last_index <= len(frames) - 3
            and frames[-1].video_time_s - last_frame.video_time_s >= 1.5
            and track["confidence"] >= 0.55
            and area_ratio >= 0.012
            and 0.1 <= center_x <= 0.9
            and 0.1 <= center_y <= 0.9
        ):
            candidates.append((area_ratio * track["confidence"], track_id, observations))
        elif (
            last_index < len(frames) - 1
            and 0.1 <= frames[-1].video_time_s - last_frame.video_time_s < 1.5
            and track["confidence"] >= 0.65
            and area_ratio >= 0.015
            and (x2 - x1) / (y2 - y1) < 0.95
            and 0.1 <= center_x <= 0.9
            and 0.1 <= center_y <= 0.9
        ):
            terminal_candidates.append((area_ratio * track["confidence"], track_id, observations))

    prioritized = (
        [(score, tid, observations, False) for score, tid, observations in sorted(candidates, reverse=True)[:4]]
        + [(score, tid, observations, True) for score, tid, observations in sorted(terminal_candidates, reverse=True)[:4]]
    )
    for _, track_id, observations, terminal in prioritized:
        last_index, last_frame, last_track, _, _ = observations[-1]
        later_frames = [frames[-1]] if terminal else frames[last_index + 1:last_index + 5]
        for frame in later_frames:
            if not 0.5 <= frame.video_time_s - last_frame.video_time_s <= 2.0:
                if not terminal or frame.video_time_s <= last_frame.video_time_s:
                    continue
            crop = None
            evidence_frame = frame
            terminal_posture = None
            if terminal:
                before_crops = list(_nearby_crops(last_frame.image, last_track["xyxy"]))
                after_crops = list(_nearby_crops(frame.image, last_track["xyxy"]))
                if before_crops and after_crops:
                    before = caption_image(before_crops[0], POSTURE_PROMPT, max_new_tokens=40)
                    after = caption_image(after_crops[0], POSTURE_PROMPT, max_new_tokens=40)
                    before_down = _describes_person_down(before)
                    after_down = _describes_person_down(after)
                    if after_down:
                        terminal_posture = "new_down" if not before_down else "persistent_down"
                        crop = after_crops[0]
                    elif before_down:
                        # Keep the earlier visible down posture even if the
                        # person recovers or the later view becomes unclear.
                        terminal_posture = "earlier_down"
                        evidence_frame = last_frame
                        crop = before_crops[0]
            else:
                for nearby in _nearby_crops(frame.image, last_track["xyxy"]):
                    answer = caption_image(nearby, GROUND_PROMPT, max_new_tokens=12)
                    if answer.strip().lower().startswith("yes"):
                        crop = nearby
                        break
            if crop is None:
                continue
            reason = (
                "VLM-assisted short-clip review candidate: YOLO tracked a person "
                f"(ID {track_id}) at {last_frame.video_time_s:.1f}s, then lost the track; "
                f"local SmolVLM saw a possible person lying on the floor or across chairs at "
                f"{evidence_frame.video_time_s:.1f}s. Earlier YOLO person-box score: "
                f"{last_track['confidence']:.0%}. This candidate has no incident "
                "confidence score. Review the original video."
            )
            if terminal:
                reason += {
                    "new_down": " The posture changed in the final sampled frame; this is an end-of-clip review proposal.",
                    "persistent_down": " A person appears down in both checked views; fall onset is not established.",
                    "earlier_down": " A person appears down in the earlier checked view; the later view does not confirm a fall onset.",
                }[terminal_posture]
            detected = DetectedEvent(
                camera_id=camera_id, category="person_down",
                confidence=0.0,  # Schema placeholder: this review proposal is unscored.
                summary="", frame_index=evidence_frame.frame_index,
                video_time_s=evidence_frame.video_time_s,
                captured_at_utc=evidence_frame.captured_at_utc, image=crop,
                track_id=track_id, rule_reason=reason,
                track_points=[
                    {
                        "time_s": prior_frame.video_time_s,
                        "x_pct": round((prior_track["xyxy"][0] + prior_track["xyxy"][2]) * 50 / image_w, 2),
                        "y_pct": round((prior_track["xyxy"][1] + prior_track["xyxy"][3]) * 50 / image_h, 2),
                    }
                    for _, prior_frame, prior_track, image_w, image_h in observations[-40:]
                ],
            )
            keyframe_path = save_keyframe(detected)
            event = Event.create(
                camera_id=camera_id, category="person_down",
                confidence=detected.confidence,
                summary=(
                    f"VLM-assisted review candidate at {evidence_frame.video_time_s:.1f}s: " +
                    ({"new_down": "a person may be down at the end of the clip. ",
                      "persistent_down": "a person appears down near the end; fall onset is unconfirmed. ",
                      "earlier_down": "a person appears down before the final frame; fall onset is unconfirmed. "
                     }[terminal_posture] if terminal else
                     "a person may be down after a person track disappeared. ") +
                    "Check the original video; this is not a confirmed fall."
                ),
                keyframe_path=keyframe_path,
            )
            if db_path is None:
                insert_event(event)
                insert_event_evidence(
                    event.id, resolved_source, evidence_frame.video_time_s,
                    evidence_frame.frame_index, track_id, reason, detected.track_points,
                )
            else:
                insert_event(event, db_path)
                insert_event_evidence(
                    event.id, resolved_source, evidence_frame.video_time_s,
                    evidence_frame.frame_index, track_id, reason, detected.track_points,
                    db_path=db_path,
                )
            return [event]
    return []
