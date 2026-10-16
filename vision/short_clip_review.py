"""Conservative VLM review proposal when short clips lose a tracked person.

This does not classify a fall. It proposes one moment for human review only
when YOLO loses an interior person track and the local VLM recognizes a
grounded person in a nearby crop. The evaluated standard pipeline is unchanged.
"""

from collections import defaultdict
from pathlib import Path

from core.db import insert_event, insert_event_evidence
from core.schema import Event
from vision.detect_track import TrackedFrame
from vision.event_rules import DetectedEvent
from vision.run_pipeline import save_keyframe
from vision.vlm import caption_image


GROUND_PROMPT = "Is a person clearly lying or collapsed on the ground in this crop? Answer Yes or No only."


def _nearby_crop(image, box):
    """Look above and around the last upright box for a person who went low."""
    image_h, image_w = image.shape[:2]
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left = max(0, int(x1 - 0.7 * width))
    right = min(image_w, int(x2 + 1.9 * width))
    top = max(0, int(y1 - 0.7 * height))
    bottom = min(image_h, int(y2 - 0.54 * height))
    if right - left < 60 or bottom - top < 60:
        return None
    return image[top:bottom, left:right]


def propose_short_clip_review(
    frames: list[TrackedFrame], source_path: str, camera_id: str,
    db_path=None,
) -> list[Event]:
    """Save at most one labeled VLM-assisted review candidate for a short clip."""
    if not 3 <= len(frames) <= 20 or not Path(source_path).is_file():
        return []
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

    candidates = []
    for track_id, observations in history.items():
        last_index, last_frame, track, image_w, image_h = observations[-1]
        x1, y1, x2, y2 = track["xyxy"]
        center_x, center_y = (x1 + x2) / 2 / image_w, (y1 + y2) / 2 / image_h
        area_ratio = (x2 - x1) * (y2 - y1) / (image_w * image_h)
        if (
            last_index <= len(frames) - 3
            and track["confidence"] >= 0.55
            and area_ratio >= 0.012
            and 0.1 <= center_x <= 0.9
            and 0.1 <= center_y <= 0.9
        ):
            candidates.append((area_ratio * track["confidence"], track_id, observations))

    for _, track_id, observations in sorted(candidates, reverse=True)[:4]:
        last_index, last_frame, last_track, _, _ = observations[-1]
        for frame in frames[last_index + 1:last_index + 5]:
            if not 0.5 <= frame.video_time_s - last_frame.video_time_s <= 2.0:
                continue
            crop = _nearby_crop(frame.image, last_track["xyxy"])
            if crop is None:
                continue
            answer = caption_image(crop, GROUND_PROMPT, max_new_tokens=12)
            if not answer.strip().lower().startswith("yes"):
                continue
            reason = (
                "VLM-assisted short-clip review candidate: YOLO tracked a person "
                f"(ID {track_id}) at {last_frame.video_time_s:.1f}s, then lost the track; "
                f"local SmolVLM saw a possible grounded person in a nearby crop at "
                f"{frame.video_time_s:.1f}s. The displayed score is the earlier YOLO "
                "person-box score, not a fall probability. Review the original video."
            )
            detected = DetectedEvent(
                camera_id=camera_id, category="person_down",
                confidence=float(last_track["confidence"]),
                summary="", frame_index=frame.frame_index,
                video_time_s=frame.video_time_s,
                captured_at_utc=frame.captured_at_utc, image=crop,
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
                    f"VLM-assisted review candidate at {frame.video_time_s:.1f}s: "
                    "a person may be on the ground after a person track disappeared. "
                    "Check the original video; this is not a confirmed fall."
                ),
                keyframe_path=keyframe_path,
            )
            if db_path is None:
                insert_event(event)
                insert_event_evidence(
                    event.id, str(Path(source_path).resolve()), frame.video_time_s,
                    frame.frame_index, track_id, reason, detected.track_points,
                )
            else:
                insert_event(event, db_path)
                insert_event_evidence(
                    event.id, str(Path(source_path).resolve()), frame.video_time_s,
                    frame.frame_index, track_id, reason, detected.track_points,
                    db_path=db_path,
                )
            return [event]
    return []
