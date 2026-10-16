# Phase 3 reference-clip baseline

Run on 2026-10-16 with the existing YOLO + ByteTrack pipeline at 2 sampled FPS, default confidence, default event rules, and `--no-vlm`. No event thresholds were tuned for these clips. The source clips are kept locally under `data/videos/` and remain excluded from Git.

These are detector outputs and visual-review notes, not a scored accuracy result. Ground-truth labels still need to be fixed in `eval/phase3_clip_labels.csv` before tuning or marking Gate 3 categories complete. Use `1` for present, `0` for absent, and record the relevant time interval in `notes`.

| Clip | Existing output | Initial visual review |
|---|---|---|
| `v011_converted 3.avi` | `person_down` at 4.5 s, score 0.3461 | People appear to be in an altercation; a person is low to the floor. Candidate positive for person-down and fighting, but the pipeline emits no fighting label. |
| `v006_converted 3.avi` | `crowd` at 0.0 s, 13 people, score 0.5593 | Busy security queue. This may be a crowd-count hit or a false alarm, depending on the agreed definition of a crowd incident. |
| `w001_converted 3.avi` | No event | The visible clip overlay describes a courthouse gunfight. This is a candidate fighting positive that the current rules miss. |
| `v001_converted 3.avi` | `crowd` at 0.0 s, 12 people, score 0.5226; `person_down` at 1.0 s, score 0.5411 | Apparent street altercation with a person on the ground. Candidate positive for person-down and fighting; no fighting category is implemented. |
| `v003_converted 3.avi` | No event | Needs full-clip label review before scoring. |
| `v026_converted 3.avi` | No event | Needs full-clip label review before scoring. |

## Gate 3 status

- The current implementation supports only `person_down` and `crowd` event rules. It has no `fighting`, `tailgating`, or `suspicious` classifier.
- This baseline alone does not satisfy the three-positive-clips-per-kept-category requirement; the expected labels have not been frozen.
- The `v006` crowd alert needs review against the team's definition of a harmful crowd incident. A count threshold can alert on normal queues.
- The gunfight clip demonstrates that the current crowd/person-down heuristics do not detect fighting by themselves.
- The first camera-filter query exposed a chatbot parser bug: IDs such as `v006` were rewritten as `camera-v006`. The parser now keeps alphabetic IDs intact and preserves the existing `camera-1` form. The regression check passes, and real queries return only the matching `v006` row or matching `v001` rows.
- Two separate AVI sources have been processed under camera IDs `v006` and `v001`; this was a sequential replay, not a simultaneous multi-camera stress test.
- `eval.test_real_chatbot` now evaluates 25 saved questions against the captured video-event rows (excluding the old mock event) and passes 25/25.
- The initial reference-clip run did not establish the later false-alarm, chatbot, or stability checks; their current evidence is recorded in the Phase 3 gate and the follow-up sections below.

## Detector recheck

Re-ran the existing person-down and crowd rules on all six supplied AVI clips on 2026-10-16 at 2 sampled FPS. This run bypassed event persistence and did not change the shared event database. `max_people` is the largest number of YOLO person tracks in any sampled frame; the crowd rule threshold is eight.

| Clip | Sampled frames | Max people | Events emitted |
|---|---:|---:|---|
| `v001_converted 3.avi` | 13 | 12 | `crowd` at 0.0 s; `person_down` at 1.0 s |
| `v003_converted 3.avi` | 6 | 5 | None |
| `v006_converted 3.avi` | 12 | 13 | `crowd` at 0.0 s |
| `v011_converted 3.avi` | 18 | 7 | `person_down` at 4.5 s |
| `v026_converted 3.avi` | 20 | 0 | None |
| `w001_converted 3.avi` | 11 | 2 | None |

The supplied clips are only 3.0–9.7 seconds long. The two clips that meet the current count threshold (`v001`, `v006`) are not enough to score crowd against a three-clip set. The visually apparent falls/altercations also need frozen, clip-level labels and a consistent definition of `person_down` before recall can be scored. The blank worksheet remains intentionally unfilled rather than treating model outputs as ground truth.

## Normal-footage false-alarm check

On 2026-10-16, ran the default person-down and crowd rules on the user-provided fixed shop-camera clip `Recording 2026-10-16 065305.mp4`. The clip is 322.1 seconds (5 minutes 22 seconds), sampled at 2 FPS for 645 frames. The full video was reviewed at 30-second intervals and each alert frame was manually checked; the scene shows ordinary shopping, reaching, and bending, with no fall or crowd incident.

| Result | Count / details |
|---|---|
| `person_down` alerts | 9, at 10.5, 115.0, 154.0, 157.0, 172.5, 177.5, 178.5, 222.0, and 300.5 seconds |
| Reviewed false alerts | 9 of 9; people were standing, bending, or reaching for shelves |
| `crowd` alerts | 0 |
| Phase 3 false-alarm criterion | **Fail**; the requirement is fewer than 3 false alarms |

A threshold-only sweep evaluated 96 combinations of person-down aspect ratio (0.8–1.4), height-drop threshold (0.25–0.55), and consecutive frames (2–5) against this clip and three candidate fall clips (`v001`, `v011`, `v003`). No tested setting both reduced the normal-clip person-down alerts below 3 and detected person-down in at least 2 of those 3 candidate clips. One setting (`aspect=1.2`, `height_drop=0.25`, `frames=5`) produced 1 normal-clip alert but detected only `v011`; it missed `v001` and `v003`. This setting was not retained. Defaults remain unchanged. The sweep is a small, clip-specific experiment, not a general accuracy claim.

## Mixed-scene follow-up clip

The user also supplied `Recording 2026-10-16 071114.mp4` (314.5 seconds). It is a montage of different outdoor, sports, play, and home scenes rather than one continuous camera view. At 2 sampled FPS, the default rules processed 629 frames and emitted 16 `person_down` events and 1 `crowd` event. The event frames and short surrounding context were reviewed. Several sequences may provide candidate fall examples (around 72–77 s, 115–120 s, and 193–196 s), while other alerts occur during play, exercise, swimming, or when a person is already resting on the ground. These events have not been assigned confirmed ground-truth labels, so this montage is not scored as a detector accuracy set or as a normal-footage false-alarm check. Tentative time ranges and candidate labels are listed in `eval/phase3_person_down_segments.csv` for review before the motion rule is evaluated.

## Opt-in posture-transition experiment

Added an opt-in experimental mode to `EventDetector` that requires an observed upright person, followed by a low/wide bounding box for three sampled observations. The default mode is unchanged. Focused rule checks pass for an upright-to-low sequence, for someone first seen already down (no transition alert), and for existing default behavior.

Replayed the cached YOLO + ByteTrack tracks at 2 FPS:

| Clip | Existing rule | Transition experiment |
|---|---:|---:|
| Normal shop, 322 s | 9 alerts | 1 alert |
| Follow-up montage, 314 s | 16 alerts | 0 alerts |
| `v001_converted 3.avi` | 1 alert | 0 alerts |
| `v011_converted 3.avi` | 1 alert | 0 alerts |
| `v003_converted 3.avi` | 0 alerts | 0 alerts |

The experiment substantially lowers alerts on this one normal clip but misses all candidate fall clips in this small replay. It is not a usable replacement and was not enabled by default. The remaining shop alert is at 193.0 s. This does not pass the Phase 3 false-alarm criterion because the default implementation still produces nine reviewed false alerts; Gate 3 remains open. Track continuity, sampling rate, camera view, and confirmed segment labels need to be improved before this motion rule can be judged fairly.
