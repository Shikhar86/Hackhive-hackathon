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
- Do not mark the five-minute false-alarm, 25-question real-event chatbot, or ten-minute stability checks complete from this run.

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
