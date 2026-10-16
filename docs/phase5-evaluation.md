# Phase 5 evaluation — working measurements

Measured on 2026-10-16 for the single-laptop prototype. The kept alert category is `person_down`. This document separates measured results from checks still needed for the Phase 5 gate.

## Accuracy on the reviewed clips

Unit of analysis: **one video clip**. A clip is predicted positive when the tuned 2 FPS rule emits at least one `person_down` alert anywhere in it. The four reviewed clips below were used during threshold selection, so these are **in-sample descriptive results**, not an estimate of accuracy on unseen video. Ground truth came from the earlier user review recorded in `eval/phase3_baseline.md` and `eval/gate_evidence.json`; no detector output was treated as a label.

| Reviewed clip | Human label | Tuned rule | Clip result |
|---|---|---|---|
| 30-second office fall demo | Person down | Alerts | True positive |
| 14-second snowy doorway slip | Person down | Alert | True positive |
| 14-second icy-walk slip | Person down | No alert | False negative |
| 322.1-second normal shop view | No person down | Two alerts | False positive |

Clip-level confusion matrix (rows are human labels; columns are predictions):

| | Alert | No alert |
|---|---:|---:|
| Person down | 2 | 1 |
| No person down | 1 | 0 |

- **Precision:** 2 / (2 + 1) = **66.7%** by clip.
- **Recall:** 2 / (2 + 1) = **66.7%** by clip.
- **Normal-footage event count:** two false alerts over 322.1 seconds (645 sampled frames), at 195.5 s and 300.5 s. This satisfies the Phase 3 criterion of fewer than three alerts on that reviewed clip; the clip-level matrix counts the shop video only once.
- **Additional confirmed positives:** both user-confirmed fall intervals in the mixed-scene montage triggered an alert. The rest of that montage has unresolved labels, so it is excluded from the matrix.
- **Chatbot correctness:** `python -m eval.test_real_chatbot` passed **25/25** saved questions against captured video events on 2026-10-16.

This tiny, selected set cannot establish broad precision, recall, or a stable false-alarm rate. The six short AVI references still have unfrozen labels and are excluded from the scored set. A new independently labeled positive and normal set is needed for a defensible held-out result.

## Stage timing

Run `python -m eval.phase5_latency` from the repository root to reproduce. The run used `data/videos/phase1_user_30s.mp4` at 2 sampled FPS on the **NVIDIA GeForce RTX 3050 A Laptop GPU**. It processed **60 frames**, created **two events**, and produced **two real SmolVLM captions without fallback**. Events, JPEGs, and the timing database were written to a temporary directory; the shared `data/events.db` was not modified. The 25 chatbot timing questions ran against a temporary copy of the existing 30-row event database with semantic search off. Each reported worst value is the maximum observed, not a guaranteed bound.

| Stage | Samples | Average | Worst observed | What was timed |
|---|---:|---:|---:|---|
| Detection + ByteTrack, including first model load | 60 frames | 244.20 ms | 9,572.69 ms | Getting the next sampled frame and its tracks; includes decode and first model load |
| Detection + ByteTrack after first frame | 59 frames | 86.09 ms | 115.76 ms | Same path after initial load |
| SmolVLM summary | 2 events | 9,885.17 ms | 17,532.96 ms | Real local caption generation; first call includes model load |
| Keyframe JPEG save | 2 events | 23.65 ms | 26.77 ms | Temporary JPEG write |
| SQLite event write | 2 events | 57.74 ms | 77.89 ms | Existing `insert_event`, including its schema checks |
| Chatbot answer, semantic off | 25 questions | 7.29 ms | 10.27 ms | Read-only parser and database query |
| Dashboard database read | 25 reads | 7.40 ms | 9.02 ms | `get_events(limit=500)` against the temporary DB copy |
| Event commit to dashboard table DOM | 3 events | 4,776 ms | 4,977 ms | Isolated Streamlit dashboard in headless Chrome; five-second auto-refresh |

The complete cold run took **34.71 seconds** for a roughly 30-second clip. This particular cold run was therefore slower than real-time playback. The earlier Phase 4 600-second repeated-video run finished in 40.8 seconds and showed sustained throughput after setup on different footage. These are different workloads; neither substitutes for a varied, live-stream latency study.

For the dashboard timing, a second Streamlit server on local port 8504 read an isolated temporary database. Three test events were inserted one at a time, and a headless Chrome browser observed when each event appeared in the dashboard table's DOM. The commit-to-DOM delays were **4,977, 4,921, and 4,430 ms**. This includes waiting for the five-second refresh and the browser data update. It does **not** measure final pixel paint or operator reaction time. The probe used no shared event rows and has only three samples; its raw result is in `eval/phase5_dashboard_probe_results.json`.

## Concrete failure cases

1. **Missed icy-walk slip:** the reviewed 14-second positive clip produced no alert at the tuned 2 FPS settings. The rule requires three qualifying sampled observations of the same tracked person, and this clip did not satisfy that trigger. A successful alert on other clips does not fix this miss.
2. **Shop alert at 195.5 s:** visual review found only a partial head at the frame edge. The bounding-box posture heuristic produced a false person-down alert.
3. **Shop alert at 300.5 s:** visual review found a standing shopper. The heuristic produced another false person-down alert.

SmolVLM captions a selected keyframe; it does not independently validate the `person_down` category. The label and confidence score come from the heuristic and should not be presented as a calibrated probability.

## Gate status and next evidence

- Accuracy and chatbot figures above are measured, but detector accuracy is in-sample and based on four clips. Keep the sample count and this limitation with every slide or spoken claim.
- Timing is measured for detection/tracking, captioning, JPEG save, database write, chatbot answer, and event commit to the dashboard table's browser DOM. The UI sample count is three and does not include pixel paint.
- The five-slide `docs/phase5-review.pptx` was checked against this report. It uses the same 2/3 positive clips, two shop false alerts, 25/25 chatbot result, 66.7% in-sample clip-level precision and recall, and rounded timing figures. Its limitations slide names the missed slip, both false-alert causes, and the cut categories.
- Run `python -m eval.phase_gate check 5` after recording the report evidence. Passing this gate documents the current evaluated prototype; it does not establish performance on unseen footage.
