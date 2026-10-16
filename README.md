# Live video event summarization system

This repository is being built in phases for a 48-hour hackathon. Development, integration, and testing happen on one Windows laptop. The current prototype includes sampled YOLO + ByteTrack detections, heuristic person-down events, local SmolVLM captions, SQLite storage, a read-only event chatbot, and a Streamlit dashboard. Run every phase gate before advancing; see [docs/phase-gates.md](docs/phase-gates.md).

## Phase 6 offline demo

Use [the Phase 6 runbook](docs/phase6-runbook.md) for the clean-checkout rehearsal, three complete dashboard demos, backup recording, and submission checklist. Keep the local video, `yolov8n.pt`, the prepared `.venv`, and the cached SmolVLM model on the presentation laptop; Git does not contain those large local assets. The gate remains open until each manual check has direct evidence. The measured Phase 5 results and limitations are in [the evaluation report](docs/phase5-evaluation.md), with [slides](docs/phase5-review.pptx).

## Problem statement 3 - acceptance target

The Honeywell brief asks for a working prototype that processes a **live or simulated video stream**, detects and tracks objects, uses an **open-source vision-language model** to summarize events, classifies them with confidence scores, stores timestamp/camera/category/summary/key frame, and answers natural-language questions about the stored events. All inference must run with open-source models on one consumer GPU or laptop, with no paid cloud API. The submission must include measured accuracy and latency.

The five named categories are examples in the brief; our 48-hour build plan treats them as targets. If a category fails evaluation, we will disclose that result and prioritize a reliable end-to-end prototype, chatbot, and measured report. Features such as semantic search, privacy blur, and incident grouping are additions from the build plan, not requirements of the brief.

## Current smoke test

```powershell
.\.venv\Scripts\python.exe -m vision.run_pipeline --mock
.\.venv\Scripts\python.exe -c "from core.db import get_events; print(get_events())"
.\.venv\Scripts\python.exe -m eval.phase_gate check 0
```

## Phase 1 local workflow

Put a test clip in `data/videos/` (video files are local-only and excluded from Git), then run the detector and tracker:

```powershell
.\.venv\Scripts\python.exe -m vision.detect_track data/videos/clip.mp4 --sample-fps 2 --jsonl data/videos/clip_tracks.jsonl
```

Run the full video-to-event pipeline and open the dashboard in another PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m vision.run_pipeline data/videos/clip.mp4
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

The video pipeline calls cached SmolVLM on each event keyframe by default. Use `--no-vlm` only for debugging the vision rules without captions.

Then open the local Streamlit address, usually `http://localhost:8501`, and check the feed, filters, keyframes, and chatbot.

The dashboard accepts an uploaded video and creates a browser-compatible H.264 Baseline copy with audio using the locally installed `imageio-ffmpeg` runtime. The original is retained, while the selected orientation of the playback copy is also used for analysis. Enter a camera name and choose **Analyze video** to run YOLO, ByteTrack, person-down rules, SmolVLM, and SQLite at the default two sampled frames per second. The dashboard shows sampled frames during analysis, then results in the event feed below. Both local files are kept under ignored `data/videos/uploads/`; they are not sent to a cloud service. A clip with no matching event will report zero events. Processing and preview conversion happen on the laptop and may take time for long clips.

For phone uploads, check **Video orientation** before analysis. Some recordings contain rotation metadata that does not match the filmed scene. The preview and detector use the same selected, browser-compatible copy. The upload control accepts MP4, MOV, M4V, AVI, WebM, and 3GP up to 1 GB; smaller clips transfer more reliably over a phone hotspot. The **Standard** person-down mode retains the evaluated 1.2 aspect and three-frame rule. **Motion-sensitive (experimental)** requires a box wider than 1.05 and shorter than 75% of that track's tallest observed box for two sampled frames. On the 45-second user-provided hallway clip, this mode flagged a possible low posture at 29.0 seconds after orientation correction; Standard found no event. This is a candidate for human review, not a confirmed fall or a measured accuracy improvement. An approximate replay of the saved 4 FPS normal-shop tracks downsampled to 2 FPS produced additional false alerts, so use the mode only when greater sensitivity is worth more manual review.

The **Incident timeline** opens a saved alert for human review. The sidebar can filter the feed by **Unreviewed**, **Confirmed**, or **False alarm** status. Its detail panel shows the keyframe, SmolVLM scene description, the rule measurements that triggered the alert, three nearby video frames, local playback near the event, and the tracked person's sampled path when available. Mark the alert **Confirmed** or **False alarm** and save an operator note. Reviews and detection evidence are stored in separate local SQLite tables; they do not change the original seven-field event format or the detector's output. Older events still open with their keyframes and summaries, but cannot show replay, track path, or detailed rule evidence unless their clip is analyzed again. The track diagram shows sampled bounding-box centers, not a continuous trajectory or proof that a fall occurred. The detector score is not a calibrated probability.

Run the Phase 1 gate with:

```powershell
.\.venv\Scripts\python.exe -m eval.phase_gate check 1
```

Current limitations: `person_down` uses a bounding-box heuristic; it misses the icy-walk slip and its tuning was evaluated at 2 FPS on the available clips. At 4 FPS the false-alert count rises, so do not change the sample rate without reevaluating. SmolVLM captions the event keyframe but does not independently validate the category. See `eval/gate_evidence.json` for the measured phase results.

### Phase 4 optional feature: local semantic search

The dashboard chatbot has a **Semantic search (local Ollama)** checkbox, off by default. With it off, the existing category, camera, confidence, and latest-event search is unchanged. With it on, those structured filters still apply, then the 100 most recent matching events are ranked by cosine similarity between the question and event summaries. Older matching events are excluded from semantic ranking; the dashboard help text and result message disclose this limit. The search uses the locally installed `nomic-embed-text` model through Ollama at `127.0.0.1:11434`; it does not call a paid cloud API. Start Ollama and confirm the model is available with `ollama list` before using this option. A local model error is shown in the dashboard rather than silently returning keyword results. Semantic ranking returns the best matches without a relevance cutoff, so inspect the summaries before treating a result as evidence.

Phase 4 checks: `python -m eval.test_semantic` exercises the switch on and off with a paraphrased question; `python -m eval.phase4_vlm_ratio` counts attempted VLM calls on 645 saved samples from the 322.1-second normal clip; and `python -m eval.phase4_stability` runs the repeated 600-second clip with real YOLO, ByteTrack, SmolVLM, and keyframe saving into an isolated event database. The VLM ratio uses a caption stub to count call sites, so it does not measure caption latency or success. Privacy blur and incident grouping are not implemented in this build.

### Phase 3 category scope

For this build and demo, `person_down` is the only kept category. Its labeled-clip check detects events in 2 of 3 positive clips, and the tuned 2 FPS rule produces 2 false alerts on the reviewed normal shop clip. The missed icy-walk clip remains documented. The following categories are cut from the standard detector/demo:

- `crowd`: cut from the standard detector/demo because the eight-person count rule also flags an ordinary queue (`v006`), so it does not establish a harmful crowd incident. The count-only experiment can be re-enabled with `--crowd-threshold 8` but is not presented as incident detection.
- `fighting`: no fighting classifier is implemented; the altercation examples (`v001`, `v003`, `v011`, and `w001`) are not detected as fighting events.
- `tailgating`: no classifier or labeled test clips are available.
- `suspicious`: no classifier or agreed labeled test set is available; a general-purpose label would be too vague to evaluate reliably.

These are scope cuts for this build, not claims that the events are absent from surveillance footage. The event schema retains all five category values for compatibility.

## Frozen event contract

Each event has exactly these seven fields: `id` (UUID string), `timestamp` (UTC ISO 8601 string), `camera_id` (string), `category` (string), `confidence` (normalized score from 0 to 1), `summary` (string), and `keyframe_path` (string or null). Confidence is a score, not a calibrated probability; different model scores should not be compared without evaluation.

Category values: `person_down`, `fighting`, `tailgating`, `crowd`, `suspicious`. These correspond to person down, fighting, tailgating, crowd incident, and suspicious activity. Teammates should use the code values in database rows and APIs.

The shared API is `Event.create(...)`, `insert_event(event)`, and `get_events()`. The SQLite file is `data/events.db` and is excluded from Git.

## Main laptop vision environment

Create a Python 3.11 virtual environment. Install the CUDA-enabled PyTorch and Torchvision wheels from the [official selector](https://pytorch.org/get-started/locally/) for Windows and your driver, then install `requirements.txt`. The build tested here was PyTorch `2.14.1+cu126` with Torchvision `0.29.1+cu126`; the command used was:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

See [docs/gpu-smoke.md](docs/gpu-smoke.md) for measured results. Model downloads are cached locally and ignored by Git.

## One-laptop work arrangement

The team roles (vision, chatbot/data, dashboard/evaluation) are work assignments within this single checkout on the main Windows laptop. Keep one shared Python environment and database on that laptop; do not set up separate laptop clones for the hackathon workflow. Review and commit the integrated Phase 1 changes at the gate checkpoint, then push that checkpoint to GitHub. See [docs/team-handoff.md](docs/team-handoff.md) for the one-laptop workflow.
