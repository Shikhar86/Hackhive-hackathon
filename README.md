# Live video event summarization system

This repository is being built in phases for a 48-hour hackathon. Development, integration, and testing happen on one Windows laptop in this checkout. Phase 1 now includes sampled YOLO + ByteTrack detections, heuristic person-down and crowd events, SQLite storage, a read-only event chatbot, and a Streamlit dashboard. Run every phase gate before advancing; see [docs/phase-gates.md](docs/phase-gates.md).

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

Then open the local Streamlit address, usually `http://localhost:8501`, and check the feed, filters, keyframes, and chatbot. Run the Phase 1 gate with:

```powershell
.\.venv\Scripts\python.exe -m eval.phase_gate check 1
```

Current limitations: `person_down` and `crowd` use geometric/count heuristics; `fighting`, `tailgating`, and `suspicious` are not implemented in the vision pipeline. SmolVLM captions the event keyframe but does not independently validate the category. See `eval/gate_evidence.json` for the measured phase results.

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
