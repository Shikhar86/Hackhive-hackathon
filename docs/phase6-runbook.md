# Phase 6 — offline demo and submission runbook

Run this on the presentation laptop. The project uses one laptop and does not need a cloud inference API. The live demo supports `person_down`; the other proposed categories remain outside this build.

## Before the demo

Keep these local-only items on the presentation laptop:

- The Python environment at `.venv` with dependencies already installed.
- `yolov8n.pt` in the repository root.
- A reviewed fall clip, such as `data/videos/phase1_user_30s.mp4`.
- The SmolVLM cache in `data/hf-cache`. Do not assume a new clone contains it.

Confirm that `docs/phase5-evaluation.md` and `docs/phase5-review.pptx` open. The report's accuracy is an in-sample result on four reviewed clips, not an unseen-video claim.

## Clean offline checkout rehearsal

Run the following in PowerShell from the current repository root. Choose an unused name for the fresh checkout. The first command needs Git installed but no internet. It does not modify the current checkout.

```powershell
$Project = (Get-Location).Path
$Fresh = Join-Path (Split-Path $Project -Parent) 'surveillance-hackathon-offline-check'
git clone --local --no-hardlinks $Project $Fresh
Copy-Item -LiteralPath (Join-Path $Project 'yolov8n.pt') -Destination $Fresh
New-Item -ItemType Directory -Path (Join-Path $Fresh 'data/videos') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Project 'data/videos/phase1_user_30s.mp4') -Destination (Join-Path $Fresh 'data/videos/demo.mp4')
$env:HF_HOME = Join-Path $Project 'data/hf-cache'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Set-Location $Fresh
& (Join-Path $Project '.venv/Scripts/python.exe') -B -m vision.run_pipeline data/videos/demo.mp4 --camera-id phase6-demo-1
```

Expected: the pipeline completes without a `VLM summary unavailable` warning. On the reviewed 30-second clip, the current build produced two `person_down` events, real SmolVLM captions, and saved JPEG keyframes. The absolute Python path intentionally reuses the prepared environment on this same laptop; this checks a clean code checkout and offline model loading, not a fresh dependency installation. To validate installation without the prepared environment, build and test an offline wheelhouse separately.

## Full live demo: repeat three times

1. Open the Streamlit dashboard from the fresh checkout in a second PowerShell window. Repeat the `$Project` and `$Fresh` assignments from above in that window, then run:

   ```powershell
   Set-Location $Fresh
   & (Join-Path $Project '.venv/Scripts/python.exe') -B -m streamlit run ui/app.py
   ```

2. Open the local address shown by Streamlit, usually `http://localhost:8501`. Keep **Auto-refresh (5 s)** checked.
3. For each run, replay the clip with a new camera ID (`phase6-demo-1`, `phase6-demo-2`, then `phase6-demo-3`). In the first PowerShell window, use the same pipeline command above with the next camera ID. Wait for the row to appear without a manual browser reload.
4. Show the new event's timestamp, camera, category, confidence, summary, and keyframe. Ask the chatbot `Show person down events from camera phase6-demo-1` (change the number for each run). Confirm that it returns the matching camera's events.
5. Note the start/end time, event count, caption success, dashboard appearance, and chatbot result for each of the three consecutive runs. A pipeline exit alone does not complete the full-demo gate.

The demonstration uses a simulated video stream. The category comes from a posture heuristic; SmolVLM describes the selected keyframe and does not validate the category. Confidence is an uncalibrated score.

## Backup recording

Record the complete flow on this laptop: show the source clip, start processing, wait for the refreshed dashboard row, open its keyframe, and ask the chatbot about that event. Save the recording outside Git, play the saved file end to end, and check that text and audio are readable. Keep the recording locally available if the live demo fails.

## Timed rehearsal and submission

Run the entire presentation twice with a timer, including the backup-video fallback. Check the event summary and dashboard before each rehearsal. Submit the README, `docs/phase5-evaluation.md`, and `docs/phase5-review.pptx` through the organizer's actual portal at least one hour before the stated deadline. Save the confirmation or receipt. Do not record the Phase 6 `submission` gate as passed until the portal confirms it.

## Gate evidence to record

| Check | Evidence required | Status |
|---|---|---|
| Fresh offline checkout | Clone path, commit, commands, no-network run result, real caption and keyframe | Pending |
| Three full demos | Three consecutive screen-observed results covering video, event, dashboard and chatbot | Pending |
| Backup video | Path, duration, playback result, full-flow review | Pending |
| Submission | Deadline and time zone, upload time, portal receipt, one-hour buffer | Pending |

Run `python -m eval.phase_gate check 6` to see the current gate. Record human checks only after the evidence exists.

## Work completed on 2026-10-16

A clean archive of commit `343b16c` was extracted into a temporary checkout. The local reviewed clip and YOLO weight were copied into it. The first pipeline run made two events but its captions fell back because the clean export did not include the SmolVLM cache. After setting `HF_HOME` to the main laptop's cache and setting `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, three consecutive pipeline runs each made two `person_down` events with real SmolVLM captions. All six new keyframe paths existed. Camera-filtered chatbot queries returned two matching events for each of the three run IDs. This validates the offline processing path with the prepared local environment.

The isolated Streamlit dashboard then displayed the saved events and keyframes. During an additional live pipeline replay, its feed changed from 8 to 10 events with the two new rows and keyframes, without a browser reload. A chatbot question restricted to that camera returned its two events. This was one browser-observed run, following three pipeline-only runs. It was an archive export, not a Git clone. Three complete browser-observed demos, backup recording review, and submission are still pending, so the Phase 6 gate remains open.
