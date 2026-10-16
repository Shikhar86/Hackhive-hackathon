# Phase-by-phase test method

Use the seven gates from the 48-hour build plan. Run each phase on the **main Windows laptop** before moving to the next. Hour 0 means the actual hackathon start, not the date this file was written. A gate stays open until every required check has evidence. Test clips and questions should be chosen before tuning against them; otherwise a pass says little about unseen footage.

## How to use the gate runner

From the repository root, with `.venv` active:

```powershell
python -m eval.phase_gate check 0
```

The command prints `PASS`, `FAIL`, or `TODO` for each check and exits with code 0 only when the gate passes. It starts with an automated schema, database, mock-pipeline, and OpenCV check. Later checks are human-reviewed until their modules exist. Record each human check only after running it, with a short result or a link to logs, labels, or a commit:

```powershell
python -m eval.phase_gate record 0 gpu_models "RTX 3050 4 GB; YOLO and SmolVLM loaded; log: docs/gpu-smoke.txt"
python -m eval.phase_gate check 0
```

Evidence is kept in `eval/gate_evidence.json`. Commit the evidence alongside the code so the team can see why a gate was marked complete. Do not record a check based on an intention or a screenshot of a different laptop. If a test fails, fix it and rerun; if a category or optional feature is cut, document the cut in the README and record the supporting test result.

## Gate 0 - setup, hours 0-3

**Automated:** schema categories and seven fields, SQLite insert/retrieve, mock pipeline, OpenCV import.

**Human:** run `nvidia-smi` and record GPU/VRAM; confirm YOLO and SmolVLM load on the main laptop; B and C clone the repository and receive the frozen schema; B gets one Ollama answer and C opens a blank Streamlit page. Also put the schema and categories in the README. Keep model-download and version details in the evidence note.

## Gate 1 - independent modules, hours 3-15

- **Automated:** `python -m eval.phase_gate check 1` creates a 10-second video, verifies 20 samples at 2 FPS and their source timestamps, checks crowd/person-down rule behavior, and runs a synthetic detections-to-SQLite integration check. Run `python -m eval.test_ingest`, `python -m eval.test_event_rules`, or `python -m eval.test_pipeline` individually to inspect focused results.
- **A:** run detection and ByteTrack on a 30-second clip; inspect IDs frame by frame. Run three labeled person-down clips and require triggers on at least two.
- **B:** seed about 200 fake events. Save 20 questions and expected rows before running them; require at least 15 correct. Try `DROP`, `UPDATE`, and `DELETE` queries and require every one to be rejected.
- **C:** open the dashboard on fake events; verify feed, key frames, filters, and chat responses.
- **All:** merge branches, install from the shared dependency files, and rerun each module on the main laptop.

Record clip IDs, question results, commands, and the merge commit in the gate evidence.

## Gate 2 - first end-to-end path, hours 15-24

Run `python -m eval.phase_gate check 2` for the automated mocked-VLM-to-database contract check. Then replay one labeled clip at its original rate. Require a correct person-down event in SQLite with a key frame and real SmolVLM summary; observe the row in the dashboard and query it through the chatbot. Confirm the model is loaded from the local cache. Run this on the main laptop from one environment. If the gate is still open at hour 24, stop new features and fix this path.

## Gate 3 - wider categories, hours 24-32

Add categories one at a time. For each kept category, use three labeled clips and require two triggers. Run a normal five-minute clip and require fewer than three false alarms. Run two camera sources and prove a camera-filtered query. Use 25 prewritten chatbot questions against real events and require 20 correct. Run continuously for 10 minutes without a crash. Mark unsupported categories `cut` in README, with the reason and tested clips.

## Gate 4 - optional novelty, hours 32-38

Start only after gate 3. For each feature kept, test its config switch both on and off. Measure the fraction of sampled frames sent to the VLM on a five-minute clip. Prove one semantic query succeeds beyond keyword matching. Run all kept features for 10 minutes and confirm replay stays at or above real-time speed. Cut a feature that fails by hour 38.

## Gate 5 - evaluation, hours 38-43

Freeze labels before scoring. Report per-category precision and recall, a confusion matrix, and chatbot correctness on the 25 saved questions. Measure average and worst-case latency for detection/tracking, VLM summary, database write, chatbot answer, and event-to-dashboard. State the device and sample count for each measurement. Include at least two concrete failure cases and check that slide numbers match the report.

## Gate 6 - freeze and demo, hours 43-48

From a fresh clone without internet, follow only the README to run the demo. Run the full demo three times consecutively on the presentation laptop. Play the backup video and verify it covers the same flow. Rehearse twice with a timer. Submit README, report, and slides with at least a one-hour buffer.

## Gate record template

For every tested clip or question, keep: `input ID`, `expected result`, `observed result`, `pass/fail`, `duration`, `hardware`, `commit`, and `notes`. This makes failures reproducible and prevents a later prompt or threshold change from silently invalidating an earlier gate.
