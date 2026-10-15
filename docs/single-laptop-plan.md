# Honeywell Hackathon: Single-Laptop Execution Plan

This project will use one laptop for development, testing, integration, evaluation, and the final demo. GitHub will provide backup, version history, and phase checkpoints.

## Phase 0 — Foundation

- Use one Python virtual environment on the laptop.
- Install dependencies from `requirements.txt`.
- Verify GPU, YOLO, SmolVLM, OpenCV, SQLite, and Streamlit.
- Keep the frozen seven-field event schema unchanged.
- Run the foundation phase gate.
- Commit and push the baseline to GitHub.

Baseline commit already pushed: `d4eaaf6`.

## Phase 1 — Independent modules

Build and test all modules on the same laptop:

- Video sampling and timestamps.
- YOLO detection.
- ByteTrack tracking.
- Person-down detection logic.
- SQLite event insertion and retrieval.
- Read-only chatbot queries.
- Dashboard event feed, filters, key frames, and chat.
- Evaluation labels and test fixtures.

For each component:

1. Implement it.
2. Run its focused smoke test.
3. Run the relevant phase-gate check.
4. Inspect the output.
5. Commit the working change.
6. Push the commit to GitHub.

Do not advance until the Phase 1 gate requirements are recorded.

## Phase 2 — First end-to-end path

Connect and verify the complete flow:

```text
video
→ sampled frames
→ YOLO + ByteTrack
→ event rule
→ SQLite event
→ key frame
→ VLM summary
→ dashboard
→ chatbot query
```

Use a labeled person-down clip and confirm the entire path on the laptop that will be used for the demo.

## Phase 3 — Wider categories

Add and test categories one at a time:

- `person_down`
- `fighting`
- `tailgating`
- `crowd`
- `suspicious`

For every category, record the clip ID, expected result, observed result, pass/fail status, duration, hardware, commit, and notes. Mark unsupported categories as cut rather than keeping unreliable behavior.

## Phase 4 — Optional features

Only after the main flow is stable, consider:

- Semantic search.
- Privacy blur.
- Incident grouping.
- Extra dashboard features.

Every kept feature must work with its switch on and off and must preserve real-time replay performance.

## Phase 5 — Evaluation

Freeze the labels before scoring. Measure and report:

- Precision and recall for each kept category.
- Confusion matrix.
- Detection and tracking latency.
- VLM summary latency.
- Database write latency.
- Chatbot response latency.
- Event-to-dashboard latency.

Record the laptop hardware and sample count for every measurement. Include at least two concrete failure cases.

## Phase 6 — Demo freeze

- Create a clean final checkout from GitHub.
- Follow only the README to run the demo.
- Run the complete demo three consecutive times.
- Test the backup video.
- Confirm the README, report, and slides agree.
- Stop adding features.
- Push the final release checkpoint.

## Working routine

Use this loop for every change:

```text
Implement
→ run focused smoke test
→ run phase gate
→ inspect output
→ commit
→ push to GitHub
```

Keep one source directory, one virtual environment, one local database, one test-video folder, one GitHub remote, and one active branch at a time.

The main rule is simple: every phase must pass on the same laptop that will run the final demo before work begins on the next phase.
