# Team handoff - Gate 0

The main laptop now passes the schema, database, OpenCV, YOLO, and SmolVLM smoke checks. Work on separate Git branches and merge at each phase gate. Do not copy a virtual environment between laptops.

## Shared contract - freeze now

Import `Event` from `core.schema` and `insert_event`, `get_events` from `core.db`. An event has seven fields: `id`, `timestamp`, `camera_id`, `category`, `confidence`, `summary`, `keyframe_path`. Category code values are `person_down`, `fighting`, `tailgating`, `crowd`, `suspicious`. `timestamp` is UTC ISO 8601. `confidence` is a 0-1 score, not a calibrated probability. See the root README for details.

Do not rename fields or category codes without telling everyone and changing the gate tests together.

## Person B - chatbot and data (CPU laptop)

1. Clone the repo, create a Python 3.11 virtual environment, and confirm `python -c "from core.schema import Event; from core.db import init_db; init_db(); print('OK')"` works.
2. Install Ollama and run one prompt with Qwen2.5 1.5B Instruct locally. Record the exact model tag, response, and elapsed time. Keep this model on CPU during the main-laptop demo.
3. In your branch, build `chatbot/seed_fake_events.py` (about 200 events), `chatbot/text2sql.py`, and `chatbot/chat.py`. Restrict generated SQL to read-only `SELECT`; reject writes. Prepare 20 questions and expected rows before tuning.
4. Send the branch or commit, dependency list, and test results to Person A by the hour-15 checkpoint.

## Person C - dashboard and evaluation

1. Clone the repo, create a Python 3.11 virtual environment, install Streamlit, and open a blank page locally. Record the command and page result.
2. In your branch, build `ui/app.py` with a fake-event feed, filters, key-frame gallery, and chatbot stub. Start `eval/labels.csv` with clip ID, true category, and timestamp; gather 20-30 short clips.
3. Prepare an evaluation sheet that records expected and observed events, pass/fail, latency, hardware, and commit. Avoid tuning against every evaluation clip.
4. Send the branch or commit, dependency list, and UI/test results to Person A by the hour-15 checkpoint.

## Gate 0 acknowledgement

Each teammate should report: branch/commit, successful clone, schema import result, and their module's one-line smoke result. Person A then records `team_handoff` and `teammate_smoke` with those concrete results using `python -m eval.phase_gate record 0 ...`.
