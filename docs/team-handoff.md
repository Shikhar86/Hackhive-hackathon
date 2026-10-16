# Team workflow - one laptop

All coding, integration, and testing happen on the main Windows laptop in this repository checkout. The vision, chatbot/data, and dashboard/evaluation roles describe areas of responsibility, not separate laptop environments. Work on the shared checkout in sequence, keep one `.venv` and SQLite database, run the phase checks, then commit and push the completed checkpoint to GitHub. Do not create teammate clones or copy virtual environments.

## Shared contract - freeze now

Import `Event` from `core.schema` and `insert_event`, `get_events` from `core.db`. An event has seven fields: `id`, `timestamp`, `camera_id`, `category`, `confidence`, `summary`, `keyframe_path`. Category code values are `person_down`, `fighting`, `tailgating`, `crowd`, `suspicious`. `timestamp` is UTC ISO 8601. `confidence` is a 0-1 score, not a calibrated probability. See the root README for details.

Do not rename fields or category codes without telling everyone and changing the gate tests together.

## Chatbot and data role

1. Work in `chatbot/` and `eval/` on the main laptop. The current `chatbot/chat.py` uses parameterized, read-only SQL; keep write requests rejected.
2. Use `chatbot/seed_fake_events.py` to populate the local database, then run the saved 20-question evaluation and record the results.
3. Keep changes in the shared checkout and tell the team when the chatbot/data work is ready for review.

## Dashboard and evaluation role

1. Work in `ui/` and `eval/` on the main laptop using the shared `.venv` and database.
2. Check the dashboard feed, category/camera filters, keyframes, and chatbot against known events.
3. Maintain clip labels and evaluation results in the repository; do not commit private/local video files.

## Vision and integration role

1. Run sampling, YOLO, ByteTrack, event rules, and the SQLite pipeline on the main laptop.
2. Review track IDs and event keyframes against labeled clips; record misses and false alerts without tuning on the entire evaluation set.
3. Run `python -m eval.phase_gate check 1` after integrating the modules. Review the changes, commit the phase checkpoint, and push it to GitHub once its gate passes.
