"""Streamlit dashboard for live surveillance events.

Run::

    streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from chatbot.chat import ask
from core.db import get_events, init_db

init_db()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Surveillance Dashboard", layout="wide")
st.title("🔒 Surveillance Event Dashboard")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

category_options = ["All", "person_down", "fighting", "tailgating", "crowd", "suspicious"]
selected_category = st.sidebar.selectbox("Category", category_options)

existing_events = get_events(limit=500)
camera_options = ["All", *sorted({event["camera_id"] for event in existing_events})]
selected_camera = st.sidebar.selectbox("Camera", camera_options)

min_confidence = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

limit = st.sidebar.number_input("Max events", min_value=1, max_value=500, value=50)

auto_refresh = st.sidebar.checkbox("Auto-refresh (5 s)", value=True)

# ---------------------------------------------------------------------------
# Filtered event feed
# ---------------------------------------------------------------------------
@st.fragment(run_every="5s" if auto_refresh else None)
def render_event_feed():
    st.header("Event Feed")
    all_events = get_events(limit=500)

    filtered = all_events
    if selected_category != "All":
        filtered = [event for event in filtered if event["category"] == selected_category]
    if selected_camera != "All":
        filtered = [event for event in filtered if event["camera_id"] == selected_camera]
    if min_confidence > 0:
        filtered = [event for event in filtered if event["confidence"] >= min_confidence]
    filtered = filtered[:limit]

    if not filtered:
        st.info("No events match the current filters.")
        return

    st.write(f"Showing **{len(filtered)}** event(s)")
    table_data = [
        {
            "Time": event["timestamp"],
            "Camera": event["camera_id"],
            "Category": event["category"],
            "Confidence": f"{event['confidence']:.0%}",
            "Summary": event["summary"][:80],
        }
        for event in filtered
    ]
    st.dataframe(table_data, use_container_width=True)

    keyframe_events = [event for event in filtered if event.get("keyframe_path")]
    if keyframe_events:
        st.header("Keyframes")
        cols = st.columns(min(len(keyframe_events), 4))
        for idx, event in enumerate(keyframe_events[:12]):
            col = cols[idx % len(cols)]
            kf_path = ROOT / event["keyframe_path"]
            if kf_path.is_file():
                col.image(str(kf_path), caption=f"{event['category']} — {event['camera_id']}", use_container_width=True)
            else:
                col.warning(f"Missing: {event['keyframe_path']}")


render_event_feed()

# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------
st.header("💬 Event Chatbot")
semantic_search = st.checkbox(
    "Semantic search (local Ollama)", value=False,
    help="Ranks up to the 100 most recent events matching filters in your question; older events are not searched.",
)
question = st.text_input("Ask a question about stored events", placeholder="e.g. Show person down events from camera 1")

if question:
    try:
        message, rows = ask(question, semantic=semantic_search)
        st.success(message)
        if rows:
            chat_table = [
                {
                    "Time": r["timestamp"],
                    "Camera": r["camera_id"],
                    "Category": r["category"],
                    "Confidence": f"{r['confidence']:.0%}",
                    "Summary": r["summary"][:80],
                }
                for r in rows
            ]
            st.dataframe(chat_table, use_container_width=True)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
