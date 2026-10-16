"""Streamlit dashboard for live surveillance events.

Run::

    streamlit run ui/app.py
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

# Ensure project root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import imageio_ffmpeg

from chatbot.chat import ask
from core.db import get_events, init_db
from vision.run_pipeline import run_pipeline

init_db()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Surveillance Dashboard", layout="wide")
st.title("🔒 Surveillance Event Dashboard")
st.caption("Review a clip, detect person-down events, and ask questions about saved results.")


def save_upload(uploaded_file) -> Path:
    """Store the original clip locally, reusing it across Streamlit reruns."""
    upload_dir = ROOT / "data" / "videos" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
    path = upload_dir / f"{digest}{Path(uploaded_file.name).suffix.lower()}"
    if not path.is_file() or path.stat().st_size != uploaded_file.size:
        uploaded_file.seek(0)
        with path.open("wb") as output:
            shutil.copyfileobj(uploaded_file, output)
    return path


def browser_preview(source_path: Path) -> Path:
    """Make a widely playable H.264 Baseline preview with its audio."""
    preview_path = source_path.with_name(f"{source_path.stem}_preview.mp4")
    if preview_path.is_file() and preview_path.stat().st_size:
        return preview_path
    temporary_path = preview_path.with_name(f"{preview_path.stem}-{uuid4().hex}.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(source_path), "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "25", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(temporary_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        temporary_path.replace(preview_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return preview_path

# ---------------------------------------------------------------------------
# Local video upload and processing
# ---------------------------------------------------------------------------
st.header("Video analysis")
uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
source_path = None

if uploaded_video is not None:
    source_path = save_upload(uploaded_video)
    with st.spinner("Preparing a browser-compatible preview..."):
        try:
            preview_path = browser_preview(source_path)
        except (OSError, subprocess.SubprocessError) as exc:
            st.warning(f"Preview unavailable: {type(exc).__name__}. The original video can still be analyzed.")
        else:
            st.video(str(preview_path))

camera_id = st.text_input("Camera name for this run", value="uploaded-camera")
if st.button("Analyze video", disabled=source_path is None or not camera_id.strip()):
    frame_slot = st.empty()
    frame_count = [0]

    def show_sampled_frame(frame):
        frame_count[0] += 1
        if frame_count[0] == 1 or frame_count[0] % 5 == 0:
            frame_slot.image(
                frame.image, channels="BGR", width="stretch",
                caption=f"Analyzing video at {frame.video_time_s:.1f} seconds",
            )

    with st.spinner("Detecting and tracking people, then describing event keyframes..."):
        try:
            created = run_pipeline(
                str(source_path), camera_id=camera_id.strip(), on_frame=show_sampled_frame,
            )
        except Exception as exc:
            st.error(f"Video analysis failed: {type(exc).__name__}: {exc}")
        else:
            if created:
                st.success(f"Analysis complete: {len(created)} event(s) saved. See the feed below.")
            else:
                st.info("Analysis complete. No person-down event was detected in this video.")

st.caption("Videos are processed on this laptop at 2 sampled frames per second. The original and its playback copy stay in the local data/videos folder.")

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
    st.dataframe(table_data, width="stretch")

    keyframe_events = [event for event in filtered if event.get("keyframe_path")]
    if keyframe_events:
        st.header("Keyframes")
        cols = st.columns(min(len(keyframe_events), 4))
        for idx, event in enumerate(keyframe_events[:12]):
            col = cols[idx % len(cols)]
            kf_path = ROOT / event["keyframe_path"]
            if kf_path.is_file():
                col.image(str(kf_path), caption=f"{event['category']} — {event['camera_id']}", width="stretch")
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
            st.dataframe(chat_table, width="stretch")
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
