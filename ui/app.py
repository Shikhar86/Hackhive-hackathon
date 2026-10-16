"""Streamlit dashboard for live surveillance events.

Run::

    streamlit run ui/app.py
"""

import hashlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Ensure project root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import imageio_ffmpeg
import cv2

from chatbot.chat import ask
from core.db import (
    get_event, get_event_evidence, get_events, get_review, get_review_map,
    init_db, set_review,
)
from vision.run_pipeline import run_pipeline

init_db()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Surveillance Intelligence", page_icon="✦", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background:
            radial-gradient(circle at 71% 0%, rgba(143, 99, 136, .25), transparent 34rem),
            radial-gradient(circle at 8% 82%, rgba(111, 68, 109, .15), transparent 30rem),
            #09080c;
        color: #f8f3f8;
        font-family: "Segoe UI", Inter, sans-serif;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1450px; padding: 2.1rem 3.2rem 5rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #111015, #09080b 74%);
        border-right: 1px solid #4c3a4b;
    }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1rem 2rem; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #f7eef6; }
    h1, h2, h3, p, label { letter-spacing: -.015em; }
    h2, h3 { color: #f8f3f8 !important; }
    [data-testid="stMarkdownContainer"] a { color: #e5b9dd; }
    .ui-hero {
        position: relative; overflow: hidden; display: flex; align-items: center;
        justify-content: space-between; gap: 2rem; padding: 2.25rem 2.5rem;
        border: 1px solid rgba(213, 169, 207, .28); border-radius: 27px;
        background: linear-gradient(110deg, #302632, #211a27 55%, #100e14);
        box-shadow: 0 20px 60px rgba(0, 0, 0, .28), inset 0 1px rgba(255,255,255,.09);
        margin: .2rem 0 1.7rem;
    }
    .ui-hero::after {
        content: ""; position: absolute; width: 24rem; height: 24rem;
        right: 4%; top: -12rem; pointer-events: none;
        background: radial-gradient(circle, rgba(226, 160, 214, .23), transparent 65%);
    }
    .ui-hero-copy, .ui-hero-status { position: relative; z-index: 1; }
    .ui-eyebrow { font-size: .7rem; letter-spacing: .2em; font-weight: 700; color: #dfb7d7; }
    .ui-hero h1 { font-size: clamp(2rem, 3vw, 3.2rem); line-height: 1.09; margin: .65rem 0 .7rem; }
    .ui-hero h1 em { font-style: normal; color: #e7b7db; }
    .ui-hero p { margin: 0; color: #d9cbd6; font-size: 1.03rem; }
    .ui-hero-status {
        white-space: nowrap; align-self: flex-start; padding: .58rem .9rem;
        border: 1px solid #886c86; border-radius: 999px;
        background: rgba(11, 9, 13, .55); color: #f0deeb;
        font-size: .75rem; font-weight: 650; letter-spacing: .07em;
    }
    .ui-hero-status b { color: #70d6a1; margin-right: .4rem; }
    .ui-steps { display: flex; flex-wrap: wrap; gap: .55rem; margin: 0 0 1.15rem; }
    .ui-steps a {
        display: inline-block; text-decoration: none;
        border: 1px solid #695367; border-radius: 999px; background: #161219;
        padding: .5rem .9rem; color: #ddcbd9; font-size: .8rem;
    }
    .ui-steps a:hover, .ui-steps a:focus {
        border-color: #d4a2ca; background: #2b202d; color: #fff;
        box-shadow: 0 0 0 2px rgba(212, 162, 202, .15);
    }
    .ui-steps a:first-child { background: linear-gradient(110deg, #694964, #402d43); color: #fff; }
    .ui-anchor { scroll-margin-top: 1rem; }
    .ui-metric {
        min-height: 118px; padding: 1.25rem 1.4rem; border-radius: 22px;
        background: linear-gradient(135deg, #1c1720, #0e0c11);
        border: 1px solid #544253;
        box-shadow: inset 0 1px rgba(255,255,255,.06), 0 14px 40px rgba(0,0,0,.2);
    }
    .ui-metric.accent { background: radial-gradient(circle at 90% 5%, #5c4059, #211922 62%, #151118); }
    .ui-metric-label { color: #d5c2d2; font-size: .82rem; }
    .ui-metric-value { margin-top: .9rem; color: #fff; font-size: 2rem; font-weight: 700; line-height: 1; }
    .ui-metric-foot { margin-top: .45rem; color: #a997a7; font-size: .72rem; }
    .ui-preview-empty {
        height: 326px; display: flex; flex-direction: column; align-items: center;
        justify-content: center; text-align: center; gap: .55rem;
        border: 1px dashed #6b5268; border-radius: 20px;
        background: radial-gradient(circle at 50% 20%, #2a1e2b, #100e13 65%);
        color: #bda9ba;
    }
    .ui-preview-empty b { font-size: 2rem; font-weight: 400; color: #e3b8dc; }
    .ui-preview-empty strong { color: #f4e9f1; font-size: 1rem; }
    .ui-preview-empty small { color: #ac99a9; }
    .ui-brand { padding: .8rem .6rem 1.4rem; border-bottom: 1px solid #3c303d; margin-bottom: 1.2rem; }
    .ui-brand strong { color: #fff; letter-spacing: .07em; font-size: 1.02rem; }
    .ui-brand strong span { color: #e5a9d9; }
    .ui-brand small { display: block; margin-top: .45rem; color: #a994a9; }
    .ui-side-label { color: #bb9eb9; font-size: .68rem; font-weight: 700; letter-spacing: .18em; margin: .5rem .2rem .8rem; }
    .ui-track-box {
        padding: .8rem; border-radius: 18px; border: 1px solid #62495f;
        background: radial-gradient(circle at 50% 50%, #2c202d, #120e16);
    }
    .ui-track-box svg { display: block; width: 100%; max-height: 250px; }
    .ui-track-foot { color: #bba8b8; font-size: .75rem; margin-top: .5rem; }
    [data-testid="stFileUploaderDropzone"], [data-testid="stVideo"],
    [data-testid="stDataFrame"], [data-testid="stVerticalBlockBorderWrapper"] {
        background: #131016; border: 1px solid #5a4558 !important; border-radius: 20px !important;
        overflow: hidden;
    }
    [data-testid="stFileUploaderDropzone"] { border-style: dashed !important; padding: 1rem; }
    [data-testid="stTextInputRootElement"], [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input {
        background: #17131b !important; border: 1px solid #674f64 !important;
        border-radius: 13px !important; color: #fff !important;
    }
    [data-testid="stTextInputField"] { color: #fff !important; }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
        border: 1px solid #b188ab; border-radius: 999px; color: #fff;
        background: linear-gradient(115deg, #674b64, #a36f9b); font-weight: 650;
        box-shadow: 0 7px 24px rgba(156, 103, 148, .22);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {
        border-color: #e4b7da; color: #fff; transform: translateY(-1px);
        box-shadow: 0 9px 30px rgba(193, 128, 182, .37);
    }
    [data-testid="stButton"] button:disabled { opacity: .45; transform: none; box-shadow: none; }
    [data-testid="stAlert"] { border-radius: 16px; }
    [data-testid="stMetric"] { padding: 1rem; border-radius: 16px; background: #17121a; border: 1px solid #5a4558; }
    [data-testid="stProgressBar"] > div { background: #211822; }
    [data-testid="stProgressBar"] > div > div { background: #d4a2ca; }
    [data-testid="stSlider"] [data-rac][style*="position: absolute"] { background: #d4a2ca !important; }
    [data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] p { color: #d4a2ca !important; }
    [data-testid="stCheckbox"] input { accent-color: #d4a2ca; }
    @media (max-width: 850px) {
        [data-testid="stMainBlockContainer"] { padding: 1.4rem 1rem 3rem; }
        .ui-hero { flex-direction: column; align-items: flex-start; padding: 1.55rem; }
        .ui-hero h1 { font-size: 2.1rem; }
        .ui-hero-status { align-self: flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="ui-hero">
      <div class="ui-hero-copy">
        <div class="ui-eyebrow">HONEYWELL HACKATHON · PROBLEM 3</div>
        <h1>Surveillance <em>Intelligence</em></h1>
        <p>Analyze footage. Review possible incidents. Ask questions about saved events.</p>
      </div>
      <div class="ui-hero-status"><b>●</b> ON-DEVICE PROTOTYPE</div>
    </div>
    <nav class="ui-steps" aria-label="Dashboard sections">
      <a href="#video-analysis">01 &nbsp; Analyze video</a>
      <a href="#review-events">02 &nbsp; Review events</a>
      <a href="#ask-assistant">03 &nbsp; Ask the assistant</a>
    </nav>
    """,
    unsafe_allow_html=True,
)
overview_events = get_events(limit=500)
overview_cameras = len({event["camera_id"] for event in overview_events})
overview_count = f"{len(overview_events)}+" if len(overview_events) == 500 else str(len(overview_events))
metric_columns = st.columns(3)
for column, css_class, label, value, foot in zip(
    metric_columns,
    ("accent", "", ""),
    ("Recent events", "Cameras with events", "Analysis sampling"),
    (overview_count, str(overview_cameras), "2 FPS"),
    ("Latest 500 stored events", "From saved analyses", "Frames checked per second of video"),
):
    with column:
        st.markdown(
            f'<div class="ui-metric {css_class}"><div class="ui-metric-label">{label}</div>'
            f'<div class="ui-metric-value">{value}</div><div class="ui-metric-foot">{foot}</div></div>',
            unsafe_allow_html=True,
        )


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


VIDEO_ROTATIONS = {
    "As shown": None,
    "Turn left 90°": "transpose=2",
    "Turn right 90°": "transpose=1",
    "Turn upside down": "hflip,vflip",
}


def browser_preview(source_path: Path, rotation: str = "As shown") -> Path:
    """Make a browser-compatible copy in the operator-selected orientation."""
    if rotation not in VIDEO_ROTATIONS:
        raise ValueError(f"Unknown video orientation: {rotation}")
    suffix = "preview" if rotation == "As shown" else f"{list(VIDEO_ROTATIONS).index(rotation)}_preview"
    preview_path = source_path.with_name(f"{source_path.stem}_{suffix}.mp4")
    if preview_path.is_file() and preview_path.stat().st_size:
        return preview_path
    temporary_path = preview_path.with_name(f"{preview_path.stem}-{uuid4().hex}.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(source_path), "-map", "0:v:0", "-map", "0:a?",
    ]
    if VIDEO_ROTATIONS[rotation]:
        command.extend(["-vf", VIDEO_ROTATIONS[rotation]])
    command.extend([
        "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "25", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(temporary_path),
    ])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        temporary_path.replace(preview_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return preview_path


def suggest_camera_name() -> str:
    """Choose a short camera name that has not been used in recent events."""
    existing = {event["camera_id"] for event in get_events(limit=500)}
    number = 1
    while f"demo-{number}" in existing:
        number += 1
    return f"demo-{number}"


def video_duration(source_path: Path) -> float | None:
    """Read video length for the progress bar without decoding the clip."""
    capture = cv2.VideoCapture(str(source_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame_count / fps if fps > 0 and frame_count > 0 else None
    finally:
        capture.release()


def display_time(timestamp: str) -> str:
    """Show stored UTC timestamps in the laptop's local time zone."""
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime("%d %b %Y · %I:%M %p")
    except ValueError:
        return timestamp


def display_video_time(seconds: float | None) -> str:
    """Format an event's exact position inside its source video."""
    if seconds is None:
        return "Unavailable"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def video_frame_at(source_path: Path, seconds: float):
    """Read one nearby frame for an incident storyboard."""
    capture = cv2.VideoCapture(str(source_path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000)
        ok, image = capture.read()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        capture.release()


def track_diagram(points: list[dict]) -> str:
    """Draw a normalized track path without altering the saved keyframe."""
    coordinates = []
    for point in points:
        try:
            x = max(0.0, min(100.0, float(point["x_pct"])))
            y = max(0.0, min(100.0, float(point["y_pct"])))
        except (KeyError, TypeError, ValueError):
            continue
        coordinates.append((x, y))
    if not coordinates:
        return ""
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    start_x, start_y = coordinates[0]
    end_x, end_y = coordinates[-1]
    return (
        '<div class="ui-track-box"><svg viewBox="0 0 100 100" role="img" '
        'aria-label="Tracked person path across the video frame">'
        '<rect x="1" y="1" width="98" height="98" rx="7" fill="none" stroke="#70576d"/>'
        '<path d="M50 2 V98 M2 50 H98" stroke="#3b2d3b" stroke-dasharray="2 3"/>'
        f'<polyline points="{path}" fill="none" stroke="#e5acd8" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="2.3" fill="#83dcb2"/>'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="2.7" fill="#f4c4ea"/>'
        '</svg><div class="ui-track-foot">Green: first observation · Pink: flagged position. '
        'Positions are sampled box centers, not a continuous route.</div></div>'
    )

# ---------------------------------------------------------------------------
# Local video upload and processing
# ---------------------------------------------------------------------------
st.markdown('<div id="video-analysis" class="ui-anchor"></div>', unsafe_allow_html=True)
st.header("Video analysis")
video_controls, video_preview = st.columns([1, 1.18], gap="large")
with video_controls:
    st.caption("Upload a clip from this computer or phone and give the analysis a unique camera name.")
    uploaded_video = st.file_uploader(
        "Choose a video from this device",
        type=["mp4", "mov", "m4v", "avi", "webm", "3gp"],
        help="Phone videos up to 1 GB are accepted. Keep this page open while the file transfers to the laptop.",
    )
    source_path = save_upload(uploaded_video) if uploaded_video is not None else None
    clip_duration = video_duration(source_path) if source_path else None
    rotation = st.selectbox(
        "Video orientation",
        list(VIDEO_ROTATIONS),
        help="If people look sideways or upside down in the preview, turn the video until they are upright before analysis.",
    )
    detection_mode = st.selectbox(
        "Person-down detection",
        ["Standard", "Motion-sensitive (experimental)"],
        help="Motion-sensitive mode can catch shorter low-posture moments, but may create more false alerts. Review each result.",
    )

    if "camera_name" not in st.session_state:
        st.session_state["camera_name"] = suggest_camera_name()
    if st.button("Use a new camera name", help="Keep each analysis separate from earlier results"):
        st.session_state["camera_name"] = suggest_camera_name()
    camera_id = st.text_input("Camera name for this run", key="camera_name")
    st.caption("A new name keeps this run separate from earlier results.")
    analyze_clicked = st.button(
        "Analyze video", type="primary", use_container_width=True,
        disabled=source_path is None or not camera_id.strip(),
    )

with video_preview:
    st.caption("VIDEO PREVIEW")
    if source_path:
        with st.spinner("Preparing a browser-compatible preview..."):
            try:
                preview_path = browser_preview(source_path, rotation)
            except (OSError, subprocess.SubprocessError) as exc:
                st.warning(f"Preview unavailable: {type(exc).__name__}. Re-select the file or try a shorter clip before analysis.")
            else:
                st.video(str(preview_path))
    else:
        st.markdown(
            '<div class="ui-preview-empty"><b>◎</b><strong>Your footage appears here</strong>'
            '<small>Choose an MP4, AVI, or MOV file to preview it.</small></div>',
            unsafe_allow_html=True,
        )

if analyze_clicked:
    frame_slot = st.empty()
    progress_slot = st.progress(0, text="Preparing video analysis...")
    frame_count = [0]

    def show_sampled_frame(frame):
        frame_count[0] += 1
        if frame_count[0] == 1 or frame_count[0] % 5 == 0:
            frame_slot.image(
                frame.image, channels="BGR", width="stretch",
                caption=f"Analyzing video at {frame.video_time_s:.1f} seconds",
            )
            if clip_duration:
                percentage = min(95, int(frame.video_time_s / clip_duration * 100))
                progress_slot.progress(
                    percentage,
                    text=f"Analyzing {frame.video_time_s:.1f} of {clip_duration:.1f} seconds",
                )

    with st.spinner("Detecting and tracking people, then describing event keyframes..."):
        try:
            # Use the same operator-checked orientation for playback and AI.
            analysis_source = browser_preview(source_path, rotation)
            created = run_pipeline(
                str(analysis_source), camera_id=camera_id.strip(), on_frame=show_sampled_frame,
                person_down_aspect=1.05 if detection_mode.startswith("Motion-sensitive") else 1.2,
                person_down_frames=2 if detection_mode.startswith("Motion-sensitive") else 3,
                person_down_motion_ratio=0.75 if detection_mode.startswith("Motion-sensitive") else None,
            )
        except Exception as exc:
            progress_slot.empty()
            st.error(f"Video analysis failed: {type(exc).__name__}: {exc}")
        else:
            progress_slot.progress(100, text="Analysis complete")
            st.session_state["latest_run"] = {
                "camera_id": camera_id.strip(),
                "video_name": uploaded_video.name,
                "events": [event.to_dict() for event in created],
            }
            if created:
                st.session_state["selected_event_id"] = created[0].id
                st.session_state["camera_filter"] = camera_id.strip()
                st.session_state["chat_question"] = (
                    f"Show person down events from camera {camera_id.strip()}"
                )
                st.success(f"Analysis complete: {len(created)} event(s) saved. See the feed below.")
            else:
                st.info("Analysis complete. No person-down event was detected in this video.")

st.caption("Check that people are upright in the preview before analysis. Videos are processed on this laptop at 2 sampled frames per second. The original and its playback copy stay in the local data/videos folder.")

latest_run = st.session_state.get("latest_run")
if latest_run:
    st.subheader("Results from the latest analysis")
    st.caption(f"{latest_run['video_name']} · camera {latest_run['camera_id']}")
    st.metric("New events", len(latest_run["events"]))
    for event in latest_run["events"]:
        with st.container(border=True):
            picture, details = st.columns([1, 2])
            with picture:
                keyframe = ROOT / event["keyframe_path"] if event["keyframe_path"] else None
                if keyframe and keyframe.is_file():
                    st.image(str(keyframe), width="stretch")
            with details:
                st.write(f"**{event['category'].replace('_', ' ').title()}** — {event['summary']}")
                st.caption(
                    f"Score {event['confidence']:.0%} · Saved {display_time(event['timestamp'])}"
                )
    if latest_run["events"] and st.button("Ask the chatbot about this run"):
        st.session_state["chat_question"] = (
            f"Show person down events from camera {latest_run['camera_id']}"
        )

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="ui-brand"><strong>✦ <span>HONEYWELL</span> VISION</strong>'
    '<small>Incident review workspace</small></div>'
    '<div class="ui-side-label">EVENT CONTROLS</div>',
    unsafe_allow_html=True,
)
st.sidebar.header("Filters")
st.sidebar.caption("The current analysis detects possible person-down events.")

category_options = ["All", "person_down", "fighting", "tailgating", "crowd", "suspicious"]
selected_category = st.sidebar.selectbox("Category", category_options)

existing_events = get_events(limit=500)
camera_options = ["All", *sorted({event["camera_id"] for event in existing_events})]
if st.session_state.get("camera_filter") not in camera_options:
    st.session_state["camera_filter"] = "All"
selected_camera = st.sidebar.selectbox("Camera", camera_options, key="camera_filter")

selected_review = st.sidebar.selectbox(
    "Review status", ["All", "Unreviewed", "Confirmed", "False alarm"],
)

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
    review_map = get_review_map()

    filtered = all_events
    if selected_category != "All":
        filtered = [event for event in filtered if event["category"] == selected_category]
    if selected_camera != "All":
        filtered = [event for event in filtered if event["camera_id"] == selected_camera]
    if min_confidence > 0:
        filtered = [event for event in filtered if event["confidence"] >= min_confidence]
    if selected_review != "All":
        filtered = [
            event for event in filtered
            if review_map.get(event["id"], "Unreviewed") == selected_review
        ]
    filtered = filtered[:limit]

    if not filtered:
        st.info("No events match the current filters.")
        return

    st.write(f"Showing **{len(filtered)}** event(s). Select a timeline card to inspect an incident.")
    table_data = [
        {
            "Time (local)": display_time(event["timestamp"]),
            "Camera": event["camera_id"],
            "Category": event["category"].replace("_", " ").title(),
            "Confidence": f"{event['confidence']:.0%}",
            "Review": review_map.get(event["id"], "Unreviewed"),
            "Summary": event["summary"][:80],
        }
        for event in filtered
    ]
    st.dataframe(table_data, width="stretch", hide_index=True)

    if st.session_state.get("selected_event_id") not in {event["id"] for event in filtered}:
        st.session_state["selected_event_id"] = filtered[0]["id"]
    st.subheader("Incident timeline")
    st.caption("Choose one of the newest alerts, or open an older event from the list.")
    timeline_columns = st.columns(3)
    for index, event in enumerate(filtered[:6]):
        status = review_map.get(event["id"], "Unreviewed")
        label = f"{display_time(event['timestamp']).split('·')[-1].strip()} · {event['camera_id']} · {status}"
        with timeline_columns[index % 3]:
            if st.button(label, key=f"timeline_{event['id']}", use_container_width=True):
                st.session_state["selected_event_id"] = event["id"]
                st.rerun()
    if len(filtered) > 6:
        older = filtered[6:]
        older_lookup = {event["id"]: event for event in older}
        older_id = st.selectbox(
            "Older incidents", options=[None, *older_lookup],
            format_func=lambda event_id: "Choose an older incident" if event_id is None else (
                f"{display_time(older_lookup[event_id]['timestamp'])} · "
                f"{older_lookup[event_id]['camera_id']}"
            ),
        )
        if older_id and st.button("Open older incident"):
            st.session_state["selected_event_id"] = older_id
            st.rerun()

    keyframe_events = [event for event in filtered if event.get("keyframe_path")]
    if keyframe_events:
        with st.expander("Browse all keyframes"):
            st.caption("Hover over a keyframe and select the expand icon to see it larger.")
            cols = st.columns(min(len(keyframe_events), 4))
            for idx, event in enumerate(keyframe_events[:12]):
                col = cols[idx % len(cols)]
                kf_path = ROOT / event["keyframe_path"]
                if kf_path.is_file():
                    col.image(str(kf_path), caption=f"{event['category']} — {event['camera_id']}", width="stretch")
                else:
                    col.warning(f"Missing: {event['keyframe_path']}")


st.markdown('<div id="review-events" class="ui-anchor"></div>', unsafe_allow_html=True)
render_event_feed()

selected_event_id = st.session_state.get("selected_event_id")
selected_event = get_event(selected_event_id) if selected_event_id else None
if selected_event:
    evidence = get_event_evidence(selected_event_id)
    review = get_review(selected_event_id)
    st.header("Incident review")
    video_position = display_video_time(evidence["video_time_s"] if evidence else None)
    st.info(
        f"**Video timestamp: {video_position}**  \n"
        f"Saved at {display_time(selected_event['timestamp'])} · "
        f"Camera {selected_event['camera_id']} · {review['status']}"
    )
    keyframe_column, description_column = st.columns([1, 1.15], gap="large")
    with keyframe_column:
        keyframe_path = (
            ROOT / selected_event["keyframe_path"]
            if selected_event.get("keyframe_path") else None
        )
        if keyframe_path and keyframe_path.is_file():
            st.image(str(keyframe_path), caption="Detected moment · saved keyframe", width="stretch")
        else:
            st.info("No saved keyframe is available for this event.")
    with description_column:
        st.subheader(selected_event["category"].replace("_", " ").title())
        st.write(selected_event["summary"])
        st.caption(f"Detector score: {selected_event['confidence']:.0%} · This is not a calibrated probability.")
        st.markdown("**Why was this flagged?**")
        if evidence and evidence["rule_reason"]:
            st.info(evidence["rule_reason"])
        else:
            st.info("Detailed rule evidence was not saved for this older event. Reanalyze its clip to add it.")
        st.caption("A flagged posture is a possible incident. Review the footage before confirming a fall.")

    if evidence:
        source_file = Path(evidence["source_path"]) if evidence["source_path"] else None
        moment = evidence["video_time_s"]
        if source_file and source_file.is_file() and moment is not None:
            st.subheader("Incident storyboard")
            storyboard_columns = st.columns(3)
            for column, label, seconds in zip(
                storyboard_columns,
                ("Before", "Flagged moment", "After"),
                (max(0.0, moment - 2), moment, moment + 2),
            ):
                with column:
                    image = video_frame_at(source_file, seconds)
                    if image is not None:
                        st.image(image, caption=f"{label} · {seconds:.1f}s", width="stretch")
                    else:
                        st.caption(f"{label}: frame unavailable")
            if st.button("Play video near this incident", key=f"replay_{selected_event_id}"):
                st.session_state["replay_event_id"] = selected_event_id
            if st.session_state.get("replay_event_id") == selected_event_id:
                with st.spinner("Preparing local playback..."):
                    try:
                        st.video(str(browser_preview(source_file)), start_time=max(0, int(moment - 3)))
                    except (OSError, subprocess.SubprocessError) as exc:
                        st.warning(f"Playback unavailable: {type(exc).__name__}.")
        elif evidence["source_path"]:
            st.caption("The original video is not on this laptop, so replay and nearby frames are unavailable.")

        st.subheader("Track view")
        if evidence["track_id"] is not None:
            st.caption(
                f"ByteTrack person ID {evidence['track_id']} · sampled frame {evidence['frame_index']} · "
                f"video time {evidence['video_time_s']:.1f}s"
            )
            diagram = track_diagram(evidence["track_points"])
            if diagram:
                st.markdown(diagram, unsafe_allow_html=True)
            else:
                st.caption("Track coordinates were unavailable for this incident.")
        else:
            st.caption("This event was not linked to one person track.")
    else:
        st.caption("This older event has no saved video time or track evidence. Its keyframe and summary remain available.")

    st.subheader("Human review")
    with st.form(key=f"review_form_{selected_event_id}"):
        statuses = ["Unreviewed", "Confirmed", "False alarm"]
        status = st.selectbox("Decision", statuses, index=statuses.index(review["status"]))
        note = st.text_area(
            "Operator note", value=review["note"],
            placeholder="Describe what you saw in the video and why you chose this decision.",
        )
        if st.form_submit_button("Save review"):
            set_review(selected_event_id, status, note)
            st.success("Review saved on this laptop.")
            st.rerun()

# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------
st.markdown('<div id="ask-assistant" class="ui-anchor"></div>', unsafe_allow_html=True)
st.header("💬 Event Chatbot")
semantic_search = st.checkbox(
    "Semantic search (local Ollama)", value=False,
    help="Ranks up to the 100 most recent events matching filters in your question; older events are not searched.",
)
question = st.text_input(
    "Ask a question about stored events",
    placeholder="e.g. Show person down events from camera demo-1",
    key="chat_question",
)

if question:
    try:
        message, rows = ask(question, semantic=semantic_search)
        st.success(message)
        if rows:
            chat_table = [
                {
                    "Time (local)": display_time(r["timestamp"]),
                    "Camera": r["camera_id"],
                    "Category": r["category"].replace("_", " ").title(),
                    "Confidence": f"{r['confidence']:.0%}",
                    "Summary": r["summary"][:80],
                }
                for r in rows
            ]
            st.dataframe(chat_table, width="stretch")
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
