"""FaceVision AI: local-first face recognition and attendance dashboard."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.attendance import dashboard_metrics
from src.database import Database, DuplicateUserError, InvalidUserError
from src.face_detection import FaceEngine
from src.face_recognition import FaceMatcher
from src.utils import (
    COOLDOWN_SECONDS,
    DATABASE_PATH,
    MATCH_THRESHOLD,
    decode_image,
    ensure_models,
)

logger = logging.getLogger("facevision")

MIN_SAMPLES = 3
MAX_SAMPLES = 5

st.set_page_config(
    page_title="FaceVision AI | Attendance",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg:#0b1117; --bg2:#101b22; --panel:#121f26; --panel-soft:#162a33; --line:#213a42;
        --text:#edf6f3; --muted:#9ab5ad; --green:#4dd39b; --green-deep:#26b974; --mint:#d8f5e6;
        --accent:#72d4ff; --danger:#ff7a7a; --mono:'Cascadia Code',Consolas,monospace;
    }
    html, body, [class*="css"] {
        font-family:'Segoe UI',sans-serif; color:var(--text); background:var(--bg);
    }
    .stApp {
        background:linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    }
    .stApp > header { background:transparent; }
    div.block-container {
        padding-top: 2rem; padding-left: 2rem; padding-right: 2rem;
        max-width: 1400px;
    }
    [data-testid="stSidebar"] {
        background:linear-gradient(180deg, #0f181d 0%, #0d171d 100%);
        border-right:1px solid var(--line);
    }
    [data-testid="stSidebar"] h1 { font-family:'Georgia',serif; font-size:1.55rem; letter-spacing:0; color:var(--text); }
    .brand-mark {
        display:inline-grid; place-items:center; width:34px; height:34px;
        background:var(--green-deep); color:#08130d; border-radius:9px; font-size:18px;
        margin-right:8px;
    }
    .eyebrow {
        font:500 11px var(--mono); text-transform:uppercase; letter-spacing:0.08em;
        color:var(--green); margin-bottom:6px;
    }
    .page-title {
        font:700 36px/1.13 'Georgia',serif; letter-spacing:0; margin:5px 0 8px; color:var(--text);
    }
    .page-subtitle { font-size:15px; color:var(--muted); margin:0 0 25px; }
    .metric {
        border:1px solid var(--line); background:linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
        border-radius:12px; padding:19px 20px 16px; min-height:105px; box-shadow:0 2px 8px rgba(0,0,0,.18);
    }
    .metric-label {
        font:500 11px var(--mono); text-transform:uppercase; color:var(--muted);
    }
    .metric-value { font:700 30px 'Georgia',serif; margin-top:6px; color:var(--text); }
    .section-label {
        font:600 12px var(--mono); text-transform:uppercase; color:var(--muted);
        border-bottom:1px solid var(--line); padding-bottom:10px; margin:23px 0 14px;
    }
    .privacy-note {
        background:rgba(77, 211, 155, 0.12); border-left:3px solid var(--green);
        padding:11px 14px; border-radius:8px; color:#dfeee9; font-size:13px;
    }
    .status-chip {
        display:inline-block; border-radius:999px; padding:5px 12px; background:rgba(77, 211, 155, 0.15);
        color:var(--mint); font:500 11px var(--mono); border:1px solid rgba(77,211,155,0.28);
    }
    .stButton > button, .stDownloadButton > button {
        border-radius:8px; font-weight:600; min-height:42px; border:1px solid var(--line);
        color:var(--text); background:var(--panel-soft);
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(180deg, var(--green) 0%, var(--green-deep) 100%);
        border-color:var(--green-deep); color:#08130d; box-shadow:none;
    }
    [data-testid="stMetric"] {
        background:var(--panel-soft); padding:14px; border:1px solid var(--line);
        border-radius:10px;
    }
    div[data-testid="stDataFrame"] {
        border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#0e171c;
    }
    .stDataFrame, .stTable, .stDataFrame > div { background:transparent; }
    .stDataFrame .dataframe { color:var(--text); }
    h2, h3 { font-family:'Georgia',serif !important; letter-spacing:0 !important; color:var(--text) !important; }
    p, li, label, .stCaptionContainer, .stMarkdown { color:var(--text); }
    .stWarning, .stInfo, .stError, .stSuccess {
        border-radius:10px; border:1px solid var(--line);
    }
    .stWarning { background:rgba(255,173,85,0.08); }
    .stInfo { background:rgba(114,212,255,0.08); }
    .stError { background:rgba(255,122,122,0.08); }
    .stSuccess { background:rgba(77,211,155,0.08); }
    .stTabs [role="tablist"] { background:#111b20; border-bottom:1px solid var(--line); }
    .stTabs [role="tab"] { color:var(--muted); }
    .stTabs [role="tab"][aria-selected="true"] { color:var(--text); }
    footer { visibility:hidden; }
    @media (max-width:700px) { .page-title { font-size:29px; } .metric { min-height:90px; } div.block-container { padding-left: 1rem; padding-right: 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Shared resources
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_database() -> Database:
    return Database(DATABASE_PATH)


@st.cache_resource
def get_engine() -> FaceEngine:
    detector_path, recognizer_path = ensure_models()
    return FaceEngine(detector_path, recognizer_path)


def get_matcher() -> FaceMatcher:
    return FaceMatcher(MATCH_THRESHOLD)


def title(eyebrow: str, heading: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{heading}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def unit_vector(vector) -> np.ndarray:
    """Return the L2-normalised float32 version of an embedding."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def check_sample_quality(image: np.ndarray, box) -> list[str]:
    """Return a list of problems with an enrollment sample (empty list = good)."""
    height, width = image.shape[:2]
    x, y, box_w, box_h = [int(v) for v in box]
    problems: list[str] = []

    if x < 0 or y < 0 or x + box_w > width or y + box_h > height:
        problems.append("Face is cut off. Keep your whole face inside the frame.")
    if min(box_w, box_h) < 110:
        problems.append("Face is too small. Move closer to the camera.")

    crop = image[max(y, 0): max(y + box_h, 0), max(x, 0): max(x + box_w, 0)]
    if crop.size == 0:
        return problems + ["Could not read the face region. Retake the photo."]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 160))  # fixed size so the blur score is comparable
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
        problems.append("Image is blurry. Hold still and retake.")
    brightness = float(gray.mean())
    if brightness < 60:
        problems.append("Too dark. Face a light source.")
    elif brightness > 200:
        problems.append("Too bright or overexposed. Reduce direct light.")
    return problems


# --------------------------------------------------------------------------- #
# Recognition helpers
# --------------------------------------------------------------------------- #
def match_faces(image: np.ndarray, engine: FaceEngine, matcher: FaceMatcher, users: list) -> list[dict]:
    """Detect faces and match each against the enrolled users."""
    matches: list[dict] = []
    for detection in engine.analyze(image):
        user, score = matcher.identify(detection["embedding"], users)
        matches.append(
            {
                "box": tuple(int(v) for v in detection["box"]),
                "user": user,
                "score": float(score),
            }
        )
    return matches


def draw_results(image: np.ndarray, matches: list[dict]) -> np.ndarray:
    """Draw boxes and labels (BGR colours) on a copy of the image."""
    annotated = image.copy()
    for match in matches:
        x, y, width, height = match["box"]
        if match["user"]:
            label = f'{match["user"]["display_name"]} · {match["score"]:.2f}'
            color = (55, 150, 83)
        else:
            label = f'Unknown · {match["score"]:.2f}'
            color = (65, 75, 205)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            annotated,
            label,
            (x, max(22, y - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def recognize_image(image: np.ndarray, record: bool = True) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Recognize faces in a single image (used for uploads)."""
    database = get_database()
    matches = match_faces(image, get_engine(), get_matcher(), database.get_users())
    results: list[dict[str, object]] = []
    for match in matches:
        user = match["user"]
        if user:
            if record:
                try:
                    database.record_attendance(str(user["user_id"]), COOLDOWN_SECONDS)
                except InvalidUserError:
                    pass
            results.append(
                {
                    "status": "Recognized",
                    "user_id": user["user_id"],
                    "name": user["display_name"],
                    "score": round(match["score"], 3),
                }
            )
        else:
            results.append({"status": "Unknown", "user_id": "", "name": "Unknown", "score": round(match["score"], 3)})
    return draw_results(image, matches), results


class LiveVideoProcessor(VideoProcessorBase):
    """Runs recognition on the live camera stream in a worker thread."""

    PROCESS_EVERY = 3  # run detection on every Nth frame, reuse boxes in between
    USERS_REFRESH_SECONDS = 5.0

    def __init__(self) -> None:
        self.engine = get_engine()
        self.matcher = get_matcher()
        self.database: Database | None = None  # opened lazily inside the worker thread
        self.users: list = []
        self.users_loaded_at = 0.0
        self.frame_count = 0
        self.last_matches: list[dict] = []
        self.last_recorded: dict[str, float] = {}
        self.last_error: str | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        try:
            if self.database is None:
                self.database = Database(DATABASE_PATH)
            now = time.monotonic()
            if now - self.users_loaded_at > self.USERS_REFRESH_SECONDS:
                self.users = self.database.get_users()
                self.users_loaded_at = now

            self.frame_count += 1
            if self.frame_count % self.PROCESS_EVERY == 1:
                self.last_matches = match_faces(image, self.engine, self.matcher, self.users)
                for match in self.last_matches:
                    user = match["user"]
                    if not user:
                        continue
                    user_id = str(user["user_id"])
                    if now - self.last_recorded.get(user_id, 0.0) < COOLDOWN_SECONDS:
                        continue
                    try:
                        self.database.record_attendance(user_id, COOLDOWN_SECONDS)
                        self.last_recorded[user_id] = now
                    except InvalidUserError:
                        pass
            image = draw_results(image, self.last_matches)
            self.last_error = None
        except Exception as exc:  # keep the video running, but do not hide the problem
            self.last_error = str(exc)
            logger.exception("Live recognition failed")
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def dashboard_page() -> None:
    metrics = dashboard_metrics(get_database())
    title("FACEVISION AI / OVERVIEW", "Attendance, at a glance.", "A private, local-first recognition workspace.")
    columns = st.columns(3)
    metric_items = [
        ("Registered users", metrics["registered_users"]),
        ("Today's attendance", metrics["today_attendance"]),
        ("All-time records", metrics["total_records"]),
    ]
    for column, (label, value) in zip(columns, metric_items):
        column.markdown(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Start a session")
        st.caption("Register a face, recognize a live camera feed, or review the attendance ledger.")
        action_cols = st.columns(2)
        if action_cols[0].button("◉  Start Camera", type="primary", use_container_width=True):
            st.session_state.navigation_request = "Live recognition"
            st.rerun()
        if action_cols[1].button("＋  Register User", use_container_width=True):
            st.session_state.navigation_request = "Register user"
            st.rerun()
        if st.button("▤  View Attendance", use_container_width=True):
            st.session_state.navigation_request = "Attendance"
            st.rerun()
    with right:
        st.markdown("#### Recent activity")
        recent = get_database().get_attendance(limit=5)
        if recent:
            st.dataframe(pd.DataFrame(recent), hide_index=True, use_container_width=True)
        else:
            st.info("No attendance recorded yet. Recognized users will appear here.")
    st.markdown(
        '<div class="privacy-note">Privacy by default: face embeddings and attendance stay in your configured SQLite database. '
        'Camera frames are processed for recognition and are not saved.</div>',
        unsafe_allow_html=True,
    )


def registration_page() -> None:
    title(
        "PEOPLE / ENROLLMENT",
        "Register a user.",
        f"Capture {MIN_SAMPLES} clear samples to build a stable face representation.",
    )
    database = get_database()
    st.session_state.setdefault("registration_samples", [])
    st.session_state.setdefault("registration_thumbs", [])
    st.session_state.setdefault("capture_sequence", 0)

    user_id = st.text_input("Unique user ID", placeholder="e.g. student-1042", max_chars=64)
    display_name = st.text_input("Display name", placeholder="e.g. Jordan Lee", max_chars=100)

    st.markdown('<div class="section-label">Face samples</div>', unsafe_allow_html=True)
    st.caption(
        "Use even lighting, face the camera, and make sure only one person is visible. "
        "Turn your head very slightly between samples."
    )

    sequence = st.session_state.capture_sequence
    capture = st.camera_input("Capture sample", key=f"registration_capture_{sequence}")

    pending_embedding = None
    pending_thumb = None
    if capture:
        try:
            image = decode_image(capture.getvalue())
            faces = get_engine().analyze(image)
            if len(faces) == 0:
                st.error("No face detected. Adjust the lighting and try again.")
            elif len(faces) > 1:
                st.error("Multiple faces detected. Capture one person at a time.")
            else:
                face = faces[0]
                x, y, box_w, box_h = [int(v) for v in face["box"]]
                problems = check_sample_quality(image, face["box"])

                shown = image.copy()
                cv2.rectangle(shown, (x, y), (x + box_w, y + box_h), (0, 0, 255) if problems else (0, 200, 0), 2)
                st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), caption="Detected face", width=320)

                embedding = unit_vector(face["embedding"])
                existing = st.session_state.registration_samples
                if problems:
                    for problem in problems:
                        st.warning(problem)
                elif existing and float(np.dot(embedding, unit_vector(existing[0]))) < MATCH_THRESHOLD:
                    st.error(
                        "This face does not match your earlier samples. "
                        "Clear the samples if you are enrolling a different person."
                    )
                elif len(existing) >= MAX_SAMPLES:
                    st.info(f"Maximum of {MAX_SAMPLES} samples reached. Save the registration.")
                else:
                    pending_embedding = embedding
                    crop = image[max(y, 0): y + box_h, max(x, 0): x + box_w]
                    pending_thumb = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2RGB)
                    st.success("Good sample. Add it to the enrollment set.")
        except Exception as exc:
            st.error(f"Could not process this capture: {exc}")

        if st.button("Add sample", type="primary", disabled=pending_embedding is None):
            st.session_state.registration_samples.append(pending_embedding.tolist())
            st.session_state.registration_thumbs.append(pending_thumb)
            st.session_state.capture_sequence = sequence + 1
            st.rerun()

    samples = st.session_state.registration_samples
    st.progress(min(len(samples) / MIN_SAMPLES, 1.0), text=f"{len(samples)} of {MIN_SAMPLES} samples captured")
    if st.session_state.registration_thumbs:
        st.image(st.session_state.registration_thumbs, width=72)
        st.caption("Samples are held in this browser session only until you save.")
        if st.button("Clear samples"):
            st.session_state.registration_samples = []
            st.session_state.registration_thumbs = []
            st.session_state.capture_sequence = sequence + 1
            st.rerun()

    ready = len(samples) >= MIN_SAMPLES and bool(user_id.strip()) and bool(display_name.strip())
    if not ready and len(samples) >= MIN_SAMPLES:
        st.caption("Enter both a user ID and a display name to enable saving.")
    if st.button("Save registration", type="primary", disabled=not ready):
        try:
            averaged = np.asarray(samples, dtype=np.float32).mean(axis=0)
            norm = np.linalg.norm(averaged)
            if norm == 0:
                raise ValueError("The combined embedding is invalid. Capture the samples again.")
            averaged = averaged / norm

            known_user, _ = get_matcher().identify(averaged, database.get_users())
            if known_user:
                raise ValueError(
                    f'This face is already registered as {known_user["display_name"]} ({known_user["user_id"]}).'
                )

            database.register_user(user_id.strip(), display_name.strip(), averaged.tolist())
            st.session_state.registration_samples = []
            st.session_state.registration_thumbs = []
            st.session_state.capture_sequence = sequence + 1
            st.success(f"Registered {display_name.strip()} ({user_id.strip()}).")
        except (DuplicateUserError, InvalidUserError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Registration could not be saved: {exc}")


def live_page() -> None:
    title("RECOGNITION / LIVE", "Know who is here.", "Recognized faces are marked in green; unmatched faces remain unknown.")
    mode = st.radio("Recognition input", ["Browser camera", "Upload image or video"], horizontal=True)
    if mode == "Browser camera":
        if not get_database().get_users():
            st.warning("Register at least one user before starting recognition.")
        try:
            get_engine()
            context = webrtc_streamer(
                key="facevision-live-camera",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=LiveVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                async_processing=True,
            )
            st.caption("Frames are processed in memory. Attendance is recorded once per user per cooldown window.")
            if context.state.playing:
                st.markdown('<span class="status-chip">CAMERA ACTIVE</span>', unsafe_allow_html=True)
                processor = context.video_processor
                if processor is not None and processor.last_error:
                    st.error(f"Recognition error: {processor.last_error}")
        except Exception as exc:
            st.error(f"Camera or model initialization failed: {exc}")
            st.info("Check browser camera permission. On hosted deployments, use the upload demo if WebRTC is unavailable.")
    else:
        upload = st.file_uploader("Choose a face photo or short video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
        if upload and st.button("Analyze media", type="primary"):
            try:
                if upload.type.startswith("image/"):
                    image = decode_image(upload.getvalue())
                    annotated, results = recognize_image(image)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Recognition result", use_container_width=True)
                    if results:
                        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
                    else:
                        st.warning("No face detected in this image.")
                    if not get_database().get_users():
                        st.info("No registered users: detected faces are shown as unknown.")
                else:
                    suffix = Path(upload.name).suffix or ".mp4"
                    identities: set[str] = set()
                    examined = 0
                    preview = None
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                        temporary.write(upload.getvalue())
                        temporary_path = Path(temporary.name)
                    try:
                        video = cv2.VideoCapture(str(temporary_path))
                        frame_index = 0
                        while examined < 90:
                            ok, frame = video.read()
                            if not ok:
                                break
                            if frame_index % 12 == 0:
                                preview, results = recognize_image(frame)
                                identities.update(str(item["name"]) for item in results if item["status"] == "Recognized")
                                examined += 1
                            frame_index += 1
                        video.release()
                    finally:
                        temporary_path.unlink(missing_ok=True)
                    if examined == 0:
                        st.error("No video frames could be read. Try MP4 or MOV.")
                    else:
                        if preview is not None:
                            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Last analyzed frame", use_container_width=True)
                        st.metric("Frames analyzed", examined)
                        st.write("Recognized identities:", ", ".join(sorted(identities)) if identities else "None")
            except Exception as exc:
                st.error(f"Media analysis failed: {exc}")
        st.caption("Uploaded media is processed for this session and is not saved by FaceVision AI.")


def attendance_page() -> None:
    title("RECORDS / ATTENDANCE", "The attendance ledger.", "Review recognized check-ins and export a portable CSV report.")
    database = get_database()
    records = database.get_attendance()
    if records:
        frame = pd.DataFrame(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.info("The ledger is empty. Recognized users will be recorded automatically.")
    st.download_button(
        "↓  Export attendance CSV",
        data=database.export_attendance_csv().encode("utf-8"),
        file_name="facevision-attendance.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.markdown('<div class="section-label">Registered users</div>', unsafe_allow_html=True)
    users = database.get_users()
    if users:
        st.dataframe(
            pd.DataFrame([{"user_id": user["user_id"], "name": user["display_name"], "registered_at": user["created_at"]} for user in users]),
            hide_index=True,
            use_container_width=True,
        )
        selected = st.selectbox("Select a user to delete", [str(user["user_id"]) for user in users])
        confirmed = st.checkbox("I understand this permanently deletes the user's embedding and attendance history.")
        if st.button("Delete user", disabled=not confirmed):
            if database.delete_user(selected):
                st.success(f"Deleted {selected} and their linked attendance records.")
                st.rerun()
    else:
        st.caption("No users are registered.")


def about_page() -> None:
    title("PROJECT / ABOUT", "Responsible recognition, locally run.", "An educational computer-vision system built for transparent attendance workflows.")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### How it works")
        st.write("YuNet detects faces in each frame. SFace aligns each detected face and creates a 128-dimensional feature vector. FaceVision compares that vector with enrolled templates using cosine similarity; matches below the configured threshold stay **Unknown**.")
        st.write("The default similarity threshold is `0.42`. Tune it for your camera and lighting, and evaluate both false matches and missed matches before any real use.")
        st.markdown("### Project boundaries")
        st.write("This is a portfolio and classroom project, not an identity-verification or access-control system. It has no liveness detection and can be affected by pose, lighting, camera quality, and demographic variation.")
    with right:
        st.markdown("### Privacy notice")
        st.warning("Biometric data is sensitive. By default, face embeddings and attendance are stored in a local SQLite file. Raw webcam frames and enrollment captures are not intentionally written to disk.")
        st.write("If hosted, camera frames or uploaded media are transmitted to that host for processing, and the database may be ephemeral or outside your control. Use synthetic or consenting test subjects only. Do not deploy with real biometric data without legal review, informed consent, security controls, and a retention policy.")
        st.markdown("### Stack")
        st.write("Python · OpenCV YuNet/SFace · Streamlit · SQLite · Pandas")


def main() -> None:
    try:
        database = get_database()
        metrics = dashboard_metrics(database)
    except Exception as exc:
        st.error(f"Database unavailable: {exc}")
        st.info("Check FACEVISION_DB_PATH and ensure the database folder is writable. Back up the database before repairing or replacing it.")
        st.stop()

    requested_page = st.session_state.get("navigation_request")
    if requested_page in ["Dashboard", "Register user", "Live recognition", "Attendance", "About"]:
        st.session_state.navigation = requested_page
        st.session_state.navigation_request = None

    with st.sidebar:
        st.markdown('<div><span class="brand-mark">◉</span><b>FaceVision AI</b></div>', unsafe_allow_html=True)
        st.caption("ATTENDANCE INTELLIGENCE")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Dashboard", "Register user", "Live recognition", "Attendance", "About"],
            key="navigation",
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(f"{metrics['registered_users']} registered · {metrics['today_attendance']} check-ins today")
        st.caption(f"Match threshold · {MATCH_THRESHOLD:.2f}")

    try:
        if page == "Dashboard":
            dashboard_page()
        elif page == "Register user":
            registration_page()
        elif page == "Live recognition":
            live_page()
        elif page == "Attendance":
            attendance_page()
        else:
            about_page()
    except Exception as exc:
        st.error(f"This view could not be loaded: {exc}")


if __name__ == "__main__":
    main()