"""Streamlit page for EPM analysis, isolated from the CPP page."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from canvas_utils import st_canvas_fixed
from epm import (
    REGION_ORDER,
    calibration_from_canvas,
    create_epm_bundle,
    draw_epm_overlay,
    extract_epm_polygons,
    numbered_canvas_overlay,
)
from epm_video import write_annotated_epm_video
from io_utils import (
    ensure_directory,
    export_dataframe_csv,
    export_warnings_text,
    get_video_metadata,
    save_uploaded_video,
)
from tracker import SingleRatTracker, TrackingConfig

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ensure_directory(APP_DIR / "runtime_data" / "epm_uploads")
RESULTS_DIR = ensure_directory(APP_DIR / "runtime_data" / "epm_results")
EPM_TRACKING_CONFIG = TrackingConfig(
    min_contour_area=150.0,
    max_jump_px=120.0,
    smoothing_alpha=0.35,
    roi_padding_px=0,
)
POINT_OPTIONS = {
    "Head-and-shoulders proxy": "head_shoulders",
    "Smoothed body centroid": "smoothed_centroid",
    "Raw body centroid": "centroid",
}


def _display_frame(frame, max_width: int = 1100) -> tuple[Image.Image, float, float]:
    height, width = frame.shape[:2]
    display_width = min(width, max_width)
    display_height = max(1, round(height * display_width / width))
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return image.resize((display_width, display_height)), width / display_width, height / display_height


def _read_calibration_frame(path: Path, frame_index: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for calibration: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Could not read calibration frame {frame_index:,}.")
    return frame


def _clear_results() -> None:
    st.session_state["epm_results"] = None


def _load_upload(uploaded_file) -> None:
    digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
    signature = f"{uploaded_file.name}:{uploaded_file.size}:{digest}"
    if signature != st.session_state.get("epm_video_signature"):
        path = save_uploaded_video(uploaded_file, UPLOAD_DIR)
        st.session_state["epm_video_path"] = str(path)
        st.session_state["epm_video_signature"] = signature
        st.session_state["epm_canvas_version"] = st.session_state.get("epm_canvas_version", 0) + 1
        st.session_state.pop("epm_calibration_seconds", None)
        for name in REGION_ORDER:
            st.session_state.pop(f"epm_map_{name}", None)
        _clear_results()


def _download(label: str, path: str | Path, mime: str) -> None:
    source = Path(path)
    st.download_button(label, data=source.read_bytes(), file_name=source.name, mime=mime, key=f"epm_download_{source.name}")


def _render_results(results: dict) -> None:
    st.subheader("EPM results")
    summary = results["summary"]
    st.dataframe(summary, hide_index=True, use_container_width=True)
    time_cols = st.columns(3)
    time_cols[0].metric("Open arm", f'{int(summary.iloc[0]["Open Arm Time (whole seconds)"])} s')
    time_cols[1].metric("Closed arm", f'{int(summary.iloc[0]["Closed Arm Time (whole seconds)"])} s')
    time_cols[2].metric("Center", f'{int(summary.iloc[0]["Center Time (whole seconds)"])} s')
    if results["warnings"]:
        with st.expander(f'Quality warnings ({len(results["warnings"])})', expanded=True):
            for warning in results["warnings"]:
                st.warning(warning)
    tabs = st.tabs(["Events", "Time by region", "QC", "Frame preview"])
    with tabs[0]:
        st.dataframe(results["events"], hide_index=True, use_container_width=True)
    with tabs[1]:
        st.dataframe(results["region_times"], hide_index=True, use_container_width=True)
    with tabs[2]:
        st.dataframe(results["qc"], hide_index=True, use_container_width=True)
    with tabs[3]:
        st.dataframe(results["per_frame_preview"], hide_index=True, use_container_width=True)
        st.caption("The download contains every frame.")
    st.caption(f'Results saved in {results["output_dir"]}. Download what you need before leaving the deployed app.')
    first = st.columns(3)
    with first[0]:
        _download("Summary CSV", results["summary_csv"], "text/csv")
        _download("Events CSV", results["events_csv"], "text/csv")
    with first[1]:
        _download("Per-frame CSV", results["per_frame_csv"], "text/csv")
        _download("Time by region CSV", results["region_csv"], "text/csv")
    with first[2]:
        _download("QC CSV", results["qc_csv"], "text/csv")
        _download("Raw tracking CSV", results["tracking_csv"], "text/csv")
    second = st.columns(3)
    with second[0]:
        _download("Calibration and settings JSON", results["calibration_json"], "application/json")
    with second[1]:
        _download("Warnings TXT", results["warnings_txt"], "text/plain")
    if results.get("annotated_video"):
        with second[2]:
            _download("Annotated MP4", results["annotated_video"], "video/mp4")


def _render_epm_sidebar() -> None:
    st.sidebar.markdown(
        """
<style>
.epm-sidebar-panel {
    box-sizing: border-box;
    padding: 1.15rem 1rem;
    border: 1px solid rgba(166, 190, 205, 0.88);
    border-radius: 22px;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(15, 36, 54, 0.07);
    color: #172a3a;
}
.epm-sidebar-kicker {
    display: inline-block;
    padding: 0.34rem 0.7rem;
    border-radius: 999px;
    background: rgba(15, 118, 110, 0.11);
    color: #0f766e;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.epm-sidebar-title {
    margin: 0.7rem 0 0.75rem;
    font-size: 1.18rem;
    font-weight: 700;
}
.epm-sidebar-section {
    margin: 0.85rem 0 0.4rem;
    padding-top: 0.7rem;
    border-top: 1px solid #d5e1ea;
    color: #0f766e;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.epm-sidebar-list {
    margin: 0;
    padding-left: 1.1rem;
    font-size: 0.87rem;
}
.epm-sidebar-list li {
    margin-bottom: 0.55rem;
    line-height: 1.45;
}
.epm-sidebar-note {
    margin-top: 0.85rem;
    padding: 0.7rem 0.8rem;
    border: 1px solid rgba(15, 118, 110, 0.2);
    border-radius: 14px;
    background: rgba(15, 118, 110, 0.08);
    font-size: 0.86rem;
    line-height: 1.45;
}
</style>
<div class="epm-sidebar-panel">
  <div class="epm-sidebar-kicker">Quick guide</div>
  <div class="epm-sidebar-title">EPM Analyzer</div>
  <div class="epm-sidebar-section">Setup</div>
  <ol class="epm-sidebar-list">
    <li><strong>Upload video.</strong> Check the duration and FPS.</li>
    <li><strong>Choose a clear frame.</strong> Select a calibration time without the experimenter over the maze.</li>
    <li><strong>Draw five regions.</strong> Outline center, two open arms, and two closed arms. Right-click to close each polygon, then Send to Streamlit.</li>
    <li><strong>Check labels.</strong> Map the numbered polygons and review their boundaries.</li>
  </ol>
  <div class="epm-sidebar-section">Analysis</div>
  <ol class="epm-sidebar-list" start="5">
    <li><strong>Use the head-and-shoulders proxy.</strong> Set dwell time and initial-arm counting to your lab rule.</li>
    <li><strong>Run analysis.</strong> Keep the page open while tracking and optional video export finish.</li>
    <li><strong>Review results.</strong> Check the event table, QC warnings, and annotated video.</li>
    <li><strong>Download outputs.</strong> Save the six-column summary and review files.</li>
  </ol>
  <div class="epm-sidebar-note">Confirm automated counts against reviewed video before using them as research measurements.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("Changelog", expanded=False):
        st.markdown(
            """
**September 20, 2026**

**EPM analyzer added**

- Added a separate EPM section with five-region calibration and the six requested counts and times.
- Added a selectable calibration frame for videos that start with the experimenter in view.
- Added head-and-shoulders proxy scoring, QC warnings, event exports, and an optional annotated MP4.
- Kept CPP scoring in its own section.
"""
        )


def main() -> None:
    st.set_page_config(page_title="EPM Rat Behavior Analyzer", layout="wide")
    _render_epm_sidebar()
    st.title("Elevated Plus Maze Analyzer")
    st.caption("Upload one fixed-camera session, choose a clear calibration frame, draw five maze regions, and score the rat.")
    st.caption("Pilot scoring: review the event table and annotated video before using results as research measurements.")
    st.info(
        "Entries use the selected scoring point crossing from center into an arm. "
        "A return from an arm into center counts as a center entry. "
        "A stable arm occupied at the start can count as one initial entry. "
        "Low-confidence or missing frames cannot create entries."
    )

    with st.container(border=True):
        st.subheader("1 · Upload video")
        uploaded = st.file_uploader("EPM video", type=["mp4", "mov", "avi", "m4v"], key=f'epm_upload_{st.session_state.get("epm_upload_version", 0)}')
        if uploaded is not None:
            _load_upload(uploaded)
        if st.button("Forget EPM video and results"):
            st.session_state["epm_video_path"] = None
            st.session_state["epm_video_signature"] = None
            st.session_state.pop("epm_calibration_seconds", None)
            st.session_state["epm_upload_version"] = st.session_state.get("epm_upload_version", 0) + 1
            st.session_state["epm_canvas_version"] = st.session_state.get("epm_canvas_version", 0) + 1
            _clear_results()
            st.rerun()
    if not st.session_state.get("epm_video_path"):
        return

    video_path = Path(st.session_state["epm_video_path"])
    try:
        metadata = get_video_metadata(video_path)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return
    st.write(
        f"**{video_path.name}** · {metadata.width} × {metadata.height} · "
        f"{metadata.frame_count:,} frames · {metadata.fps:.3f} FPS · {metadata.duration_seconds:.1f} s"
    )
    if metadata.notes:
        for note in metadata.notes:
            st.warning(note)
    manual_fps = st.checkbox("Use a manual FPS for time conversion", value=False, key="epm_manual_fps")
    fps = float(metadata.fps)
    if manual_fps:
        fps = float(st.number_input("Actual FPS", min_value=0.1, max_value=240.0, value=float(metadata.fps), step=0.1, key="epm_fps"))

    max_frame = max(0, metadata.frame_count - 1)
    max_seconds = max_frame / metadata.fps
    calibration_seconds = st.number_input(
        "Calibration frame time (seconds)", min_value=0.0, max_value=float(max_seconds),
        value=float(min(10.0, max_seconds)), step=1.0, key="epm_calibration_seconds",
        help="Choose a moment with a clear view of the maze. This changes only the drawing frame; the full video is scored.",
    )
    calibration_frame_index = min(max_frame, round(calibration_seconds * metadata.fps))
    try:
        calibration_frame = _read_calibration_frame(video_path, calibration_frame_index)
    except ValueError as error:
        st.error(str(error))
        return
    image, scale_x, scale_y = _display_frame(calibration_frame)
    with st.container(border=True):
        st.subheader("2 · Define the maze")
        st.write(
            "Draw **five polygons** on the selected frame: center, two open arms, and two closed arms. "
            "Click around each walking surface and right-click to close that polygon. "
            "Keep the room, equipment, and shadows outside the outlines. "
            "Place the center-to-arm boundaries where your lab scores entry. "
            "After drawing all five, click the canvas toolbar icon labeled Send to Streamlit."
        )
        st.caption(
            "The reference video has narrow vertical arms and darker horizontal arms; "
            "identify open and closed arms from the apparatus, then verify the labels below."
        )
        if st.button("Clear EPM drawing"):
            st.session_state["epm_canvas_version"] = st.session_state.get("epm_canvas_version", 0) + 1
            _clear_results()
            st.rerun()
        canvas = st_canvas_fixed(
            fill_color="rgba(14, 165, 233, 0.12)",
            stroke_width=3,
            stroke_color="#0ea5e9",
            background_image=image,
            update_streamlit=True,
            height=image.height,
            width=image.width,
            drawing_mode="polygon",
            key=f'epm_canvas_{st.session_state.get("epm_canvas_version", 0)}_{st.session_state["epm_video_signature"][:16]}_{calibration_frame_index}',
        )

    calibration = None
    if canvas.json_data:
        try:
            polygons = extract_epm_polygons(canvas.json_data, scale_x, scale_y)
            st.caption(f"{len(polygons)} of 5 polygons drawn.")
            if polygons:
                st.image(
                    cv2.cvtColor(numbered_canvas_overlay(calibration_frame, polygons), cv2.COLOR_BGR2RGB),
                    caption="Numbers show drawing order. Map each number to a region below.",
                    use_container_width=True,
                )
            if len(polygons) == 5:
                with st.container(border=True):
                    st.subheader("3 · Label and review regions")
                    mapping = {}
                    cols = st.columns(5)
                    labels = ["Center", "Open arm 1", "Open arm 2", "Closed arm 1", "Closed arm 2"]
                    for index, (name, label) in enumerate(zip(REGION_ORDER, labels)):
                        with cols[index]:
                            mapping[name] = st.selectbox(label, options=list(range(1, 6)), index=index, key=f"epm_map_{name}") - 1
                    calibration = calibration_from_canvas(
                        canvas.json_data, mapping, metadata.width, metadata.height, scale_x, scale_y
                    )
                    preview = draw_epm_overlay(calibration_frame, calibration)
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Confirm every region and shared boundary before analysis.", use_container_width=True)
                    st.success("Five regions are ready. Tracking will be limited to their union.")
            elif len(polygons) > 5:
                st.error("There are more than five polygons. Clear the drawing and try again.")
        except (ValueError, TypeError, KeyError) as error:
            st.warning(f"Calibration needs attention: {error}")

    with st.container(border=True):
        st.subheader("4 · Choose scoring and run")
        point_label = st.selectbox("Position used for region and entry scoring", list(POINT_OPTIONS), key="epm_point")
        st.caption("The head-and-shoulders option is a motion-based proxy, not anatomical pose tracking.")
        dwell = float(st.number_input(
            "Minimum continuous time in a new region before an entry counts (seconds)",
            min_value=0.0, max_value=2.0, value=0.3, step=0.05, key="epm_dwell",
            help="Set to 0 for an immediate boundary crossing. The default requires about nine frames at 30 FPS.",
        ))
        initial_entry = st.checkbox(
            "Count the initial arm if the rat starts in an arm",
            value=True, key="epm_initial_entry",
            help="Matches sessions with one arm entry but no center entry. Initial center occupancy is never a center entry.",
        )
        export_video = st.checkbox("Create annotated QC video", value=True, key="epm_export_video")
        draw_trajectory = st.checkbox("Show trajectory in QC video", value=True, key="epm_trajectory")
        if calibration is None:
            st.warning("Complete and verify all five polygons before running analysis.")
        run = st.button("Run EPM analysis", type="primary", disabled=calibration is None)

    if calibration is None:
        return
    settings = {
        "video_signature": st.session_state["epm_video_signature"],
        "calibration": calibration.to_dict(),
        "timing_fps": fps,
        "scoring_point": POINT_OPTIONS[point_label],
        "min_dwell_seconds": dwell,
        "count_initial_arm": initial_entry,
        "export_annotated_video": export_video,
        "draw_trajectory": draw_trajectory,
        "tracker_config": vars(EPM_TRACKING_CONFIG),
    }
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    if run:
        progress = st.progress(0, text="Preparing EPM tracking")
        try:
            tracker = SingleRatTracker(EPM_TRACKING_CONFIG)

            def track_progress(current: int, total: int) -> None:
                if current % 120 == 0 or current == total:
                    progress.progress(min(70, 5 + int(65 * current / max(total, 1))), text=f"Tracking frame {current:,} of {total:,}")

            tracking = tracker.track_video(video_path, arena_mask=calibration.tracking_mask(), progress_callback=track_progress, fps_override=fps)
            progress.progress(75, text="Scoring regions and entries")
            bundle = create_epm_bundle(tracking, calibration, fps, POINT_OPTIONS[point_label], dwell, initial_entry)
            output_dir = ensure_directory(RESULTS_DIR / f'{video_path.stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}')
            paths = {
                "tracking_csv": export_dataframe_csv(tracking, output_dir / "tracking_raw.csv"),
                "per_frame_csv": export_dataframe_csv(bundle.per_frame, output_dir / "per_frame_assignments.csv"),
                "summary_csv": export_dataframe_csv(bundle.summary, output_dir / "summary.csv"),
                "events_csv": export_dataframe_csv(bundle.events, output_dir / "events.csv"),
                "region_csv": export_dataframe_csv(bundle.region_times, output_dir / "region_times.csv"),
                "qc_csv": export_dataframe_csv(bundle.qc_metrics, output_dir / "qc_metrics.csv"),
                "warnings_txt": export_warnings_text(bundle.warnings, output_dir / "warnings.txt"),
            }
            calibration_json = output_dir / "calibration_and_settings.json"
            calibration_json.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            paths["calibration_json"] = calibration_json
            payload = {
                "signature": signature,
                "output_dir": str(output_dir),
                "summary": bundle.summary,
                "events": bundle.events,
                "region_times": bundle.region_times,
                "qc": bundle.qc_metrics,
                "warnings": bundle.warnings,
                "per_frame_preview": bundle.per_frame.head(300),
                "annotated_video": None,
                **{name: str(path) for name, path in paths.items()},
            }
            st.session_state["epm_results"] = payload
            if export_video:
                def video_progress(current: int, total: int) -> None:
                    progress.progress(min(99, 80 + int(19 * current / max(total, 1))), text=f"Writing annotated frame {current:,} of {total:,}")

                payload["annotated_video"] = str(write_annotated_epm_video(
                    video_path, output_dir / "annotated_output.mp4", bundle.per_frame,
                    calibration, draw_trajectory, video_progress
                ))
            progress.progress(100, text="EPM analysis complete")
            st.success("EPM analysis complete.")
        except Exception as error:
            _clear_results()
            progress.empty()
            st.exception(error)
            return

    results = st.session_state.get("epm_results")
    if results and results.get("signature") == signature:
        _render_results(results)
    elif results:
        st.info("The video, calibration, or scoring settings changed. Run EPM analysis again to update the results.")
