"""Streamlit page for EPM analysis, isolated from the CPP page."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from canvas_utils import st_canvas_fixed
from epm import (
    ARM_DWELL_SECONDS,
    CENTER_DWELL_SECONDS,
    REGION_ORDER,
    calibration_from_canvas,
    calibration_from_saved_settings,
    create_epm_bundle,
    draw_epm_overlay,
    extract_epm_polygons,
    numbered_canvas_overlay,
)
from epm_video import write_annotated_epm_video
from epm_tracker import EPMTracker
from epm_visuals import (
    inject_epm_visual_theme,
    render_epm_creator,
    render_epm_empty_state,
    render_epm_hero,
    render_epm_progress,
)
from io_utils import (
    ensure_directory,
    export_dataframe_csv,
    export_warnings_text,
    get_video_metadata,
    save_uploaded_video,
)
from tracker import TrackingConfig

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ensure_directory(APP_DIR / "runtime_data" / "epm_uploads")
RESULTS_DIR = ensure_directory(APP_DIR / "runtime_data" / "epm_results")
EPM_TRACKING_CONFIG = TrackingConfig(
    min_contour_area=150.0,
    max_jump_px=120.0,
    smoothing_alpha=0.35,
    roi_padding_px=0,
)
EPM_SCORING_POINT = "smoothed_centroid"
CREATOR_PHOTO_PATH = APP_DIR / "assets" / "austin_dean_headshot.png"


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
        st.session_state.pop("epm_start_seconds", None)
        st.session_state.pop("epm_end_seconds", None)
        for name in REGION_ORDER:
            st.session_state.pop(f"epm_map_{name}", None)
        _clear_results()


def _download(label: str, path: str | Path, mime: str) -> None:
    source = Path(path)
    st.download_button(label, data=source.read_bytes(), file_name=source.name, mime=mime, key=f"epm_download_{source.name}")


def _render_results(results: dict) -> None:
    st.subheader("EPM results")
    qc_values = results["qc"].set_index("metric")["value"]
    st.error(
        f'Needs manual review · assigned body time {float(qc_values["scored_coverage_percent"]):.1f}% · '
        f'{int(qc_values["inferred_frames"])} inferred frames · '
        f'{float(qc_values["unclassified_seconds"]):.1f} s unclassified · '
        f'{int(qc_values.get("possible_transitions_for_review", 0))} possible transitions. '
        "Review the raw video and event table before using any measurement."
    )
    summary = results["summary"]
    st.caption("Entries use the smoothed body centroid. Arm dwell is 0.5 seconds; center dwell is 0.1 seconds. Review inferred frames and events against the raw video.")
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
    padding: 1rem;
    border: 1px solid #d5e1ea;
    border-radius: 7px;
    background: #ffffff;
    color: #172a3a;
}
.epm-sidebar-kicker {
    display: inline-block;
    padding: 0;
    color: #0f766e;
    font-size: 0.78rem;
    font-weight: 700;
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
    <li><strong>Check the session.</strong> The entire uploaded video is scored at its embedded FPS.</li>
    <li><strong>Choose a clear frame.</strong> Select a calibration time without the experimenter over the maze.</li>
    <li><strong>Draw five regions.</strong> Outline center, two open arms, and two closed arms. Right-click to close each polygon, then Send to Streamlit.</li>
    <li><strong>Check labels.</strong> Map the numbered polygons and review their boundaries.</li>
  </ol>
  <div class="epm-sidebar-section">Analysis</div>
  <ol class="epm-sidebar-list" start="6">
    <li><strong>Check the fixed scoring rule.</strong> Smoothed body centroid is used. Arm entry requires 0.5 seconds; center entry requires 0.1 second.</li>
    <li><strong>Run analysis.</strong> Keep the page open while tracking and optional video export finish.</li>
    <li><strong>Review results.</strong> Check inferred frames, rejected candidates, the event table, and observed versus inferred markers in the annotated video.</li>
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
**September 22, 2026 — EPM page and scoring update**

- Set the fixed arm entry dwell to 0.5 seconds; center dwell remains 0.1 seconds.
- Simplified the page to use video FPS, the full recording, and smoothed body centroid automatically.
- Added EPM visuals, a rat progress panel, and the creator section.

**September 21, 2026 — temporal scoring revision**

- Fixed arm dwell at 1 second and center dwell at 0.1 second; brief arm peeks count as center time.
- Added trajectory and neighboring-frame assignments for uncertain body positions, with per-frame QC flags and a jump ceiling.
- Marked inferred body positions separately in the annotated video.

**September 21, 2026 — controlled pilot candidate**

- Defaulted provisional entries to smoothed body centroid while scoring body occupancy separately from optional entry proxies.
- Added same-video calibration reload, possible-transition review rows, and separate body/proxy markers in QC video.
- Rechecked illumination artifacts and reacquisition on the original recording; manual verification remains required.

**September 20, 2026**

**EPM reliability review (local)**

- Added EPM-only candidate gating, explicit reacquisition, unknown time, and an analysis interval.
- Separated rejected raw candidates from accepted scoring points and broke trails at gaps.
- Added a manual-review status and expanded per-frame/event QC.

**EPM analyzer added**

- Added a separate EPM section with five-region calibration and the six requested counts and times.
- Added a selectable calibration frame for videos that start with the experimenter in view.
- Added head-and-shoulders proxy scoring, QC warnings, event exports, and an optional annotated MP4.
- Kept CPP scoring in its own section.
"""
        )


def main() -> None:
    st.set_page_config(page_title="Rat Behavior Analysis Suite", layout="wide")
    inject_epm_visual_theme()
    _render_epm_sidebar()
    render_epm_hero()

    with st.container(border=True):
        st.subheader("1 · Upload video")
        uploaded = st.file_uploader("EPM video", type=["mp4", "mov", "avi", "m4v"], key=f'epm_upload_{st.session_state.get("epm_upload_version", 0)}')
        if uploaded is not None:
            _load_upload(uploaded)
        if st.button("Forget EPM video and results"):
            st.session_state["epm_video_path"] = None
            st.session_state["epm_video_signature"] = None
            st.session_state.pop("epm_calibration_seconds", None)
            st.session_state.pop("epm_start_seconds", None)
            st.session_state.pop("epm_end_seconds", None)
            st.session_state["epm_upload_version"] = st.session_state.get("epm_upload_version", 0) + 1
            st.session_state["epm_canvas_version"] = st.session_state.get("epm_canvas_version", 0) + 1
            _clear_results()
            st.rerun()
    if not st.session_state.get("epm_video_path"):
        render_epm_empty_state()
        render_epm_creator(CREATOR_PHOTO_PATH)
        return

    video_path = Path(st.session_state["epm_video_path"])
    try:
        metadata = get_video_metadata(video_path)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        render_epm_creator(CREATOR_PHOTO_PATH)
        return
    st.write(
        f"**{video_path.name}** · {metadata.width} × {metadata.height} · "
        f"{metadata.frame_count:,} frames · {metadata.fps:.3f} FPS · {metadata.duration_seconds:.1f} s"
    )
    if metadata.notes:
        for note in metadata.notes:
            st.warning(note)
    fps = float(metadata.fps)
    start_seconds, end_seconds = 0.0, metadata.frame_count / fps
    start_frame, end_frame = 0, metadata.frame_count
    st.caption("The full recording is scored using its embedded frame rate. Select a separate calibration frame below without changing the scored interval.")

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
        render_epm_creator(CREATOR_PHOTO_PATH)
        return
    image, scale_x, scale_y = _display_frame(calibration_frame)
    with st.container(border=True):
        st.subheader("2 · Define the maze")
        saved_calibration = st.file_uploader(
            "Or load calibration_and_settings.json from this same video",
            type=["json"], key=f'epm_saved_calibration_{hashlib.sha256(st.session_state["epm_video_signature"].encode()).hexdigest()[:16]}',
        )
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
    if saved_calibration is not None:
        try:
            calibration = calibration_from_saved_settings(
                json.loads(saved_calibration.getvalue()), metadata.width, metadata.height,
                st.session_state["epm_video_signature"],
            )
            st.image(
                cv2.cvtColor(draw_epm_overlay(calibration_frame, calibration), cv2.COLOR_BGR2RGB),
                caption="Saved polygons loaded. Check all five regions against this frame.",
                use_container_width=True,
            )
            st.success("Saved calibration loaded for this recording.")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            calibration = None
            st.error(f"Saved calibration could not be used: {error}")

    with st.container(border=True):
        st.subheader("4 · Run analysis")
        dwell = ARM_DWELL_SECONDS
        st.caption("Fixed scoring: smoothed body centroid; 0.5 seconds in an arm and 0.1 seconds in center. Short arm peeks count as center time.")
        initial_entry = st.checkbox(
            "Count the initial arm if the rat starts in an arm",
            value=True, key="epm_initial_entry",
            help="Matches sessions with one arm entry but no center entry. Initial center occupancy is never a center entry.",
        )
        export_video = st.checkbox("Create annotated QC video", value=True, key="epm_export_video")
        draw_trajectory = st.checkbox("Show trajectory in QC video", value=True, key="epm_trajectory")
        if calibration is None:
            st.warning("Load a matching saved calibration or complete and verify all five polygons before running analysis.")
        run = st.button("Run EPM analysis", type="primary", disabled=calibration is None)

    if calibration is None:
        render_epm_creator(CREATOR_PHOTO_PATH)
        return
    settings = {
        "video_signature": st.session_state["epm_video_signature"],
        "calibration": calibration.to_dict(),
        "timing_fps": fps,
        "analysis_start_seconds": start_seconds,
        "analysis_end_seconds": end_seconds,
        "analysis_start_frame": start_frame,
        "analysis_end_frame_exclusive": end_frame,
        "scoring_point": EPM_SCORING_POINT,
        "occupancy_point": "tracked_smoothed_body_centroid",
        "scoring_version": "epm_fixed_centroid_half_second_v5",
        "min_dwell_seconds": dwell,
        "count_initial_arm": initial_entry,
        "export_annotated_video": export_video,
        "draw_trajectory": draw_trajectory,
        "tracker_config": vars(EPM_TRACKING_CONFIG),
    }
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    if run:
        progress_panel = st.empty()
        started_at = time.time()
        status_notes: list[str] = []

        def update_status(fraction: float, stage: str, detail: str, technical: str, note: str | None = None) -> None:
            if note and (not status_notes or status_notes[-1] != note):
                status_notes.append(note)
            render_epm_progress(progress_panel, fraction, stage, detail, technical,
                                time.time() - started_at, status_notes)

        update_status(.02, "Preparing the EPM analysis", "Checking the video and five maze regions.",
                      f"{video_path.name} · {fps:.3f} FPS · full recording", "Analysis started.")
        try:
            tracker = EPMTracker(EPM_TRACKING_CONFIG)

            def track_progress(current: int, total: int) -> None:
                if current % 120 == 0 or current == total:
                    fraction = .08 + .62 * current / max(total, 1)
                    elapsed = max(time.time() - started_at, 1e-6)
                    speed = current / elapsed
                    remaining = (total - current) / speed if speed > 0 else 0
                    update_status(fraction, "Tracking the rat through the maze",
                                  f"Scanning frame {current:,} of {total:,} within the five mapped regions.",
                                  f"{speed:.1f} frames/sec · approximately {remaining:.0f} s remaining",
                                  "Frame-by-frame tracking is running.")

            tracking = tracker.track_video(video_path, calibration=calibration, progress_callback=track_progress, fps_override=fps)
            update_status(.74, "Scoring occupancy and entries", "Applying fixed arm and center dwell rules.",
                          "Smoothed centroid · 0.5 s arm · 0.1 s center", "Tracking finished; scoring regions.")
            bundle = create_epm_bundle(
                tracking, calibration, fps, EPM_SCORING_POINT, dwell, initial_entry,
                analysis_start_seconds=start_seconds, analysis_end_seconds=end_seconds,
            )
            update_status(.80, "Saving measurements and QC", "Writing frame assignments, events, and summary files.",
                          f"{len(bundle.per_frame):,} frames scored", "Exporting review files.")
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
            flagged_frames = bundle.per_frame.loc[
                (bundle.per_frame["tracking_status"] != "tracked")
                | bundle.per_frame["event"].ne("") | bundle.per_frame["review_event"].ne("")
            ]
            preview = pd.concat([
                bundle.per_frame.iloc[::max(1, len(bundle.per_frame) // 100)],
                flagged_frames.head(150),
                bundle.per_frame.tail(50),
            ]).sort_values("frame_index").drop_duplicates("frame_index")
            payload = {
                "signature": signature,
                "output_dir": str(output_dir),
                "summary": bundle.summary,
                "events": bundle.events,
                "region_times": bundle.region_times,
                "qc": bundle.qc_metrics,
                "warnings": bundle.warnings,
                "per_frame_preview": preview,
                "annotated_video": None,
                **{name: str(path) for name, path in paths.items()},
            }
            st.session_state["epm_results"] = payload
            if export_video:
                def video_progress(current: int, total: int) -> None:
                    if current % 120 == 0 or current == total:
                        update_status(.84 + .15 * current / max(total, 1), "Creating the annotated video",
                                      f"Writing annotated frame {current:,} of {total:,}.",
                                      "Observed points: green · inferred points: yellow",
                                      "Annotated MP4 export is running.")

                payload["annotated_video"] = str(write_annotated_epm_video(
                    video_path, output_dir / "annotated_output.mp4", bundle.per_frame,
                    calibration, draw_trajectory, video_progress
                ))
            update_status(1., "Analysis complete", "Results and downloads are ready below.",
                          f"Saved to {output_dir}", "Analysis complete.")
            st.success("EPM analysis complete.")
        except Exception as error:
            _clear_results()
            update_status(1., "Analysis stopped", "The run could not finish.", str(error), "Review the error and retry.")
            st.exception(error)
            render_epm_creator(CREATOR_PHOTO_PATH)
            return

    results = st.session_state.get("epm_results")
    if results and results.get("signature") == signature:
        _render_results(results)
    elif results:
        st.info("The video, calibration, or scoring settings changed. Run EPM analysis again to update the results.")
    render_epm_creator(CREATOR_PHOTO_PATH)
