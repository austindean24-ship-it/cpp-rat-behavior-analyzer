from __future__ import annotations

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# `streamlit-drawable-canvas` still expects `streamlit.elements.image.image_to_url`,
# but newer Streamlit versions moved that helper. This shim keeps the drawing widget
# working without asking the user to edit packages by hand.
try:
    import streamlit.elements.image as _st_image_module

    if not hasattr(_st_image_module, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

        def _compat_image_to_url(
            image,
            width_or_layout_config=None,
            clamp: bool = False,
            channels: str = "RGB",
            output_format: str = "auto",
            image_id: str | None = None,
        ) -> str:
            if hasattr(width_or_layout_config, "width"):
                layout_config = width_or_layout_config
            else:
                layout_width = int(width_or_layout_config) if isinstance(width_or_layout_config, int) else width_or_layout_config
                layout_config = _LayoutConfig(width=layout_width)
            return _image_to_url(
                image,
                layout_config,
                clamp,
                channels,
                output_format,
                image_id or "",
            )

        _st_image_module.image_to_url = _compat_image_to_url
except Exception:
    pass

from streamlit_drawable_canvas import st_canvas

from analysis import create_analysis_bundle
from demo_generator import generate_synthetic_cpp_video
from io_utils import (
    ensure_directory,
    export_dataframe_csv,
    export_warnings_text,
    get_video_metadata,
    read_first_frame,
    save_uploaded_video,
    write_annotated_video,
)
from regions import (
    ArenaCalibration,
    build_calibration_from_canvas_box,
    build_calibration_from_canvas_polygons,
    draw_calibration_overlay,
)
from tracker import SingleRatTracker, TrackingConfig


APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = ensure_directory(APP_DIR / "runtime_data")
UPLOAD_DIR = ensure_directory(RUNTIME_DIR / "uploads")
RESULTS_DIR = ensure_directory(RUNTIME_DIR / "results")
DEMO_DIR = ensure_directory(RUNTIME_DIR / "demo")


def ensure_session_state() -> None:
    defaults = {
        "video_path": None,
        "video_signature": None,
        "analysis_results": None,
        "canvas_reset_counter": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_run_state() -> None:
    st.session_state["analysis_results"] = None
    st.session_state["canvas_reset_counter"] = st.session_state.get("canvas_reset_counter", 0) + 1


def load_video_into_session(video_path: Path, signature: str) -> None:
    if st.session_state.get("video_signature") != signature:
        st.session_state["video_path"] = str(video_path)
        st.session_state["video_signature"] = signature
        clear_run_state()


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def resize_for_canvas(frame, max_width: int = 900) -> tuple[Image.Image, float, float]:
    height, width = frame.shape[:2]
    display_width = min(max_width, width)
    scale = width / display_width if display_width else 1.0
    display_height = int(round(height / scale))
    image = frame_to_pil(frame).resize((display_width, display_height))
    scale_x = width / display_width
    scale_y = height / display_height
    return image, scale_x, scale_y


def calibration_signature(calibration: ArenaCalibration | None) -> str:
    if calibration is None:
        return "no_calibration"
    calibration_text = str(calibration.to_dict()).encode("utf-8")
    return hashlib.sha256(calibration_text).hexdigest()


def results_signature(
    video_signature: str,
    calibration: ArenaCalibration,
    tracker_config: TrackingConfig,
    use_smoothed_centroid: bool,
    fps_for_timing: float,
    assignment_point_mode: str,
) -> str:
    text = "|".join(
        [
            video_signature,
            calibration_signature(calibration),
            str(tracker_config),
            str(use_smoothed_centroid),
            str(fps_for_timing),
            assignment_point_mode,
        ]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_sidebar_help() -> None:
    st.sidebar.header("How To Use This App")
    st.sidebar.markdown(
        """
1. Upload one rat video, or create the synthetic demo video.
2. Draw the apparatus.
3. Click **Run analysis**.
4. Review the table, CSV files, and optional annotated video.

The easiest setup is **one arena box split into thirds**.
"""
    )


def show_metadata(metadata) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Width", f"{metadata.width}px")
    col2.metric("Height", f"{metadata.height}px")
    col3.metric("FPS used", f"{metadata.fps:.2f}")
    col4.metric("Duration", f"{metadata.duration_seconds:.2f}s")
    for note in metadata.notes:
        st.info(note)


def calibration_from_canvas(
    mode: str,
    canvas_json,
    metadata,
    scale_x: float,
    scale_y: float,
    boundary_margin_px: float,
) -> ArenaCalibration | None:
    if not canvas_json or not canvas_json.get("objects"):
        return None
    if mode == "Split one arena box into left / center / right (Recommended)":
        return build_calibration_from_canvas_box(
            canvas_json=canvas_json,
            frame_width=metadata.width,
            frame_height=metadata.height,
            image_scale_x=scale_x,
            image_scale_y=scale_y,
            boundary_margin_px=boundary_margin_px,
        )
    return build_calibration_from_canvas_polygons(
        canvas_json=canvas_json,
        frame_width=metadata.width,
        frame_height=metadata.height,
        image_scale_x=scale_x,
        image_scale_y=scale_y,
        boundary_margin_px=boundary_margin_px,
    )


def analysis_output_dir(video_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return ensure_directory(RESULTS_DIR / f"{video_path.stem}_{stamp}")


def main() -> None:
    st.set_page_config(page_title="CPP Rat Behavior Analyzer", layout="wide")
    ensure_session_state()
    render_sidebar_help()

    st.title("Three-Chamber CPP Rat Behavior Analyzer")
    st.write(
        "This local app measures how long one rat spends in the left, center, and right chambers of a three-chamber CPP box."
    )
    st.write(
        "You upload a fixed-camera video, draw the chamber layout on the first frame, and the app tracks the rat body centroid across the video."
    )

    st.subheader("Step 1: Pick A Video")
    uploaded_file = st.file_uploader(
        "Upload one top-down or near top-down video",
        type=["mp4", "mov", "avi", "m4v"],
        help="If you are new to the app, try the synthetic demo first. It gives you a fake practice video.",
    )

    demo_col, clear_col = st.columns([1, 1])
    if demo_col.button("Create demo video for practice"):
        demo_files = generate_synthetic_cpp_video(DEMO_DIR)
        load_video_into_session(Path(demo_files["video_path"]), signature=f"demo::{demo_files['video_path']}")
        st.success("Practice video created. The app loaded it for you.")

    if clear_col.button("Forget current analysis"):
        st.session_state["video_path"] = None
        st.session_state["video_signature"] = None
        clear_run_state()
        st.rerun()

    if uploaded_file is not None:
        signature = f"{uploaded_file.name}-{uploaded_file.size}"
        saved_video = save_uploaded_video(uploaded_file, UPLOAD_DIR)
        load_video_into_session(saved_video, signature=signature)

    if not st.session_state.get("video_path"):
        st.info("Upload a video or click the demo button to begin.")
        return

    video_path = Path(st.session_state["video_path"])
    metadata = get_video_metadata(video_path)
    first_frame = read_first_frame(video_path)
    show_metadata(metadata)

    st.subheader("Step 1B: Time Conversion")
    st.write("If your total chamber time looks too long or too short, the most common reason is incorrect FPS metadata in the video file.")
    use_manual_fps = st.checkbox(
        "Use a manual FPS value for timing",
        value=False,
        help="Only change this if the reported total time obviously does not match the real video length.",
    )
    fps_for_timing = float(metadata.fps)
    if use_manual_fps:
        fps_for_timing = float(
            st.number_input(
                "FPS to use for converting frames into seconds",
                min_value=0.1,
                max_value=240.0,
                value=float(metadata.fps),
                step=0.1,
                help="Example: if the file says 15 FPS but the recording system actually used 30 FPS, type 30 here.",
            )
        )
        st.info(f"The app will use {fps_for_timing:.3f} FPS for the time calculations.")
    else:
        st.caption(f"The app will use the video's reported FPS: {metadata.fps:.3f}")

    frame_image, scale_x, scale_y = resize_for_canvas(first_frame)
    st.subheader("Step 2: Define The Three Chambers")
    st.write("The app uses your drawing on the first frame to decide where the left, center, and right chambers are.")

    boundary_margin_px = st.number_input(
        "Optional neutral boundary margin in pixels",
        min_value=0,
        max_value=50,
        value=0,
        help="Use 0 for the simplest rule. If you enter a value larger than 0, frames very close to shared borders can be labeled as boundary.",
    )

    mode = st.radio(
        "Choose the chamber setup method",
        options=[
            "Split one arena box into left / center / right (Recommended)",
            "Draw 3 chamber rectangles manually (Easy)",
            "Draw 3 chamber polygons manually (Advanced)",
        ],
        horizontal=False,
    )

    if mode == "Split one arena box into left / center / right (Recommended)":
        drawing_mode = "rect"
        st.caption("Draw one rectangle around the full apparatus. The app will split it into 3 equal-width chambers.")
    elif mode == "Draw 3 chamber rectangles manually (Easy)":
        drawing_mode = "rect"
        st.caption("Draw 3 separate rectangles, one per chamber. Draw left first if you want, but the app will still sort them from left to right automatically.")
    else:
        drawing_mode = "polygon"
        st.caption("Draw exactly 3 polygons. This is the least beginner-friendly mode because the drawing widget can be finicky.")
        st.info(
            "Polygon tip: click to add corner points, then double-click near the end to finish one shape. "
            "If that feels frustrating, switch to 'Draw 3 chamber rectangles manually (Easy)'."
        )

    reset_col, image_col = st.columns([1, 4])
    if reset_col.button("Clear drawing"):
        clear_run_state()
        st.rerun()

    canvas_result = image_col.empty()
    canvas = canvas_result.container()
    with canvas:
        canvas_data = st_canvas(
            fill_color="rgba(255, 0, 0, 0.12)",
            stroke_width=3,
            stroke_color="#ff5a36",
            background_image=frame_image,
            update_streamlit=True,
            height=frame_image.height,
            width=frame_image.width,
            drawing_mode=drawing_mode,
            point_display_radius=5,
            key=f"canvas_{st.session_state['canvas_reset_counter']}_{drawing_mode}_{video_path.stem}",
        )

    calibration: ArenaCalibration | None = None
    if canvas_data.json_data and canvas_data.json_data.get("objects"):
        try:
            calibration = calibration_from_canvas(
                mode=mode,
                canvas_json=canvas_data.json_data,
                metadata=metadata,
                scale_x=scale_x,
                scale_y=scale_y,
                boundary_margin_px=float(boundary_margin_px),
            )
            preview = draw_calibration_overlay(first_frame.copy(), calibration)
            st.success("Chamber drawing loaded.")
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption="Calibration preview. This is the layout the app will use.",
                use_container_width=True,
            )
        except Exception as error:  # noqa: BLE001
            st.warning(f"Drawing not ready yet: {error}")

    st.subheader("Step 3: Run The Tracking")
    st.write("If you are unsure which settings to change, keep the defaults. They were chosen to be simple and safe.")
    assignment_point_mode = st.selectbox(
        "What should count as the rat's chamber position?",
        options=[
            "Head-and-shoulders proxy (Recommended for CPP boundary scoring)",
            "Smoothed body centroid",
            "Raw body centroid",
        ],
        help=(
            "The first option tries to place the chamber-assignment point toward the front of the rat's body, "
            "using recent motion as a simple estimate of heading. This better matches a head-and-shoulders rule, "
            "but it is still an approximation, not full pose tracking."
        ),
    )
    use_smoothed_centroid = assignment_point_mode == "Smoothed body centroid"
    if assignment_point_mode == "Head-and-shoulders proxy (Recommended for CPP boundary scoring)":
        st.caption("This mode uses a single front-of-body proxy point, not two chamber labels. It still assigns only one chamber per frame.")
    elif assignment_point_mode == "Smoothed body centroid":
        st.caption("This mode uses the body centroid with light smoothing to reduce flicker near borders.")
    else:
        st.caption("This mode uses the raw body centroid with no smoothing.")
    export_annotated = st.checkbox("Create annotated output video", value=True)
    draw_trajectory = st.checkbox("Draw trajectory overlay in annotated video", value=True)

    suggested_min_area = max(80, int(metadata.width * metadata.height * 0.0003))
    with st.expander("Optional advanced tracking settings"):
        min_contour_area = st.number_input(
            "Minimum contour area (pixels)",
            min_value=20,
            max_value=max(5000, suggested_min_area * 10),
            value=suggested_min_area,
            help="If the tracker grabs tiny dust-like blobs, raise this number a little.",
        )
        diff_threshold = st.slider(
            "Background difference threshold",
            min_value=5,
            max_value=80,
            value=28,
            help="Higher numbers make the tracker stricter. Lower numbers make it more sensitive.",
        )
        frame_diff_threshold = st.slider(
            "Frame-to-frame motion threshold",
            min_value=5,
            max_value=60,
            value=16,
            help="This helps when the rat moves quickly.",
        )
        max_jump_px = st.slider(
            "Maximum expected centroid jump per frame",
            min_value=20,
            max_value=300,
            value=150,
            help="If the tracker suddenly jumps across the arena, lowering this can help.",
        )
        smoothing_alpha = st.slider(
            "Smoothing strength",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            help="Higher means the tracker reacts faster. Lower means smoother movement.",
        )

    tracker_config = TrackingConfig(
        min_contour_area=float(min_contour_area),
        diff_threshold=int(diff_threshold),
        frame_diff_threshold=int(frame_diff_threshold),
        max_jump_px=float(max_jump_px),
        smoothing_alpha=float(smoothing_alpha),
    )

    if st.button("Run analysis", type="primary", disabled=calibration is None):
        if calibration is None:
            st.error("Please draw the chambers before running the analysis.")
        else:
            run_signature = results_signature(
                video_signature=st.session_state["video_signature"],
                calibration=calibration,
                tracker_config=tracker_config,
                use_smoothed_centroid=use_smoothed_centroid,
                fps_for_timing=fps_for_timing,
                assignment_point_mode=assignment_point_mode,
            )
            progress_bar = st.progress(0, text="Starting tracking...")

            def on_progress(current: int, total: int) -> None:
                if total <= 0:
                    progress_bar.progress(0, text="Tracking...")
                    return
                fraction = min(current / total, 1.0)
                progress_bar.progress(fraction, text=f"Tracking frames: {current}/{total}")

            with st.spinner("Tracking the rat and measuring chamber occupancy..."):
                tracker = SingleRatTracker(config=tracker_config)
                tracking_df = tracker.track_video(
                    video_path=video_path,
                    arena_mask=calibration.arena_mask(),
                    progress_callback=on_progress,
                    fps_override=fps_for_timing,
                )
                if assignment_point_mode == "Head-and-shoulders proxy (Recommended for CPP boundary scoring)":
                    analysis_point_mode = "head_shoulders"
                elif assignment_point_mode == "Raw body centroid":
                    analysis_point_mode = "centroid"
                else:
                    analysis_point_mode = "smoothed_centroid"
                analysis_bundle = create_analysis_bundle(
                    tracking_df=tracking_df,
                    calibration=calibration,
                    fps=fps_for_timing,
                    use_smoothed_centroid=use_smoothed_centroid,
                    boundary_margin_px=float(boundary_margin_px),
                    assignment_point_mode=analysis_point_mode,
                )

                output_dir = analysis_output_dir(video_path)
                tracking_csv = export_dataframe_csv(tracking_df, output_dir / "tracking_raw.csv")
                per_frame_csv = export_dataframe_csv(analysis_bundle.per_frame, output_dir / "per_frame_assignments.csv")
                summary_csv = export_dataframe_csv(analysis_bundle.summary, output_dir / "summary.csv")
                qc_csv = export_dataframe_csv(analysis_bundle.qc_metrics, output_dir / "qc_metrics.csv")
                warnings_txt = export_warnings_text(analysis_bundle.warnings, output_dir / "warnings.txt")

                annotated_video_path = None
                if export_annotated:
                    annotated_video_path = write_annotated_video(
                        input_video_path=video_path,
                        output_video_path=output_dir / "annotated_output.mp4",
                        per_frame_df=analysis_bundle.per_frame,
                        calibration=calibration,
                        draw_trajectory=draw_trajectory,
                    )

            progress_bar.progress(1.0, text="Analysis complete.")
            st.session_state["analysis_results"] = {
                "signature": run_signature,
                "output_dir": str(output_dir),
                "summary": analysis_bundle.summary,
                "qc_metrics": analysis_bundle.qc_metrics,
                "warnings": analysis_bundle.warnings,
                "fps_for_timing": fps_for_timing,
                "assignment_point_mode": assignment_point_mode,
                "tracking_csv": str(tracking_csv),
                "per_frame_csv": str(per_frame_csv),
                "summary_csv": str(summary_csv),
                "qc_csv": str(qc_csv),
                "warnings_txt": str(warnings_txt),
                "annotated_video": str(annotated_video_path) if annotated_video_path else None,
                "per_frame_preview": analysis_bundle.per_frame.head(300),
            }

    results = st.session_state.get("analysis_results")
    if not results:
        return

    st.subheader("Step 4: Read The Results")
    st.write("The table below shows how much time the rat spent in each chamber.")
    left_center_right = results["summary"][results["summary"]["chamber"].isin(["left", "center", "right"])]
    chamber_seconds_total = float(left_center_right["seconds"].sum())
    st.caption(
        f"Timing used {results['fps_for_timing']:.3f} FPS. "
        f"Left+center+right chamber time totals {chamber_seconds_total:.3f} seconds."
    )
    st.caption(f"Chamber assignment rule used: {results['assignment_point_mode']}")
    st.dataframe(results["summary"], use_container_width=True)

    st.write("Quality-control metrics help you judge whether the tracker looked stable.")
    st.dataframe(results["qc_metrics"], use_container_width=True)

    warnings = results.get("warnings", [])
    if warnings:
        st.warning("Warnings found:")
        for warning in warnings:
            st.write(f"- {warning}")
    else:
        st.success("No major QC warnings were raised.")

    if results.get("annotated_video"):
        st.subheader("Annotated Output Video")
        st.video(results["annotated_video"])

    st.subheader("Download Files")
    download_col1, download_col2, download_col3, download_col4 = st.columns(4)

    summary_path = Path(results["summary_csv"])
    per_frame_path = Path(results["per_frame_csv"])
    qc_path = Path(results["qc_csv"])
    tracking_path = Path(results["tracking_csv"])

    download_col1.download_button(
        "Download summary CSV",
        data=summary_path.read_bytes(),
        file_name=summary_path.name,
        mime="text/csv",
    )
    download_col2.download_button(
        "Download per-frame CSV",
        data=per_frame_path.read_bytes(),
        file_name=per_frame_path.name,
        mime="text/csv",
    )
    download_col3.download_button(
        "Download QC CSV",
        data=qc_path.read_bytes(),
        file_name=qc_path.name,
        mime="text/csv",
    )
    download_col4.download_button(
        "Download raw tracking CSV",
        data=tracking_path.read_bytes(),
        file_name=tracking_path.name,
        mime="text/csv",
    )

    if results.get("annotated_video"):
        video_path = Path(results["annotated_video"])
        st.download_button(
            "Download annotated MP4",
            data=video_path.read_bytes(),
            file_name=video_path.name,
            mime="video/mp4",
        )

    st.subheader("Per-Frame Preview")
    st.write("This is a short preview of the frame-by-frame table. The full file is in the CSV download.")
    st.dataframe(results["per_frame_preview"], use_container_width=True)
    st.caption(f"Saved output folder: {results['output_dir']}")


if __name__ == "__main__":
    main()
