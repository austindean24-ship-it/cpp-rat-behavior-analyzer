from __future__ import annotations

import html
import hashlib
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from canvas_utils import st_canvas_fixed as st_canvas

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


def svg_data_uri(svg: str) -> str:
    compact_svg = "".join(line.strip() for line in svg.splitlines())
    return f"data:image/svg+xml;utf8,{quote(compact_svg)}"


def inject_visual_theme() -> None:
    hero_silhouette = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 220">
  <g fill="none" stroke="#B58C74" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity="0.48">
    <path d="M54 148c0-34 32-58 84-58 23 0 44 5 61 16 18 11 30 28 30 48 0 36-32 59-74 59H118c-37 0-64-28-64-65z" fill="#E8D8CB" stroke="#C6A28B"/>
    <circle cx="114" cy="89" r="12" fill="#E8D8CB"/>
    <circle cx="157" cy="89" r="12" fill="#E8D8CB"/>
    <path d="M71 154c-8 17-20 30-35 39"/>
    <path d="M116 212c-11 0-18-7-18-17"/>
    <path d="M164 212c-11 0-18-7-18-17"/>
    <path d="M247 136c53 6 94 26 124 58 26 27 49 33 74 29"/>
    <path d="M202 117c16-18 25-33 28-48 5-24 18-34 40-34 17 0 31 8 42 24"/>
    <circle cx="182" cy="117" r="6" fill="#B58C74" stroke="none"/>
  </g>
</svg>
"""
    )
    lab_grid = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240">
  <g fill="none" stroke="#D6C7B7" stroke-width="1" opacity="0.46">
    <path d="M0 40h240M0 80h240M0 120h240M0 160h240M0 200h240"/>
    <path d="M40 0v240M80 0v240M120 0v240M160 0v240M200 0v240"/>
  </g>
</svg>
"""
    )
    section_divider = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 28">
  <g fill="none" stroke="#C7B6A6" stroke-width="1.4" opacity="0.9">
    <path d="M0 14h82M158 14h82"/>
    <path d="M114 10c6-5 14-7 22-5 8 2 12 8 12 13 0 5-4 9-10 9-8 0-15-8-27-8-8 0-15 3-20 8"/>
    <circle cx="104" cy="14" r="2.4" fill="#C7B6A6" stroke="none"/>
  </g>
</svg>
"""
    )
    rat_runner = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 34">
  <g fill="none" stroke="#73483C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10 20c0-6 6-10 15-10 4 0 8 1 11 3 3 2 5 6 5 9 0 6-5 10-13 10H22c-7 0-12-5-12-12z" fill="#B67763" stroke="#73483C"/>
    <circle cx="21" cy="9" r="2.8" fill="#D6B3A5" stroke="#73483C"/>
    <circle cx="28" cy="9" r="2.8" fill="#D6B3A5" stroke="#73483C"/>
    <path d="M39 20c8 0 14 3 18 9"/>
    <path d="M16 30c-2 2-4 3-7 3"/>
    <path d="M30 31c-2 2-4 3-7 3"/>
    <circle cx="34" cy="18" r="1.3" fill="#73483C" stroke="none"/>
  </g>
</svg>
"""
    )
    chamber_target = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 38 38">
  <rect x="2" y="6" width="34" height="26" rx="6" fill="#FFF8F1" stroke="#C0A48F" stroke-width="2"/>
  <path d="M14 8v22M24 8v22" stroke="#C0A48F" stroke-width="2"/>
  <circle cx="19" cy="19" r="4.5" fill="#B46C58" opacity="0.88"/>
</svg>
"""
    )
    empty_state_art = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 220">
  <rect width="360" height="220" rx="24" fill="#FBF6F1"/>
  <g opacity="0.65">
    <rect x="66" y="54" width="228" height="112" rx="18" fill="none" stroke="#CBB39F" stroke-width="4"/>
    <path d="M142 54v112M218 54v112" stroke="#D3BEAE" stroke-width="3"/>
    <path d="M118 122c0-20 17-34 42-34 11 0 20 2 28 7 9 5 15 14 15 26 0 21-18 34-39 34h-19c-16 0-27-14-27-33z" fill="#E9D8CB" stroke="#BE9A84" stroke-width="3"/>
    <circle cx="147" cy="86" r="7" fill="#E9D8CB" stroke="#BE9A84" stroke-width="3"/>
    <circle cx="172" cy="86" r="7" fill="#E9D8CB" stroke="#BE9A84" stroke-width="3"/>
    <path d="M203 116c34 4 59 18 79 42" fill="none" stroke="#BE9A84" stroke-width="3" stroke-linecap="round"/>
  </g>
</svg>
"""
    )
    st.markdown(
        f"""
<style>
:root {{
    --lab-border: #ded0c1;
    --lab-shadow: 0 18px 40px rgba(57, 47, 39, 0.08);
    --lab-shadow-soft: 0 8px 24px rgba(57, 47, 39, 0.06);
    --lab-text: #24353b;
    --lab-muted: #66777c;
    --lab-muted-soft: #86969b;
    --lab-accent: #b46c58;
    --lab-accent-deep: #8f5445;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, #fffdf8 0%, rgba(255, 253, 248, 0.95) 18%, rgba(244, 239, 232, 0.92) 38%, rgba(240, 233, 224, 0.92) 100%),
        url("{lab_grid}");
    background-size: auto, 240px 240px;
    color: var(--lab-text);
}}

[data-testid="stAppViewContainer"] > .main {{
    background: transparent;
}}

.block-container {{
    padding-top: 1.7rem;
    padding-bottom: 3.5rem;
    max-width: 1180px;
}}

h1, h2, h3, h4 {{
    color: var(--lab-text);
    letter-spacing: -0.025em;
}}

p, label, li {{
    color: var(--lab-text);
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid var(--lab-border);
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255,255,255,0.82) 0%, rgba(255,251,247,0.72) 100%);
    box-shadow: var(--lab-shadow);
    padding: 0.45rem 0.5rem;
    backdrop-filter: blur(10px);
}}

section[data-testid="stSidebar"] {{
    background:
        linear-gradient(180deg, rgba(251,247,241,0.98) 0%, rgba(246,240,232,0.98) 100%),
        url("{lab_grid}");
    background-size: auto, 220px 220px;
    border-right: 1px solid rgba(205, 185, 166, 0.7);
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.6rem;
}}

.lab-sidebar-panel {{
    position: relative;
    overflow: hidden;
    padding: 1.2rem 1rem 1rem 1rem;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(252,248,243,0.88) 100%);
    border: 1px solid rgba(207, 189, 172, 0.8);
    box-shadow: var(--lab-shadow-soft);
}}

.lab-sidebar-panel::after {{
    content: "";
    position: absolute;
    inset: auto -18px -18px auto;
    width: 120px;
    height: 80px;
    background: url("{hero_silhouette}") no-repeat center / contain;
    opacity: 0.14;
    pointer-events: none;
}}

.lab-sidebar-kicker {{
    display: inline-flex;
    border-radius: 999px;
    background: rgba(180,108,88,0.1);
    color: var(--lab-accent-deep);
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.36rem 0.72rem;
    margin-bottom: 0.8rem;
}}

.lab-sidebar-title {{
    font-size: 1.18rem;
    font-weight: 700;
    margin: 0 0 0.55rem 0;
}}

.lab-sidebar-copy {{
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--lab-muted);
    margin-bottom: 0.75rem;
}}

.lab-sidebar-list {{
    margin: 0;
    padding-left: 1.1rem;
    color: var(--lab-text);
}}

.lab-sidebar-list li {{
    margin-bottom: 0.52rem;
    line-height: 1.45;
}}

.lab-sidebar-note {{
    margin-top: 0.9rem;
    padding: 0.7rem 0.8rem;
    border-radius: 14px;
    background: rgba(127, 150, 132, 0.08);
    border: 1px solid rgba(127, 150, 132, 0.2);
    font-size: 0.86rem;
    color: var(--lab-muted);
}}

.lab-hero {{
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 1.6fr) minmax(250px, 0.8fr);
    gap: 1.4rem;
    padding: 1.5rem 1.5rem 1.35rem 1.5rem;
    border-radius: 28px;
    border: 1px solid rgba(207, 189, 172, 0.86);
    background:
        linear-gradient(140deg, rgba(255, 253, 248, 0.94) 0%, rgba(250, 244, 237, 0.92) 52%, rgba(244, 235, 227, 0.94) 100%),
        url("{lab_grid}");
    background-size: auto, 220px 220px;
    box-shadow: 0 24px 56px rgba(57, 47, 39, 0.11);
    margin-bottom: 1.15rem;
}}

.lab-hero::after {{
    content: "";
    position: absolute;
    right: -36px;
    bottom: -22px;
    width: 240px;
    height: 140px;
    background: url("{hero_silhouette}") no-repeat center / contain;
    opacity: 0.18;
    pointer-events: none;
}}

.lab-hero__eyebrow {{
    display: inline-flex;
    border-radius: 999px;
    background: rgba(180,108,88,0.1);
    color: var(--lab-accent-deep);
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.45rem 0.78rem;
    margin-bottom: 0.9rem;
}}

.lab-hero__title {{
    font-size: clamp(2rem, 3vw, 2.85rem);
    line-height: 1.02;
    margin: 0 0 0.85rem 0;
    max-width: 14ch;
}}

.lab-hero__copy {{
    font-size: 1rem;
    line-height: 1.7;
    color: var(--lab-muted);
    max-width: 70ch;
    margin: 0;
}}

.lab-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1rem;
}}

.lab-badge {{
    display: inline-flex;
    border-radius: 999px;
    padding: 0.42rem 0.78rem;
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(207, 189, 172, 0.86);
    color: var(--lab-text);
    font-size: 0.85rem;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(57, 47, 39, 0.05);
}}

.lab-hero__meta {{
    display: grid;
    gap: 0.8rem;
    align-content: start;
}}

.lab-hero__stat {{
    border-radius: 18px;
    padding: 0.95rem 1rem;
    background: rgba(255,255,255,0.74);
    border: 1px solid rgba(207, 189, 172, 0.86);
    box-shadow: 0 10px 22px rgba(57, 47, 39, 0.05);
}}

.lab-hero__stat-label {{
    font-size: 0.77rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--lab-muted-soft);
    margin-bottom: 0.32rem;
}}

.lab-hero__stat-value {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--lab-text);
}}

.lab-section-header {{
    position: relative;
    padding: 0.15rem 0 0.9rem 0;
    margin-bottom: 0.25rem;
}}

.lab-section-header::after {{
    content: "";
    display: block;
    width: 158px;
    height: 18px;
    margin-top: 0.95rem;
    background: url("{section_divider}") no-repeat left center / contain;
    opacity: 0.9;
}}

.lab-section-step {{
    display: inline-flex;
    border-radius: 999px;
    background: rgba(127, 150, 132, 0.12);
    color: #5e7562;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.38rem 0.72rem;
    margin-bottom: 0.75rem;
}}

.lab-section-title {{
    margin: 0;
    font-size: clamp(1.35rem, 2vw, 1.8rem);
}}

.lab-section-copy {{
    margin: 0.45rem 0 0 0;
    color: var(--lab-muted);
    line-height: 1.65;
    max-width: 76ch;
}}

.lab-empty-state {{
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(240px, 0.8fr);
    gap: 1.4rem;
    align-items: center;
    padding: 1.45rem 1.35rem;
    border-radius: 26px;
    border: 1px solid rgba(207, 189, 172, 0.86);
    background: linear-gradient(180deg, rgba(255,255,255,0.86) 0%, rgba(251,247,241,0.9) 100%);
    box-shadow: var(--lab-shadow);
    margin-top: 0.8rem;
}}

.lab-empty-state__art {{
    width: 100%;
    min-height: 220px;
    background: url("{empty_state_art}") no-repeat center / contain;
}}

.lab-empty-state__title {{
    font-size: 1.45rem;
    margin: 0 0 0.6rem 0;
}}

.lab-empty-state__copy {{
    color: var(--lab-muted);
    line-height: 1.7;
    margin: 0;
}}

.lab-empty-state__list {{
    margin: 0.95rem 0 0 0;
    padding-left: 1.05rem;
    color: var(--lab-text);
}}

.lab-empty-state__list li {{
    margin-bottom: 0.5rem;
}}

.lab-results-strip {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.8rem;
    margin: 0.2rem 0 1rem 0;
}}

.lab-results-card {{
    padding: 0.95rem 1rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(207, 189, 172, 0.86);
    box-shadow: var(--lab-shadow-soft);
}}

.lab-results-card__label {{
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--lab-muted-soft);
    margin-bottom: 0.32rem;
}}

.lab-results-card__value {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--lab-text);
}}

.lab-results-card__value--accent {{
    color: var(--lab-accent-deep);
}}

div[data-testid="stMetric"] {{
    border: 1px solid rgba(207, 189, 172, 0.82);
    border-radius: 18px;
    padding: 0.85rem 1rem;
    background: rgba(255,255,255,0.68);
    box-shadow: var(--lab-shadow-soft);
}}

div[data-testid="stMetricLabel"] p {{
    color: var(--lab-muted-soft);
    font-weight: 600;
}}

div[data-testid="stMetricValue"] {{
    color: var(--lab-text);
}}

div[data-testid="stFileUploaderDropzone"] {{
    background: linear-gradient(180deg, rgba(255, 250, 244, 0.96) 0%, rgba(248, 242, 235, 0.96) 100%);
    border: 1.5px dashed rgba(180,108,88,0.55);
    border-radius: 20px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}}

div[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--lab-accent);
    box-shadow: 0 12px 28px rgba(180,108,88,0.12);
    transform: translateY(-1px);
}}

.stButton > button,
.stDownloadButton > button {{
    border-radius: 14px;
    border: 1px solid rgba(207, 189, 172, 0.9);
    background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(245,238,230,0.96) 100%);
    color: var(--lab-text);
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(57, 47, 39, 0.06);
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    transform: translateY(-1px);
    border-color: rgba(180,108,88,0.76);
    box-shadow: 0 12px 24px rgba(57, 47, 39, 0.09);
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--lab-accent) 0%, var(--lab-accent-deep) 100%);
    color: white;
    border-color: transparent;
}}

.stButton > button:focus,
.stDownloadButton > button:focus,
div[data-baseweb="input"] input:focus,
div[data-baseweb="select"] input:focus {{
    outline: none;
    box-shadow: 0 0 0 4px rgba(180,108,88,0.18) !important;
}}

.stButton > button:disabled {{
    opacity: 0.55;
}}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {{
    border-radius: 14px !important;
    border-color: rgba(207, 189, 172, 0.95) !important;
    background: rgba(255,255,255,0.86) !important;
}}

div[data-baseweb="input"] > div:hover,
div[data-baseweb="select"] > div:hover {{
    border-color: rgba(180,108,88,0.7) !important;
}}

label[data-testid="stWidgetLabel"] p,
.stCheckbox label p,
.stRadio label p,
.stSelectbox label p {{
    color: var(--lab-text);
    font-weight: 600;
}}

div[data-testid="stExpander"] {{
    border-radius: 18px;
    border: 1px solid rgba(207, 189, 172, 0.86);
    overflow: hidden;
    background: rgba(255,255,255,0.56);
}}

div[data-testid="stExpander"] summary {{
    background: linear-gradient(180deg, rgba(250,244,237,0.9) 0%, rgba(246,240,232,0.9) 100%);
}}

div[data-testid="stAlert"] {{
    border-radius: 18px;
    border-width: 1px;
}}

div[data-testid="stDataFrame"] {{
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(207, 189, 172, 0.86);
    box-shadow: var(--lab-shadow-soft);
}}

div[data-testid="stCaptionContainer"] p {{
    color: var(--lab-muted);
    font-size: 0.86rem;
}}

iframe[title="streamlit_drawable_canvas.st_canvas"] {{
    border-radius: 20px;
    border: 1px solid rgba(207, 189, 172, 0.9);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.8), var(--lab-shadow-soft);
    background: rgba(255,255,255,0.7);
}}

video {{
    border-radius: 22px;
    border: 1px solid rgba(207, 189, 172, 0.9);
    box-shadow: var(--lab-shadow);
}}

.cpp-progress-card {{
    border: 1px solid rgba(207, 189, 172, 0.86);
    background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(250,245,239,0.92) 100%);
    border-radius: 22px;
    padding: 18px 18px 14px 18px;
    box-shadow: var(--lab-shadow);
    margin: 0 0 1rem 0;
}}

.cpp-progress-topline {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 10px;
    font-size: 0.93rem;
    color: var(--lab-muted);
}}

.cpp-progress-stage {{
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--lab-text);
    margin-bottom: 6px;
}}

.cpp-progress-detail {{
    font-size: 0.96rem;
    color: var(--lab-text);
    margin-bottom: 4px;
}}

.cpp-progress-tech {{
    font-size: 0.88rem;
    color: var(--lab-muted);
    margin-bottom: 14px;
}}

.cpp-progress-track {{
    position: relative;
    height: 18px;
    border-radius: 999px;
    background: linear-gradient(90deg, #eee3d6 0%, #f4ece3 100%);
    overflow: visible;
    margin-bottom: 16px;
}}

.cpp-progress-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #d49a83 0%, #b46c58 52%, #8f5445 100%);
    transition: width 0.2s ease;
}}

.cpp-progress-rat {{
    position: absolute;
    top: -18px;
    width: 38px;
    height: 24px;
    transition: left 0.2s ease;
    background: url("{rat_runner}") no-repeat center / contain;
}}

.cpp-progress-target {{
    position: absolute;
    right: -2px;
    top: -15px;
    width: 24px;
    height: 24px;
    background: url("{chamber_target}") no-repeat center / contain;
}}

.cpp-progress-updates-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--lab-muted);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.cpp-progress-updates {{
    margin: 0;
    padding-left: 18px;
    color: var(--lab-text);
    font-size: 0.9rem;
}}

.cpp-progress-updates li {{
    margin: 0 0 4px 0;
}}

@media (max-width: 980px) {{
    .lab-hero,
    .lab-empty-state,
    .lab-results-strip {{
        grid-template-columns: 1fr;
    }}

    .lab-hero__title {{
        max-width: none;
    }}
}}

@media (max-width: 720px) {{
    .block-container {{
        padding-top: 1rem;
    }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_lab_hero() -> None:
    st.markdown(
        """
<section class="lab-hero">
  <div>
    <div class="lab-hero__eyebrow">Internal Research Tool</div>
    <h1 class="lab-hero__title">Three-Chamber CPP Rat Behavior Analyzer</h1>
    <p class="lab-hero__copy">
      A polished local workflow for fixed-camera CPP videos, designed for day-to-day lab use with clean exports,
      quality-control summaries, and subtle research-minded guidance throughout the analysis flow.
    </p>
    <div class="lab-badges">
      <span class="lab-badge">Single-rat tracking</span>
      <span class="lab-badge">Three-chamber CPP</span>
      <span class="lab-badge">QC + CSV exports</span>
    </div>
  </div>
  <div class="lab-hero__meta">
    <div class="lab-hero__stat">
      <div class="lab-hero__stat-label">Workflow</div>
      <div class="lab-hero__stat-value">Upload, map chambers, track, review, export</div>
    </div>
    <div class="lab-hero__stat">
      <div class="lab-hero__stat-label">Designed for</div>
      <div class="lab-hero__stat-value">Fixed apparatus, single Sprague Dawley-style rat videos</div>
    </div>
    <div class="lab-hero__stat">
      <div class="lab-hero__stat-label">Visual tone</div>
      <div class="lab-hero__stat-value">Warm, clean, subtle lab aesthetic with zero workflow changes</div>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_section_header(step_label: str, title: str, description: str) -> None:
    st.markdown(
        f"""
<div class="lab-section-header">
  <div class="lab-section-step">{html.escape(step_label)}</div>
  <h2 class="lab-section-title">{html.escape(title)}</h2>
  <p class="lab-section-copy">{html.escape(description)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
<section class="lab-empty-state">
  <div>
    <h3 class="lab-empty-state__title">Ready for a new assay run</h3>
    <p class="lab-empty-state__copy">
      Upload a top-down CPP video or generate the synthetic demo to test the full interface. Nothing about the
      tracking workflow changes here; this is simply the cleanest place to begin.
    </p>
    <ul class="lab-empty-state__list">
      <li>Use a fixed-camera recording whenever possible.</li>
      <li>The simplest chamber setup is one arena box split into thirds.</li>
      <li>Local and Streamlit versions now share the same visual styling.</li>
    </ul>
  </div>
  <div class="lab-empty-state__art" aria-hidden="true"></div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_results_highlights(results: dict, chamber_seconds_total: float) -> None:
    warning_count = len(results.get("warnings", []))
    output_folder = Path(results["output_dir"]).name
    st.markdown(
        f"""
<div class="lab-results-strip">
  <div class="lab-results-card">
    <div class="lab-results-card__label">Combined chamber time</div>
    <div class="lab-results-card__value lab-results-card__value--accent">{chamber_seconds_total:.3f} seconds</div>
  </div>
  <div class="lab-results-card">
    <div class="lab-results-card__label">Timing FPS used</div>
    <div class="lab-results-card__value">{results['fps_for_timing']:.3f}</div>
  </div>
  <div class="lab-results-card">
    <div class="lab-results-card__label">Assignment rule</div>
    <div class="lab-results-card__value">{html.escape(results['assignment_point_mode'])}</div>
  </div>
  <div class="lab-results-card">
    <div class="lab-results-card__label">Warnings raised</div>
    <div class="lab-results-card__value">{warning_count} | {html.escape(output_folder)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_help() -> None:
    st.sidebar.markdown(
        """
<div class="lab-sidebar-panel">
  <div class="lab-sidebar-kicker">Lab workflow</div>
  <div class="lab-sidebar-title">How To Use This App</div>
  <p class="lab-sidebar-copy">
    The workflow stays exactly the same. The interface is just cleaner, calmer, and easier to scan during daily lab use.
  </p>
  <ol class="lab-sidebar-list">
    <li>Upload one rat video, or create the synthetic demo video.</li>
    <li>Draw the apparatus on the first frame.</li>
    <li>Click <strong>Run analysis</strong>.</li>
    <li>Review the tables, CSV files, and optional annotated video.</li>
  </ol>
  <div class="lab-sidebar-note">
    Best starting point: draw one arena box and let the app split it into left, center, and right chambers.
  </div>
</div>
""",
        unsafe_allow_html=True,
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


def format_elapsed(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_rat_progress(
    container,
    fraction: float,
    stage_title: str,
    detail_text: str,
    technical_text: str,
    elapsed_seconds: float,
    recent_updates: list[str],
) -> None:
    percent = max(0.0, min(100.0, fraction * 100.0))
    rat_left = max(2.0, min(96.0, percent))
    updates_html = "".join(f"<li>{html.escape(item)}</li>" for item in recent_updates[-4:])
    container.markdown(
        f"""
<div class="cpp-progress-card">
  <div class="cpp-progress-topline">
    <span><strong>{percent:.1f}% complete</strong></span>
    <span>Elapsed: {html.escape(format_elapsed(elapsed_seconds))}</span>
  </div>
  <div class="cpp-progress-stage">{html.escape(stage_title)}</div>
  <div class="cpp-progress-detail">{html.escape(detail_text)}</div>
  <div class="cpp-progress-tech">{html.escape(technical_text)}</div>
  <div class="cpp-progress-track">
    <div class="cpp-progress-fill" style="width: {percent:.2f}%"></div>
    <div class="cpp-progress-rat" style="left: calc({rat_left:.2f}% - 14px)"></div>
    <div class="cpp-progress-target"></div>
  </div>
  <div class="cpp-progress-updates-title">Latest status notes</div>
  <ul class="cpp-progress-updates">{updates_html}</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="CPP Rat Behavior Analyzer", layout="wide")
    ensure_session_state()
    inject_visual_theme()
    render_sidebar_help()
    render_lab_hero()

    with st.container(border=True):
        render_section_header(
            "Step 1",
            "Pick a video",
            "Upload one fixed-camera CPP recording or generate the synthetic demo. This section only changes presentation, not the upload workflow itself.",
        )
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
        render_empty_state()
        return

    video_path = Path(st.session_state["video_path"])
    metadata = get_video_metadata(video_path)
    first_frame = read_first_frame(video_path)
    with st.container(border=True):
        render_section_header(
            "Video Snapshot",
            "Check the recording details",
            "Confirm the dimensions, FPS, and duration before moving on. These values are unchanged from the original app, only presented more clearly.",
        )
        show_metadata(metadata)

    with st.container(border=True):
        render_section_header(
            "Step 1B",
            "Time conversion",
            "If your total chamber time looks too long or too short, the most common reason is incorrect FPS metadata in the video file.",
        )
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
    with st.container(border=True):
        render_section_header(
            "Step 2",
            "Define the three chambers",
            "Use the first frame as your calibration surface. The drawing behavior is unchanged; this section now just has cleaner hierarchy and more breathing room.",
        )

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
            with st.container(border=True):
                render_section_header(
                    "Preview",
                    "Calibration preview",
                    "This is the exact chamber layout the app will use for analysis. The preview is unchanged in function, only framed more clearly.",
                )
                st.success("Chamber drawing loaded.")
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption="Calibration preview. This is the layout the app will use.",
                    use_container_width=True,
                )
        except Exception as error:  # noqa: BLE001
            st.warning(f"Drawing not ready yet: {error}")

    with st.container(border=True):
        render_section_header(
            "Step 3",
            "Run the tracking pipeline",
            "The underlying behavior is unchanged. This panel simply organizes the same chamber-position choices, export toggles, and advanced settings into a cleaner control surface.",
        )
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
            progress_panel = st.empty()
            analysis_started_at = time.time()
            status_log: list[str] = []
            tracking_started_at: float | None = None
            annotation_started_at: float | None = None
            last_tracking_reported_frame = 0
            last_annotation_reported_frame = 0

            def update_status(
                fraction: float,
                stage_title: str,
                detail_text: str,
                technical_text: str,
                log_message: str | None = None,
            ) -> None:
                if log_message and (not status_log or status_log[-1] != log_message):
                    status_log.append(log_message)
                render_rat_progress(
                    container=progress_panel,
                    fraction=fraction,
                    stage_title=stage_title,
                    detail_text=detail_text,
                    technical_text=technical_text,
                    elapsed_seconds=time.time() - analysis_started_at,
                    recent_updates=status_log or ["Waiting to begin analysis."],
                )

            update_status(
                0.01,
                "Preparing the analysis run",
                "Checking your video, chamber drawing, and analysis settings before the heavy work begins.",
                f"Video file: {video_path.name} | Timing FPS: {fps_for_timing:.3f} | Chamber rule: {assignment_point_mode}",
                log_message="Starting a new analysis run.",
            )

            def on_progress(current: int, total: int) -> None:
                nonlocal tracking_started_at, last_tracking_reported_frame
                if total <= 0:
                    return
                if tracking_started_at is None:
                    tracking_started_at = time.time()
                if current != total and current - last_tracking_reported_frame < 120:
                    return
                last_tracking_reported_frame = current
                fraction = 0.10 + (0.60 * min(current / total, 1.0))
                tracking_elapsed = max(time.time() - tracking_started_at, 1e-6)
                frames_per_second = current / tracking_elapsed
                remaining_seconds = ((total - current) / frames_per_second) if frames_per_second > 0 else 0.0
                update_status(
                    fraction,
                    "Tracking the rat through the video",
                    f"Scanning frame {current:,} of {total:,} to find the rat and estimate one chamber label for that frame.",
                    (
                        f"Tracker speed: {frames_per_second:.1f} frames/sec | "
                        f"Estimated tracking time remaining: {format_elapsed(remaining_seconds)}"
                    ),
                    log_message="Main frame-by-frame tracking pass is running.",
                )

            def on_annotation_progress(current: int, total: int) -> None:
                nonlocal annotation_started_at, last_annotation_reported_frame
                if total <= 0:
                    return
                if annotation_started_at is None:
                    annotation_started_at = time.time()
                if current != total and current - last_annotation_reported_frame < 120:
                    return
                last_annotation_reported_frame = current
                fraction = 0.94 + (0.05 * min(current / total, 1.0))
                annotation_elapsed = max(time.time() - annotation_started_at, 1e-6)
                frames_per_second = current / annotation_elapsed
                remaining_seconds = ((total - current) / frames_per_second) if frames_per_second > 0 else 0.0
                update_status(
                    fraction,
                    "Exporting the annotated video",
                    f"Writing annotated frame {current:,} of {total:,} into the output MP4 file.",
                    (
                        f"Video writer speed: {frames_per_second:.1f} frames/sec | "
                        f"Estimated annotation time remaining: {format_elapsed(remaining_seconds)}"
                    ),
                    log_message="Annotated MP4 export is in progress.",
                )

            update_status(
                0.06,
                "Preparing the empty-arena background model",
                "Sampling representative frames so the app can learn what the apparatus looks like without the rat.",
                "This makes it easier to separate the moving rat from the static box and walls.",
                log_message="Preparing the background model used for motion-based tracking.",
            )

            try:
                tracker = SingleRatTracker(config=tracker_config)
                tracking_df = tracker.track_video(
                    video_path=video_path,
                    arena_mask=calibration.arena_mask(),
                    progress_callback=on_progress,
                    fps_override=fps_for_timing,
                )
                update_status(
                    0.74,
                    "Assigning each frame to a chamber",
                    "The tracker has finished finding the rat. Now the app is turning those points into left, center, right, boundary, or missing labels.",
                    "This stage applies the chamber rule you selected to every tracked frame.",
                    log_message="Tracking pass finished. Starting chamber assignment.",
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
                update_status(
                    0.80,
                    "Computing summaries and quality-control checks",
                    "Building the chamber time table, quality metrics, and warnings from the frame-by-frame results.",
                    "This includes success rate, missing-frame counts, contour-area stability, and border warnings.",
                    log_message="Summary statistics and QC metrics are being calculated.",
                )

                output_dir = analysis_output_dir(video_path)
                update_status(
                    0.83,
                    "Creating the output folder",
                    "Making a results folder so the app can save the CSV files and optional video.",
                    f"Output folder: {output_dir}",
                    log_message="Output folder created for this run.",
                )
                tracking_csv = export_dataframe_csv(tracking_df, output_dir / "tracking_raw.csv")
                update_status(
                    0.86,
                    "Saving the raw tracking CSV",
                    "Writing the first CSV file with the frame-by-frame tracker output before chamber scoring.",
                    f"Saved file: {tracking_csv.name}",
                    log_message="Saved raw tracking CSV.",
                )
                per_frame_csv = export_dataframe_csv(analysis_bundle.per_frame, output_dir / "per_frame_assignments.csv")
                update_status(
                    0.89,
                    "Saving the per-frame chamber assignments",
                    "Writing the detailed CSV that contains one row per frame with the chamber label and assignment point.",
                    f"Saved file: {per_frame_csv.name}",
                    log_message="Saved per-frame chamber assignment CSV.",
                )
                summary_csv = export_dataframe_csv(analysis_bundle.summary, output_dir / "summary.csv")
                update_status(
                    0.91,
                    "Saving the summary table",
                    "Writing the compact table with total time spent in each chamber.",
                    f"Saved file: {summary_csv.name}",
                    log_message="Saved chamber summary CSV.",
                )
                qc_csv = export_dataframe_csv(analysis_bundle.qc_metrics, output_dir / "qc_metrics.csv")
                update_status(
                    0.93,
                    "Saving quality-control metrics",
                    "Writing the QC table so you can inspect missing frames, low-confidence frames, and tracking stability.",
                    f"Saved file: {qc_csv.name}",
                    log_message="Saved QC metrics CSV.",
                )
                warnings_txt = export_warnings_text(analysis_bundle.warnings, output_dir / "warnings.txt")
                update_status(
                    0.94,
                    "Saving the warning notes",
                    "Writing a simple text file with any warnings raised during analysis.",
                    f"Saved file: {warnings_txt.name}",
                    log_message="Saved warnings text file.",
                )

                annotated_video_path = None
                if export_annotated:
                    update_status(
                        0.945,
                        "Starting annotated MP4 export",
                        "Opening the original video again so the app can draw the chamber lines, tracking point, and optional trajectory overlay.",
                        "This is usually the slowest part after tracking finishes.",
                        log_message="Starting annotated video export.",
                    )
                    annotated_video_path = write_annotated_video(
                        input_video_path=video_path,
                        output_video_path=output_dir / "annotated_output.mp4",
                        per_frame_df=analysis_bundle.per_frame,
                        calibration=calibration,
                        draw_trajectory=draw_trajectory,
                        progress_callback=on_annotation_progress,
                    )
                    update_status(
                        0.995,
                        "Annotated MP4 export finished",
                        "The output video has been written successfully.",
                        f"Saved file: {annotated_video_path.name}",
                        log_message="Annotated video export finished.",
                    )

                update_status(
                    1.0,
                    "Analysis complete",
                    "Everything finished successfully. The result tables and download buttons are ready below.",
                    f"Results saved in: {output_dir}",
                    log_message="Analysis complete. Results are ready.",
                )
            except Exception as error:  # noqa: BLE001
                update_status(
                    1.0,
                    "Analysis stopped because of an error",
                    "The app hit a problem before finishing this run.",
                    f"Error details: {error}",
                    log_message="The analysis run stopped because of an error.",
                )
                raise

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

    left_center_right = results["summary"][results["summary"]["chamber"].isin(["left", "center", "right"])]
    chamber_seconds_total = float(left_center_right["seconds"].sum())
    with st.container(border=True):
        render_section_header(
            "Step 4",
            "Read the results",
            "Everything below uses the same analysis outputs as before, now grouped into clearer summary, QC, export, and preview panels.",
        )
        render_results_highlights(results, chamber_seconds_total)
        st.caption(
            f"Timing used {results['fps_for_timing']:.3f} FPS. "
            f"Left+center+right chamber time totals {chamber_seconds_total:.3f} seconds."
        )
        st.caption(f"Chamber assignment rule used: {results['assignment_point_mode']}")
        st.dataframe(results["summary"], use_container_width=True)

    with st.container(border=True):
        render_section_header(
            "QC",
            "Quality-control view",
            "Use this panel to judge whether the tracker looked stable and whether any warnings need review before you trust the exported numbers.",
        )
        st.dataframe(results["qc_metrics"], use_container_width=True)

        warnings = results.get("warnings", [])
        if warnings:
            st.warning("Warnings found:")
            for warning in warnings:
                st.write(f"- {warning}")
        else:
            st.success("No major QC warnings were raised.")

    if results.get("annotated_video"):
        with st.container(border=True):
            render_section_header(
                "Output Video",
                "Annotated output video",
                "Preview the exported MP4 with chamber outlines, tracking points, and any enabled overlay styling.",
            )
            st.video(results["annotated_video"])

    with st.container(border=True):
        render_section_header(
            "Exports",
            "Download files",
            "The same exports are available as before. They are simply grouped into a cleaner, easier-to-scan download area.",
        )
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

    with st.container(border=True):
        render_section_header(
            "Preview",
            "Per-frame preview",
            "This is still the same preview table, with the full detailed dataset available through the exported CSV file.",
        )
        st.dataframe(results["per_frame_preview"], use_container_width=True)
        st.caption(f"Saved output folder: {results['output_dir']}")


if __name__ == "__main__":
    main()
