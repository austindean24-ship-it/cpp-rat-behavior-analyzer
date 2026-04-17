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
    build_calibration_from_canvas_polygons,
    draw_calibration_overlay,
)
from tracker import SingleRatTracker, TrackingConfig


APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = ensure_directory(APP_DIR / "runtime_data")
UPLOAD_DIR = ensure_directory(RUNTIME_DIR / "uploads")
RESULTS_DIR = ensure_directory(RUNTIME_DIR / "results")
DEMO_DIR = ensure_directory(RUNTIME_DIR / "demo")
FIXED_TRACKING_CONFIG = TrackingConfig(
    min_contour_area=900.0,
    diff_threshold=40,
    frame_diff_threshold=10,
    max_jump_px=50.0,
    smoothing_alpha=0.10,
    roi_padding_px=0,
)


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


def resize_for_canvas(frame: np.ndarray, max_width: int = 900) -> tuple[Image.Image, float, float]:
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
  <g fill="none" stroke="#7A9CB0" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity="0.48">
    <path d="M54 148c0-34 32-58 84-58 23 0 44 5 61 16 18 11 30 28 30 48 0 36-32 59-74 59H118c-37 0-64-28-64-65z" fill="#DCEAF1" stroke="#9CB8C8"/>
    <circle cx="114" cy="89" r="12" fill="#DCEAF1"/>
    <circle cx="157" cy="89" r="12" fill="#DCEAF1"/>
    <path d="M71 154c-8 17-20 30-35 39"/>
    <path d="M116 212c-11 0-18-7-18-17"/>
    <path d="M164 212c-11 0-18-7-18-17"/>
    <path d="M247 136c53 6 94 26 124 58 26 27 49 33 74 29"/>
    <path d="M202 117c16-18 25-33 28-48 5-24 18-34 40-34 17 0 31 8 42 24"/>
    <circle cx="182" cy="117" r="6" fill="#5D879C" stroke="none"/>
  </g>
</svg>
"""
    )
    lab_grid = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240">
  <g fill="none" stroke="#D4E1EA" stroke-width="1" opacity="0.46">
    <path d="M0 40h240M0 80h240M0 120h240M0 160h240M0 200h240"/>
    <path d="M40 0v240M80 0v240M120 0v240M160 0v240M200 0v240"/>
  </g>
</svg>
"""
    )
    section_divider = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 28">
  <g fill="none" stroke="#9FB6C8" stroke-width="1.4" opacity="0.9">
    <path d="M0 14h82M158 14h82"/>
    <path d="M114 10c6-5 14-7 22-5 8 2 12 8 12 13 0 5-4 9-10 9-8 0-15-8-27-8-8 0-15 3-20 8"/>
    <circle cx="104" cy="14" r="2.4" fill="#9FB6C8" stroke="none"/>
  </g>
</svg>
"""
    )
    rat_runner = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 34">
  <g fill="none" stroke="#174A5C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10 20c0-6 6-10 15-10 4 0 8 1 11 3 3 2 5 6 5 9 0 6-5 10-13 10H22c-7 0-12-5-12-12z" fill="#3D8EA1" stroke="#174A5C"/>
    <circle cx="21" cy="9" r="2.8" fill="#B7D8E4" stroke="#174A5C"/>
    <circle cx="28" cy="9" r="2.8" fill="#B7D8E4" stroke="#174A5C"/>
    <path d="M39 20c8 0 14 3 18 9"/>
    <path d="M16 30c-2 2-4 3-7 3"/>
    <path d="M30 31c-2 2-4 3-7 3"/>
    <circle cx="34" cy="18" r="1.3" fill="#174A5C" stroke="none"/>
  </g>
</svg>
"""
    )
    chamber_target = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 38 38">
  <rect x="2" y="6" width="34" height="26" rx="6" fill="#F5FBFD" stroke="#8BA8BA" stroke-width="2"/>
  <path d="M14 8v22M24 8v22" stroke="#8BA8BA" stroke-width="2"/>
  <circle cx="19" cy="19" r="4.5" fill="#0F766E" opacity="0.88"/>
</svg>
"""
    )
    empty_state_art = svg_data_uri(
        """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 220">
  <rect width="360" height="220" rx="24" fill="#F5FAFC"/>
  <g opacity="0.65">
    <rect x="66" y="54" width="228" height="112" rx="18" fill="none" stroke="#A8BFCE" stroke-width="4"/>
    <path d="M142 54v112M218 54v112" stroke="#C0D2DE" stroke-width="3"/>
    <path d="M118 122c0-20 17-34 42-34 11 0 20 2 28 7 9 5 15 14 15 26 0 21-18 34-39 34h-19c-16 0-27-14-27-33z" fill="#DCEAF1" stroke="#7A9CB0" stroke-width="3"/>
    <circle cx="147" cy="86" r="7" fill="#DCEAF1" stroke="#7A9CB0" stroke-width="3"/>
    <circle cx="172" cy="86" r="7" fill="#DCEAF1" stroke="#7A9CB0" stroke-width="3"/>
    <path d="M203 116c34 4 59 18 79 42" fill="none" stroke="#7A9CB0" stroke-width="3" stroke-linecap="round"/>
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

.lab-quick-guide .lab-sidebar-title {{
    margin-bottom: 0.75rem;
}}

.lab-guide-section-title {{
    margin: 0.9rem 0 0.42rem 0;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(213, 225, 234, 0.85);
    color: var(--lab-accent-deep);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

.lab-guide-section-title--first {{
    margin-top: 0.15rem;
    padding-top: 0;
    border-top: 0;
}}

.lab-quick-guide-list {{
    font-size: 0.87rem;
    padding-left: 1.05rem;
}}

.lab-quick-guide-list li {{
    margin-bottom: 0.58rem;
}}

.lab-quick-guide-list strong,
.lab-sidebar-note strong {{
    color: var(--lab-text);
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

iframe[title*="st_canvas"] {{
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

:root {{
    --lab-border: #d5e1ea;
    --lab-shadow: 0 18px 40px rgba(15, 36, 54, 0.08);
    --lab-shadow-soft: 0 8px 24px rgba(15, 36, 54, 0.06);
    --lab-text: #172a3a;
    --lab-muted: #536879;
    --lab-muted-soft: #718699;
    --lab-accent: #0f766e;
    --lab-accent-deep: #0b4f6c;
}}

.stApp {{
    background:
        radial-gradient(circle at top left, #ffffff 0%, rgba(246, 250, 252, 0.98) 32%, rgba(238, 245, 249, 0.96) 100%),
        url("{lab_grid}");
    background-size: auto, 240px 240px;
}}

div[data-testid="stVerticalBlockBorderWrapper"],
.lab-sidebar-panel,
.lab-results-card,
div[data-testid="stMetric"],
.cpp-progress-card {{
    border-color: var(--lab-border);
    background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248, 252, 254, 0.9) 100%);
}}

section[data-testid="stSidebar"] {{
    background:
        linear-gradient(180deg, rgba(248,252,254,0.98) 0%, rgba(239,247,250,0.98) 100%),
        url("{lab_grid}");
    border-right: 1px solid rgba(181, 202, 216, 0.72);
}}

.lab-hero {{
    border-color: var(--lab-border);
    background:
        linear-gradient(140deg, rgba(255,255,255,0.96) 0%, rgba(244,250,252,0.95) 52%, rgba(232,244,247,0.94) 100%),
        url("{lab_grid}");
    box-shadow: 0 24px 56px rgba(15, 36, 54, 0.10);
}}

.lab-sidebar-kicker,
.lab-hero__eyebrow {{
    background: rgba(15, 118, 110, 0.11);
    color: var(--lab-accent-deep);
}}

.lab-sidebar-note {{
    background: rgba(15, 118, 110, 0.08);
    border-color: rgba(15, 118, 110, 0.20);
}}

.lab-badge,
.lab-hero__stat {{
    border-color: var(--lab-border);
    background: rgba(255,255,255,0.82);
}}

.lab-section-step {{
    background: rgba(11, 79, 108, 0.10);
    color: var(--lab-accent-deep);
}}

.lab-empty-state {{
    border-color: var(--lab-border);
    background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(246,250,252,0.94) 100%);
}}

div[data-testid="stFileUploaderDropzone"] {{
    background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(241,248,251,0.98) 100%);
    border-color: rgba(15, 118, 110, 0.48);
}}

div[data-testid="stFileUploaderDropzone"]:hover {{
    box-shadow: 0 12px 28px rgba(15,118,110,0.12);
}}

.stButton > button,
.stDownloadButton > button {{
    border-color: var(--lab-border);
    background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(239,247,250,0.98) 100%);
    box-shadow: 0 8px 20px rgba(15, 36, 54, 0.06);
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    border-color: rgba(15,118,110,0.62);
    box-shadow: 0 12px 24px rgba(15, 36, 54, 0.09);
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #0f766e 0%, #0b4f6c 100%);
}}

.stButton > button:focus,
.stDownloadButton > button:focus,
div[data-baseweb="input"] input:focus,
div[data-baseweb="select"] input:focus {{
    box-shadow: 0 0 0 4px rgba(15,118,110,0.18) !important;
}}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {{
    border-color: var(--lab-border) !important;
    background: rgba(255,255,255,0.92) !important;
}}

div[data-baseweb="input"] > div:hover,
div[data-baseweb="select"] > div:hover {{
    border-color: rgba(15,118,110,0.58) !important;
}}

div[data-testid="stDataFrame"],
iframe[title*="st_canvas"],
video {{
    border-color: var(--lab-border);
}}

.cpp-progress-track {{
    background: linear-gradient(90deg, #dceaf1 0%, #eef6f9 100%);
}}

.cpp-progress-fill {{
    background: linear-gradient(90deg, #4fb3aa 0%, #0f766e 52%, #0b4f6c 100%);
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
      Track one fixed-camera CPP video, review QC, and export results.
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
      <div class="lab-hero__stat-value">Upload, draw, run, export</div>
    </div>
    <div class="lab-hero__stat">
      <div class="lab-hero__stat-label">Video type</div>
      <div class="lab-hero__stat-value">Single rat, fixed apparatus</div>
    </div>
    <div class="lab-hero__stat">
      <div class="lab-hero__stat-label">Outputs</div>
      <div class="lab-hero__stat-value">Summary, QC, CSV, annotated MP4</div>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_section_header(step_label: str, title: str, description: str | None = None) -> None:
    description_html = f'<p class="lab-section-copy">{html.escape(description)}</p>' if description else ""
    st.markdown(
        f"""
<div class="lab-section-header">
  <div class="lab-section-step">{html.escape(step_label)}</div>
  <h2 class="lab-section-title">{html.escape(title)}</h2>
  {description_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
<section class="lab-empty-state">
  <div>
    <h3 class="lab-empty-state__title">Start a new run</h3>
    <p class="lab-empty-state__copy">
      Upload a CPP video or create the demo video, then draw one chamber box for each chamber on the first frame.
    </p>
  </div>
  <div class="lab-empty-state__art" aria-hidden="true"></div>
</section>
""",
        unsafe_allow_html=True,
    )


def render_results_highlights(results: dict, chamber_seconds_total: float) -> None:
    warning_count = len(results.get("warnings", []))
    st.markdown(
        f"""
<div class="lab-results-strip">
  <div class="lab-results-card">
    <div class="lab-results-card__label">Chamber time</div>
    <div class="lab-results-card__value lab-results-card__value--accent">{chamber_seconds_total:.3f} seconds</div>
  </div>
  <div class="lab-results-card">
    <div class="lab-results-card__label">Timing FPS</div>
    <div class="lab-results-card__value">{results['fps_for_timing']:.3f}</div>
  </div>
  <div class="lab-results-card">
    <div class="lab-results-card__label">Assignment rule</div>
    <div class="lab-results-card__value">{html.escape(results['assignment_point_mode'])}</div>
  </div>
  <div class="lab-results-card">
    <div class="lab-results-card__label">Warnings</div>
    <div class="lab-results-card__value">{warning_count}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_analysis_results(results: dict) -> None:
    """Render saved analysis outputs.

    This is used both during a run, after CSVs are ready, and after a rerun when
    results are loaded from session state.
    """

    left_center_right = results["summary"][results["summary"]["chamber"].isin(["left", "center", "right"])]
    chamber_seconds_total = float(left_center_right["seconds"].sum())
    render_state = "pending_video" if results.get("annotated_video_pending") else "final"
    widget_suffix = f"{results.get('signature', 'results')}_{render_state}"
    with st.container(border=True):
        render_section_header(
            "Step 4",
            "Results",
            "Summary, QC, exports, and preview.",
        )
        render_results_highlights(results, chamber_seconds_total)
        st.caption(
            f"Timing FPS: {results['fps_for_timing']:.3f}. "
            f"Left+center+right total: {chamber_seconds_total:.3f} seconds."
        )
        st.caption(f"Chamber assignment rule used: {results['assignment_point_mode']}")
        st.dataframe(results["summary"], use_container_width=True)

    with st.container(border=True):
        render_section_header(
            "QC",
            "Quality control",
        )
        st.dataframe(results["qc_metrics"], use_container_width=True)

        warnings = results.get("warnings", [])
        if warnings:
            st.warning("Warnings")
            for warning in warnings:
                st.write(f"- {warning}")
        else:
            st.success("No QC warnings.")

    with st.container(border=True):
        render_section_header(
            "Exports",
            "Downloads",
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
            key=f"summary_csv_{widget_suffix}",
        )
        download_col2.download_button(
            "Download per-frame CSV",
            data=per_frame_path.read_bytes(),
            file_name=per_frame_path.name,
            mime="text/csv",
            key=f"per_frame_csv_{widget_suffix}",
        )
        download_col3.download_button(
            "Download QC CSV",
            data=qc_path.read_bytes(),
            file_name=qc_path.name,
            mime="text/csv",
            key=f"qc_csv_{widget_suffix}",
        )
        download_col4.download_button(
            "Download raw tracking CSV",
            data=tracking_path.read_bytes(),
            file_name=tracking_path.name,
            mime="text/csv",
            key=f"tracking_csv_{widget_suffix}",
        )

        if results.get("annotated_video"):
            video_path = Path(results["annotated_video"])
            st.download_button(
                "Download annotated MP4",
                data=video_path.read_bytes(),
                file_name=video_path.name,
                mime="video/mp4",
                key=f"annotated_video_{widget_suffix}",
            )
        elif results.get("annotated_video_pending"):
            st.info("CSV outputs are ready. Annotated video is still being created.")

    if results.get("annotated_video"):
        with st.container(border=True):
            render_section_header(
                "Output Video",
                "Annotated video",
            )
            st.video(results["annotated_video"])
    elif results.get("annotated_video_pending"):
        with st.container(border=True):
            render_section_header(
                "Output Video",
                "Annotated video",
            )
            st.info("Annotated video export is running. This section will update when the MP4 is ready.")

    with st.container(border=True):
        render_section_header(
            "Preview",
            "Frame preview",
        )
        st.dataframe(results["per_frame_preview"], use_container_width=True)
        st.caption(f"Saved folder: {results['output_dir']}")


def render_sidebar_help() -> None:
    st.sidebar.markdown(
        """
<div class="lab-sidebar-panel lab-quick-guide">
  <div class="lab-sidebar-kicker">Quick guide</div>
  <div class="lab-sidebar-title">CPP Analyzer</div>

  <div class="lab-guide-section-title lab-guide-section-title--first">Setup</div>
  <ol class="lab-sidebar-list lab-quick-guide-list">
    <li><strong>Upload video.</strong> Pick a CPP video or create the demo video.</li>
    <li><strong>Check duration.</strong> A standard 15-minute session should be about 900 seconds.</li>
    <li><strong>Keep timing default.</strong> Leave manual FPS unchecked unless the video duration is clearly wrong.</li>
    <li><strong>Define chambers.</strong> Draw 3 rectangles on the same image, one per chamber.</li>
    <li><strong>Fit boxes tightly.</strong> Avoid extra floor or background when you can.</li>
    <li><strong>Check preview.</strong> Make sure all 3 boxes sit correctly before running analysis.</li>
  </ol>

  <div class="lab-guide-section-title">Analysis</div>
  <ol class="lab-sidebar-list lab-quick-guide-list" start="7">
    <li><strong>Use head-and-shoulders mode.</strong> Recommended for CPP boundary scoring.</li>
    <li><strong>Annotated video is optional.</strong> Leave it off for routine scoring; turn it on to inspect tracking.</li>
    <li><strong>Run analysis.</strong> Let processing finish before closing the page.</li>
    <li><strong>Read chamber time.</strong> Use the first results table for routine CPP scoring.</li>
  </ol>

  <div class="lab-sidebar-note">
    <strong>Color key:</strong> Left = white chamber, Center = middle chamber, Right = black chamber.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_changelog() -> None:
    with st.sidebar.expander("Changelog", expanded=False):
        st.markdown(
            """
**April 16, 2026**

**Exact chamber-only tracking**

- Removed the tracking tolerance around the drawn chamber boxes.
- The tracker now considers movement only inside the user-drawn chamber quadrilaterals.
- Shadows, reflections, hands, or room movement outside the drawn chambers are excluded from the tracking mask.
- Kept chamber scoring, CSV outputs, annotated video export, and the user workflow unchanged.

**Synthetic demo video**

- Replaced the jumpy synthetic demo movement with a smoother continuous path.
- Demo rat now steers between chamber targets with small randomized rat-like wandering instead of teleporting between segment centers.
- Updated the demo drawing to use a simple body, head, tail, and shadow silhouette.
- Demo ground truth now records the chamber where the generated rat actually appears on each frame.
- Creating a new demo now refreshes the loaded demo video even when it reuses the same filename.
- Demo video length is now 5 minutes for a more realistic practice run.

**April 12, 2026**

**Tracking and speed**

- Added chamber-cropped tracking. The tracker now focuses on the drawn chamber area instead of processing the full camera frame.
- The tracker now uses the exact drawn chamber mask with no extra outside padding.
- Continues saving full-video coordinates in the CSV files, so chamber scoring and annotated video overlays stay aligned with the original video.
- Keeps the chamber assignment rule unchanged: chamber occupancy is still based on the selected rat position point.

**Results workflow**

- Results tables now appear as soon as tracking, chamber assignment, and CSV export are complete.
- CSV downloads are shown before optional annotated video export finishes.
- If annotated video is enabled, the app keeps exporting the MP4 and updates the results area when the video is ready.
- This makes routine scoring faster to review because users do not have to wait for the optional MP4 before seeing chamber times.

**Fixed tracking configuration**

- Removed the Advanced Settings dropdown from the user interface.
- Hardcoded the lab-recommended tracker settings for more consistent usage:
- Minimum contour area: 900 px.
- Background difference threshold: 40.
- Frame-to-frame motion threshold: 10.
- Maximum expected jump: 50 px.
- Smoothing strength: 0.10.

**Guide and interface**

- Replaced the old sidebar workflow text with a compact quick guide based on the lab quick-guide document.
- Removed obsolete instructions about Advanced Settings.
- Updated the empty start screen so it matches the current 3-rectangle chamber drawing workflow.
- Added this collapsible changelog so version notes stay available without cluttering the main workflow.

**Visual design**

- Updated the app theme from the tan palette to the current slate/teal palette.
- Kept the subtle lab/rat visual style while improving contrast and readability.
- Kept the same user flow and controls while polishing the interface.

**Validation**

- Added a crop-specific automated test to confirm cropped tracking still reports coordinates in the original full-video coordinate system.
- Local test suite passed after these updates.
- Synthetic validation remained accurate after the chamber-cropping change.

**April 7, 2026**

**Chamber drawing**

- Restored the preferred one-image chamber setup.
- Users draw 3 click-and-drag rectangles on the same first-frame preview: left, center, and right.
- Removed the multi-preview chamber box adjustment workflow.

**Streamlit Cloud compatibility**

- Improved the drawing background handling so the uploaded video preview can appear behind the drawing canvas on Streamlit Cloud.
- Added static file serving support for canvas background images.
- Kept the local drawing workflow and deployed drawing workflow aligned as closely as possible.

**Tracking behavior**

- Added the head-and-shoulders proxy option for CPP boundary scoring.
- Preserved centroid-based options for comparison and troubleshooting.
- Added clearer low-confidence and QC reporting around tracking results.
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
    render_sidebar_changelog()
    render_lab_hero()

    with st.container(border=True):
        render_section_header(
            "Step 1",
            "Pick a video",
            "Upload a video or use the demo.",
        )
        uploaded_file = st.file_uploader(
            "Upload one top-down or near top-down video",
            type=["mp4", "mov", "avi", "m4v"],
            help="New here? Try the demo video first.",
        )

        demo_col, clear_col = st.columns([1, 1])
        if demo_col.button("Create demo video for practice"):
            demo_files = generate_synthetic_cpp_video(DEMO_DIR)
            demo_video_path = Path(demo_files["video_path"])
            load_video_into_session(
                demo_video_path,
                signature=f"demo::{demo_video_path}::{demo_video_path.stat().st_mtime_ns}",
            )
            st.success("Demo video created and loaded.")

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
            "Video details",
            "Check FPS, duration, and size.",
        )
        show_metadata(metadata)

    with st.container(border=True):
        render_section_header(
            "Step 1B",
            "Time conversion",
            "Override FPS only if timing looks wrong.",
        )
        use_manual_fps = st.checkbox(
            "Use a manual FPS value for timing",
            value=False,
            help="Use this only if the reported video time is clearly wrong.",
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
                    help="Example: if the file says 15 FPS but the camera really used 30 FPS, enter 30.",
                )
            )
            st.info(f"Using {fps_for_timing:.3f} FPS for timing.")
        else:
            st.caption(f"Using video FPS: {metadata.fps:.3f}")

    frame_image, scale_x, scale_y = resize_for_canvas(first_frame)
    with st.container(border=True):
        render_section_header(
            "Step 2",
            "Define chambers",
            "Draw one box for each chamber on the same image.",
        )

        boundary_margin_px = st.number_input(
            "Optional neutral boundary margin in pixels",
            min_value=0,
            max_value=50,
            value=0,
            help="0 means no boundary zone. Values above 0 can label border frames as boundary.",
        )
        st.caption("Draw exactly 3 rectangles: left, center, and right.")
        st.info("Use the rectangle tool and draw the 3 chamber boxes on this one image.")

        if st.button("Clear drawing"):
            clear_run_state()
            st.rerun()

        canvas_data = st_canvas(
            fill_color="rgba(15, 118, 110, 0.12)",
            stroke_width=3,
            stroke_color="#0f766e",
            background_image=frame_image,
            update_streamlit=True,
            height=frame_image.height,
            width=frame_image.width,
            drawing_mode="rect",
            point_display_radius=5,
            key=f"canvas_{st.session_state['canvas_reset_counter']}_{video_path.stem}",
        )

    calibration: ArenaCalibration | None = None
    if canvas_data.json_data and canvas_data.json_data.get("objects"):
        try:
            calibration = build_calibration_from_canvas_polygons(
                canvas_json=canvas_data.json_data,
                frame_width=metadata.width,
                frame_height=metadata.height,
                image_scale_x=scale_x,
                image_scale_y=scale_y,
                boundary_margin_px=float(boundary_margin_px),
            )
            preview = draw_calibration_overlay(first_frame.copy(), calibration)
            with st.container(border=True):
                render_section_header(
                    "Preview",
                    "Calibration preview",
                    "Review the chamber layout.",
                )
                st.success("Chamber layout ready.")
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption="Chamber layout preview",
                    use_container_width=True,
                )
        except Exception as error:  # noqa: BLE001
            st.warning(f"Drawing not ready yet: {error}")

    with st.container(border=True):
        render_section_header(
            "Step 3",
            "Run analysis",
            "Choose scoring and export settings.",
        )
        assignment_point_mode = st.selectbox(
            "What should count as the rat's chamber position?",
            options=[
                "Head-and-shoulders proxy (Recommended for CPP boundary scoring)",
                "Smoothed body centroid",
                "Raw body centroid",
            ],
            help=(
                "The first option shifts the scoring point toward the rat's front using recent motion. It is still an approximation."
            ),
        )
        use_smoothed_centroid = assignment_point_mode == "Smoothed body centroid"
        if assignment_point_mode == "Head-and-shoulders proxy (Recommended for CPP boundary scoring)":
            st.caption("Uses one front-of-body proxy point per frame.")
        elif assignment_point_mode == "Smoothed body centroid":
            st.caption("Uses a lightly smoothed body centroid.")
        else:
            st.caption("Uses the raw body centroid.")
        export_annotated = st.checkbox("Create annotated output video", value=True)
        draw_trajectory = st.checkbox("Draw trajectory overlay in annotated video", value=True)

    tracker_config = FIXED_TRACKING_CONFIG
    run_was_executed = False

    if st.button("Run analysis", type="primary", disabled=calibration is None):
        run_was_executed = True
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
            results_panel = st.empty()
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
                    f"Scanning chamber-cropped frame {current:,} of {total:,} to find the rat and estimate one chamber label.",
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
                "Sampling the drawn chamber area so the app can learn what the apparatus looks like without the rat.",
                "Tracking now ignores pixels outside the exact drawn chamber quadrilaterals.",
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

                results_payload = {
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
                    "annotated_video": None,
                    "annotated_video_pending": bool(export_annotated),
                    "per_frame_preview": analysis_bundle.per_frame.head(300),
                }
                st.session_state["analysis_results"] = results_payload
                update_status(
                    0.94,
                    "Results and CSV files are ready",
                    "The chamber-time tables and CSV downloads are available below while optional video export continues.",
                    f"Results saved in: {output_dir}",
                    log_message="Tables and CSV downloads are ready.",
                )
                with results_panel.container():
                    render_analysis_results(results_payload)

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
                    results_payload["annotated_video"] = str(annotated_video_path)
                    results_payload["annotated_video_pending"] = False
                    st.session_state["analysis_results"] = results_payload
                    update_status(
                        0.995,
                        "Annotated MP4 export finished",
                        "The output video has been written successfully.",
                        f"Saved file: {annotated_video_path.name}",
                        log_message="Annotated video export finished.",
                    )
                    with results_panel.container():
                        render_analysis_results(results_payload)
                else:
                    results_payload["annotated_video_pending"] = False
                    st.session_state["analysis_results"] = results_payload

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
            return

    results = st.session_state.get("analysis_results")
    if run_was_executed or not results:
        return
    render_analysis_results(results)


if __name__ == "__main__":
    main()
