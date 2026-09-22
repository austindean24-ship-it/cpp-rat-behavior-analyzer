"""EPM-only presentation elements; no changes to CPP visuals or scoring."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from urllib.parse import quote

import streamlit as st


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote("".join(line.strip() for line in svg.splitlines()))


RAT_ICON = _svg_uri("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 34">
 <g stroke="#245365" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M10 20c0-6 6-10 15-10 9 0 16 5 16 12 0 6-5 10-13 10H22c-7 0-12-5-12-12z" fill="#6D9EAD"/>
  <circle cx="20" cy="9" r="3" fill="#D6E6EA"/><circle cx="28" cy="9" r="3" fill="#D6E6EA"/>
  <path d="M39 20c8 0 14 3 19 9M16 30l-6 3M30 31l-6 3" fill="none"/>
  <circle cx="34" cy="18" r="1.4" fill="#245365" stroke="none"/>
 </g>
</svg>
""")

MAZE_ICON = _svg_uri("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
 <path d="M19 3h10v16h16v10H29v16H19V29H3V19h16z" fill="#F5FAFB" stroke="#467D8C" stroke-width="2"/>
 <circle cx="24" cy="24" r="3" fill="#2D827D"/>
</svg>
""")

HERO_ART = _svg_uri("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 300">
 <rect width="440" height="300" fill="#FFFFFF"/>
 <path d="M186 18h68v97h-68zM186 185h68v97h-68z" fill="#E7F6F3" stroke="#70AFA7" stroke-width="3"/>
 <path d="M22 115h164v70H22zM254 115h164v70H254z" fill="#E9F0F4" stroke="#7695A6" stroke-width="3"/>
 <rect x="186" y="115" width="68" height="70" fill="#F2F8FA" stroke="#4A7D8B" stroke-width="3"/>
 <path d="M186 115h68M186 185h68" stroke="#70AFA7" stroke-width="2" stroke-dasharray="5 4"/>
 <path d="M186 115v70M254 115v70" stroke="#7695A6" stroke-width="2" stroke-dasharray="5 4"/>
 <path d="M210 153c0-7 5-12 13-12 9 0 15 5 15 13 0 7-6 12-15 12-8 0-13-5-13-13z" fill="#6D9EAD" stroke="#245365" stroke-width="2"/>
 <circle cx="218" cy="140" r="4" fill="#D6E6EA" stroke="#245365" stroke-width="1.5"/>
 <path d="M237 155c14 2 22 8 27 19" fill="none" stroke="#245365" stroke-width="2" stroke-linecap="round"/>
 <circle cx="224" cy="150" r="1.5" fill="#245365"/>
 <text x="220" y="11" text-anchor="middle" fill="#3C827B" font-family="sans-serif" font-size="11" letter-spacing="2">OPEN</text>
 <text x="220" y="298" text-anchor="middle" fill="#3C827B" font-family="sans-serif" font-size="11" letter-spacing="2">OPEN</text>
 <text x="45" y="107" fill="#58788A" font-family="sans-serif" font-size="11" letter-spacing="2">CLOSED</text>
 <text x="337" y="107" fill="#58788A" font-family="sans-serif" font-size="11" letter-spacing="2">CLOSED</text>
</svg>
""")


def inject_epm_visual_theme() -> None:
    st.markdown(f"""
<style>
.epm-hero {{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(230px,.6fr);align-items:center;gap:1.2rem;padding:.35rem 0 .85rem;border-bottom:1px solid #D6E5E8;margin:0 0 .8rem}}
.epm-hero__eyebrow,.epm-visual-label {{color:#277B79;font-size:.76rem;font-weight:750;letter-spacing:.06em}}
.epm-hero h1 {{color:#173645;font-size:clamp(1.8rem,2.6vw,2.5rem);line-height:1.15;margin:.3rem 0 .55rem}}
.epm-hero p {{color:#4F6875;font-size:.96rem;line-height:1.5;max-width:46rem;margin:0}}
.epm-hero__art {{height:160px;background:#FFFFFF url('{HERO_ART}') no-repeat center/contain}}
.epm-region-guide {{display:flex;flex-wrap:wrap;gap:.3rem 1rem;margin:.1rem 0 1rem;padding-bottom:.65rem;border-bottom:1px solid #E3ECEE}}
.epm-region-guide__item {{display:flex;align-items:center;gap:.4rem;padding:.2rem .9rem .2rem 0;color:#526D79;font-size:.82rem}}
.epm-region-guide__mark {{width:5px;height:23px;border-radius:2px;background:#70AFA7;flex:none}}
.epm-region-guide__item:nth-child(2) .epm-region-guide__mark {{background:#7695A6}}
.epm-region-guide__item:nth-child(3) .epm-region-guide__mark {{background:#9DC5CB}}
.epm-region-guide strong {{display:block;color:#173645;font-size:.85rem}}
.epm-empty {{display:flex;align-items:center;gap:1rem;padding:.8rem 0;border-top:1px solid #E3ECEE;color:#4F6875;margin:.6rem 0 1.3rem}}
.epm-empty__art {{width:70px;height:60px;flex:none;background:url('{MAZE_ICON}') no-repeat center/50px}}
.epm-empty strong {{display:block;color:#173645;margin-bottom:.3rem}}
.epm-progress {{border:1px solid #D6E5E8;border-radius:7px;background:#FFFFFF;padding:.9rem 1rem;margin:0 0 1rem}}
.epm-progress__top {{display:flex;justify-content:space-between;gap:.7rem;color:#58717B;font-size:.88rem}}
.epm-progress__stage {{color:#173645;font-size:1.08rem;font-weight:750;margin:.5rem 0 .25rem}}
.epm-progress__detail {{color:#4F6875;font-size:.93rem}}
.epm-progress__tech {{color:#68808A;font-size:.82rem;margin:.3rem 0 1.2rem}}
.epm-progress__track {{height:16px;border-radius:999px;background:#E7F0F2;position:relative;margin:0 20px 1.2rem 0}}
.epm-progress__fill {{height:100%;border-radius:999px;background:#3D8F92;transition:width .2s ease}}
.epm-progress__rat {{position:absolute;top:-19px;width:42px;height:27px;background:url('{RAT_ICON}') no-repeat center/contain;transition:left .2s ease}}
.epm-progress__target {{position:absolute;right:-25px;top:-19px;width:32px;height:32px;background:url('{MAZE_ICON}') no-repeat center/contain}}
.epm-progress__notes {{color:#5F7782;font-size:.83rem;margin:0;padding-left:1.2rem}}
.epm-progress__notes li {{margin:.16rem 0}}
.epm-creator {{display:grid;grid-template-columns:110px minmax(0,1fr);gap:1rem;align-items:center;border-top:1px solid #D6E5E8;margin:2.2rem 0 .8rem;padding:1.4rem 0;color:#526C78;line-height:1.55}}
.epm-creator img {{width:105px;height:105px;object-fit:cover;border-radius:12px;border:1px solid #D6E5E8}}
.epm-creator h3 {{font-size:1.13rem;color:#173645;margin:.25rem 0}}
.epm-creator p {{margin:.3rem 0 0;font-size:.9rem}}
.epm-creator__meta {{font-size:.82rem;font-weight:650}}
@media(max-width:760px) {{.epm-hero {{grid-template-columns:1fr}}.epm-hero__art {{height:145px}}.epm-creator {{grid-template-columns:1fr}}}}
</style>
""", unsafe_allow_html=True)


def render_epm_hero() -> None:
    st.markdown("""
<section class="epm-hero">
 <div><div class="epm-hero__eyebrow">Behavioral analysis · Elevated plus maze</div>
 <h1>Elevated Plus Maze Analyzer</h1>
 <p>Map five maze regions, track one rat, and review provisional arm entries and occupancy alongside the annotated recording.</p></div>
 <div class="epm-hero__art" role="img" aria-label="Top-down diagram of a rat in a five-region elevated plus maze"></div>
</section>
<div class="epm-region-guide" aria-label="Maze region guide">
 <div class="epm-region-guide__item"><span class="epm-region-guide__mark"></span><span><strong>Open arms</strong>Exploration time and entries</span></div>
 <div class="epm-region-guide__item"><span class="epm-region-guide__mark"></span><span><strong>Closed arms</strong>Enclosed-arm time and entries</span></div>
 <div class="epm-region-guide__item"><span class="epm-region-guide__mark"></span><span><strong>Center</strong>Transitions between arms</span></div>
</div>
""", unsafe_allow_html=True)


def render_epm_empty_state() -> None:
    st.markdown("""
<div class="epm-empty"><div class="epm-empty__art" aria-hidden="true"></div>
<div><strong>Ready for one EPM recording</strong>Upload a video to choose a clear calibration frame and mark the five walking surfaces.</div></div>
""", unsafe_allow_html=True)


def render_epm_progress(container, fraction: float, stage: str, detail: str, technical: str,
                        elapsed_seconds: float, updates: list[str]) -> None:
    percent = max(0., min(100., fraction * 100.))
    rat_left = max(2., min(96., percent))
    elapsed = f"{int(elapsed_seconds // 60)}m {int(elapsed_seconds % 60):02d}s"
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in updates[-4:])
    container.markdown(f"""
<div class="epm-progress" role="status" aria-live="polite">
 <div class="epm-progress__top"><strong>{percent:.1f}% complete</strong><span>Elapsed: {elapsed}</span></div>
 <div class="epm-progress__stage">{html.escape(stage)}</div>
 <div class="epm-progress__detail">{html.escape(detail)}</div>
 <div class="epm-progress__tech">{html.escape(technical)}</div>
 <div class="epm-progress__track" role="progressbar" aria-valuenow="{percent:.1f}" aria-valuemin="0" aria-valuemax="100">
  <div class="epm-progress__fill" style="width:{percent:.2f}%"></div>
  <div class="epm-progress__rat" style="left:calc({rat_left:.2f}% - 18px)"></div>
  <div class="epm-progress__target"></div>
 </div>
 <ul class="epm-progress__notes">{notes}</ul>
</div>
""", unsafe_allow_html=True)


def render_epm_creator(asset_path: Path) -> None:
    photo = ("data:image/png;base64," + base64.b64encode(asset_path.read_bytes()).decode("ascii")) if asset_path.exists() else ""
    photo_html = f'<img src="{photo}" alt="Portrait of Austin Dean">' if photo else ""
    st.markdown(f"""
<footer class="epm-creator">
 <div>{photo_html}</div>
 <div><div class="epm-visual-label">About the creator</div><h3>Austin Dean</h3>
 <div class="epm-creator__meta">Biochemistry major, Leadership Studies minor | Fernandez Lab | Christopher Newport University</div>
 <p>Austin built these rat behavior tools to make video scoring easier to inspect and repeat. The EPM section pairs automated measurements with frame-level evidence so researchers can check entries, timing, and uncertain tracking before using the results.</p></div>
</footer>
""", unsafe_allow_html=True)
