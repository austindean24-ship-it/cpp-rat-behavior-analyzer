"""Landing page for the two independent rat-behavior analyzers."""

from __future__ import annotations

from urllib.parse import quote

import streamlit as st


def _uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote("".join(line.strip() for line in svg.splitlines()))


CPP_ART = _uri("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 190">
 <rect width="340" height="190" rx="16" fill="#F5FAFC"/>
 <path d="M45 45h250v100H45z" fill="#FFF" stroke="#7D9BAC" stroke-width="3"/>
 <path d="M128 45v100M212 45v100" stroke="#A9C1CD" stroke-width="3"/>
 <rect x="54" y="54" width="65" height="82" fill="#E8F1F5"/>
 <rect x="137" y="54" width="66" height="82" fill="#EDF6F4"/>
 <rect x="221" y="54" width="65" height="82" fill="#EAF0F7"/>
 <path d="M147 100c0-10 9-16 22-16 14 0 24 7 24 17s-10 17-24 17c-13 0-22-7-22-18z" fill="#7DA8B1" stroke="#3D6572" stroke-width="2"/>
 <circle cx="159" cy="82" r="5" fill="#D5E7E9" stroke="#3D6572" stroke-width="2"/>
 <path d="M191 102c17 2 28 9 39 21" fill="none" stroke="#3D6572" stroke-width="2" stroke-linecap="round"/>
 <path d="M45 153h250" stroke="#BDD0D9" stroke-width="1" stroke-dasharray="4 5"/>
</svg>
""")

EPM_ART = _uri("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 190">
 <rect width="340" height="190" rx="16" fill="#F5FAFC"/>
 <path d="M143 15h54v62h105v38H197v62h-54v-62H38V77h105z" fill="#F1F8F9" stroke="#5C93A0" stroke-width="3"/>
 <path d="M143 15h54v62h-54zM143 115h54v62h-54z" fill="#E5F5F0" stroke="#70AFA7" stroke-width="2"/>
 <path d="M38 77h105v38H38zM197 77h105v38H197z" fill="#E9F0F5" stroke="#809CAC" stroke-width="2"/>
 <path d="M155 96c0-6 5-10 12-10 8 0 14 5 14 11s-6 11-14 11c-7 0-12-5-12-12z" fill="#7DA8B1" stroke="#3D6572" stroke-width="2"/>
 <circle cx="163" cy="85" r="3" fill="#D5E7E9" stroke="#3D6572" stroke-width="1.5"/>
 <path d="M180 98c13 1 20 6 27 16" fill="none" stroke="#3D6572" stroke-width="2" stroke-linecap="round"/>
</svg>
""")


def render_home(cpp_page, epm_page) -> None:
    st.set_page_config(page_title="Rat Behavior Analysis Suite", layout="wide")
    st.markdown(f"""
<style>
.suite-hero {{padding:.6rem 0 1.15rem;margin:0 0 1.2rem;border-bottom:1px solid #D8E7E9}}
.suite-eyebrow {{color:#287B79;font-size:.79rem;font-weight:700;letter-spacing:.04em}}
.suite-hero h1 {{font-size:clamp(1.9rem,2.7vw,2.7rem);color:#173645;line-height:1.15;margin:.35rem 0 .5rem}}
.suite-hero p {{font-size:.98rem;color:#516B77;line-height:1.55;max-width:46rem;margin:0}}
.suite-section {{color:#173645;font-size:1.28rem;font-weight:750;margin:.2rem 0 .25rem}}
.suite-muted {{color:#617B86;font-size:.91rem;margin-bottom:1.1rem}}
.suite-card-art {{height:155px;background:#FFFFFF no-repeat center/contain;margin-bottom:.65rem}}
.suite-card-art--cpp {{background-image:url('{CPP_ART}')}}
.suite-card-art--epm {{background-image:url('{EPM_ART}')}}
.suite-card-kicker {{font-size:.76rem;color:#277B79;font-weight:700}}
.suite-card-title {{font-size:1.35rem;color:#173645;font-weight:750;margin:.25rem 0 .45rem}}
.suite-card-copy {{color:#526D79;font-size:.92rem;line-height:1.55;min-height:4rem}}
.suite-footnote {{border-top:1px solid #DBE7EA;color:#667D86;font-size:.88rem;margin-top:1.8rem;padding-top:1rem}}
@media(max-width:760px) {{.suite-hero {{padding:.35rem 0 .9rem}}}}
</style>
<section class="suite-hero">
 <div class="suite-eyebrow">Research video tools</div>
 <h1>Rat Behavior Analysis Suite</h1>
 <p>Choose an assay to map its apparatus, analyze a recording, and inspect frame-level evidence before using the results.</p>
</section>
<div class="suite-section">Choose an analyzer</div>
<div class="suite-muted">Each tool has its own calibration, scoring rules, and exports.</div>
""", unsafe_allow_html=True)

    cpp_col, epm_col = st.columns(2, gap="large")
    with cpp_col:
        with st.container(border=True):
            st.markdown("""
<div class="suite-card-art suite-card-art--cpp" role="img" aria-label="Three-chamber place preference apparatus"></div>
<div class="suite-card-kicker">Three-chamber assay</div>
<div class="suite-card-title">CPP Analyzer</div>
<div class="suite-card-copy">Score chamber occupancy and transitions in a conditioned place preference recording. Review the trajectory, quality checks, and video overlay.</div>
""", unsafe_allow_html=True)
            st.page_link(cpp_page, label="Open CPP Analyzer", use_container_width=True)
    with epm_col:
        with st.container(border=True):
            st.markdown("""
<div class="suite-card-art suite-card-art--epm" role="img" aria-label="Five-region elevated plus maze apparatus"></div>
<div class="suite-card-kicker">Five-region assay</div>
<div class="suite-card-title">EPM Analyzer</div>
<div class="suite-card-copy">Measure open-arm, closed-arm, and center time with provisional entries. Inspect inferred positions and flagged events against the video.</div>
""", unsafe_allow_html=True)
            st.page_link(epm_page, label="Open EPM Analyzer", use_container_width=True)
    st.markdown('<div class="suite-footnote">Use the sidebar to move between Home, CPP Analyzer, and EPM Analyzer at any time.</div>', unsafe_allow_html=True)
