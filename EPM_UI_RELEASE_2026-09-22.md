# Rat Behavior Analysis Suite UI release — 22 September 2026

- Home is now the default route. It links to the existing CPP Analyzer and EPM
  Analyzer; the Streamlit sidebar retains links to all three pages. CPP
  tracking, scoring, calibration, and export code were not changed.
- The EPM page uses embedded video FPS, the full video, and the smoothed body
  centroid. Its calibration-frame time control remains adjustable. Arm dwell
  is fixed at 0.5 seconds; center dwell remains 0.1 seconds.
- EPM has a maze diagram, region guide, empty state, rat progress panel, and
  creator footer. These are presentation-only components in `epm_visuals.py`.
- Home, CPP, and EPM use a restrained, flat visual style while retaining the
  apparatus drawings, CPP rat progress indicator, and visible QC guidance.

Re-scoring the prior 9,006-frame tracking CSV at the new arm threshold gave
3 open-arm entries, 11 closed-arm entries, 20 center entries, 928 flagged
inferred frames, and 7 possible transitions for review. The earlier one-second
run produced 2 / 9 / 17 entries. These are provisional and are **not** human
validated. Keep the raw video and exported event/per-frame files for review.

Local UI inspection verified the home, CPP, and EPM routes at narrow and
desktop widths, the EPM sidebar guide, and creator footer. The complete suite
with the original real-video fixture passed 52 tests with two known duplicate
CPP timing failures excluded; those failures reproduce on the prior main branch.
