# EPM controlled pilot handoff — 21 September 2026

## Status and measurement contract

This branch changes EPM code only. The six-column summary remains, but entry
counts are **provisional centroid crossings**, not head, shoulder, or paw
measurements. Smoothed body centroid is the pilot default for occupancy and
entries. The raw centroid and motion-derived front proxy remain labeled
comparison modes. Missing or conflicting body evidence contributes unknown
time. A possible transition is exported for review and never added to the
six-column counts. No comprehensive human event labels are available.

The original input and exact saved calibration were matched by SHA-256, size,
name, frame count (9,006), and FPS (30.003). Video, calibration JSON, CSV
exports, and diagnostic frames remain local and are **not in Git**.

## Independent review of the proposed ZIP

| Proposed component | Counterexample / evidence | Decision |
| --- | --- | --- |
| Dark illumination patch and 90-frame spatial quarantine | At frame 460 the false lower-arm patch is **darker** than the modeled floor (median signed difference about -36), despite its bright-looking image context. Disabling quarantine alone removed that lock but created another false lock at frames 724–727. A rat entering a blanket quarantine would be hidden. Synthetic bright and dark patches, then a small and an elongated rat in the same area, were tested. | Detect nearly full-width, single-sign patches of either polarity. Retain only a 20-frame, area/sign/width-specific fading-patch signature; smaller or narrow rat candidates can pass. Still an imperfect appearance heuristic. |
| Bounding-box fragment union | Frame 1000 has several real fragments and the ZIP groups them, but transitive pairwise unions can bridge unrelated objects and exceed the intended extent. | Require aggregate extent and complete-link proximity before grouping. Weighted centroid uses the original components; the convex hull is used only for the optional front proxy. A nearby shadow can still contaminate a group. |
| Largest-area unanchored reacquisition | The original run chose an empty-arm illumination blob at frame 460. The ZIP avoids that case through patch filtering but retains largest-area fallback. The provisional tracker later falsely reacquired a smaller, irregular lower-arm artifact near frame 8795 while the rat was visibly left. | Multiple unanchored candidates are ambiguous. Retain the last accepted location for a bounded two-second reacquisition window, with path and area checks; frame 8795 is unknown, and the left-arm rat is recovered near frame 8820. This may increase misses after true long occlusions. |
| Wider motion gate | The original effective gate was about 22 px/frame despite a 120 px setting; a 36 px center fragment shift was rejected. The ZIP's about 56 px/frame is still a geometric heuristic, not measured animal speed. | Keep a capped, maze-scale gate, extend the short-gap association radius to three arm widths, and check the walkway path. A seven-pixel morphological close is applied **only** to the path mask to cross tiny calibration seams; detection still uses the exact maze mask. Synthetic fast motion, distant switching, and off-floor diagonals were tested. |
| New median background | The provided background image is visually clean in sampled regions. A stationary synthetic rat can create a ghost if the temporal median is naively interpolated. Start/end lighting or experimenter changes remain a risk. | Retain bimodal-pixel handling and the synthetic absence regression; no claim of lighting invariance. |
| V2 event state machine | The ZIP emitted reviewer rows before target dwell and could clear uncertainty during pending transitions. A gap followed by return to the same region should not erase a later witnessed crossing. | Replaced with one state machine in `epm.py`: target dwell precedes either a confirmed **proxy** event or one reviewer-only transition. Tested gaps, peeks, direct arm changes, trim, and contiguous boundary observations. |
| Body occupancy versus entry point | The saved run lost 1,905 accepted frames because motion heading was unavailable. Centroid and front-proxy event counts diverge sharply on the same tracking data. Smoothing can lag at a boundary. | Score body occupancy separately. If raw and smoothed body labels disagree, time is unknown; the frame may bridge event continuity because both points are observed on the same accepted segment. Default entries to smoothed centroid per the user's updated preference. |
| Split scoring API | The ZIP would have introduced `epm_scoring_v2.py` while leaving the old `epm.py` scorer callable. | Integrated one authoritative `create_epm_bundle` in `epm.py`; the UI imports that function directly. No duplicate V2 scorer ships. |
| Annotation patch | The ZIP marked body position but did not draw the actual entry proxy, so the video could imply the wrong coordinate was scored. | Green circle = body occupancy, cyan cross = entry proxy, orange X = rejected raw candidate, yellow ring = unavailable optional head orientation. Possible transitions and provisional events have separate text. Continuous trails stop at gaps and off-maze paths. |
| Auto installer | The supplied script would push after tests and could publish an unreviewed heuristic. | Did not run it. Changes were integrated, tested, staged, and reviewed manually in an isolated clone. |
| Synthetic-only validation | A bright circular toy rat cannot establish performance in dark closed arms or around experimenters. | Used exact original calibration for a full 9,006-frame regression, reviewed source/overlay frames across the session, and kept real-video assertions enabled for the recorded fixture. This is a regression, not a numerical accuracy estimate. |

## Before/after on the provided recording

| Diagnostic | Saved prior export | This branch, centroid default |
| --- | ---: | ---: |
| Accepted tracked frames | 7,848 | 8,552 |
| Track segments | 126 | 51 |
| Unknown body/region frames | Prior head-mode `unclassified`: 3,280 | 928, including 765 unclassified and 163 outside calibrated polygons |
| Frame 460 | Wrong lower open arm | Rat in right closed arm near (1180, 513) |
| Frame 1000 | Ambiguous | Rat near center at (1011, 499) |
| Frames 8788–8820 | Not previously flagged | Newly found provisional false lock removed; uncertain until left-arm recovery |
| Provisional events, open / closed / center | Prior saved table: 0 / 1 / 0 | 0 / 4 / 1; **33** possible transitions require review |

The unknown-frame rows are **not directly comparable**: the prior run erased
otherwise accepted body time when head orientation was unavailable. The new
unknown metric is body occupancy. Improved coverage is not proof of identity
accuracy. In a 25-frame spot sheet spanning the recording, accepted points
were visually plausible at the reviewed samples and the known lower-arm false
lock was absent; two sampled frames around 719/724 were uncertain while the rat
was visible. This is not a complete false-positive or sensitivity audit.

The same final tracking yielded different provisional event counts by proxy:
smoothed centroid **0/4/1**, raw centroid **1/3/5**, and motion-derived front
**3/14/10** (open/closed/center). Do not choose a mode by its tally or compare
the counts to paw-based scoring. The reviewer queue is part of the pilot.

## Tests and trial

Run from the repository root with dependencies in `requirements.txt`:

```bash
EPM_REGRESSION_VIDEO='/absolute/path/to/original.mp4' \
EPM_REGRESSION_CALIBRATION='/absolute/path/to/calibration_and_settings.json' \
python -m pytest -q
python -m compileall -q epm.py epm_tracker.py epm_ui.py epm_video.py tests
git diff --check
```

The EPM regression test requires both environment variables; without them it
skips. The full suite has one **pre-existing CPP synthetic timing failure**:
`test_validation.py::test_synthetic_validation_stays_within_reasonable_error`
reports 1.0663 s versus its 0.75 s limit on both this branch and untouched
`main` (`22aa319`). CPP source files were not changed. The exact final pass
count and QC-video frame count are recorded in the pull request.

For a controlled trial after deployment: upload the original recording,
load its `calibration_and_settings.json`, leave the full interval and default
smoothed-centroid entry proxy unless protocol review dictates otherwise, run
analysis, then inspect the annotated MP4, every event, and every possible
transition against the raw video. Export all CSVs, warnings, and the saved
settings. Retain any disagreement frames for a later human-labeled evaluation.

**Pilot readiness:** code is integrated for a controlled, manually reviewed
trial. **Research validation:** no; comprehensive human event and time labels
and multi-video accuracy tolerances are still missing.
