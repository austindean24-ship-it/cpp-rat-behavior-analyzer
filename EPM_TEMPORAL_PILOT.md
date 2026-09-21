# EPM temporal scoring pilot — 21 September 2026

This revision changes only the EPM page and scorer. The CPP tracker and scorer
are untouched. The Streamlit EPM default remains smoothed body centroid.

## Rules now used by the page

- Arm occupancy must persist for **1.0 second** to make an arm entry. Shorter
  arm visits next to center are assigned to center time, without an arm entry
  or a new center return entry.
- A center occupancy of **0.1 second** can make a center entry.
- A missing body position between accepted positions is interpolated only for
  short gaps with an in-maze path and displacement below the per-frame jump
  ceiling. Other gaps hold the nearest accepted position. These frames have
  `assignment_uncertain=true` and an `assignment_method` in the per-frame CSV.
  Long unobserved starts/ends and recordings with no observation remain unknown.
- The video draws inferred positions in yellow and observed ones in green.
  Plausible inferred transitions count provisionally with
  `review_state=inferred_provisional`; unsupported crossings remain review rows.

## Original-video comparison using the saved 9,006-frame tracking CSV

| Metric | Previous pilot | This revision |
| --- | ---: | ---: |
| Open / closed / center entries | 0 / 4 / 1 | 2 / 9 / 17 |
| Unclassified body frames | 928 | 0 |
| Inferred body frames | 0 | 928 (317 interpolated, 611 held) |
| Short arm peek frames assigned to center | Not applied | 96 |
| Possible transitions for review | 33 | 7 |

The changed counts reflect new operational rules and inferred transitions;
they are **not an accuracy improvement estimate**. In particular, the 17 center
entries require manual inspection against the video. Holding a position during
a long occlusion can assign the wrong arm time if the animal moved unseen.
The QC columns and raw tracking CSV permit review of those stretches.

For the next controlled trial, upload the same recording and matching saved
calibration, run the fixed dwell rules, and inspect the annotated video,
`events.csv`, and all `assignment_uncertain` rows in
`per_frame_assignments.csv`. Keep the new exports if a frame or event is wrong.
These measurements remain provisional and are not validated for research use.
