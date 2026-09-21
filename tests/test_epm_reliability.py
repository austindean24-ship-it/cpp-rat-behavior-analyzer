from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from epm import create_epm_bundle
from epm_tracker import Candidate, EPMTrackState, EPMTracker
from tests.test_epm import calibration, tracking_rows
from tracker import TrackingConfig


def candidate(x: int, y: int, radius: int = 8) -> Candidate:
    contour = cv2.ellipse2Poly((x, y), (radius, radius), 0, 0, 360, 10).reshape(-1, 1, 2)
    return Candidate((float(x), float(y)), float(cv2.contourArea(contour)), contour)


def test_raw_jump_rejected_before_smoothing_and_reacquisition_is_unscored() -> None:
    maze = calibration()
    state = EPMTrackState(maze.tracking_mask(), TrackingConfig(max_jump_px=120, smoothing_alpha=0.35), 10, 30)
    for frame, x in enumerate((100, 105, 110)):
        accepted, _, status, _, _ = state.update([candidate(x, 100)], frame)
        if accepted:
            state.score_points(accepted, frame)
    assert status == "tracked"
    saved = state.smoothed
    accepted, raw, status, reason, displacement = state.update([candidate(150, 100)], 3)
    assert accepted is None and raw.point == (150.0, 100.0)
    assert (status, reason) == ("rejected", "raw_jump")
    assert displacement > state.max_step and state.smoothed == saved
    for frame in (4, 5):
        accepted, _, status, _, _ = state.update([candidate(112, 100)], frame)
        assert accepted is None and status == "reacquiring"
    accepted, _, status, _, _ = state.update([candidate(112, 100)], 6)
    assert accepted is not None and status == "tracked" and state.segment == 2


def test_external_motion_shadow_and_single_frame_glare_cannot_score() -> None:
    maze = calibration()
    config = TrackingConfig(min_contour_area=30, diff_threshold=20, max_jump_px=60)
    tracker = EPMTracker(config)
    background = np.full((200, 200), 90, np.uint8)
    outside = background.copy()
    cv2.circle(outside, (10, 10), 9, 240, -1)
    assert tracker._candidates(outside, background, maze.tracking_mask())[0] == []
    shadow = background.copy()
    cv2.rectangle(shadow, (85, 20), (115, 80), 50, -1)
    assert tracker._candidates(shadow, background, maze.tracking_mask())[0] == []
    glare = background.copy()
    cv2.circle(glare, (100, 100), 8, 240, -1)
    candidates, _ = tracker._candidates(glare, background, maze.tracking_mask())
    assert candidates
    state = EPMTrackState(maze.tracking_mask(), config, 10, 30)
    accepted, _, status, _, _ = state.update(candidates, 0)
    assert accepted is None and status == "reacquiring"
    assert state.update([], 1)[0] is None


def test_untrusted_frames_have_unknown_time_and_no_proxy_fallback() -> None:
    rows = tracking_rows(["center", "center", "open_1", "open_1", "open_1"])
    rows.loc[2, "low_confidence"] = True
    rows.loc[3, "tracking_status"] = "carried_forward"
    rows.loc[3, "carried_forward"] = True
    rows["head_estimate_source"] = "motion_heading"
    rows.loc[4, "head_estimate_source"] = "centroid_fallback"
    bundle = create_epm_bundle(rows, calibration(), 10, min_dwell_seconds=0.1)
    assert bundle.region_times.set_index("region").loc["unclassified", "frames"] == 2
    assert bundle.per_frame.loc[4, "entry_point_source"] == "missing_head_orientation"
    assert bundle.summary.loc[0, "Open Arm Entries"] == 0
    assert bundle.qc_metrics.set_index("metric").loc["unclassified_frames", "value"] == 2
    assert bundle.qc_metrics.set_index("metric").loc["missing_head_proxy_frames", "value"] == 1


def test_trimmed_interval_can_count_initial_arm_occupancy() -> None:
    rows = tracking_rows(["center"] * 10 + ["closed_1"] * 4)
    bundle = create_epm_bundle(
        rows, calibration(), fps=10, min_dwell_seconds=0.2,
        analysis_start_seconds=1.0, analysis_end_seconds=1.4,
    )
    assert bundle.summary.loc[0, "Closed Arm Entries"] == 1
    assert bundle.summary.loc[0, "Center Entry"] == 0


def test_boundary_peeks_do_not_make_entries_but_confirmed_crossings_do() -> None:
    rows = tracking_rows(["center"] * 2 + ["center"] * 4 + ["closed_2"] * 2 + ["center"] * 2)
    rows.loc[2:5, "smoothed_head_shoulder_x"] = [121, 119, 121, 119]
    rows.loc[6:7, "smoothed_head_shoulder_x"] = 130
    rows.loc[8:9, "smoothed_head_shoulder_x"] = 110
    bundle = create_epm_bundle(rows, calibration(), 10, min_dwell_seconds=0.2)
    assert bundle.events.loc[bundle.events.review_state == "confirmed_proxy", "event"].tolist() == ["closed_arm_entry", "center_entry"]
    assert int((bundle.per_frame["entry_boundary"]).sum()) == 4


def test_interval_excludes_setup_frames_and_preserves_total_with_unknown() -> None:
    rows = tracking_rows(["closed_1"] * 4 + ["center"] * 4 + ["open_1"] * 4)
    rows.loc[9, "tracking_status"] = "rejected"
    rows.loc[9, "low_confidence"] = True
    bundle = create_epm_bundle(
        rows, calibration(), fps=10, min_dwell_seconds=0.2,
        analysis_start_seconds=0.4, analysis_end_seconds=1.2,
    )
    assert bundle.per_frame.loc[:3, "region"].eq("excluded").all()
    assert bundle.summary.loc[0, "Closed Arm Entries"] == 0
    assert bundle.region_times.set_index("region").loc["unclassified", "frames"] == 1
    assert bundle.qc_metrics.set_index("metric").loc["analysis_duration_seconds", "value"] == 0.8


def test_segment_change_never_invents_a_return_from_an_unseen_arm() -> None:
    rows = tracking_rows(["center"] * 2 + ["open_1"] * 2 + ["missing"] + ["center"] * 3)
    rows["track_segment_id"] = [1, 1, 1, 1, np.nan, 2, 2, 2]
    bundle = create_epm_bundle(rows, calibration(), 10, min_dwell_seconds=0.2)
    assert bundle.events.loc[bundle.events.review_state == "confirmed_proxy", "event"].tolist() == ["open_arm_entry"]
    assert bundle.events.loc[bundle.events.review_state == "requires_manual_review", "event"].tolist() == ["possible_transition"]
    assert bundle.region_times.set_index("region").loc["unclassified", "frames"] == 1


def test_trail_refuses_gaps_and_off_maze_diagonals() -> None:
    from epm_video import _valid_trail_edge

    mask = calibration().tracking_mask()
    assert _valid_trail_edge((1, (100, 100), 1), (2, (110, 100), 1), mask)
    assert not _valid_trail_edge((1, (100, 150), 1), (3, (150, 100), 1), mask)
    assert not _valid_trail_edge((1, (100, 150), 1), (2, (150, 100), 1), mask)
    assert not _valid_trail_edge((1, (100, 100), 1), (2, (110, 100), 2), mask)


def test_video_does_not_follow_person_outside_static_rat(tmp_path) -> None:
    maze = calibration()
    path = tmp_path / "person_outside.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 200))
    assert writer.isOpened()
    for index in range(80):
        frame = np.full((200, 200, 3), 35, np.uint8)
        for region in maze.regions:
            cv2.fillPoly(frame, [region.as_int_polygon()], (90, 90, 90))
        # External movement is present even when the rat is absent.
        cv2.circle(frame, (10 + index % 20, 10), 9, (240, 240, 240), -1)
        if 20 <= index < 60:
            cv2.circle(frame, (100, 100), 8, (245, 245, 245), -1)
        writer.write(frame)
    writer.release()
    tracked = EPMTracker(TrackingConfig(min_contour_area=30, diff_threshold=20)).track_video(path, maze)
    assert tracked.iloc[:20]["centroid_x"].isna().all()
    assert tracked.iloc[60:]["centroid_x"].isna().all()
    stable = tracked.iloc[24:55]
    assert stable["tracking_status"].eq("tracked").mean() > 0.8
    assert stable["centroid_x"].dropna().between(90, 110).all()
    result = create_epm_bundle(tracked, maze, 10, mode="centroid")
    assert result.region_times.set_index("region").loc["unclassified", "frames"] >= 20
    assert result.summary.loc[0, "Open Arm Entries"] == 0


def test_realistic_normal_and_fast_center_to_arm_motion_is_retained() -> None:
    maze = calibration()
    for step in (5, 12):
        state = EPMTrackState(maze.tracking_mask(), TrackingConfig(max_jump_px=120, smoothing_alpha=1), 10, 30)
        y_values = [100] * 5 + list(range(100 - step, 45, -step))
        rows = []
        for index, y in enumerate(y_values):
            accepted, _, status, _, _ = state.update([candidate(100, y)], index)
            points = state.score_points(accepted, index) if accepted else {}
            anchor = points.get("head_smoothed", (np.nan, np.nan))
            rows.append({
                "frame_index": index, "time_seconds": index / 10,
                "centroid_x": 100 if accepted else np.nan, "centroid_y": y if accepted else np.nan,
                "smoothed_head_shoulder_x": anchor[0], "smoothed_head_shoulder_y": anchor[1],
                "head_estimate_source": points.get("head_source", "missing"),
                "tracking_status": status, "low_confidence": status != "tracked",
                "carried_forward": False, "contour_area": accepted.area if accepted else np.nan,
                "distance_from_previous_px": step, "track_segment_id": state.segment if accepted else np.nan,
            })
        assert all(row["tracking_status"] == "tracked" for row in rows[5:])
        bundle = create_epm_bundle(pd.DataFrame(rows), maze, 10, min_dwell_seconds=0.1, count_initial_arm=False)
        assert bundle.summary.loc[0, "Open Arm Entries"] == 1
