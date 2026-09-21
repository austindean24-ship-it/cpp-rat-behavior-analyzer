"""Synthetic guardrails for the EPM-only tracker; no human-score claims."""
from __future__ import annotations

import cv2
import numpy as np

from epm import EPMCalibration, MazeRegion
from epm_tracker import Candidate, EPMTracker, EPMTrackState, _join_fragments
from tracker import TrackingConfig


def maze():
    boxes = {"center": (80, 80, 40, 40), "open_1": (85, 20, 30, 60),
             "open_2": (85, 120, 30, 60), "closed_1": (20, 85, 60, 30),
             "closed_2": (120, 85, 60, 30)}
    regions = [MazeRegion(n, np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], np.float32))
               for n, (x, y, w, h) in boxes.items()]
    return EPMCalibration(regions, 200, 200)


def candidate(x, y):
    contour = cv2.ellipse2Poly((x, y), (8, 8), 0, 0, 360, 10).reshape(-1, 1, 2)
    return Candidate((float(x), float(y)), float(cv2.contourArea(contour)), contour)


def test_fast_plausible_motion_kept_distant_jump_rejected():
    state = EPMTrackState(maze().tracking_mask(), TrackingConfig(max_jump_px=120), fps=30, arm_width=62.7)
    for frame, x in enumerate((100, 102, 104)):
        accepted, _, status, _, _ = state.update([candidate(x, 100)], frame)
    assert status == 'tracked'
    accepted, _, status, reason, _ = state.update([candidate(139, 100)], 3)
    assert status == 'tracked' and accepted is not None  # 35px, previously gated at 22
    accepted, _, status, reason, _ = state.update([candidate(100, 160)], 4)
    assert accepted is None and status == 'rejected' and reason in ('raw_jump', 'off_maze_path')


def test_group_nearby_fragments_does_not_group_far_artifact():
    target = maze()
    tracker = EPMTracker(TrackingConfig(min_contour_area=25, diff_threshold=20))
    bg = np.full((200, 200), 90, np.uint8)
    fg = bg.copy()
    cv2.circle(fg, (90, 96), 5, 230, -1)
    cv2.circle(fg, (109, 101), 5, 230, -1)
    cv2.circle(fg, (100, 160), 6, 230, -1)
    candidates, _ = tracker._candidates(fg, bg, target.tracking_mask())
    assert len(candidates) == 2
    center = min(candidates, key=lambda c: abs(c.point[1]-100))
    assert center.evidence == 'grouped_fragments'
    assert 90 <= center.point[0] <= 110


def test_fragment_chain_cannot_merge_distant_objects():
    items = [candidate(x, 100) for x in (45, 68, 91)]
    groups = _join_fragments(items, max_extent=100)
    assert len(groups) >= 2


def test_unanchored_tracker_does_not_choose_largest_blob():
    target = maze()
    state = EPMTrackState(target.tracking_mask(), TrackingConfig(), 10, 30)
    large = Candidate(candidate(100, 100).point, 1000, candidate(100, 100).contour)
    small = candidate(150, 100)
    for frame in range(5):
        accepted, _, status, reason, _ = state.update([large, small], frame)
        assert accepted is None and status == 'ambiguous'
        assert reason == 'multiple_unanchored_candidates'


def test_short_gap_does_not_reacquire_distant_single_artifact():
    target = maze()
    state = EPMTrackState(target.tracking_mask(), TrackingConfig(max_jump_px=120), 10, 30)
    for frame in range(3):
        state.update([candidate(50, 100)], frame)
    assert state.last is not None
    for frame in range(3, 10):
        state.update([], frame)
    for frame in range(10, 14):
        accepted, _, status, reason, _ = state.update([candidate(100, 150)], frame)
        assert accepted is None and status == 'rejected'
        assert reason in {'raw_jump', 'off_maze_path'}
    for frame in (14, 15):
        assert state.update([candidate(55, 100)], frame)[2] == 'reacquiring'
    accepted, _, status, _, _ = state.update([candidate(55, 100)], 16)
    assert accepted is not None and status == 'tracked'


def test_small_calibration_seam_is_passable_but_floor_diagonal_is_not():
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (85, 20), (115, 78), 255, -1)
    cv2.rectangle(mask, (80, 82), (120, 120), 255, -1)
    state = EPMTrackState(mask, TrackingConfig(), 10, 30)
    assert state._walkway_path((100, 75), (100, 85))
    assert not state._walkway_path((100, 50), (150, 100))


def test_signed_light_patches_do_not_blanket_exclude_a_later_rat():
    target = maze()
    tracker = EPMTracker(TrackingConfig(min_contour_area=30, diff_threshold=20))
    tracker._arm_width = 30
    tracker._min_patch_length = 36
    tracker._region_masks = []
    tracker._region_areas = []
    for region in target.regions:
        region_mask = np.zeros((200, 200), np.uint8)
        cv2.fillPoly(region_mask, [region.as_int_polygon()], 255)
        tracker._region_masks.append(region_mask)
        tracker._region_areas.append(cv2.countNonZero(region_mask))
    bg = np.full((200, 200), 90, np.uint8)
    full_mask = np.full((200, 200), 255, np.uint8)
    for value in (40, 150):
        tracker._illumination_zones = []
        patch = bg.copy()
        cv2.rectangle(patch, (85, 25), (115, 69), value, -1)
        candidates, reason = tracker._candidates(patch, bg, full_mask)
        assert not candidates and reason == 'suspected_illumination_patch'
        rat = bg.copy()
        cv2.circle(rat, (100, 48), 7, value, -1)
        candidates, _ = tracker._candidates(rat, bg, full_mask)
        assert len(candidates) == 1
        elongated = bg.copy()
        cv2.ellipse(elongated, (100, 48), (12, 23), 0, 0, 360, value, -1)
        candidates, _ = tracker._candidates(elongated, bg, full_mask)
        assert len(candidates) == 1


def test_real_video_regression_when_available():
    """Use a local video fixture only; recordings must not enter Git history."""
    import os
    import pytest
    path = os.environ.get('EPM_REGRESSION_VIDEO')
    if not path:
        pytest.skip('Set EPM_REGRESSION_VIDEO to the private original video')
    import json
    cfg = json.load(open(os.environ['EPM_REGRESSION_CALIBRATION']))['calibration']
    target = EPMCalibration([
        MazeRegion(r['name'], np.asarray(r['polygon'], np.float32)) for r in cfg['regions']
    ], cfg['frame_width'], cfg['frame_height'])
    result = EPMTracker(TrackingConfig(min_contour_area=150., max_jump_px=120.)).track_video(path, target)
    assert result.iloc[460].tracking_status == 'tracked'
    assert 1140 < result.iloc[460].centroid_x < 1250
    assert 480 < result.iloc[460].centroid_y < 545
    assert not ((result.iloc[430:530].tracking_status == 'tracked') &
                (result.iloc[430:530].centroid_y > 650)).any()
    assert not ((result.iloc[716:730].tracking_status == 'tracked') &
                (result.iloc[716:730].centroid_y > 650)).any()
    assert not ((result.iloc[8788:8820].tracking_status == 'tracked') &
                (result.iloc[8788:8820].centroid_y > 650)).any()
    assert result.iloc[1000].tracking_status == 'tracked'
    assert result.iloc[1000].centroid_x > 950
    assert result.iloc[1450].tracking_status == 'tracked'
    assert 950 < result.iloc[1450].centroid_x < 1050
    for frame in (3000, 5100):
        assert result.iloc[frame].tracking_status == 'tracked'
        assert 1150 < result.iloc[frame].centroid_x < 1300


def test_synthetic_absence_recovery_without_ghost(tmp_path):
    target = maze()
    video = tmp_path / "rat_absent_then_present.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 200))
    assert writer.isOpened()
    for i in range(80):
        frame = np.full((200, 200, 3), 35, np.uint8)
        for region in target.regions:
            cv2.fillPoly(frame, [region.as_int_polygon()], (90, 90, 90))
        cv2.circle(frame, (10 + i % 20, 10), 9, (240, 240, 240), -1)
        if 20 <= i < 60:
            cv2.circle(frame, (100, 100), 8, (245, 245, 245), -1)
        writer.write(frame)
    writer.release()
    tracked = EPMTracker(TrackingConfig(min_contour_area=30, diff_threshold=20)).track_video(video, target)
    assert tracked.iloc[:20].centroid_x.isna().all()
    assert tracked.iloc[60:].centroid_x.isna().all()
    assert tracked.iloc[24:55].tracking_status.eq("tracked").all()
    assert tracked.iloc[24:55].centroid_x.between(90, 110).all()
