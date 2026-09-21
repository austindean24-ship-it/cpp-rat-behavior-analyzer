from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from epm import (
    EPMCalibration,
    MazeRegion,
    SUMMARY_COLUMNS,
    assign_epm_frames,
    calibration_from_canvas,
    create_epm_bundle,
)
from regions import rectangle_to_polygon
from tracker import SingleRatTracker, TrackingConfig


def calibration() -> EPMCalibration:
    boxes = {
        "center": (80, 80, 40, 40),
        "open_1": (85, 20, 30, 60),
        "open_2": (85, 120, 30, 60),
        "closed_1": (20, 85, 60, 30),
        "closed_2": (120, 85, 60, 30),
    }
    return EPMCalibration(
        [MazeRegion(name, rectangle_to_polygon(*boxes[name])) for name in boxes], 200, 200
    )


def tracking_rows(labels: list[str]) -> pd.DataFrame:
    points = {
        "center": (100, 100),
        "open_1": (100, 50),
        "open_2": (100, 150),
        "closed_1": (50, 100),
        "closed_2": (150, 100),
        "missing": (np.nan, np.nan),
    }
    rows = []
    for frame, label in enumerate(labels):
        x, y = points[label]
        rows.append({
            "frame_index": frame, "time_seconds": frame / 10,
            "centroid_x": x, "centroid_y": y,
            "smoothed_x": x, "smoothed_y": y,
            "smoothed_head_shoulder_x": x, "smoothed_head_shoulder_y": y,
            "contour_area": 300 if label != "missing" else np.nan,
            "tracking_status": "tracked" if label != "missing" else "missing",
            "low_confidence": label == "missing", "carried_forward": False,
            "distance_from_previous_px": 0.0,
        })
    return pd.DataFrame(rows)


def test_calibration_masks_only_maze_and_rejects_overlap() -> None:
    maze = calibration()
    mask = maze.tracking_mask()
    assert mask[100, 100] == 255
    assert mask[50, 100] == 255
    assert mask[5, 5] == 0
    assert maze.regions[0].name == "center"
    bad = maze.regions.copy()
    bad[1] = MazeRegion("open_1", rectangle_to_polygon(90, 65, 30, 40))
    try:
        EPMCalibration(bad, 200, 200)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Overlapping behavioral regions must be rejected.")


def test_canvas_mapping_and_original_coordinates() -> None:
    boxes = [(40, 40, 20, 20), (42.5, 10, 15, 30), (42.5, 60, 15, 30),
             (10, 42.5, 30, 15), (60, 42.5, 30, 15)]
    canvas = {"objects": [
        {"type": "rect", "left": x, "top": y, "width": w, "height": h}
        for x, y, w, h in boxes
    ]}
    maze = calibration_from_canvas(
        canvas, {name: i for i, name in enumerate(("center", "open_1", "open_2", "closed_1", "closed_2"))},
        200, 200, 2.0, 2.0,
    )
    assert maze.regions[0].center() == (100.0, 100.0)
    assert maze.tracking_mask()[100, 100] == 255


def test_head_proxy_drives_region_classification() -> None:
    rows = tracking_rows(["center"])
    rows.loc[0, ["smoothed_head_shoulder_x", "smoothed_head_shoulder_y"]] = [100, 50]
    assert assign_epm_frames(rows, calibration(), mode="head_shoulders").loc[0, "entry_region"] == "open_1"
    assert assign_epm_frames(rows, calibration(), mode="head_shoulders").loc[0, "region"] == "center"
    assert assign_epm_frames(rows, calibration(), mode="centroid").loc[0, "region"] == "center"


def test_entries_require_stable_crossings_and_do_not_bridge_missing_frames() -> None:
    labels = (
        ["closed_1"] * 2 + ["center"] * 2 + ["open_1"] * 2 +
        ["center"] + ["open_1"] * 2 + ["center"] * 2 +
        ["open_2"] * 2 + ["missing"] + ["closed_2"] * 2
    )
    result = create_epm_bundle(tracking_rows(labels), calibration(), fps=10, min_dwell_seconds=0.2)
    row = result.summary.iloc[0]
    assert list(result.summary.columns) == list(SUMMARY_COLUMNS)
    assert (row["Open Arm Entries"], row["Closed Arm Entries"], row["Center Entry"]) == (2, 1, 2)
    assert sum(int(row[column]) for column in SUMMARY_COLUMNS[3:]) <= round(len(labels) / 10)
    assert result.events.loc[result.events.review_state == "confirmed_proxy", "event"].tolist() == [
        "closed_arm_entry", "center_entry", "open_arm_entry", "center_entry", "open_arm_entry"
    ]
    assert result.events.loc[result.events.review_state == "requires_manual_review", "event"].tolist() == ["possible_transition"]
    assert result.region_times.set_index("region").loc["unclassified", "frames"] == 1


def test_initial_arm_is_configurable_and_not_recounted_after_a_gap() -> None:
    rows = tracking_rows(["closed_1", "closed_1", "missing", "open_1", "open_1"])
    with_initial = create_epm_bundle(rows, calibration(), 10, min_dwell_seconds=0.2)
    without_initial = create_epm_bundle(rows, calibration(), 10, min_dwell_seconds=0.2, count_initial_arm=False)
    assert with_initial.summary.loc[0, "Closed Arm Entries"] == 1
    assert with_initial.summary.loc[0, "Open Arm Entries"] == 0
    assert without_initial.summary.loc[0, "Closed Arm Entries"] == 0


def test_tracker_on_synthetic_epm_video_ignores_motion_outside_maze(tmp_path) -> None:
    path = tmp_path / "synthetic_epm.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 200))
    assert writer.isOpened()
    maze = calibration()
    for frame_index in range(40):
        frame = np.full((200, 200, 3), 35, np.uint8)
        for region in maze.regions:
            cv2.fillPoly(frame, [region.as_int_polygon()], (85, 85, 85))
        # Track: center -> closed right -> center -> open top.
        if frame_index < 10:
            x, y = 100 + 5 * frame_index, 100
        elif frame_index < 20:
            x, y = 150 - 5 * (frame_index - 10), 100
        elif frame_index < 30:
            x, y = 100, 100 - 5 * (frame_index - 20)
        else:
            x, y = 100, 50 + 5 * (frame_index - 30)
        cv2.circle(frame, (x, y), 8, (245, 245, 245), -1)
        cv2.circle(frame, (10 + frame_index * 3, 10), 8, (245, 245, 245), -1)
        writer.write(frame)
    writer.release()
    tracker = SingleRatTracker(TrackingConfig(min_contour_area=30, max_jump_px=30))
    tracked = tracker.track_video(path, arena_mask=maze.tracking_mask())
    assert len(tracked) == 40
    assert tracked["centroid_x"].notna().mean() >= 0.8
    assert tracked["centroid_x"].dropna().between(70, 180).mean() >= 0.9
    assert tracked["centroid_y"].dropna().between(18, 180).mean() >= 0.9
    bundle = create_epm_bundle(tracked, maze, 10, min_dwell_seconds=0.1)
    assert len(bundle.per_frame) == 40
    assert bundle.summary.loc[0, "Closed Arm Time (whole seconds)"] >= 1
    from epm_video import write_annotated_epm_video

    annotated = write_annotated_epm_video(path, tmp_path / "annotated.mp4", bundle.per_frame, maze)
    assert annotated.stat().st_size > 0
    rendered = cv2.VideoCapture(str(annotated))
    assert int(rendered.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    rendered.release()


def test_fabric_polygon_path_ignores_handle_and_preserves_canvas_coordinates() -> None:
    from epm import extract_epm_polygons

    canvas = {"objects": [
        {"type": "circle", "left": 40, "top": 30, "radius": 3},
        {
            "type": "path", "left": 50, "top": 40, "width": 20, "height": 20,
            "originX": "center", "originY": "center",
            "pathOffset": {"x": 50, "y": 40},
            "path": [["M", 40, 30], ["L", 60, 30], ["L", 60, 50], ["L", 40, 50], ["z"]],
        },
        {"type": "line", "x1": 0, "y1": 0, "x2": 1, "y2": 1},
    ]}
    polygons = extract_epm_polygons(canvas, 2, 2)
    assert len(polygons) == 1
    assert np.allclose(polygons[0], [[80, 60], [120, 60], [120, 100], [80, 100]])
