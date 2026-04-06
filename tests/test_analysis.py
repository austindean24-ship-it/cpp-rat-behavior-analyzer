from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import create_analysis_bundle
from regions import create_three_chambers_from_box


def test_analysis_bundle_summarizes_time_by_chamber() -> None:
    calibration = create_three_chambers_from_box((0, 0, 300, 90), frame_width=300, frame_height=90)
    tracking_df = pd.DataFrame(
        [
            {"frame_index": 0, "time_seconds": 0.0, "centroid_x": 50.0, "centroid_y": 45.0, "smoothed_x": 50.0, "smoothed_y": 45.0, "contour_area": 300.0, "tracking_status": "tracked", "low_confidence": False, "carried_forward": False, "distance_from_previous_px": np.nan},
            {"frame_index": 1, "time_seconds": 0.5, "centroid_x": 60.0, "centroid_y": 44.0, "smoothed_x": 60.0, "smoothed_y": 44.0, "contour_area": 305.0, "tracking_status": "tracked", "low_confidence": False, "carried_forward": False, "distance_from_previous_px": 10.0},
            {"frame_index": 2, "time_seconds": 1.0, "centroid_x": 150.0, "centroid_y": 45.0, "smoothed_x": 150.0, "smoothed_y": 45.0, "contour_area": 310.0, "tracking_status": "tracked", "low_confidence": False, "carried_forward": False, "distance_from_previous_px": 90.0},
            {"frame_index": 3, "time_seconds": 1.5, "centroid_x": 160.0, "centroid_y": 46.0, "smoothed_x": 160.0, "smoothed_y": 46.0, "contour_area": 290.0, "tracking_status": "tracked", "low_confidence": False, "carried_forward": False, "distance_from_previous_px": 10.0},
            {"frame_index": 4, "time_seconds": 2.0, "centroid_x": 250.0, "centroid_y": 44.0, "smoothed_x": 250.0, "smoothed_y": 44.0, "contour_area": 315.0, "tracking_status": "tracked", "low_confidence": False, "carried_forward": False, "distance_from_previous_px": 90.0},
            {"frame_index": 5, "time_seconds": 2.5, "centroid_x": 255.0, "centroid_y": 43.0, "smoothed_x": 255.0, "smoothed_y": 43.0, "contour_area": 320.0, "tracking_status": "tracked", "low_confidence": False, "carried_forward": False, "distance_from_previous_px": 5.0},
        ]
    )

    bundle = create_analysis_bundle(tracking_df=tracking_df, calibration=calibration, fps=2.0)
    summary = bundle.summary.set_index("chamber")

    assert summary.loc["left", "seconds"] == 1.0
    assert summary.loc["center", "seconds"] == 1.0
    assert summary.loc["right", "seconds"] == 1.0
    assert bundle.warnings == []


def test_head_shoulders_assignment_mode_uses_head_proxy_columns() -> None:
    calibration = create_three_chambers_from_box((0, 0, 300, 90), frame_width=300, frame_height=90)
    tracking_df = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "time_seconds": 0.0,
                "centroid_x": 95.0,
                "centroid_y": 45.0,
                "smoothed_x": 95.0,
                "smoothed_y": 45.0,
                "head_shoulder_x": 120.0,
                "head_shoulder_y": 45.0,
                "smoothed_head_shoulder_x": 120.0,
                "smoothed_head_shoulder_y": 45.0,
                "contour_area": 300.0,
                "tracking_status": "tracked",
                "low_confidence": False,
                "carried_forward": False,
                "distance_from_previous_px": np.nan,
            }
        ]
    )

    bundle = create_analysis_bundle(
        tracking_df=tracking_df,
        calibration=calibration,
        fps=30.0,
        assignment_point_mode="head_shoulders",
    )

    assert bundle.per_frame.loc[0, "assignment_x"] == 120.0
    assert bundle.per_frame.loc[0, "chamber"] == "center"
