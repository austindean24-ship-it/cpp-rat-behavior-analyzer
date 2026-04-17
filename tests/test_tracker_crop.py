from __future__ import annotations

import cv2
import numpy as np

from regions import create_three_chambers_from_box
from tracker import SingleRatTracker, TrackingConfig


def test_chamber_crop_keeps_tracker_coordinates_in_original_frame(tmp_path) -> None:
    video_path = tmp_path / "small_cpp_crop_test.mp4"
    width, height = 180, 110
    fps = 12.0
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_index in range(24):
        frame = np.full((height, width, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (45, 25), (145, 85), (170, 170, 170), -1)
        cv2.rectangle(frame, (45, 25), (145, 85), (70, 70, 70), 2)
        cv2.line(frame, (78, 25), (78, 85), (90, 90, 90), 1)
        cv2.line(frame, (112, 25), (112, 85), (90, 90, 90), 1)

        # Distractor motion outside the chamber area should be ignored by the ROI mask.
        cv2.circle(frame, (18 + frame_index, 16), 8, (25, 25, 25), -1)

        rat_x = 68 + (frame_index * 2)
        cv2.circle(frame, (rat_x, 56), 6, (20, 20, 20), -1)
        writer.write(frame)

    writer.release()

    calibration = create_three_chambers_from_box(
        box=(45, 25, 100, 60),
        frame_width=width,
        frame_height=height,
    )
    tracker = SingleRatTracker(
        TrackingConfig(
            background_sample_count=8,
            diff_threshold=18,
            frame_diff_threshold=6,
            min_contour_area=20.0,
            max_jump_px=35.0,
            smoothing_alpha=0.25,
            roi_padding_px=0,
        )
    )

    tracking_df = tracker.track_video(video_path=video_path, arena_mask=calibration.arena_mask())
    detected = tracking_df[tracking_df["centroid_x"].notna()]

    assert len(detected) >= 18
    assert detected["centroid_x"].median() > 80.0
    assert detected["centroid_y"].median() > 45.0
    assert detected["centroid_x"].median() < 125.0
    assert detected["centroid_y"].median() < 70.0
