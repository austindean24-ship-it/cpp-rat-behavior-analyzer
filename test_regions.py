from __future__ import annotations

from regions import assign_point_to_chamber, build_calibration_from_rectangle_boxes, create_three_chambers_from_box


def test_assigns_points_to_each_chamber() -> None:
    calibration = create_three_chambers_from_box((0, 0, 300, 90), frame_width=300, frame_height=90)
    assert assign_point_to_chamber((50, 45), calibration).label == "left"
    assert assign_point_to_chamber((150, 45), calibration).label == "center"
    assert assign_point_to_chamber((250, 45), calibration).label == "right"


def test_border_is_deterministic_without_margin() -> None:
    calibration = create_three_chambers_from_box((0, 0, 300, 90), frame_width=300, frame_height=90)
    assert assign_point_to_chamber((100, 45), calibration).label == "left"


def test_neutral_boundary_margin_can_mark_shared_border() -> None:
    calibration = create_three_chambers_from_box((0, 0, 300, 90), frame_width=300, frame_height=90, boundary_margin_px=5.0)
    assignment = assign_point_to_chamber((100, 45), calibration)
    assert assignment.label == "boundary"
    assert assignment.on_boundary is True


def test_manual_rectangle_boxes_are_sorted_left_to_right() -> None:
    calibration = build_calibration_from_rectangle_boxes(
        boxes=[
            {"left": 200, "top": 0, "width": 80, "height": 90},
            {"left": 0, "top": 0, "width": 80, "height": 90},
            {"left": 100, "top": 0, "width": 80, "height": 90},
        ],
        frame_width=300,
        frame_height=90,
    )

    assert calibration.chamber_names() == ["left", "center", "right"]
    assert assign_point_to_chamber((40, 45), calibration).label == "left"
    assert assign_point_to_chamber((140, 45), calibration).label == "center"
    assert assign_point_to_chamber((240, 45), calibration).label == "right"
