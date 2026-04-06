from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


DEFAULT_CHAMBER_ORDER = ("left", "center", "right")
DEFAULT_CHAMBER_COLORS = {
    "left": (235, 99, 71),
    "center": (56, 162, 140),
    "right": (63, 81, 181),
}


@dataclass
class ChamberRegion:
    """A user-defined chamber polygon."""

    name: str
    polygon: np.ndarray
    color: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float32)
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2:
            raise ValueError("Each chamber polygon must be an Nx2 array of points.")
        if len(self.polygon) < 3:
            raise ValueError("Each chamber polygon needs at least 3 points.")

    def as_int_polygon(self) -> np.ndarray:
        return np.round(self.polygon).astype(np.int32)

    def center(self) -> tuple[float, float]:
        return tuple(np.mean(self.polygon, axis=0).tolist())

    def signed_distance(self, point: tuple[float, float]) -> float:
        return float(cv2.pointPolygonTest(self.polygon, point, True))


@dataclass
class ArenaCalibration:
    """The three chamber definitions for one video."""

    chambers: list[ChamberRegion]
    frame_width: int
    frame_height: int
    boundary_margin_px: float = 0.0
    neutral_label: str = "boundary"
    outside_label: str = "outside"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.chambers) != 3:
            raise ValueError("Exactly 3 chambers are required for a three-chamber CPP apparatus.")

    def chamber_names(self) -> list[str]:
        return [chamber.name for chamber in self.chambers]

    def arena_mask(self) -> np.ndarray:
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        for chamber in self.chambers:
            cv2.fillPoly(mask, [chamber.as_int_polygon()], 255)
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "boundary_margin_px": self.boundary_margin_px,
            "neutral_label": self.neutral_label,
            "outside_label": self.outside_label,
            "metadata": self.metadata,
            "chambers": [
                {
                    "name": chamber.name,
                    "color": list(chamber.color),
                    "polygon": chamber.polygon.tolist(),
                }
                for chamber in self.chambers
            ],
        }


@dataclass
class ChamberAssignment:
    """Assignment result for one centroid."""

    label: str
    on_boundary: bool
    inside_arena: bool
    used_margin_px: float
    distances_px: dict[str, float]


def rectangle_to_polygon(x: float, y: float, width: float, height: float) -> np.ndarray:
    return np.asarray(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float32,
    )


def scale_polygon(points: Sequence[Sequence[float]], scale_x: float, scale_y: float) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32)
    polygon[:, 0] *= scale_x
    polygon[:, 1] *= scale_y
    return polygon


def _make_chamber(name: str, polygon: np.ndarray) -> ChamberRegion:
    return ChamberRegion(
        name=name,
        polygon=polygon,
        color=DEFAULT_CHAMBER_COLORS.get(name, (255, 255, 255)),
    )


def create_three_chambers_from_box(
    box: Sequence[float],
    frame_width: int,
    frame_height: int,
    boundary_margin_px: float = 0.0,
) -> ArenaCalibration:
    """Split one arena box into left, center, and right thirds."""

    x, y, width, height = [float(value) for value in box]
    third_width = width / 3.0

    left = rectangle_to_polygon(x, y, third_width, height)
    center = rectangle_to_polygon(x + third_width, y, third_width, height)
    right = rectangle_to_polygon(x + (2.0 * third_width), y, third_width, height)

    return ArenaCalibration(
        chambers=[
            _make_chamber("left", left),
            _make_chamber("center", center),
            _make_chamber("right", right),
        ],
        frame_width=frame_width,
        frame_height=frame_height,
        boundary_margin_px=boundary_margin_px,
        metadata={"mode": "box_split"},
    )


def _extract_path_points(obj: Mapping[str, Any]) -> np.ndarray:
    path = obj.get("path", [])
    points: list[list[float]] = []
    for command in path:
        if not command:
            continue
        code = command[0]
        if code in {"M", "L"} and len(command) >= 3:
            points.append([float(command[1]), float(command[2])])
        elif code in {"Q", "C"} and len(command) >= 3:
            points.append([float(command[-2]), float(command[-1])])
    if not points:
        raise ValueError("Unable to read polygon points from the drawn path.")
    return np.asarray(points, dtype=np.float32)


def canvas_object_to_polygon(obj: Mapping[str, Any]) -> np.ndarray:
    """Convert a Fabric.js canvas object into a polygon in canvas pixels."""

    scale_x = float(obj.get("scaleX", 1.0))
    scale_y = float(obj.get("scaleY", 1.0))
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    obj_type = str(obj.get("type", "")).lower()

    if obj_type == "rect":
        width = float(obj.get("width", 0.0)) * scale_x
        height = float(obj.get("height", 0.0)) * scale_y
        return rectangle_to_polygon(left, top, width, height)

    if obj_type == "polygon":
        raw_points = obj.get("points", [])
        polygon = np.asarray(
            [
                [left + (float(point["x"]) * scale_x), top + (float(point["y"]) * scale_y)]
                for point in raw_points
            ],
            dtype=np.float32,
        )
        return polygon

    if obj_type == "path":
        polygon = _extract_path_points(obj)
        polygon[:, 0] = left + (polygon[:, 0] * scale_x)
        polygon[:, 1] = top + (polygon[:, 1] * scale_y)
        return polygon

    raise ValueError(f"Unsupported drawing object type: {obj_type or 'unknown'}")


def extract_polygons_from_canvas(
    canvas_json: Mapping[str, Any] | None,
    image_scale_x: float,
    image_scale_y: float,
) -> list[np.ndarray]:
    if not canvas_json:
        return []

    polygons: list[np.ndarray] = []
    for obj in canvas_json.get("objects", []):
        polygon = canvas_object_to_polygon(obj)
        polygon[:, 0] *= image_scale_x
        polygon[:, 1] *= image_scale_y
        polygons.append(polygon.astype(np.float32))
    return polygons


def build_calibration_from_canvas_box(
    canvas_json: Mapping[str, Any] | None,
    frame_width: int,
    frame_height: int,
    image_scale_x: float,
    image_scale_y: float,
    boundary_margin_px: float = 0.0,
) -> ArenaCalibration:
    polygons = extract_polygons_from_canvas(canvas_json, image_scale_x=image_scale_x, image_scale_y=image_scale_y)
    if not polygons:
        raise ValueError("Draw one rectangle around the full apparatus first.")

    polygon = polygons[-1]
    x, y, width, height = cv2.boundingRect(np.round(polygon).astype(np.int32))
    return create_three_chambers_from_box(
        box=(x, y, width, height),
        frame_width=frame_width,
        frame_height=frame_height,
        boundary_margin_px=boundary_margin_px,
    )


def build_calibration_from_canvas_polygons(
    canvas_json: Mapping[str, Any] | None,
    frame_width: int,
    frame_height: int,
    image_scale_x: float,
    image_scale_y: float,
    boundary_margin_px: float = 0.0,
) -> ArenaCalibration:
    polygons = extract_polygons_from_canvas(canvas_json, image_scale_x=image_scale_x, image_scale_y=image_scale_y)
    if len(polygons) != 3:
        raise ValueError("Draw exactly 3 chamber polygons on the frame.")

    sorted_polygons = sorted(polygons, key=lambda item: float(np.mean(item[:, 0])))
    chambers = [
        _make_chamber(name, polygon)
        for name, polygon in zip(DEFAULT_CHAMBER_ORDER, sorted_polygons)
    ]
    return ArenaCalibration(
        chambers=chambers,
        frame_width=frame_width,
        frame_height=frame_height,
        boundary_margin_px=boundary_margin_px,
        metadata={"mode": "manual_polygons"},
    )


def assign_point_to_chamber(
    point: tuple[float, float] | None,
    calibration: ArenaCalibration,
    boundary_margin_px: float | None = None,
) -> ChamberAssignment:
    """Assign a centroid to a chamber with deterministic border behavior.

    Rule:
    - If a point is exactly on a shared border and no neutral margin is used, the
      earlier chamber in left/center/right order wins.
    - If a neutral margin is used and the point is close to more than one chamber,
      the frame is labeled as the neutral boundary class.
    """

    if point is None:
        return ChamberAssignment(
            label="missing",
            on_boundary=False,
            inside_arena=False,
            used_margin_px=0.0,
            distances_px={},
        )

    x, y = point
    if np.isnan(x) or np.isnan(y):
        return ChamberAssignment(
            label="missing",
            on_boundary=False,
            inside_arena=False,
            used_margin_px=0.0,
            distances_px={},
        )

    margin = calibration.boundary_margin_px if boundary_margin_px is None else float(boundary_margin_px)
    distances = {
        chamber.name: chamber.signed_distance((float(x), float(y)))
        for chamber in calibration.chambers
    }

    exact_or_inside = [name for name in calibration.chamber_names() if distances[name] >= 0.0]
    fuzzy_inside = [name for name in calibration.chamber_names() if distances[name] >= (-1.0 * margin)]

    if margin > 0.0 and len(fuzzy_inside) > 1:
        return ChamberAssignment(
            label=calibration.neutral_label,
            on_boundary=True,
            inside_arena=True,
            used_margin_px=margin,
            distances_px=distances,
        )

    if exact_or_inside:
        return ChamberAssignment(
            label=exact_or_inside[0],
            on_boundary=len(exact_or_inside) > 1,
            inside_arena=True,
            used_margin_px=margin,
            distances_px=distances,
        )

    if margin > 0.0 and fuzzy_inside:
        return ChamberAssignment(
            label=fuzzy_inside[0],
            on_boundary=False,
            inside_arena=True,
            used_margin_px=margin,
            distances_px=distances,
        )

    return ChamberAssignment(
        label=calibration.outside_label,
        on_boundary=False,
        inside_arena=False,
        used_margin_px=margin,
        distances_px=distances,
    )


def draw_calibration_overlay(frame: np.ndarray, calibration: ArenaCalibration) -> np.ndarray:
    overlay = frame.copy()
    for chamber in calibration.chambers:
        polygon = chamber.as_int_polygon()
        cv2.polylines(overlay, [polygon], True, chamber.color, 3, cv2.LINE_AA)
        center_x, center_y = chamber.center()
        cv2.putText(
            overlay,
            chamber.name.upper(),
            (int(center_x) - 40, int(center_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            chamber.color,
            2,
            cv2.LINE_AA,
        )
    return overlay
