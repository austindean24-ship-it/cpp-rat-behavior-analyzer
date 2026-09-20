"""Elevated plus maze scoring. CPP files remain unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd

from analysis import compute_qc_metrics
from regions import extract_polygons_from_canvas

REGION_ORDER = ("center", "open_1", "open_2", "closed_1", "closed_2")
COLORS = {
    "center": (50, 190, 165),
    "open_1": (65, 170, 245),
    "open_2": (65, 170, 245),
    "closed_1": (200, 125, 245),
    "closed_2": (200, 125, 245),
}
SUMMARY_COLUMNS = (
    "Open Arm Entries",
    "Closed Arm Entries",
    "Center Entry",
    "Open Arm Time (whole seconds)",
    "Closed Arm Time (whole seconds)",
    "Center Time (whole seconds)",
)


@dataclass
class MazeRegion:
    name: str
    polygon: np.ndarray

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float32)
        if self.name not in REGION_ORDER:
            raise ValueError(f"Unknown maze region: {self.name}")
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2 or len(self.polygon) < 3:
            raise ValueError("Each maze region needs at least three polygon points.")
        if not np.isfinite(self.polygon).all() or abs(cv2.contourArea(self.polygon)) < 10:
            raise ValueError(f"The {self.name} polygon is empty or invalid.")

    @property
    def color(self) -> tuple[int, int, int]:
        return COLORS[self.name]

    def as_int_polygon(self) -> np.ndarray:
        return np.rint(self.polygon).astype(np.int32)

    def center(self) -> tuple[float, float]:
        x, y = np.mean(self.polygon, axis=0)
        return float(x), float(y)

    def signed_distance(self, point: tuple[float, float]) -> float:
        return float(cv2.pointPolygonTest(self.polygon, point, True))


@dataclass
class EPMCalibration:
    regions: list[MazeRegion]
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        if tuple(region.name for region in self.regions) != REGION_ORDER:
            raise ValueError("Define center, two open arms, and two closed arms exactly once.")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("Invalid video frame size.")
        masks: list[np.ndarray] = []
        for region in self.regions:
            x, y = region.polygon[:, 0], region.polygon[:, 1]
            if (x < 0).any() or (x >= self.frame_width).any() or (y < 0).any() or (y >= self.frame_height).any():
                raise ValueError(f"The {region.name} polygon extends outside the video frame.")
            mask = np.zeros((self.frame_height, self.frame_width), np.uint8)
            cv2.fillPoly(mask, [region.as_int_polygon()], 255)
            masks.append(mask)
        # Shared edge pixels are acceptable; overlapping interiors are ambiguous.
        interiors = [cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2) for mask in masks]
        for i in range(5):
            for j in range(i + 1, 5):
                if cv2.countNonZero(cv2.bitwise_and(interiors[i], interiors[j])):
                    raise ValueError("Maze regions overlap. Adjust their shared boundaries.")
        tolerance = max(3, min(self.frame_width, self.frame_height) // 200)
        near_center = cv2.dilate(masks[0], np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8))
        for region, mask in zip(self.regions[1:], masks[1:]):
            if not cv2.countNonZero(cv2.bitwise_and(near_center, mask)):
                raise ValueError(f"The {region.name} polygon must meet the center polygon.")

    def tracking_mask(self) -> np.ndarray:
        """Union of maze surfaces; classification still uses individual polygons."""
        mask = np.zeros((self.frame_height, self.frame_width), np.uint8)
        for region in self.regions:
            cv2.fillPoly(mask, [region.as_int_polygon()], 255)
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "regions": [{"name": r.name, "polygon": r.polygon.tolist()} for r in self.regions],
        }


def extract_epm_polygons(
    canvas_json: Mapping[str, Any],
    image_scale_x: float,
    image_scale_y: float,
) -> list[np.ndarray]:
    """Read completed Fabric polygons; ignore drawing handles and guide lines.

    The bundled polygon tool creates fabric.Path objects whose path coordinates
    are already in canvas space. Fabric then stores an origin and pathOffset, so
    simply adding left/top a second time would shift every region.
    """
    polygons: list[np.ndarray] = []
    for obj in canvas_json.get("objects", []):
        if str(obj.get("type", "")).lower() != "path":
            continue
        points = []
        for command in obj.get("path", []):
            if not command:
                continue
            if command[0] in {"M", "L"} and len(command) >= 3:
                points.append([float(command[1]), float(command[2])])
            elif command[0] in {"Q", "C"} and len(command) >= 3:
                points.append([float(command[-2]), float(command[-1])])
        if len(points) < 3:
            continue
        raw = np.asarray(points, dtype=np.float32)
        offset = obj.get("pathOffset") or {}
        path_center = np.asarray([
            float(offset.get("x", (raw[:, 0].min() + raw[:, 0].max()) / 2)),
            float(offset.get("y", (raw[:, 1].min() + raw[:, 1].max()) / 2)),
        ], dtype=np.float32)
        width, height = float(obj.get("width", 0)), float(obj.get("height", 0))
        origin_x = {"left": width / 2, "center": 0, "right": -width / 2}.get(obj.get("originX", "left"), 0)
        origin_y = {"top": height / 2, "center": 0, "bottom": -height / 2}.get(obj.get("originY", "top"), 0)
        local = (raw - path_center + np.asarray([origin_x, origin_y], dtype=np.float32))
        local *= np.asarray([float(obj.get("scaleX", 1)), float(obj.get("scaleY", 1))], dtype=np.float32)
        angle = math.radians(float(obj.get("angle", 0)))
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
        polygon = local @ rotation.T + np.asarray([float(obj.get("left", 0)), float(obj.get("top", 0))], dtype=np.float32)
        polygon *= np.asarray([image_scale_x, image_scale_y], dtype=np.float32)
        if abs(cv2.contourArea(polygon)) >= 10:
            polygons.append(polygon)
    # The EPM interface uses polygon mode, but accepting rectangles in tests
    # and programmatic calibration keeps this function easy to verify.
    if not polygons and canvas_json.get("objects"):
        polygons = [
            p for p in extract_polygons_from_canvas(canvas_json, image_scale_x, image_scale_y)
            if abs(cv2.contourArea(p)) >= 10
        ]
    return polygons


def calibration_from_canvas(
    canvas_json: Mapping[str, Any],
    object_for_region: Mapping[str, int],
    frame_width: int,
    frame_height: int,
    image_scale_x: float,
    image_scale_y: float,
) -> EPMCalibration:
    polygons = extract_epm_polygons(canvas_json, image_scale_x, image_scale_y)
    if len(polygons) != 5:
        raise ValueError("Draw exactly five maze polygons.")
    indices = [int(object_for_region[name]) for name in REGION_ORDER]
    if sorted(indices) != list(range(5)):
        raise ValueError("Assign each numbered polygon to exactly one maze region.")
    return EPMCalibration(
        [MazeRegion(name, polygons[index]) for name, index in zip(REGION_ORDER, indices)],
        frame_width,
        frame_height,
    )


def numbered_canvas_overlay(frame: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    output = frame.copy()
    for index, polygon in enumerate(polygons, start=1):
        shape = np.rint(polygon).astype(np.int32)
        cv2.polylines(output, [shape], True, (0, 220, 255), 3, cv2.LINE_AA)
        x, y = np.mean(polygon, axis=0)
        cv2.putText(output, str(index), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 255), 3)
    return output


def draw_epm_overlay(frame: np.ndarray, calibration: EPMCalibration) -> np.ndarray:
    output = frame.copy()
    for region in calibration.regions:
        cv2.polylines(output, [region.as_int_polygon()], True, region.color, 3, cv2.LINE_AA)
        x, y = region.center()
        cv2.putText(output, region.name.upper(), (int(x) - 25, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, region.color, 2)
    return output


def _scoring_point(row: Any, mode: str) -> tuple[float, float] | None:
    if mode == "head_shoulders":
        columns = (
            ("smoothed_head_shoulder_x", "smoothed_head_shoulder_y"),
            ("head_shoulder_x", "head_shoulder_y"),
            ("smoothed_x", "smoothed_y"),
            ("centroid_x", "centroid_y"),
        )
    elif mode == "smoothed_centroid":
        columns = (("smoothed_x", "smoothed_y"), ("centroid_x", "centroid_y"))
    elif mode == "centroid":
        columns = (("centroid_x", "centroid_y"),)
    else:
        raise ValueError(f"Unknown scoring mode: {mode}")
    for x_name, y_name in columns:
        x, y = getattr(row, x_name, np.nan), getattr(row, y_name, np.nan)
        if pd.notna(x) and pd.notna(y):
            return float(x), float(y)
    return None


def assign_epm_frames(tracking_df: pd.DataFrame, calibration: EPMCalibration, mode: str = "head_shoulders") -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in tracking_df.itertuples(index=False):
        point = _scoring_point(row, mode)
        record = row._asdict()
        record["assignment_x"] = point[0] if point else np.nan
        record["assignment_y"] = point[1] if point else np.nan
        record["assignment_point_mode"] = mode
        if point is None:
            label, boundary = "missing", False
        else:
            distances = [region.signed_distance(point) for region in calibration.regions]
            inside = [i for i, distance in enumerate(distances) if distance >= 0]
            label = calibration.regions[inside[0]].name if inside else "outside"
            boundary = len(inside) > 1 or any(abs(distance) < 0.5 for distance in distances)
        record["region"] = label
        record["arm_type"] = "open" if label.startswith("open_") else "closed" if label.startswith("closed_") else label
        record["on_boundary"] = boundary
        record["inside_arena"] = label in REGION_ORDER
        records.append(record)
    return pd.DataFrame(records)


def _whole_seconds(frame_counts: list[int], fps: float) -> list[int]:
    """Largest remainders make reported whole seconds sum to rounded video time."""
    exact = [count / fps for count in frame_counts]
    whole = [math.floor(value) for value in exact]
    target = math.floor(sum(exact) + 0.5)
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - whole[i]), i))
    for index in order[: target - sum(whole)]:
        whole[index] += 1
    return whole


def detect_epm_events(
    per_frame: pd.DataFrame,
    fps: float,
    min_dwell_seconds: float = 0.3,
    count_initial_arm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Confirm stable region transitions; uncertain frames break continuity."""
    if fps <= 0 or min_dwell_seconds < 0:
        raise ValueError("FPS must be positive and minimum dwell cannot be negative.")
    dwell_frames = max(1, math.ceil(min_dwell_seconds * fps))
    data = per_frame.copy()
    data["event"] = ""
    events: list[dict[str, Any]] = []
    stable: str | None = None
    candidate: str | None = None
    candidate_start = -1
    candidate_count = 0
    first_eligible_frame: int | None = None
    first_baseline_committed = False
    skipped_direct_arm_changes = 0

    def commit(destination: str, start: int, confirmed: int, source: str | None) -> None:
        nonlocal skipped_direct_arm_changes, first_baseline_committed
        event_type = ""
        if source is None:
            if (not first_baseline_committed and count_initial_arm and destination != "center"
                    and first_eligible_frame is not None and first_eligible_frame <= dwell_frames):
                event_type = "open_arm_entry" if destination.startswith("open_") else "closed_arm_entry"
        elif source == "center" and destination.startswith("open_"):
            event_type = "open_arm_entry"
        elif source == "center" and destination.startswith("closed_"):
            event_type = "closed_arm_entry"
        elif source != "center" and destination == "center":
            event_type = "center_entry"
        elif source != "center" and destination != "center":
            skipped_direct_arm_changes += 1
        first_baseline_committed = True
        if event_type:
            first, last = data.iloc[start], data.iloc[confirmed]
            events.append({
                "event": event_type,
                "from_region": source or "start",
                "to_region": destination,
                "frame_index": int(first["frame_index"]),
                "time_seconds": float(first["frame_index"]) / fps,
                "confirmed_frame_index": int(last["frame_index"]),
            })
            data.at[data.index[confirmed], "event"] = event_type

    for position, (_, row) in enumerate(data.iterrows()):
        region = str(row["region"])
        eligible = (
            region in REGION_ORDER
            and row.get("tracking_status") == "tracked"
            and not bool(row.get("low_confidence", False))
            and not bool(row.get("carried_forward", False))
        )
        if not eligible:
            stable = candidate = None
            candidate_count = 0
            continue
        if first_eligible_frame is None:
            first_eligible_frame = int(row["frame_index"])
        if region == stable:
            candidate = None
            candidate_count = 0
            continue
        if region != candidate:
            candidate = region
            candidate_start = position
            candidate_count = 1
        else:
            candidate_count += 1
        if candidate_count >= dwell_frames:
            commit(region, candidate_start, position, stable)
            stable = region
            candidate = None
            candidate_count = 0
    event_columns = ["event", "from_region", "to_region", "frame_index", "time_seconds", "confirmed_frame_index"]
    return data, pd.DataFrame(events, columns=event_columns), skipped_direct_arm_changes


@dataclass
class EPMBundle:
    per_frame: pd.DataFrame
    summary: pd.DataFrame
    region_times: pd.DataFrame
    events: pd.DataFrame
    qc_metrics: pd.DataFrame
    warnings: list[str]


def create_epm_bundle(
    tracking_df: pd.DataFrame,
    calibration: EPMCalibration,
    fps: float,
    mode: str = "head_shoulders",
    min_dwell_seconds: float = 0.3,
    count_initial_arm: bool = True,
) -> EPMBundle:
    if fps <= 0 or tracking_df.empty:
        raise ValueError("A video with frames and positive FPS is required.")
    assigned = assign_epm_frames(tracking_df, calibration, mode)
    per_frame, events, skipped = detect_epm_events(assigned, fps, min_dwell_seconds, count_initial_arm)
    counts = {name: int((per_frame["region"] == name).sum()) for name in REGION_ORDER}
    open_frames = counts["open_1"] + counts["open_2"]
    closed_frames = counts["closed_1"] + counts["closed_2"]
    center_frames = counts["center"]
    unknown_frames = len(per_frame) - open_frames - closed_frames - center_frames
    open_seconds, closed_seconds, center_seconds, unknown_seconds = _whole_seconds(
        [open_frames, closed_frames, center_frames, unknown_frames], fps
    )
    event_counts = events["event"].value_counts()
    summary = pd.DataFrame([{
        "Open Arm Entries": int(event_counts.get("open_arm_entry", 0)),
        "Closed Arm Entries": int(event_counts.get("closed_arm_entry", 0)),
        "Center Entry": int(event_counts.get("center_entry", 0)),
        "Open Arm Time (whole seconds)": open_seconds,
        "Closed Arm Time (whole seconds)": closed_seconds,
        "Center Time (whole seconds)": center_seconds,
    }], columns=SUMMARY_COLUMNS)
    region_times = pd.DataFrame([
        {"region": name, "frames": counts[name], "seconds": round(counts[name] / fps, 3)}
        for name in REGION_ORDER
    ] + [{"region": "unclassified", "frames": unknown_frames, "seconds": round(unknown_frames / fps, 3)}])
    qc, warnings = compute_qc_metrics(per_frame)
    qc = pd.concat([qc, pd.DataFrame([
        {"metric": "unclassified_frames", "value": unknown_frames},
        {"metric": "unclassified_whole_seconds", "value": unknown_seconds},
        {"metric": "skipped_direct_arm_changes", "value": skipped},
        {"metric": "event_dwell_frames", "value": max(1, math.ceil(min_dwell_seconds * fps))},
    ])], ignore_index=True)
    warnings = [warning.replace("chamber", "maze region") for warning in warnings]
    if unknown_frames:
        warnings.append(f"{unknown_frames} frames were outside the five regions or missing. Inspect the maze drawing and tracking video.")
    if skipped:
        warnings.append(f"{skipped} direct arm-to-arm changes lacked an observed center crossing and were not counted as entries.")
    if bool(per_frame["carried_forward"].fillna(False).any()):
        warnings.append("Carried-forward positions contribute to time estimates but never create entry events; review their frames.")
    return EPMBundle(per_frame, summary, region_times, events, qc, warnings)
