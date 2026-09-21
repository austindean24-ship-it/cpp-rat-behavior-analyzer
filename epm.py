"""Elevated plus maze scoring. CPP files remain unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd

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


def calibration_from_saved_settings(
    settings: Mapping[str, Any], frame_width: int, frame_height: int,
    video_signature: str | None = None,
) -> EPMCalibration:
    """Load the exact saved maze geometry only for the matching recording."""
    saved_signature = settings.get("video_signature")
    if saved_signature and video_signature and saved_signature != video_signature:
        raise ValueError("The saved calibration belongs to a different video. Draw new regions for this recording.")
    source = settings.get("calibration", settings)
    if source.get("frame_width") != frame_width or source.get("frame_height") != frame_height:
        raise ValueError("Saved calibration dimensions differ from this video.")
    regions = [MazeRegion(item["name"], np.asarray(item["polygon"], dtype=np.float32))
               for item in source["regions"]]
    return EPMCalibration(regions, frame_width, frame_height)


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


@dataclass
class EPMBundle:
    per_frame: pd.DataFrame
    summary: pd.DataFrame
    region_times: pd.DataFrame
    events: pd.DataFrame
    qc_metrics: pd.DataFrame
    warnings: list[str]


def _region(point, calibration: EPMCalibration, margin: float):
    if point is None or not np.isfinite(point).all():
        return 'unclassified', '', False
    distances = [r.signed_distance(point) for r in calibration.regions]
    inside = [i for i, d in enumerate(distances) if d >= 0]
    if len(inside) > 1:
        return 'unclassified', calibration.regions[inside[0]].name, True
    if not inside:
        return 'outside', '', False
    label = calibration.regions[inside[0]].name
    return ('unclassified' if abs(distances[inside[0]]) < margin else label), label, abs(distances[inside[0]]) < margin


def _xy(row, cols):
    for xname, yname in cols:
        x = row.get(xname, np.nan)
        y = row.get(yname, np.nan)
        if pd.notna(x) and pd.notna(y) and np.isfinite(float(x)) and np.isfinite(float(y)):
            return float(x), float(y), xname.removesuffix('_x')
    return np.nan, np.nan, 'missing'


def assign_epm_frames(tracking_df, calibration: EPMCalibration, mode='head_shoulders', boundary_margin_px=2.0):
    if mode not in {'head_shoulders', 'smoothed_centroid', 'centroid'}:
        raise ValueError(f'Unknown EPM entry proxy: {mode}')
    if boundary_margin_px < 0:
        raise ValueError('Boundary margin cannot be negative')
    records = []
    for row in tracking_df.to_dict('records'):
        accepted = row.get('tracking_status') == 'tracked' and not bool(row.get('low_confidence', False)) and not bool(row.get('carried_forward', False))
        cx, cy, body_source = _xy(row, [('smoothed_x', 'smoothed_y'), ('centroid_x', 'centroid_y')]) if accepted else (np.nan, np.nan, 'missing')
        body_point = (cx, cy) if np.isfinite(cx) else None
        body_region, body_candidate, body_boundary = _region(body_point, calibration, boundary_margin_px)
        raw_x, raw_y, _ = _xy(row, [('centroid_x', 'centroid_y')]) if accepted else (np.nan, np.nan, 'missing')
        raw_region, _, _ = _region((raw_x, raw_y) if np.isfinite(raw_x) else None, calibration, boundary_margin_px)
        centroid_disagreement = body_region in REGION_ORDER and raw_region != body_region
        if centroid_disagreement:
            body_region = 'unclassified'
            body_boundary = True
        if not accepted:
            body_region = 'unclassified'
        if mode == 'head_shoulders':
            proxy_valid = row.get('head_estimate_source', 'provided_head_proxy') in {'motion_heading', 'provided_head_proxy'}
            ex, ey, entry_source = _xy(row, [('smoothed_head_shoulder_x', 'smoothed_head_shoulder_y'), ('head_shoulder_x', 'head_shoulder_y')]) if accepted and proxy_valid else (np.nan, np.nan, 'missing_head_orientation')
        elif mode == 'smoothed_centroid':
            ex, ey, entry_source = cx, cy, body_source
        else:
            ex, ey, entry_source = _xy(row, [('centroid_x', 'centroid_y')]) if accepted else (np.nan, np.nan, 'missing')
        entry_region, entry_candidate, entry_boundary = _region((ex, ey) if np.isfinite(ex) else None, calibration, boundary_margin_px)
        if mode == 'smoothed_centroid' and centroid_disagreement:
            entry_region = 'unclassified'
            entry_boundary = True
        row.update({
            'assignment_x': cx, 'assignment_y': cy, 'assignment_point_mode': 'body_occupancy',
            'assignment_point_source': body_source, 'entry_assignment_x': ex, 'entry_assignment_y': ey,
            'entry_point_mode': mode, 'entry_point_source': entry_source,
            'entry_region': entry_region, 'entry_candidate_region': entry_candidate,
            'entry_boundary': entry_boundary,
            'raw_body_region': raw_region, 'centroid_disagreement': centroid_disagreement,
            'candidate_region': body_candidate, 'region': body_region,
            'arm_type': 'open' if body_region.startswith('open_') else 'closed' if body_region.startswith('closed_') else body_region,
            'on_boundary': body_boundary, 'inside_arena': body_region in REGION_ORDER,
            'boundary_hold': bool(entry_boundary and accepted),
            'body_position_valid': bool(accepted and body_region in REGION_ORDER),
            'anatomical_entry_point_valid': False,  # motion heading is NOT anatomy
        })
        records.append(row)
    return pd.DataFrame.from_records(records)


def detect_epm_events(assigned, fps, min_dwell_seconds=0.3, count_initial_arm=True):
    """Only uninterrupted observed proxy transitions contribute to counts.

    A gap preserves the last region as reviewer context, never as evidence of a
    crossing. Reviewer items are emitted after target dwell, not for every peek.
    """
    if fps <= 0 or min_dwell_seconds < 0:
        raise ValueError("FPS must be positive and dwell must be nonnegative")
    dwell = max(1, math.ceil(fps * min_dwell_seconds))
    data = assigned.copy()
    data["event"] = ""
    data["review_event"] = ""
    events = []
    included = np.flatnonzero(data["analysis_included"].to_numpy()) if "analysis_included" in data else np.arange(len(data))
    interval_start = int(included[0]) if len(included) else 0
    stable = pending = None
    pending_start = -1
    pending_count = 0
    pending_uncertain = False
    interrupted = True
    last_observed_frame = last_observed_segment = None
    last_stable_frame = None
    boundary_frames = 0
    pending_boundary_frames = 0
    first_valid = None
    baselined = False
    skipped = 0

    def emit(name, source, target, onset, confirmation, review, reason, source_frame, boundary_count):
        first, last = data.iloc[onset], data.iloc[confirmation]
        events.append({
            "event": name, "from_region": source or "start", "to_region": target,
            "frame_index": int(first["frame_index"]),
            "time_seconds": float(first["frame_index"]) / fps,
            "confirmed_frame_index": int(last["frame_index"]),
            "confirmed_time_seconds": float(last["frame_index"]) / fps,
            "supporting_frames": confirmation - onset + 1,
            "source_last_frame_index": source_frame if source_frame is not None else np.nan,
            "observed_boundary_frames": boundary_count,
            "track_segment_id": first.get("track_segment_id", np.nan),
            "point_source": first.get("entry_point_source", ""),
            "evidence": reason,
            "review_state": review,
        })
        field = "event" if review == "confirmed_proxy" else "review_event"
        data.at[data.index[confirmation], field] = name

    for pos, row in enumerate(data.to_dict("records")):
        frame = int(row["frame_index"])
        segment = row.get("track_segment_id", 0)
        region = row.get("entry_region", row.get("region", "unclassified"))
        tracked = (row.get("tracking_status") == "tracked"
                   and not bool(row.get("low_confidence", False))
                   and not bool(row.get("carried_forward", False))
                   and bool(row.get("analysis_included", True)))
        observed = tracked and pd.notna(segment) and (region in REGION_ORDER or bool(row.get("boundary_hold", False)))
        if not observed:
            interrupted = True
            pending = None
            pending_count = 0
            last_observed_frame = last_observed_segment = None
            boundary_frames = 0
            continue
        adjacent = last_observed_frame == frame - 1 and last_observed_segment == segment
        if not adjacent:
            interrupted = True
            pending = None
            pending_count = 0
            boundary_frames = 0
        last_observed_frame, last_observed_segment = frame, segment
        if bool(row.get("boundary_hold", False)):
            # A measured border point can bridge region labels only while the
            # same track is present on every frame.
            boundary_frames += 1
            pending = None
            pending_count = 0
            continue
        if first_valid is None:
            first_valid = pos
        if region == stable:
            pending = None
            pending_count = 0
            interrupted = False
            last_stable_frame = frame
            boundary_frames = 0
            continue
        if pending != region:
            pending = region
            pending_start = pos
            pending_count = 1
            pending_uncertain = interrupted or stable is None
            pending_boundary_frames = boundary_frames
        else:
            pending_count += 1
        if pending_count < dwell:
            continue
        if stable is None:
            if not baselined and count_initial_arm and region != "center" and first_valid - interval_start <= dwell:
                emit("open_arm_entry" if region.startswith("open_") else "closed_arm_entry",
                     None, region, pending_start, pos, "confirmed_proxy", "initial_arm_occupancy",
                     None, pending_boundary_frames)
        else:
            if stable == "center" and region.startswith("open_"):
                name = "open_arm_entry"
            elif stable == "center" and region.startswith("closed_"):
                name = "closed_arm_entry"
            elif stable != "center" and region == "center":
                name = "center_entry"
            else:
                name = "possible_transition"
                skipped += 1
            uncertain = pending_uncertain or name == "possible_transition"
            emit("possible_transition" if uncertain else name, stable, region,
                 pending_start, pos, "requires_manual_review" if uncertain else "confirmed_proxy",
                 "unobserved_interval_or_track_change" if pending_uncertain else
                 "direct_arm_change_without_observed_center" if name == "possible_transition" else
                 "continuous_observed_proxy_crossing", last_stable_frame, pending_boundary_frames)
        stable = region
        baselined = True
        pending = None
        pending_count = 0
        interrupted = False
        last_stable_frame = frame
        boundary_frames = 0
    columns = ["event", "from_region", "to_region", "frame_index", "time_seconds",
               "confirmed_frame_index", "confirmed_time_seconds", "supporting_frames",
               "source_last_frame_index", "observed_boundary_frames", "track_segment_id",
               "point_source", "evidence", "review_state"]
    return data, pd.DataFrame(events, columns=columns), skipped

def _whole_seconds(counts, fps):
    exact = [v / fps for v in counts]
    values = [math.floor(x) for x in exact]
    target = math.floor(sum(exact) + 0.5)
    ranking = sorted(range(len(values)), key=lambda i: (-(exact[i] - values[i]), i))
    for idx in ranking[:target - sum(values)]:
        values[idx] += 1
    return values


def create_epm_bundle(tracking_df, calibration, fps, mode='head_shoulders',
                      min_dwell_seconds=0.3, count_initial_arm=True,
                      analysis_start_seconds=0., analysis_end_seconds=None,
                      boundary_margin_px=2.):
    if tracking_df.empty or fps <= 0:
        raise ValueError('Need nonempty tracking and positive FPS')
    first = round(analysis_start_seconds * fps)
    end = len(tracking_df) if analysis_end_seconds is None else round(analysis_end_seconds * fps)
    if not 0 <= first < end <= len(tracking_df):
        raise ValueError('Analysis interval must be nonempty and inside video')
    data = assign_epm_frames(tracking_df, calibration, mode, boundary_margin_px)
    data['analysis_included'] = data['frame_index'].between(first, end - 1)
    data.loc[~data['analysis_included'], ['region', 'arm_type', 'entry_region']] = 'excluded'
    per_frame, events, skipped = detect_epm_events(data, fps, min_dwell_seconds, count_initial_arm)
    selected = per_frame[per_frame['analysis_included']]
    counts = {name: int((selected['region'] == name).sum()) for name in REGION_ORDER}
    unknown = len(selected) - sum(counts.values())
    secs = _whole_seconds([counts['open_1'] + counts['open_2'],
                           counts['closed_1'] + counts['closed_2'], counts['center'], unknown], fps)
    confirmed = events.loc[events['review_state'] == 'confirmed_proxy', 'event'].value_counts() if len(events) else pd.Series(dtype=int)
    summary = pd.DataFrame([{
        'Open Arm Entries': int(confirmed.get('open_arm_entry', 0)),
        'Closed Arm Entries': int(confirmed.get('closed_arm_entry', 0)),
        'Center Entry': int(confirmed.get('center_entry', 0)),
        'Open Arm Time (whole seconds)': secs[0],
        'Closed Arm Time (whole seconds)': secs[1],
        'Center Time (whole seconds)': secs[2],
    }], columns=SUMMARY_COLUMNS)
    region_times = pd.DataFrame([{'region': k, 'frames': v, 'seconds': round(v / fps, 3)} for k, v in counts.items()] +
                                [{'region': 'unclassified', 'frames': unknown, 'seconds': round(unknown / fps, 3)}])
    statuses = selected['tracking_status']
    rejected = int((statuses == 'rejected').sum())
    review_count = int((events['review_state'] == 'requires_manual_review').sum())
    qc_dict = {
        'analysis_start_frame': first, 'analysis_end_frame_exclusive': end,
        'analysis_duration_seconds': round(len(selected) / fps, 3),
        'scored_coverage_percent': round(100 * (len(selected) - unknown) / len(selected), 2),
        'unclassified_frames': unknown, 'unclassified_seconds': round(unknown / fps, 3),
        'unclassified_whole_seconds': secs[3],
        'low_confidence_frames': int(selected['low_confidence'].fillna(False).sum()),
        'lost_frames': int((statuses == 'lost').sum()),
        'reacquiring_frames': int((statuses == 'reacquiring').sum()),
        'rejected_candidate_frames': rejected,
        'rejected_raw_jumps': int((selected['rejection_reason'] == 'raw_jump').sum()) if 'rejection_reason' in selected else 0,
        'boundary_unclassified_frames': int(selected['on_boundary'].sum()),
        'missing_head_proxy_frames': int(((statuses == 'tracked') &
            ~selected['head_estimate_source'].isin({'motion_heading', 'provided_head_proxy'})).sum()) if 'head_estimate_source' in selected else 0,
        'skipped_direct_arm_changes': skipped,
        'possible_transitions_for_review': review_count,
        'event_dwell_frames': max(1, math.ceil(min_dwell_seconds * fps)),
        'occupancy_source': 'tracked_body_centroid',
        'entry_proxy': mode,
        'review_state': 'needs_manual_review',
    }
    qc = pd.DataFrame([{'metric': k, 'value': v} for k, v in qc_dict.items()] +
                      [{'metric': f'rejection_reason_{k}', 'value': int(v)} for k, v in
                       selected.get('rejection_reason', pd.Series(dtype=str)).dropna().value_counts().items() if k])
    warnings = ['Needs manual review: operational entry proxies are NOT verified four-paw scoring.']
    warnings += [f'{unknown} frames remain unclassified; uncertain time is not invented.',
                 'Arm time uses tracked body centroid; entry events use selected operational proxy.']
    if review_count:
        warnings.append(f'{review_count} possible transitions require manual adjudication and were NOT counted.')
    if rejected:
        warnings.append(f'{rejected} frames contained rejected candidates; inspect the QC video.')
    if qc_dict['scored_coverage_percent'] < 90:
        warnings.append('Body-position coverage below 90%; inspect tracking and calibration.')
    return EPMBundle(per_frame, summary, region_times, events, qc, warnings)
