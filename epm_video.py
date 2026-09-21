"""Annotated EPM video for visual quality review."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from epm import EPMCalibration
from io_utils import ensure_directory, get_video_metadata

ProgressCallback = Callable[[int, int], None]


def _valid_trail_edge(start: tuple[int, tuple[int, int], int], end: tuple[int, tuple[int, int], int], mask: np.ndarray) -> bool:
    if end[0] != start[0] + 1 or end[2] != start[2]:
        return False
    a, b = start[1], end[1]
    distance = float(np.linalg.norm(np.subtract(a, b)))
    for t in np.linspace(0, 1, max(2, int(distance / 2))):
        x, y = round(a[0] + t * (b[0] - a[0])), round(a[1] + t * (b[1] - a[1]))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]):
            return False
    return True


def write_annotated_epm_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    per_frame: pd.DataFrame,
    calibration: EPMCalibration,
    draw_trajectory: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    metadata = get_video_metadata(input_video_path)
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError("Could not reopen the input video for annotation.")
    output_path = Path(output_video_path)
    ensure_directory(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the annotated MP4.")
    rows = {int(row.frame_index): row for row in per_frame.itertuples(index=False)}
    trail: deque[tuple[int, tuple[int, int], int] | None] = deque(maxlen=120)
    walkway = calibration.tracking_mask()
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row = rows.get(processed)
            for region in calibration.regions:
                cv2.polylines(frame, [region.as_int_polygon()], True, region.color, 2, cv2.LINE_AA)
                x, y = region.center()
                cv2.putText(frame, region.name.upper(), (int(x) - 35, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, region.color, 2, cv2.LINE_AA)
            label = "missing"
            status = "no tracking row"
            event = ""
            reason = ""
            if row is not None:
                label = str(row.region)
                status = str(row.tracking_status)
                event = "" if pd.isna(row.event) else str(row.event)
                review_value = getattr(row, "review_event", "")
                review_event = "" if pd.isna(review_value) else str(review_value)
                reason_value = getattr(row, "rejection_reason", "")
                reason = "" if pd.isna(reason_value) else str(reason_value)
                if pd.notna(row.assignment_x) and pd.notna(row.assignment_y) and label not in {"excluded", "outside", "unclassified"}:
                    point = (int(row.assignment_x), int(row.assignment_y))
                    segment_id = int(row.track_segment_id) if hasattr(row, "track_segment_id") and pd.notna(row.track_segment_id) else 0
                    trail.append((processed, point, segment_id))
                    cv2.circle(frame, point, 7, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, point, 13, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    trail.append(None)
                if status != "tracked" and hasattr(row, "raw_candidate_x") and pd.notna(row.raw_candidate_x):
                    raw = (int(row.raw_candidate_x), int(row.raw_candidate_y))
                    cv2.drawMarker(frame, raw, (0, 140, 255), cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
                elif status == "tracked" and pd.isna(row.assignment_x) and pd.notna(row.centroid_x):
                    cv2.circle(frame, (int(row.centroid_x), int(row.centroid_y)), 8, (0, 210, 255), 2, cv2.LINE_AA)
                if (getattr(row, "entry_region", "unclassified") in {"center", "open_1", "open_2", "closed_1", "closed_2"}
                        and pd.notna(getattr(row, "entry_assignment_x", np.nan))
                        and pd.notna(getattr(row, "entry_assignment_y", np.nan))):
                    proxy = (int(row.entry_assignment_x), int(row.entry_assignment_y))
                    cv2.drawMarker(frame, proxy, (255, 220, 0), cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
                elif (status == "tracked" and getattr(row, "entry_point_mode", "") == "head_shoulders"
                      and getattr(row, "entry_point_source", "") == "missing_head_orientation"
                      and pd.notna(row.centroid_x)):
                    cv2.circle(frame, (int(row.centroid_x), int(row.centroid_y)), 18, (0, 210, 255), 2, cv2.LINE_AA)
            else:
                trail.append(None)
            if draw_trajectory and len(trail) > 1:
                for start, end in zip(list(trail)[:-1], list(trail)[1:]):
                    if start is not None and end is not None and _valid_trail_edge(start, end, walkway):
                        cv2.line(frame, start[1], end[1], (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (8, 6), (min(metadata.width - 8, 1120), 143), (20, 20, 20), -1)
            cv2.putText(frame, f"Frame {processed}  |  {(float(row.time_seconds) if row is not None else processed / metadata.fps):.2f} s  |  {label}", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Tracking: {status}  {reason[:55]}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255) if row is not None and row.low_confidence else (200, 255, 200), 2, cv2.LINE_AA)
            if event:
                cv2.putText(frame, f"PROVISIONAL ENTRY: {event.replace('_', ' ').upper()}", (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
            elif row is not None and review_event:
                cv2.putText(frame, "POSSIBLE TRANSITION - REVIEW RAW VIDEO", (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Green: body occupancy  Cyan +: entry proxy  Orange X: rejected candidate", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "White: continuous body path  Yellow ring: head orientation unavailable  |  Review all entries", (18, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            processed += 1
            if progress_callback and (processed % 60 == 0 or processed == len(per_frame)):
                progress_callback(processed, len(per_frame))
    finally:
        cap.release()
        writer.release()
    if processed != len(per_frame):
        raise ValueError(f"Annotated video frame count ({processed}) did not match tracking rows ({len(per_frame)}).")
    return output_path
