"""Annotated EPM video for visual quality review."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd

from epm import EPMCalibration
from io_utils import ensure_directory, get_video_metadata

ProgressCallback = Callable[[int, int], None]


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
    trail: deque[tuple[int, int]] = deque(maxlen=120)
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
            if row is not None:
                label = str(row.region)
                status = str(row.tracking_status)
                event = str(row.event)
                if pd.notna(row.assignment_x) and pd.notna(row.assignment_y):
                    point = (int(row.assignment_x), int(row.assignment_y))
                    trail.append(point)
                    color = (0, 210, 255) if bool(row.low_confidence) else (0, 255, 0)
                    cv2.circle(frame, point, 7, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, point, 13, color, 2, cv2.LINE_AA)
            if draw_trajectory and len(trail) > 1:
                for start, end in zip(list(trail)[:-1], list(trail)[1:]):
                    cv2.line(frame, start, end, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (8, 6), (min(metadata.width - 8, 820), 91 if event else 68), (20, 20, 20), -1)
            cv2.putText(frame, f"Frame {processed}  |  {(float(row.time_seconds) if row is not None else processed / metadata.fps):.2f} s  |  {label}", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Tracking: {status}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255) if row is not None and row.low_confidence else (200, 255, 200), 2, cv2.LINE_AA)
            if event:
                cv2.putText(frame, event.replace("_", " ").upper(), (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
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
