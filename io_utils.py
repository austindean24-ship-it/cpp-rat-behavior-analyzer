from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pandas as pd

from regions import ArenaCalibration


DEFAULT_FPS_FALLBACK = 30.0
ProgressCallback = Callable[[int, int], None]


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    fps_was_inferred: bool = False
    notes: list[str] = field(default_factory=list)


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def sanitize_filename(name: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in name)
    return safe or "uploaded_video.mp4"


def save_uploaded_video(uploaded_file, output_dir: str | Path) -> Path:
    output_dir_path = ensure_directory(output_dir)
    safe_name = sanitize_filename(getattr(uploaded_file, "name", "uploaded_video.mp4"))
    output_path = output_dir_path / safe_name
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def get_video_metadata(video_path: str | Path, fps_fallback: float = DEFAULT_FPS_FALLBACK) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open the video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    notes: list[str] = []
    fps_was_inferred = False
    if fps <= 1.0 or fps > 240.0 or np.isnan(fps):
        fps = float(fps_fallback)
        fps_was_inferred = True
        notes.append(
            f"Video FPS metadata looked missing or unusual, so the app used {fps_fallback:.1f} FPS."
        )

    duration_seconds = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return VideoMetadata(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        fps_was_inferred=fps_was_inferred,
        notes=notes,
    )


def read_first_frame(video_path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    cap.release()
    if not success or frame is None:
        raise ValueError("Could not read the first frame from the video.")
    return frame


def export_dataframe_csv(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_directory(output.parent)
    dataframe.to_csv(output, index=False)
    return output


def export_warnings_text(warnings: Iterable[str], output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_directory(output.parent)
    output.write_text("\n".join(warnings), encoding="utf-8")
    return output


def write_annotated_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    per_frame_df: pd.DataFrame,
    calibration: ArenaCalibration,
    draw_trajectory: bool = True,
    trail_length: int = 120,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for annotation: {input_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS_FALLBACK)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or calibration.frame_width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or calibration.frame_height)

    output_path = Path(output_video_path)
    ensure_directory(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    indexed_rows = per_frame_df.set_index("frame_index")
    valid_points: list[tuple[int, int]] = []
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), len(per_frame_df), 1)
    processed_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        row = indexed_rows.loc[frame_index]

        overlay = frame.copy()
        for chamber in calibration.chambers:
            polygon = chamber.as_int_polygon()
            cv2.polylines(overlay, [polygon], True, chamber.color, 2, cv2.LINE_AA)
            center_x, center_y = chamber.center()
            cv2.putText(
                overlay,
                chamber.name.upper(),
                (int(center_x) - 45, int(center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                chamber.color,
                2,
                cv2.LINE_AA,
            )

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0.0)

        if not np.isnan(row["assignment_x"]) and not np.isnan(row["assignment_y"]):
            point = (int(row["assignment_x"]), int(row["assignment_y"]))
            valid_points.append(point)
            if len(valid_points) > trail_length:
                valid_points.pop(0)
            point_color = (0, 255, 0) if not bool(row["low_confidence"]) else (0, 215, 255)
            cv2.circle(frame, point, 6, point_color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, 12, point_color, 2, cv2.LINE_AA)

        if draw_trajectory and len(valid_points) > 1:
            for start, end in zip(valid_points[:-1], valid_points[1:]):
                cv2.line(frame, start, end, (255, 255, 255), 2, cv2.LINE_AA)

        chamber_label = str(row["chamber"])
        status_text = f"Frame {frame_index} | Chamber: {chamber_label}"
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if bool(row["low_confidence"]):
            cv2.putText(
                frame,
                "Low confidence frame",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        processed_frames += 1
        if progress_callback is not None and (processed_frames == total_frames or processed_frames % 60 == 0):
            progress_callback(processed_frames, total_frames)

    cap.release()
    writer.release()
    return output_path
