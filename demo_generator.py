from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from io_utils import ensure_directory
from regions import create_three_chambers_from_box


@dataclass
class SegmentSpec:
    chamber: str
    duration_seconds: float


def default_segments() -> list[SegmentSpec]:
    return [
        SegmentSpec("left", 4.0),
        SegmentSpec("center", 2.5),
        SegmentSpec("right", 4.0),
        SegmentSpec("center", 1.5),
        SegmentSpec("left", 2.0),
    ]


def generate_synthetic_cpp_video(
    output_dir: str | Path,
    video_name: str = "synthetic_cpp_demo.mp4",
    fps: float = 15.0,
    frame_size: tuple[int, int] = (960, 360),
    segments: Sequence[SegmentSpec] | None = None,
    seed: int = 7,
) -> dict[str, Path]:
    """Create a simple synthetic three-chamber video and matching ground truth."""

    rng = np.random.default_rng(seed)
    output_directory = ensure_directory(output_dir)
    video_path = output_directory / video_name
    truth_path = output_directory / f"{Path(video_name).stem}_ground_truth.csv"
    summary_path = output_directory / f"{Path(video_name).stem}_expected_summary.csv"

    width, height = frame_size
    arena_box = (
        int(width * 0.06),
        int(height * 0.18),
        int(width * 0.88),
        int(height * 0.64),
    )
    calibration = create_three_chambers_from_box(
        box=arena_box,
        frame_width=width,
        frame_height=height,
        boundary_margin_px=0.0,
    )

    specs = list(segments or default_segments())
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    records: list[dict[str, float | str | int]] = []
    frame_index = 0
    rat_radius = max(10, width // 60)

    for spec in specs:
        segment_frames = max(1, int(round(spec.duration_seconds * fps)))
        chamber = next(chamber for chamber in calibration.chambers if chamber.name == spec.chamber)
        xs = chamber.polygon[:, 0]
        ys = chamber.polygon[:, 1]
        min_x = int(xs.min()) + 25
        max_x = int(xs.max()) - 25
        min_y = int(ys.min()) + 25
        max_y = int(ys.max()) - 25

        base_x = (min_x + max_x) / 2.0
        base_y = (min_y + max_y) / 2.0

        for local_index in range(segment_frames):
            frame = np.full((height, width, 3), 210, dtype=np.uint8)
            gradient = np.linspace(0, 20, width, dtype=np.float32)[None, :]
            green_channel = frame[:, :, 1].astype(np.float32) + gradient
            red_channel = frame[:, :, 2].astype(np.float32) + (gradient / 2.0)
            frame[:, :, 1] = np.clip(green_channel, 0, 255).astype(np.uint8)
            frame[:, :, 2] = np.clip(red_channel, 0, 255).astype(np.uint8)

            x, y, box_w, box_h = arena_box
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (55, 55, 55), 4)
            divider_1 = x + int(box_w / 3)
            divider_2 = x + int(2 * box_w / 3)
            cv2.line(frame, (divider_1, y), (divider_1, y + box_h), (80, 80, 80), 3)
            cv2.line(frame, (divider_2, y), (divider_2, y + box_h), (80, 80, 80), 3)

            phase = (local_index / max(segment_frames - 1, 1)) * (2.0 * np.pi)
            centroid_x = base_x + (0.22 * (max_x - min_x) * np.sin(phase))
            centroid_y = base_y + (0.18 * (max_y - min_y) * np.cos(phase * 1.5))
            centroid_x += rng.normal(0.0, 3.0)
            centroid_y += rng.normal(0.0, 2.0)

            centroid_x = float(np.clip(centroid_x, min_x, max_x))
            centroid_y = float(np.clip(centroid_y, min_y, max_y))

            shadow_offset = 7
            cv2.circle(frame, (int(centroid_x) + shadow_offset, int(centroid_y) + shadow_offset), rat_radius + 2, (140, 140, 140), -1)
            cv2.circle(frame, (int(centroid_x), int(centroid_y)), rat_radius, (30, 30, 30), -1)

            writer.write(frame)
            records.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / fps,
                    "expected_chamber": spec.chamber,
                    "expected_centroid_x": centroid_x,
                    "expected_centroid_y": centroid_y,
                }
            )
            frame_index += 1

    writer.release()

    truth_df = pd.DataFrame(records)
    truth_df.to_csv(truth_path, index=False)

    summary_df = (
        truth_df.groupby("expected_chamber")
        .size()
        .reset_index(name="frames")
        .rename(columns={"expected_chamber": "chamber"})
    )
    summary_df["seconds"] = summary_df["frames"] / fps
    summary_df["percent_of_video"] = summary_df["frames"] / max(len(truth_df), 1) * 100.0
    summary_df.to_csv(summary_path, index=False)

    calibration_path = output_directory / f"{Path(video_name).stem}_calibration.json"
    calibration_path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")

    return {
        "video_path": video_path,
        "ground_truth_csv": truth_path,
        "summary_csv": summary_path,
        "calibration_json": calibration_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic three-chamber CPP practice video.")
    parser.add_argument(
        "--output-dir",
        default="demo_outputs",
        help="Folder where the demo video and CSV files should be saved.",
    )
    args = parser.parse_args()

    outputs = generate_synthetic_cpp_video(output_dir=args.output_dir)
    print("Synthetic demo video created.")
    print(f"Video: {outputs['video_path']}")
    print(f"Ground truth CSV: {outputs['ground_truth_csv']}")
    print(f"Expected summary CSV: {outputs['summary_csv']}")


if __name__ == "__main__":
    main()
