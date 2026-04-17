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
from regions import ArenaCalibration, ChamberRegion, assign_point_to_chamber, create_three_chambers_from_box


@dataclass
class SegmentSpec:
    chamber: str
    duration_seconds: float


def default_segments() -> list[SegmentSpec]:
    return [
        SegmentSpec("left", 55.0),
        SegmentSpec("center", 25.0),
        SegmentSpec("right", 70.0),
        SegmentSpec("center", 20.0),
        SegmentSpec("left", 60.0),
        SegmentSpec("right", 45.0),
        SegmentSpec("center", 25.0),
    ]


def _draw_demo_arena(frame: np.ndarray, arena_box: tuple[int, int, int, int], width: int) -> None:
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


def _chamber_bounds(chamber: ChamberRegion, margin_px: float) -> tuple[float, float, float, float]:
    xs = chamber.polygon[:, 0]
    ys = chamber.polygon[:, 1]
    return (
        float(xs.min() + margin_px),
        float(xs.max() - margin_px),
        float(ys.min() + margin_px),
        float(ys.max() - margin_px),
    )


def _sample_chamber_target(
    chamber: ChamberRegion,
    margin_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    min_x, max_x, min_y, max_y = _chamber_bounds(chamber, margin_px)
    return np.asarray(
        [
            rng.uniform(min_x, max_x),
            rng.uniform(min_y, max_y),
        ],
        dtype=np.float32,
    )


def _make_segment_targets(
    calibration: ArenaCalibration,
    specs: Sequence[SegmentSpec],
    rat_radius: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Create smoothly connected chamber targets for the synthetic rat."""

    chambers = {chamber.name: chamber for chamber in calibration.chambers}
    margin_px = max(float(rat_radius * 2.4), 26.0)
    targets: list[np.ndarray] = []
    previous_target: np.ndarray | None = None

    for spec in specs:
        chamber = chambers[spec.chamber]
        target = _sample_chamber_target(chamber, margin_px=margin_px, rng=rng)

        # Avoid tiny moves within the same chamber so the demo still shows visible exploration.
        if previous_target is not None:
            for _ in range(8):
                if float(np.linalg.norm(target - previous_target)) >= rat_radius * 4:
                    break
                target = _sample_chamber_target(chamber, margin_px=margin_px, rng=rng)

        targets.append(target)
        previous_target = target

    return targets


def _draw_rat(frame: np.ndarray, position: np.ndarray, previous_position: np.ndarray, rat_radius: int) -> None:
    motion = position - previous_position
    motion_norm = float(np.linalg.norm(motion))
    if motion_norm > 1e-6:
        heading = motion / motion_norm
    else:
        heading = np.asarray([1.0, 0.0], dtype=np.float32)

    perpendicular = np.asarray([-heading[1], heading[0]], dtype=np.float32)
    body_center = position
    head_center = position + (heading * rat_radius * 0.72)
    tail_base = position - (heading * rat_radius * 0.92)
    tail_tip = tail_base - (heading * rat_radius * 2.4) + (perpendicular * rat_radius * 0.45)

    shadow_offset = np.asarray([6.0, 7.0], dtype=np.float32)
    cv2.ellipse(
        frame,
        tuple(np.round(body_center + shadow_offset).astype(int)),
        (int(rat_radius * 1.35), int(rat_radius * 0.82)),
        float(np.degrees(np.arctan2(heading[1], heading[0]))),
        0,
        360,
        (142, 142, 142),
        -1,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        tuple(np.round(tail_base + shadow_offset).astype(int)),
        tuple(np.round(tail_tip + shadow_offset).astype(int)),
        (146, 146, 146),
        max(2, rat_radius // 5),
        cv2.LINE_AA,
    )
    cv2.ellipse(
        frame,
        tuple(np.round(body_center).astype(int)),
        (int(rat_radius * 1.28), int(rat_radius * 0.78)),
        float(np.degrees(np.arctan2(heading[1], heading[0]))),
        0,
        360,
        (32, 32, 32),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, tuple(np.round(head_center).astype(int)), max(4, int(rat_radius * 0.48)), (28, 28, 28), -1, cv2.LINE_AA)
    cv2.line(
        frame,
        tuple(np.round(tail_base).astype(int)),
        tuple(np.round(tail_tip).astype(int)),
        (36, 36, 36),
        max(1, rat_radius // 6),
        cv2.LINE_AA,
    )


def generate_synthetic_cpp_video(
    output_dir: str | Path,
    video_name: str = "synthetic_cpp_demo.mp4",
    fps: float = 15.0,
    frame_size: tuple[int, int] = (960, 360),
    segments: Sequence[SegmentSpec] | None = None,
    seed: int = 7,
) -> dict[str, Path]:
    """Create a synthetic three-chamber video and matching ground truth."""

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
    targets = _make_segment_targets(calibration=calibration, specs=specs, rat_radius=rat_radius, rng=rng)
    position = targets[0].copy()
    previous_position = position.copy()
    velocity = np.asarray([0.0, 0.0], dtype=np.float32)

    for segment_index, spec in enumerate(specs):
        segment_frames = max(1, int(round(spec.duration_seconds * fps)))
        current_target = targets[segment_index]
        next_target = targets[min(segment_index + 1, len(targets) - 1)]

        for local_index in range(segment_frames):
            frame = np.full((height, width, 3), 210, dtype=np.uint8)
            _draw_demo_arena(frame=frame, arena_box=arena_box, width=width)

            segment_progress = local_index / max(segment_frames - 1, 1)
            if segment_progress > 0.72 and segment_index < len(specs) - 1:
                transition_progress = (segment_progress - 0.72) / 0.28
                target = ((1.0 - transition_progress) * current_target) + (transition_progress * next_target)
            else:
                wander = np.asarray(
                    [
                        np.sin((frame_index * 0.071) + segment_index) * rat_radius * 2.8,
                        np.cos((frame_index * 0.047) + (segment_index * 1.7)) * rat_radius * 1.8,
                    ],
                    dtype=np.float32,
                )
                target = current_target + wander

            steer = (target - position) * 0.045
            jitter = rng.normal(0.0, 0.42, size=2).astype(np.float32)
            velocity = (velocity * 0.84) + steer + jitter
            speed = float(np.linalg.norm(velocity))
            max_speed = max(1.8, width / 360.0)
            if speed > max_speed:
                velocity = velocity / speed * max_speed

            previous_position = position.copy()
            position = position + velocity
            position[0] = float(np.clip(position[0], arena_box[0] + rat_radius * 1.6, arena_box[0] + arena_box[2] - rat_radius * 1.6))
            position[1] = float(np.clip(position[1], arena_box[1] + rat_radius * 1.8, arena_box[1] + arena_box[3] - rat_radius * 1.8))

            _draw_rat(frame=frame, position=position, previous_position=previous_position, rat_radius=rat_radius)

            writer.write(frame)
            actual_chamber = assign_point_to_chamber((float(position[0]), float(position[1])), calibration).label
            records.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / fps,
                    "expected_chamber": actual_chamber,
                    "scripted_target_chamber": spec.chamber,
                    "expected_centroid_x": float(position[0]),
                    "expected_centroid_y": float(position[1]),
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
