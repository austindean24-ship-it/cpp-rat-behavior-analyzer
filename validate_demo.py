from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import create_analysis_bundle
from demo_generator import generate_synthetic_cpp_video
from io_utils import ensure_directory, export_dataframe_csv, get_video_metadata
from regions import create_three_chambers_from_box
from tracker import SingleRatTracker, TrackingConfig


def run_validation(output_dir: str | Path = "demo_outputs") -> dict[str, Path | float]:
    output_root = ensure_directory(output_dir)
    demo_files = generate_synthetic_cpp_video(output_root)
    video_path = Path(demo_files["video_path"])
    truth_df = pd.read_csv(demo_files["ground_truth_csv"])

    metadata = get_video_metadata(video_path)
    arena_box = (
        int(metadata.width * 0.06),
        int(metadata.height * 0.18),
        int(metadata.width * 0.88),
        int(metadata.height * 0.64),
    )
    calibration = create_three_chambers_from_box(
        box=arena_box,
        frame_width=metadata.width,
        frame_height=metadata.height,
        boundary_margin_px=0.0,
    )

    tracker = SingleRatTracker(
        TrackingConfig(
            background_sample_count=40,
            min_contour_area=120.0,
            max_jump_px=110.0,
            smoothing_alpha=0.4,
        )
    )
    tracking_df = tracker.track_video(video_path=video_path, arena_mask=calibration.arena_mask())
    analysis_bundle = create_analysis_bundle(tracking_df=tracking_df, calibration=calibration, fps=metadata.fps)

    measured_summary = analysis_bundle.summary[analysis_bundle.summary["chamber"].isin(["left", "center", "right"])].copy()
    truth_summary = (
        truth_df.groupby("expected_chamber")
        .size()
        .reset_index(name="expected_frames")
        .rename(columns={"expected_chamber": "chamber"})
    )
    truth_summary["expected_seconds"] = truth_summary["expected_frames"] / metadata.fps

    comparison = measured_summary.merge(truth_summary, on="chamber", how="outer").fillna(0)
    comparison["seconds_error"] = comparison["seconds"] - comparison["expected_seconds"]
    comparison["percent_error"] = comparison["seconds_error"] / comparison["expected_seconds"].replace(0, 1.0) * 100.0

    comparison_path = export_dataframe_csv(comparison, output_root / "validation_comparison.csv")
    per_frame_path = export_dataframe_csv(analysis_bundle.per_frame, output_root / "validation_per_frame.csv")

    max_abs_seconds_error = float(comparison["seconds_error"].abs().max())
    max_abs_percent_error = float(comparison["percent_error"].abs().max())

    return {
        "video_path": video_path,
        "comparison_csv": comparison_path,
        "per_frame_csv": per_frame_path,
        "max_abs_seconds_error": max_abs_seconds_error,
        "max_abs_percent_error": max_abs_percent_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CPP rat tracker on a synthetic demo video.")
    parser.add_argument(
        "--output-dir",
        default="demo_outputs",
        help="Folder where the demo video and validation CSV files should be saved.",
    )
    args = parser.parse_args()

    results = run_validation(output_dir=args.output_dir)
    print("Validation finished.")
    print(f"Demo video: {results['video_path']}")
    print(f"Comparison CSV: {results['comparison_csv']}")
    print(f"Per-frame CSV: {results['per_frame_csv']}")
    print(f"Max absolute timing error (seconds): {results['max_abs_seconds_error']:.3f}")
    print(f"Max absolute timing error (percent): {results['max_abs_percent_error']:.2f}")


if __name__ == "__main__":
    main()
