from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from regions import ArenaCalibration, assign_point_to_chamber


@dataclass
class AnalysisBundle:
    per_frame: pd.DataFrame
    summary: pd.DataFrame
    qc_metrics: pd.DataFrame
    warnings: list[str]


def _pick_assignment_point(row: Any, assignment_point_mode: str) -> tuple[float, float] | None:
    candidate_columns: list[tuple[str, str]]

    if assignment_point_mode == "head_shoulders":
        candidate_columns = [
            ("smoothed_head_shoulder_x", "smoothed_head_shoulder_y"),
            ("head_shoulder_x", "head_shoulder_y"),
            ("smoothed_x", "smoothed_y"),
            ("centroid_x", "centroid_y"),
        ]
    elif assignment_point_mode == "centroid":
        candidate_columns = [
            ("centroid_x", "centroid_y"),
        ]
    else:
        candidate_columns = [
            ("smoothed_x", "smoothed_y"),
            ("centroid_x", "centroid_y"),
        ]

    for x_name, y_name in candidate_columns:
        x_value = getattr(row, x_name, np.nan)
        y_value = getattr(row, y_name, np.nan)
        if not np.isnan(x_value) and not np.isnan(y_value):
            return (float(x_value), float(y_value))
    return None


def assign_chambers(
    tracking_df: pd.DataFrame,
    calibration: ArenaCalibration,
    use_smoothed_centroid: bool = True,
    boundary_margin_px: float | None = None,
    assignment_point_mode: str = "smoothed_centroid",
) -> pd.DataFrame:
    """Attach chamber labels to each tracked frame."""

    frame_records: list[dict[str, Any]] = []
    for row in tracking_df.itertuples(index=False):
        if assignment_point_mode == "smoothed_centroid" and not use_smoothed_centroid:
            effective_mode = "centroid"
        else:
            effective_mode = assignment_point_mode
        point = _pick_assignment_point(row=row, assignment_point_mode=effective_mode)

        assignment = assign_point_to_chamber(point=point, calibration=calibration, boundary_margin_px=boundary_margin_px)
        record = row._asdict()
        record["assignment_x"] = point[0] if point is not None else np.nan
        record["assignment_y"] = point[1] if point is not None else np.nan
        record["assignment_point_mode"] = effective_mode
        record["chamber"] = assignment.label
        record["on_boundary"] = assignment.on_boundary
        record["inside_arena"] = assignment.inside_arena
        frame_records.append(record)
    return pd.DataFrame(frame_records)


def summarize_chamber_time(
    per_frame_df: pd.DataFrame,
    fps: float,
    calibration: ArenaCalibration,
) -> pd.DataFrame:
    chamber_labels = calibration.chamber_names() + [
        calibration.neutral_label,
        calibration.outside_label,
        "missing",
    ]
    total_frames = int(len(per_frame_df))

    rows: list[dict[str, Any]] = []
    for chamber_label in chamber_labels:
        frame_count = int((per_frame_df["chamber"] == chamber_label).sum())
        seconds = frame_count / fps if fps > 0 else 0.0
        percent = (frame_count / total_frames * 100.0) if total_frames else 0.0
        rows.append(
            {
                "chamber": chamber_label,
                "frames": frame_count,
                "seconds": round(seconds, 3),
                "percent_of_video": round(percent, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_qc_metrics(per_frame_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    total_frames = max(int(len(per_frame_df)), 1)
    contour_detected_mask = per_frame_df["tracking_status"].isin(["tracked", "fallback_threshold"])
    centroid_available_mask = per_frame_df["assignment_x"].notna()
    low_confidence_mask = per_frame_df["low_confidence"].fillna(False)

    tracking_success_rate = centroid_available_mask.mean() * 100.0
    direct_detection_rate = contour_detected_mask.mean() * 100.0
    missing_centroid_frames = int((~centroid_available_mask).sum())
    low_confidence_frames = int(low_confidence_mask.sum())
    mean_contour_area = float(per_frame_df["contour_area"].dropna().mean()) if per_frame_df["contour_area"].notna().any() else 0.0
    contour_area_std = float(per_frame_df["contour_area"].dropna().std()) if per_frame_df["contour_area"].notna().sum() > 1 else 0.0
    boundary_frames = int(per_frame_df["on_boundary"].fillna(False).sum())
    mean_jump = float(per_frame_df["distance_from_previous_px"].dropna().mean()) if per_frame_df["distance_from_previous_px"].notna().any() else 0.0

    qc_metrics = pd.DataFrame(
        [
            {"metric": "tracking_success_rate_percent", "value": round(tracking_success_rate, 2)},
            {"metric": "direct_detection_rate_percent", "value": round(direct_detection_rate, 2)},
            {"metric": "missing_centroid_frames", "value": missing_centroid_frames},
            {"metric": "low_confidence_frames", "value": low_confidence_frames},
            {"metric": "mean_contour_area_px", "value": round(mean_contour_area, 2)},
            {"metric": "contour_area_std_px", "value": round(contour_area_std, 2)},
            {"metric": "boundary_frames", "value": boundary_frames},
            {"metric": "mean_centroid_jump_px", "value": round(mean_jump, 2)},
            {"metric": "total_frames", "value": total_frames},
        ]
    )

    warnings: list[str] = []
    if tracking_success_rate < 95.0:
        warnings.append("Tracking success rate is below 95%. Review the video and chamber drawing.")
    if direct_detection_rate < 90.0:
        warnings.append("The tracker had to rely on fallback logic often. Results may need manual review.")
    if low_confidence_frames / total_frames > 0.1:
        warnings.append("More than 10% of frames were low confidence.")
    if contour_area_std > 0 and mean_contour_area > 0 and (contour_area_std / mean_contour_area) > 0.75:
        warnings.append("Contour size varied a lot across the video. This can happen with shadows or reflections.")
    if boundary_frames / total_frames > 0.1:
        warnings.append("A large number of frames were near chamber borders. Consider increasing the arena box accuracy.")

    return qc_metrics, warnings


def create_analysis_bundle(
    tracking_df: pd.DataFrame,
    calibration: ArenaCalibration,
    fps: float,
    use_smoothed_centroid: bool = True,
    boundary_margin_px: float | None = None,
    assignment_point_mode: str = "smoothed_centroid",
) -> AnalysisBundle:
    per_frame_df = assign_chambers(
        tracking_df=tracking_df,
        calibration=calibration,
        use_smoothed_centroid=use_smoothed_centroid,
        boundary_margin_px=boundary_margin_px,
        assignment_point_mode=assignment_point_mode,
    )
    summary_df = summarize_chamber_time(per_frame_df=per_frame_df, fps=fps, calibration=calibration)
    qc_metrics_df, warnings = compute_qc_metrics(per_frame_df=per_frame_df)
    return AnalysisBundle(
        per_frame=per_frame_df,
        summary=summary_df,
        qc_metrics=qc_metrics_df,
        warnings=warnings,
    )
