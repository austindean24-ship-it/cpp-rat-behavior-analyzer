from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from io_utils import DEFAULT_FPS_FALLBACK, get_video_metadata


ProgressCallback = Callable[[int, int], None]


@dataclass
class TrackingConfig:
    background_sample_count: int = 60
    diff_threshold: int = 28
    frame_diff_threshold: int = 16
    min_contour_area: float = 150.0
    max_contour_area_ratio: float = 0.12
    gaussian_blur_size: int = 5
    morph_kernel_size: int = 5
    smoothing_alpha: float = 0.35
    max_jump_px: float = 150.0
    fallback_search_radius_px: float = 120.0
    max_carried_frames: int = 8
    fps_fallback: float = DEFAULT_FPS_FALLBACK
    min_heading_motion_px: float = 4.0
    head_shoulder_fraction: float = 0.65


def _odd_kernel(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _prepare_gray(frame: np.ndarray, blur_size: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_size = _odd_kernel(max(3, blur_size))
    return cv2.GaussianBlur(gray, (blur_size, blur_size), 0)


def estimate_background(
    video_path: str | Path,
    sample_count: int = 60,
    blur_size: int = 5,
) -> np.ndarray:
    """Estimate a static background using the median of sampled frames."""

    metadata = get_video_metadata(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for background estimation: {video_path}")

    if metadata.frame_count > 0:
        sample_indices = np.linspace(
            0,
            max(metadata.frame_count - 1, 0),
            num=min(sample_count, max(metadata.frame_count, 1)),
            dtype=int,
        )
    else:
        sample_indices = np.arange(sample_count, dtype=int)

    samples: list[np.ndarray] = []
    for frame_index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()
        if not success or frame is None:
            continue
        samples.append(_prepare_gray(frame, blur_size))

    cap.release()
    if not samples:
        raise ValueError("Could not collect frames to build a background image.")

    stacked = np.stack(samples, axis=0)
    return np.median(stacked, axis=0).astype(np.uint8)


def _cleanup_mask(mask: np.ndarray, kernel_size: int, arena_mask: np.ndarray | None) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_kernel(max(3, kernel_size)), _odd_kernel(max(3, kernel_size))))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    if arena_mask is not None:
        cleaned = cv2.bitwise_and(cleaned, arena_mask)
    return cleaned


def _contour_centroid(contour: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))


def _search_mask_near_point(mask: np.ndarray, point: tuple[float, float], radius: float) -> np.ndarray:
    focus = np.zeros_like(mask)
    cv2.circle(focus, (int(point[0]), int(point[1])), int(radius), 255, -1)
    return cv2.bitwise_and(mask, focus)


def _normalize_vector(vector: np.ndarray | tuple[float, float] | None) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-6:
        return None
    return array / norm


def _estimate_head_shoulder_point(
    contour: np.ndarray,
    centroid: tuple[float, float],
    heading_unit: np.ndarray | None,
    head_fraction: float,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    """Estimate a head-facing anchor from the contour.

    This does not attempt full pose estimation. Instead it uses the rat's recent
    movement direction as a proxy for heading and places the chamber-assignment
    point partway between the body centroid and the leading edge of the contour.
    """

    if heading_unit is None:
        return centroid, centroid, "centroid_fallback"

    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) == 0:
        return centroid, centroid, "centroid_fallback"

    projections = points @ heading_unit
    front_point = points[int(np.argmax(projections))]
    centroid_array = np.asarray(centroid, dtype=np.float32)
    anchor_point = centroid_array + (float(head_fraction) * (front_point - centroid_array))
    return (
        (float(front_point[0]), float(front_point[1])),
        (float(anchor_point[0]), float(anchor_point[1])),
        "motion_heading",
    )


def _score_contour(
    contour: np.ndarray,
    previous_point: tuple[float, float] | None,
    frame_area: float,
    config: TrackingConfig,
) -> tuple[float, tuple[float, float] | None, float]:
    area = float(cv2.contourArea(contour))
    max_area = config.max_contour_area_ratio * frame_area
    if area < config.min_contour_area or area > max_area:
        return (-1.0, None, area)

    centroid = _contour_centroid(contour)
    if centroid is None:
        return (-1.0, None, area)

    score = area
    if previous_point is not None:
        distance = float(np.linalg.norm(np.asarray(centroid) - np.asarray(previous_point)))
        if distance > config.max_jump_px:
            score -= (distance - config.max_jump_px) * 3.0
        else:
            score += max(config.max_jump_px - distance, 0.0)
    return (score, centroid, area)


def _find_best_contour(
    mask: np.ndarray,
    previous_point: tuple[float, float] | None,
    config: TrackingConfig,
) -> tuple[np.ndarray | None, tuple[float, float] | None, float, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, None, np.nan, -1.0)

    frame_area = float(mask.shape[0] * mask.shape[1])
    best_contour: np.ndarray | None = None
    best_centroid: tuple[float, float] | None = None
    best_area = np.nan
    best_score = -1.0

    for contour in contours:
        score, centroid, area = _score_contour(contour, previous_point, frame_area, config)
        if score > best_score and centroid is not None:
            best_score = score
            best_contour = contour
            best_centroid = centroid
            best_area = area

    return best_contour, best_centroid, best_area, best_score


class SingleRatTracker:
    """Simple single-animal tracker for fixed-camera CPP videos."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def _build_primary_mask(
        self,
        gray_frame: np.ndarray,
        background_gray: np.ndarray,
        previous_gray: np.ndarray | None,
        subtractor,
        arena_mask: np.ndarray | None,
    ) -> np.ndarray:
        diff_background = cv2.absdiff(gray_frame, background_gray)
        _, background_mask = cv2.threshold(diff_background, self.config.diff_threshold, 255, cv2.THRESH_BINARY)

        if previous_gray is not None:
            diff_previous = cv2.absdiff(gray_frame, previous_gray)
            _, motion_mask = cv2.threshold(diff_previous, self.config.frame_diff_threshold, 255, cv2.THRESH_BINARY)
        else:
            motion_mask = np.zeros_like(background_mask)

        mog_mask = subtractor.apply(gray_frame)
        _, mog_mask = cv2.threshold(mog_mask, 200, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(background_mask, motion_mask)
        combined = cv2.bitwise_or(combined, mog_mask)
        return _cleanup_mask(combined, self.config.morph_kernel_size, arena_mask)

    def _build_fallback_masks(
        self,
        gray_frame: np.ndarray,
        previous_point: tuple[float, float] | None,
        arena_mask: np.ndarray | None,
    ) -> list[np.ndarray]:
        _, dark_mask = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, light_mask = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_mask = _cleanup_mask(dark_mask, self.config.morph_kernel_size, arena_mask)
        light_mask = _cleanup_mask(light_mask, self.config.morph_kernel_size, arena_mask)
        if previous_point is not None:
            dark_mask = _search_mask_near_point(dark_mask, previous_point, self.config.fallback_search_radius_px)
            light_mask = _search_mask_near_point(light_mask, previous_point, self.config.fallback_search_radius_px)
        return [dark_mask, light_mask]

    def track_video(
        self,
        video_path: str | Path,
        arena_mask: np.ndarray | None = None,
        progress_callback: ProgressCallback | None = None,
        fps_override: float | None = None,
    ) -> pd.DataFrame:
        metadata = get_video_metadata(video_path, fps_fallback=self.config.fps_fallback)
        effective_fps = float(fps_override) if fps_override and fps_override > 0 else metadata.fps
        background = estimate_background(
            video_path=video_path,
            sample_count=self.config.background_sample_count,
            blur_size=self.config.gaussian_blur_size,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video for tracking: {video_path}")

        subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)

        previous_gray: np.ndarray | None = None
        previous_smoothed: tuple[float, float] | None = None
        previous_heading_unit: np.ndarray | None = None
        previous_head_shoulder_smoothed: tuple[float, float] | None = None
        missing_streak = 0
        results: list[dict[str, float | int | bool | str]] = []

        frame_index = 0
        total_frames = max(metadata.frame_count, 1)

        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break

            gray = _prepare_gray(frame, self.config.gaussian_blur_size)
            primary_mask = self._build_primary_mask(
                gray_frame=gray,
                background_gray=background,
                previous_gray=previous_gray,
                subtractor=subtractor,
                arena_mask=arena_mask,
            )
            contour, centroid, area, primary_score = _find_best_contour(
                mask=primary_mask,
                previous_point=previous_smoothed,
                config=self.config,
            )

            tracking_status = "tracked"
            low_confidence = False
            carried_forward = False
            front_point: tuple[float, float] | None = None
            head_shoulder_point: tuple[float, float] | None = None
            head_estimate_source = "missing"

            if contour is None:
                fallback_candidates = self._build_fallback_masks(
                    gray_frame=gray,
                    previous_point=previous_smoothed,
                    arena_mask=arena_mask,
                )
                best_fallback = (None, None, np.nan, -1.0)
                for fallback_mask in fallback_candidates:
                    current = _find_best_contour(
                        mask=fallback_mask,
                        previous_point=previous_smoothed,
                        config=self.config,
                    )
                    if current[3] > best_fallback[3]:
                        best_fallback = current
                contour, centroid, area, _ = best_fallback
                if contour is not None and best_fallback[3] > primary_score:
                    tracking_status = "fallback_threshold"
                    low_confidence = True

            if contour is None and previous_smoothed is not None and missing_streak < self.config.max_carried_frames:
                centroid = previous_smoothed
                carried_forward = True
                low_confidence = True
                tracking_status = "carried_forward"
                area = np.nan
                missing_streak += 1
                if previous_head_shoulder_smoothed is not None:
                    head_shoulder_point = previous_head_shoulder_smoothed
                    head_estimate_source = "carried_forward"
            elif contour is None:
                centroid = None
                tracking_status = "missing"
                low_confidence = True
                area = np.nan
                missing_streak += 1
            else:
                missing_streak = 0

            if centroid is not None:
                motion_vector = (
                    np.asarray(centroid, dtype=np.float32) - np.asarray(previous_smoothed, dtype=np.float32)
                    if previous_smoothed is not None
                    else None
                )
                motion_norm = float(np.linalg.norm(motion_vector)) if motion_vector is not None else 0.0
                if motion_vector is not None and motion_norm >= self.config.min_heading_motion_px:
                    heading_unit = _normalize_vector(motion_vector)
                else:
                    heading_unit = previous_heading_unit

                if contour is not None:
                    front_point, head_shoulder_point, head_estimate_source = _estimate_head_shoulder_point(
                        contour=contour,
                        centroid=centroid,
                        heading_unit=heading_unit,
                        head_fraction=self.config.head_shoulder_fraction,
                    )
                    if head_estimate_source == "centroid_fallback":
                        low_confidence = True
                elif head_shoulder_point is None:
                    head_shoulder_point = centroid
                    head_estimate_source = "centroid_fallback"

                if previous_smoothed is None:
                    smoothed = centroid
                else:
                    smoothed_x = (self.config.smoothing_alpha * centroid[0]) + ((1.0 - self.config.smoothing_alpha) * previous_smoothed[0])
                    smoothed_y = (self.config.smoothing_alpha * centroid[1]) + ((1.0 - self.config.smoothing_alpha) * previous_smoothed[1])
                    smoothed = (smoothed_x, smoothed_y)

                if head_shoulder_point is None:
                    head_shoulder_point = centroid

                if previous_head_shoulder_smoothed is None:
                    smoothed_head_shoulder = head_shoulder_point
                else:
                    smoothed_head_shoulder_x = (
                        self.config.smoothing_alpha * head_shoulder_point[0]
                    ) + ((1.0 - self.config.smoothing_alpha) * previous_head_shoulder_smoothed[0])
                    smoothed_head_shoulder_y = (
                        self.config.smoothing_alpha * head_shoulder_point[1]
                    ) + ((1.0 - self.config.smoothing_alpha) * previous_head_shoulder_smoothed[1])
                    smoothed_head_shoulder = (smoothed_head_shoulder_x, smoothed_head_shoulder_y)

                distance_from_previous = (
                    float(np.linalg.norm(np.asarray(smoothed) - np.asarray(previous_smoothed)))
                    if previous_smoothed is not None
                    else np.nan
                )
                if previous_smoothed is not None and not np.isnan(distance_from_previous) and distance_from_previous > self.config.max_jump_px:
                    low_confidence = True
                previous_smoothed = smoothed
                previous_head_shoulder_smoothed = smoothed_head_shoulder
                previous_heading_unit = heading_unit
            else:
                smoothed = (np.nan, np.nan)
                smoothed_head_shoulder = (np.nan, np.nan)
                distance_from_previous = np.nan

            results.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": frame_index / effective_fps if effective_fps > 0 else 0.0,
                    "centroid_x": float(centroid[0]) if centroid is not None else np.nan,
                    "centroid_y": float(centroid[1]) if centroid is not None else np.nan,
                    "smoothed_x": float(smoothed[0]) if centroid is not None else np.nan,
                    "smoothed_y": float(smoothed[1]) if centroid is not None else np.nan,
                    "front_x": float(front_point[0]) if front_point is not None else np.nan,
                    "front_y": float(front_point[1]) if front_point is not None else np.nan,
                    "head_shoulder_x": float(head_shoulder_point[0]) if head_shoulder_point is not None else np.nan,
                    "head_shoulder_y": float(head_shoulder_point[1]) if head_shoulder_point is not None else np.nan,
                    "smoothed_head_shoulder_x": float(smoothed_head_shoulder[0]) if centroid is not None else np.nan,
                    "smoothed_head_shoulder_y": float(smoothed_head_shoulder[1]) if centroid is not None else np.nan,
                    "head_estimate_source": head_estimate_source,
                    "contour_area": float(area) if not np.isnan(area) else np.nan,
                    "tracking_status": tracking_status,
                    "low_confidence": low_confidence,
                    "carried_forward": carried_forward,
                    "distance_from_previous_px": distance_from_previous,
                }
            )

            previous_gray = gray
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, total_frames)

        cap.release()
        return pd.DataFrame(results)
