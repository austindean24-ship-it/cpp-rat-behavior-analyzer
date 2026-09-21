"""Conservative, maze-constrained EPM tracking; the CPP tracker is untouched."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from epm import EPMCalibration
from io_utils import get_video_metadata
from tracker import (
    TrackingConfig,
    _crop_array,
    _crop_bounds_from_mask,
    _estimate_head_shoulder_point,
    _normalize_vector,
    _offset_point,
    _prepare_gray,
    estimate_background,
)


@dataclass
class Candidate:
    point: tuple[float, float]
    area: float
    contour: np.ndarray


class EPMTrackState:
    """Require a continuous plausible contour before admitting it for scoring."""

    def __init__(self, mask: np.ndarray, config: TrackingConfig, fps: float, arm_width: float) -> None:
        self.mask = mask
        self.config = config
        self.max_step = min(config.max_jump_px, max(8.0, 0.35 * arm_width, 5.0 * arm_width / fps))
        self.acquire_frames = 3
        self.max_gap_frames = max(3, round(0.2 * fps))
        self.last: Candidate | None = None
        self.last_frame = -1
        self.pending: Candidate | None = None
        self.pending_count = 0
        self.smoothed: tuple[float, float] | None = None
        self.head_smoothed: tuple[float, float] | None = None
        self.heading: np.ndarray | None = None
        self.segment = 0

    def _walkway_path(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        distance = float(np.linalg.norm(np.subtract(end, start)))
        if distance < 2:
            return True
        count = max(2, int(distance / 2))
        inside = 0
        for fraction in np.linspace(0, 1, count):
            x = round(start[0] + (end[0] - start[0]) * fraction)
            y = round(start[1] + (end[1] - start[1]) * fraction)
            inside += int(0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0] and self.mask[y, x] > 0)
        return inside / count >= 0.85

    def _compatible(self, old: Candidate, new: Candidate, frames: int) -> str:
        distance = float(np.linalg.norm(np.subtract(old.point, new.point)))
        if distance > self.max_step * min(frames, self.max_gap_frames):
            return "raw_jump"
        if not 0.35 <= new.area / max(old.area, 1) <= 2.8:
            return "area_change"
        if not self._walkway_path(old.point, new.point):
            return "off_maze_path"
        return ""

    def update(self, candidates: list[Candidate], frame_index: int) -> tuple[Candidate | None, Candidate | None, str, str, float]:
        """Return accepted, raw proposal, status, rejection reason, raw step."""
        reference = self.last if self.last is not None and frame_index - self.last_frame <= self.max_gap_frames else None
        accepted_choices: list[Candidate] = []
        reasons: list[str] = []
        for item in candidates:
            reason = self._compatible(reference, item, max(1, frame_index - self.last_frame)) if reference else ""
            if reason:
                reasons.append(reason)
            else:
                accepted_choices.append(item)
        raw = min(candidates, key=lambda item: np.linalg.norm(np.subtract(item.point, reference.point))) if reference and candidates else (
            max(candidates, key=lambda item: item.area) if candidates else None
        )
        displacement = float(np.linalg.norm(np.subtract(raw.point, reference.point))) if reference and raw else np.nan
        if not accepted_choices:
            self.pending = None
            self.pending_count = 0
            return None, raw, "rejected" if candidates else "lost", reasons[0] if reasons else "no_foreground_candidate", displacement

        accepted_choices.sort(key=lambda item: (
            np.linalg.norm(np.subtract(item.point, reference.point)) if reference else -item.area
        ))
        choice = accepted_choices[0]
        if len(accepted_choices) > 1:
            first_score = (np.linalg.norm(np.subtract(choice.point, reference.point)) if reference else choice.area)
            next_score = (np.linalg.norm(np.subtract(accepted_choices[1].point, reference.point)) if reference else accepted_choices[1].area)
            ambiguous = next_score <= first_score + self.max_step * 0.2 if reference else next_score >= first_score * 0.75
            if ambiguous:
                self.pending = None
                self.pending_count = 0
                return None, choice, "ambiguous", "multiple_plausible_candidates", displacement

        accepted_step = float(np.linalg.norm(np.subtract(choice.point, reference.point))) if reference else np.nan
        if self.last_frame == frame_index - 1 and self.last is not None:
            self.last = choice
            self.last_frame = frame_index
            return choice, choice, "tracked", "", accepted_step

        if self.pending is not None and not self._compatible(self.pending, choice, 1):
            self.pending_count += 1
        else:
            self.pending_count = 1
        self.pending = choice
        if self.pending_count < self.acquire_frames:
            return None, choice, "reacquiring", "awaiting_continuity", displacement

        self.pending = None
        self.pending_count = 0
        self.last = choice
        self.last_frame = frame_index
        self.smoothed = None
        self.head_smoothed = None
        self.heading = None
        self.segment += 1
        return choice, choice, "tracked", "", accepted_step

    def score_points(self, candidate: Candidate, frame_index: int) -> dict:
        continuous = self.smoothed is not None and frame_index - self.last_frame <= 1
        prior = self.smoothed if continuous else None
        motion = np.subtract(candidate.point, prior) if prior is not None else None
        if motion is not None and np.linalg.norm(motion) >= self.config.min_heading_motion_px:
            self.heading = _normalize_vector(motion)
        front, anchor, source = _estimate_head_shoulder_point(
            candidate.contour, candidate.point, self.heading, self.config.head_shoulder_fraction
        )
        alpha = self.config.smoothing_alpha
        self.smoothed = tuple(alpha * np.asarray(candidate.point) + (1 - alpha) * np.asarray(prior)) if prior is not None else candidate.point
        self.head_smoothed = tuple(alpha * np.asarray(anchor) + (1 - alpha) * np.asarray(self.head_smoothed)) if continuous and self.head_smoothed is not None else anchor
        return {
            "smoothed": self.smoothed, "front": front, "head": anchor,
            "head_smoothed": self.head_smoothed, "head_source": source,
        }


class EPMTracker:
    """Tracks a rat only on calibrated walkway pixels, with unknown gaps."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def _candidates(self, gray: np.ndarray, background: np.ndarray, mask: np.ndarray) -> tuple[list[Candidate], str]:
        difference = cv2.absdiff(gray, background)
        _, foreground = cv2.threshold(difference, self.config.diff_threshold, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(foreground, mask)  # Never dilate room activity into the walkway.
        coverage = cv2.countNonZero(foreground) / max(cv2.countNonZero(mask), 1)
        if coverage > 0.35:
            return [], "widespread_foreground_or_occlusion"
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        foreground = cv2.bitwise_and(foreground, mask)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = min(gray.size * self.config.max_contour_area_ratio, cv2.countNonZero(mask) * 0.06)
        candidates: list[Candidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.config.min_contour_area <= area <= max_area:
                continue
            rect = cv2.minAreaRect(contour)
            short, long = sorted(rect[1])
            if short < 2 or long / short > 5.5 or area / max(cv2.contourArea(cv2.convexHull(contour)), 1) < 0.45:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            point = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            x, y = round(point[0]), round(point[1])
            if not 0 <= x < mask.shape[1] or not 0 <= y < mask.shape[0] or not mask[y, x]:
                continue
            candidates.append(Candidate(point, area, contour))
        return candidates, "no_suitable_contour" if not candidates else ""

    def track_video(
        self,
        video_path: str | Path,
        calibration: EPMCalibration,
        progress_callback: Callable[[int, int], None] | None = None,
        fps_override: float | None = None,
    ) -> pd.DataFrame:
        metadata = get_video_metadata(video_path, fps_fallback=self.config.fps_fallback)
        fps = float(fps_override) if fps_override and fps_override > 0 else metadata.fps
        full_mask = calibration.tracking_mask()
        if full_mask.shape != (metadata.height, metadata.width):
            raise ValueError("Calibration and video dimensions differ.")
        crop = _crop_bounds_from_mask(full_mask, metadata.width, metadata.height)
        mask = _crop_array(full_mask, crop)
        background = estimate_background(video_path, self.config.background_sample_count, self.config.gaussian_blur_size, crop)
        arm_width = min(min(cv2.minAreaRect(region.polygon)[1]) for region in calibration.regions[1:])
        state = EPMTrackState(mask, self.config, fps, arm_width)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Could not open EPM video.")
        rows: list[dict] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = len(rows)
                gray = _prepare_gray(_crop_array(frame, crop), self.config.gaussian_blur_size)
                candidates, detection_reason = self._candidates(gray, background, mask)
                accepted, raw, status, reason, raw_step = state.update(candidates, index)
                if accepted is None and status == "lost":
                    reason = detection_reason or reason
                points = state.score_points(accepted, index) if accepted is not None else {}

                def full(point: tuple[float, float] | None) -> tuple[float, float]:
                    return _offset_point(point, crop) if point is not None else (np.nan, np.nan)

                raw_x, raw_y = full(raw.point if raw else None)
                x, y = full(accepted.point if accepted else None)
                smooth_x, smooth_y = full(points.get("smoothed"))
                front_x, front_y = full(points.get("front"))
                head_x, head_y = full(points.get("head"))
                hs_x, hs_y = full(points.get("head_smoothed"))
                rows.append({
                    "frame_index": index, "time_seconds": index / fps,
                    "raw_candidate_x": raw_x, "raw_candidate_y": raw_y,
                    "centroid_x": x, "centroid_y": y,
                    "smoothed_x": smooth_x, "smoothed_y": smooth_y,
                    "front_x": front_x, "front_y": front_y,
                    "head_shoulder_x": head_x, "head_shoulder_y": head_y,
                    "smoothed_head_shoulder_x": hs_x, "smoothed_head_shoulder_y": hs_y,
                    "head_estimate_source": points.get("head_source", "missing"),
                    "contour_area": accepted.area if accepted else np.nan,
                    "raw_candidate_area": raw.area if raw else np.nan,
                    "tracking_status": status, "rejection_reason": reason,
                    "low_confidence": status != "tracked", "carried_forward": False,
                    "distance_from_previous_px": raw_step if status == "tracked" else np.nan,
                    "raw_displacement_px": raw_step,
                    "track_segment_id": state.segment if accepted is not None else np.nan,
                })
                if progress_callback and (len(rows) % 120 == 0 or len(rows) == metadata.frame_count):
                    progress_callback(len(rows), metadata.frame_count)
        finally:
            capture.release()
        return pd.DataFrame(rows)
