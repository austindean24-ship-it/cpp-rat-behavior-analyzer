"""EPM-only rat tracking with audited candidates and conservative unknown frames.

CPP's tracker.py is intentionally not modified.  A foreground blob is a
hypothesis, not an animal: reject illumination footprints, group close
silhouette fragments, and associate against an existing track before scoring.
"""
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
    TrackingConfig, _crop_array, _crop_bounds_from_mask,
    _estimate_head_shoulder_point, _normalize_vector,
    _offset_point, _prepare_gray,
)


@dataclass
class Candidate:
    point: tuple[float, float]
    area: float
    contour: np.ndarray
    evidence: str = "foreground"


def _background(video_path: str | Path, metadata, crop, count: int, blur: int) -> np.ndarray:
    """Median background without the artificial mid-level of a bimodal pixel.

    If two recurring pixel modes straddle the median, pick the mode nearest
    the beginning AND end when those agree.  Do not treat the median of
    'floor 90' and 'animal 245' (167) as a physically observed background.
    """
    cap = cv2.VideoCapture(str(video_path))
    samples = []
    try:
        for i in np.linspace(0, max(0, metadata.frame_count - 1), min(count, metadata.frame_count), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            good, frame = cap.read()
            if good:
                samples.append(_prepare_gray(_crop_array(frame, crop), blur))
    finally:
        cap.release()
    if not samples:
        raise ValueError("Could not construct EPM background from video.")
    stack = np.stack(samples).astype(np.uint8)
    med = np.median(stack, axis=0).astype(np.uint8)
    lo, hi = np.percentile(stack, [35, 65], axis=0)
    # Strongly bimodal pixels with same opening/closing value: likely background
    # temporarily occluded. Both endpoints must agree to avoid experimenter
    # movement being mistaken for a stable background.
    starts, ends = stack[0].astype(np.int16), stack[-1].astype(np.int16)
    bimodal = (hi - lo > 35) & (np.abs(starts - ends) < 15)
    med[bimodal] = ((starts[bimodal] + ends[bimodal]) // 2).astype(np.uint8)
    return med


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, bx - (ax + aw), ax - (bx + bw))
    dy = max(0, by - (ay + ah), ay - (by + bh))
    return float(np.hypot(dx, dy))


def _join_fragments(parts: list[Candidate], max_extent: float = 165.0) -> list[Candidate]:
    """Group fragments of one silhouette; leave distant hypotheses separate."""
    if len(parts) < 2:
        return parts
    parent = list(range(len(parts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [cv2.boundingRect(item.contour) for item in parts]
    pairs = sorted(((_bbox_gap(boxes[i], boxes[j]), i, j)
                    for i in range(len(parts)) for j in range(i + 1, len(parts))))
    for gap, i, j in pairs:
        if gap > 13:
            break
        left, right = find(i), find(j)
        if left == right:
            continue
        members = [k for k in range(len(parts)) if find(k) in (left, right)]
        if any(_bbox_gap(boxes[a], boxes[b]) > 20 for a in members for b in members):
            continue
        pts = np.concatenate([parts[k].contour.reshape(-1, 2) for k in members])
        bounds = cv2.boundingRect(pts)
        if bounds[2] <= max_extent and bounds[3] <= max_extent:
            parent[right] = left
    grouped: dict[int, list[Candidate]] = {}
    for i, item in enumerate(parts):
        grouped.setdefault(find(i), []).append(item)
    result = []
    for chunk in grouped.values():
        if len(chunk) == 1:
            result.append(chunk[0])
            continue
        total = sum(item.area for item in chunk)
        location = np.sum([np.asarray(item.point) * item.area for item in chunk], axis=0) / total
        hull = cv2.convexHull(np.concatenate([item.contour for item in chunk]))
        result.append(Candidate(tuple(map(float, location)), total, hull, "grouped_fragments"))
    return result


class EPMTrackState:
    """Identity association: candidate is not trusted until continuity is shown."""

    def __init__(self, mask: np.ndarray, config: TrackingConfig, fps: float, arm_width: float) -> None:
        self.mask = mask
        # Calibrated polygons can leave a few unassigned pixels at shared
        # borders. Close only the path-check mask; detection stays in the
        # exact five-region union.
        self.path_mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        self.config = config
        # Previous 0.35 * arm_width gave ~22px despite max_jump_px=120.
        # A center-fragment can jump ~36px within one frame. Use measured
        # maze scale but keep the configured jump a hard ceiling.
        self.max_step = min(config.max_jump_px, max(12.0, 0.9 * arm_width, 8.0 * arm_width / fps))
        self.acquire_frames = 3
        self.max_gap_frames = max(3, round(0.2 * fps))
        self.reacquire_horizon_frames = max(self.max_gap_frames, round(2.0 * fps))
        self.reacquire_radius = 3.0 * arm_width
        self.max_heading_age_frames = max(3, round(0.5 * fps))
        self.last: Candidate | None = None
        self.last_frame = -1
        self.pending: Candidate | None = None
        self.pending_count = 0
        self.smoothed: tuple[float, float] | None = None
        self.head_smoothed: tuple[float, float] | None = None
        self.heading: np.ndarray | None = None
        self.last_heading_frame = -1
        self.segment = 0

    def _walkway_path(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        length = float(np.linalg.norm(np.subtract(a, b)))
        if length < 2:
            return True
        n = max(2, int(length / 2))
        inside = 0
        for t in np.linspace(0, 1, n):
            x, y = round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)
            inside += int(0 <= x < self.path_mask.shape[1] and 0 <= y < self.path_mask.shape[0] and self.path_mask[y, x] > 0)
        return inside / n >= 0.85

    def _compatible(self, old: Candidate, new: Candidate, frames: int) -> str:
        d = float(np.linalg.norm(np.subtract(old.point, new.point)))
        allowed = self.max_step if frames <= 1 else min(
            self.max_step * min(frames, self.max_gap_frames), self.reacquire_radius
        )
        if d > allowed:
            return "raw_jump"
        if not 0.2 <= new.area / max(old.area, 1) <= 5.0:
            return "area_change"
        if not self._walkway_path(old.point, new.point):
            return "off_maze_path"
        return ""

    def update(self, candidates: list[Candidate], frame_index: int) -> tuple[Candidate | None, Candidate | None, str, str, float]:
        reference = self.last if self.last is not None and frame_index - self.last_frame <= self.reacquire_horizon_frames else None
        choices, reasons = [], []
        for candidate in candidates:
            reason = self._compatible(reference, candidate, max(1, frame_index - self.last_frame)) if reference else ""
            if reason:
                reasons.append(reason)
            else:
                choices.append(candidate)
        raw = (min(candidates, key=lambda c: np.linalg.norm(np.subtract(c.point, reference.point)))
               if reference and candidates else candidates[0] if len(candidates) == 1 else None)
        step = float(np.linalg.norm(np.subtract(raw.point, reference.point))) if reference and raw else np.nan
        if not choices:
            self.pending = None
            self.pending_count = 0
            return None, raw, "rejected" if candidates else "lost", reasons[0] if reasons else "no_foreground_candidate", step
        if reference is None and len(choices) > 1:
            self.pending = None
            self.pending_count = 0
            return None, None, "ambiguous", "multiple_unanchored_candidates", np.nan
        choices.sort(key=lambda c: np.linalg.norm(np.subtract(c.point, reference.point)) if reference else 0)
        chosen = choices[0]
        if len(choices) > 1:
            if reference:
                d0 = np.linalg.norm(np.subtract(choices[0].point, reference.point))
                d1 = np.linalg.norm(np.subtract(choices[1].point, reference.point))
                ambiguous = d1 - d0 < self.max_step * 0.12
            else:
                ambiguous = True
            if ambiguous:
                self.pending = None
                self.pending_count = 0
                return None, chosen, "ambiguous", "multiple_plausible_candidates", step
        accepted_step = float(np.linalg.norm(np.subtract(chosen.point, reference.point))) if reference else np.nan
        if self.last is not None and self.last_frame == frame_index - 1:
            self.last, self.last_frame = chosen, frame_index
            return chosen, chosen, "tracked", "", accepted_step
        if self.pending is not None and not self._compatible(self.pending, chosen, 1):
            self.pending_count += 1
        else:
            self.pending_count = 1
        self.pending = chosen
        if self.pending_count < self.acquire_frames:
            return None, chosen, "reacquiring", "awaiting_continuity", step
        self.pending = None
        self.pending_count = 0
        self.last, self.last_frame = chosen, frame_index
        self.smoothed = self.head_smoothed = self.heading = None
        self.last_heading_frame = -1
        self.segment += 1
        return chosen, chosen, "tracked", "", accepted_step

    def score_points(self, candidate: Candidate, frame_index: int) -> dict:
        prior = self.smoothed
        motion = np.subtract(candidate.point, prior) if prior is not None else None
        if motion is not None and np.linalg.norm(motion) >= self.config.min_heading_motion_px:
            self.heading = _normalize_vector(motion)
            self.last_heading_frame = frame_index
        elif frame_index - self.last_heading_frame > self.max_heading_age_frames:
            self.heading = None
        front, anchor, source = _estimate_head_shoulder_point(
            candidate.contour, candidate.point, self.heading, self.config.head_shoulder_fraction
        )
        alpha = self.config.smoothing_alpha
        self.smoothed = tuple(alpha * np.asarray(candidate.point) + (1 - alpha) * np.asarray(prior)) if prior is not None else candidate.point
        self.head_smoothed = tuple(alpha * np.asarray(anchor) + (1 - alpha) * np.asarray(self.head_smoothed)) if self.head_smoothed is not None else anchor
        return {"smoothed": self.smoothed, "front": front, "head": anchor,
                "head_smoothed": self.head_smoothed, "head_source": source}


class EPMTracker:
    """Use background subtraction plus spatial/photometric object evidence."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._region_masks: list[np.ndarray] | None = None
        self._region_areas: list[int] | None = None
        # Short-lived signatures for fading illumination footprints, not
        # blanket spatial exclusion: a smaller animal may enter this area.
        self._illumination_zones: list[tuple[tuple[int, int, int, int], int, float, int]] = []
        self._min_patch_length = 36.0
        self._arm_width = 30.0

    def _candidates(self, gray: np.ndarray, background: np.ndarray, mask: np.ndarray) -> tuple[list[Candidate], str]:
        self._illumination_zones = [
            (rect, ttl - 1, area, sign) for rect, ttl, area, sign in self._illumination_zones if ttl > 1
        ]
        signed = gray.astype(np.int16) - background.astype(np.int16)
        foreground = ((np.abs(signed) > self.config.diff_threshold) & (mask > 0)).astype(np.uint8) * 255
        if np.count_nonzero(foreground) > 0.35 * np.count_nonzero(mask):
            return [], "widespread_foreground_or_occlusion"
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        foreground = cv2.bitwise_and(foreground, mask)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = min(gray.size * self.config.max_contour_area_ratio, np.count_nonzero(mask) * 0.06)
        fragments = []
        lighting_rejected = False
        for c in contours:
            area = float(cv2.contourArea(c))
            if not self.config.min_contour_area <= area <= max_area:
                continue
            rr = cv2.minAreaRect(c)
            dims = sorted(rr[1])
            if dims[0] < 2 or dims[1] / dims[0] > 7.5:
                continue
            hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1)
            solidity = area / hull_area
            if solidity < 0.4:
                continue
            moments = cv2.moments(c)
            if moments["m00"] <= 0:
                continue
            point = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
            px, py = np.rint(point).astype(int)
            if not (0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and mask[py, px]):
                continue
            # Illumination changes are large, uniformly signed, almost full-
            # width rectangular patches of a walkway. Test all these cues, not
            # area alone: a true elongated or brightly lit rat must survive.
            x, y, w, h = cv2.boundingRect(c)
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [c - np.array([x, y]).reshape(1, 1, 2)], -1, 255, -1)
            sign = signed[y:y+h, x:x+w][local > 0]
            same_sign = max(np.mean(sign > 0), np.mean(sign < 0)) if len(sign) else 0
            fill = area / (w * h)
            region_fraction = 0.0
            if self._region_masks is not None and self._region_areas is not None:
                memberships = [bool(region[py, px]) for region in self._region_masks]
                for inside, region_area in zip(memberships, self._region_areas):
                    if inside:
                        region_fraction = max(region_fraction, area / max(region_area, 1))
            if (region_fraction > 0.20 and min(w, h) > 0.85 * self._arm_width
                    and max(w, h) > self._min_patch_length
                    and solidity > 0.85 and fill > 0.65
                    and same_sign > 0.97):
                lighting_rejected = True
                rect = (max(0, x - 10), max(0, y - 10), min(gray.shape[1], x + w + 10), min(gray.shape[0], y + h + 10))
                patch_sign = 1 if np.mean(sign > 0) >= 0.5 else -1
                self._illumination_zones.append((rect, 20, area, patch_sign))
                continue
            candidate_sign = 1 if np.mean(sign > 0) >= 0.5 else -1
            if any(
                x0 <= px < x1 and y0 <= py < y1 and candidate_sign == patch_sign
                and area >= 0.15 * patch_area and min(w, h) > 0.85 * self._arm_width
                for (x0, y0, x1, y1), _, patch_area, patch_sign in self._illumination_zones
            ):
                lighting_rejected = True
                continue
            fragments.append(Candidate(point, area, c))
        candidates = _join_fragments(fragments, max_extent=max(60.0, 2.7 * self._arm_width))
        reason = "suspected_illumination_patch" if lighting_rejected and not candidates else "no_suitable_contour" if not candidates else ""
        return candidates, reason

    def track_video(self, video_path: str | Path, calibration: EPMCalibration,
                    progress_callback: Callable[[int, int], None] | None = None,
                    fps_override: float | None = None) -> pd.DataFrame:
        metadata = get_video_metadata(video_path, fps_fallback=self.config.fps_fallback)
        fps = float(fps_override) if fps_override and fps_override > 0 else metadata.fps
        full_mask = calibration.tracking_mask()
        if full_mask.shape != (metadata.height, metadata.width):
            raise ValueError("Calibration and video dimensions differ.")
        crop = _crop_bounds_from_mask(full_mask, metadata.width, metadata.height)
        mask = _crop_array(full_mask, crop)
        self._region_masks = []
        self._region_areas = []
        self._illumination_zones = []
        for r in calibration.regions:
            m = np.zeros_like(full_mask)
            cv2.fillPoly(m, [r.as_int_polygon()], 255)
            m = _crop_array(m, crop)
            self._region_masks.append(m)
            self._region_areas.append(cv2.countNonZero(m))
        background = _background(video_path, metadata, crop,
                                 self.config.background_sample_count, self.config.gaussian_blur_size)
        arm_width = min(min(cv2.minAreaRect(r.polygon)[1]) for r in calibration.regions[1:])
        self._min_patch_length = 1.2 * arm_width
        self._arm_width = arm_width
        state = EPMTrackState(mask, self.config, fps, arm_width)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open EPM video.")
        rows = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index = len(rows)
                gray = _prepare_gray(_crop_array(frame, crop), self.config.gaussian_blur_size)
                candidates, detection_reason = self._candidates(gray, background, mask)
                accepted, raw, status, reason, raw_step = state.update(candidates, index)
                if accepted is None and status == "lost":
                    reason = detection_reason or reason
                points = state.score_points(accepted, index) if accepted is not None else {}
                def full(point):
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
                    "candidate_evidence": accepted.evidence if accepted else raw.evidence if raw else "none",
                    "candidate_count": len(candidates),
                    "tracking_status": status, "rejection_reason": reason,
                    "low_confidence": status != "tracked", "carried_forward": False,
                    "distance_from_previous_px": raw_step if status == "tracked" else np.nan,
                    "raw_displacement_px": raw_step,
                    "track_segment_id": state.segment if accepted is not None else np.nan,
                })
                if progress_callback and (len(rows) % 120 == 0 or len(rows) == metadata.frame_count):
                    progress_callback(len(rows), metadata.frame_count)
        finally:
            cap.release()
        return pd.DataFrame(rows)
