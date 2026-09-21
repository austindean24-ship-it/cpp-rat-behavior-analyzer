"""EPM-only, evidence-aware occupancy and witnessed-entry scoring.

Body location is not head orientation. Time uses the tracked body; entries use
an explicitly selected, labeled proxy. Unseen crossings are review items,
never silently counted. Neither proxy establishes a four-paw entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from epm import REGION_ORDER, SUMMARY_COLUMNS, EPMCalibration


@dataclass
class EPMBundle:
    per_frame: pd.DataFrame
    summary: pd.DataFrame
    region_times: pd.DataFrame
    events: pd.DataFrame
    qc_metrics: pd.DataFrame
    warnings: list[str]


def _region(point, calibration: EPMCalibration, margin: float):
    if point is None or not np.isfinite(point).all():
        return 'unclassified', '', False
    distances = [r.signed_distance(point) for r in calibration.regions]
    inside = [i for i, d in enumerate(distances) if d >= 0]
    if len(inside) > 1:
        return 'unclassified', calibration.regions[inside[0]].name, True
    if not inside:
        return 'outside', '', False
    label = calibration.regions[inside[0]].name
    return ('unclassified' if abs(distances[inside[0]]) < margin else label), label, abs(distances[inside[0]]) < margin


def _xy(row, cols):
    for xname, yname in cols:
        x = row.get(xname, np.nan)
        y = row.get(yname, np.nan)
        if pd.notna(x) and pd.notna(y) and np.isfinite(float(x)) and np.isfinite(float(y)):
            return float(x), float(y), xname.removesuffix('_x')
    return np.nan, np.nan, 'missing'


def assign_epm_frames(tracking_df, calibration: EPMCalibration, mode='head_shoulders', boundary_margin_px=2.0):
    if mode not in {'head_shoulders', 'smoothed_centroid', 'centroid'}:
        raise ValueError(f'Unknown EPM entry proxy: {mode}')
    if boundary_margin_px < 0:
        raise ValueError('Boundary margin cannot be negative')
    records = []
    for row in tracking_df.to_dict('records'):
        accepted = row.get('tracking_status') == 'tracked' and not bool(row.get('low_confidence', False)) and not bool(row.get('carried_forward', False))
        cx, cy, body_source = _xy(row, [('smoothed_x', 'smoothed_y'), ('centroid_x', 'centroid_y')]) if accepted else (np.nan, np.nan, 'missing')
        body_point = (cx, cy) if np.isfinite(cx) else None
        body_region, body_candidate, body_boundary = _region(body_point, calibration, boundary_margin_px)
        if not accepted:
            body_region = 'unclassified'
        if mode == 'head_shoulders':
            proxy_valid = row.get('head_estimate_source', 'provided_head_proxy') in {'motion_heading', 'provided_head_proxy'}
            ex, ey, entry_source = _xy(row, [('smoothed_head_shoulder_x', 'smoothed_head_shoulder_y'), ('head_shoulder_x', 'head_shoulder_y')]) if accepted and proxy_valid else (np.nan, np.nan, 'missing_head_orientation')
        elif mode == 'smoothed_centroid':
            ex, ey, entry_source = cx, cy, body_source
        else:
            ex, ey, entry_source = _xy(row, [('centroid_x', 'centroid_y')]) if accepted else (np.nan, np.nan, 'missing')
        entry_region, entry_candidate, entry_boundary = _region((ex, ey) if np.isfinite(ex) else None, calibration, boundary_margin_px)
        row.update({
            'assignment_x': cx, 'assignment_y': cy, 'assignment_point_mode': 'body_occupancy',
            'assignment_point_source': body_source, 'entry_assignment_x': ex, 'entry_assignment_y': ey,
            'entry_point_mode': mode, 'entry_point_source': entry_source,
            'entry_region': entry_region, 'entry_candidate_region': entry_candidate,
            'entry_boundary': entry_boundary,
            'candidate_region': body_candidate, 'region': body_region,
            'arm_type': 'open' if body_region.startswith('open_') else 'closed' if body_region.startswith('closed_') else body_region,
            'on_boundary': body_boundary, 'inside_arena': body_region in REGION_ORDER,
            'boundary_hold': bool(entry_boundary and accepted),
            'body_position_valid': bool(accepted and body_region in REGION_ORDER),
            'anatomical_entry_point_valid': False,
        })
        records.append(row)
    return pd.DataFrame.from_records(records)


def detect_epm_events(assigned, fps, min_dwell_seconds=0.3, count_initial_arm=True):
    """Count transitions with witnessed contiguous center-to-arm evidence only."""
    if fps <= 0 or min_dwell_seconds < 0:
        raise ValueError('FPS must be positive and dwell must be nonnegative')
    dwell = max(1, math.ceil(fps * min_dwell_seconds))
    data = assigned.copy()
    data['event'] = ''
    events = []
    stable = None
    pending = None
    pending_start = -1
    pending_count = 0
    prior_region = None
    prior_segment = None
    previous_eligible_frame = None
    observed_boundary = False
    invalid_between = False
    first_valid = None
    ever_baselined = False
    skipped = 0
    included = np.flatnonzero(data['analysis_included'].to_numpy()) if 'analysis_included' in data else np.arange(len(data))
    interval_start = int(included[0]) if len(included) else 0

    def emit(name, source, target, start, end, status='confirmed', reason='observed_transition'):
        beginning, ending = data.iloc[start], data.iloc[end]
        events.append({
            'event': name, 'from_region': source or 'start', 'to_region': target,
            'frame_index': int(beginning['frame_index']),
            'time_seconds': float(beginning['frame_index']) / fps,
            'confirmed_frame_index': int(ending['frame_index']),
            'confirmed_time_seconds': float(ending['frame_index']) / fps,
            'supporting_frames': end - start + 1,
            'track_segment_id': beginning.get('track_segment_id', np.nan),
            'point_source': beginning.get('entry_point_source', ''),
            'evidence': reason,
            'review_state': status,
        })
        if status == 'confirmed':
            data.at[data.index[end], 'event'] = name

    for position, row in enumerate(data.to_dict('records')):
        region = row.get('entry_region', row.get('region', 'unclassified'))
        segment = row.get('track_segment_id', np.nan)
        valid_seg = pd.notna(segment)
        seg_change = valid_seg and prior_segment is not None and segment != prior_segment
        if valid_seg:
            prior_segment = segment
        eligible = (region in REGION_ORDER and row.get('tracking_status') == 'tracked'
                    and not bool(row.get('low_confidence', False))
                    and not bool(row.get('carried_forward', False))
                    and bool(row.get('analysis_included', True)))
        if seg_change:
            invalid_between = True
            pending = None
            pending_count = 0
        if row.get('boundary_hold', False) and bool(row.get('analysis_included', True)):
            observed_boundary = True
            pending = None
            pending_count = 0
            continue
        if not eligible:
            invalid_between = True
            observed_boundary = False
            pending = None
            pending_count = 0
            continue
        if first_valid is None:
            first_valid = position
        consecutive = previous_eligible_frame is not None and row['frame_index'] == previous_eligible_frame + 1
        previous_eligible_frame = int(row['frame_index'])
        if stable == region:
            pending = None
            pending_count = 0
            invalid_between = False
            observed_boundary = False
            continue
        if stable is not None and (invalid_between or (not consecutive and not observed_boundary)):
            if prior_region != region:
                emit('possible_transition', stable, region, position, position,
                     'requires_manual_review', 'location_changed_during_unobserved_interval')
            stable = None
            pending = None
            pending_count = 0
        if pending != region:
            pending, pending_start, pending_count = region, position, 1
        else:
            pending_count += 1
        if pending_count < dwell:
            invalid_between = False
            observed_boundary = False
            continue
        if stable is None:
            if not ever_baselined and count_initial_arm and region != 'center' and first_valid - interval_start <= dwell:
                emit('open_arm_entry' if region.startswith('open_') else 'closed_arm_entry',
                     None, region, pending_start, position, 'confirmed', 'initial_arm_occupancy')
        elif stable == 'center' and region.startswith(('open_', 'closed_')):
            emit('open_arm_entry' if region.startswith('open_') else 'closed_arm_entry',
                 stable, region, pending_start, position)
        elif stable != 'center' and region == 'center':
            emit('center_entry', stable, region, pending_start, position)
        elif stable != 'center' and region != 'center':
            skipped += 1
            emit('possible_transition', stable, region, pending_start, position,
                 'requires_manual_review', 'direct_arm_change_without_observed_center')
        stable, prior_region = region, region
        pending = None
        pending_count = 0
        invalid_between = False
        observed_boundary = False
        ever_baselined = True
    columns = ['event', 'from_region', 'to_region', 'frame_index', 'time_seconds',
               'confirmed_frame_index', 'confirmed_time_seconds', 'supporting_frames',
               'track_segment_id', 'point_source', 'evidence', 'review_state']
    return data, pd.DataFrame(events, columns=columns), skipped


def _whole_seconds(counts, fps):
    exact = [v / fps for v in counts]
    values = [math.floor(x) for x in exact]
    target = math.floor(sum(exact) + 0.5)
    ranking = sorted(range(len(values)), key=lambda i: (-(exact[i] - values[i]), i))
    for idx in ranking[:target - sum(values)]:
        values[idx] += 1
    return values


def create_epm_bundle(tracking_df, calibration, fps, mode='head_shoulders',
                      min_dwell_seconds=0.3, count_initial_arm=True,
                      analysis_start_seconds=0., analysis_end_seconds=None,
                      boundary_margin_px=2.):
    if tracking_df.empty or fps <= 0:
        raise ValueError('Need nonempty tracking and positive FPS')
    first = round(analysis_start_seconds * fps)
    end = len(tracking_df) if analysis_end_seconds is None else round(analysis_end_seconds * fps)
    if not 0 <= first < end <= len(tracking_df):
        raise ValueError('Analysis interval must be nonempty and inside video')
    data = assign_epm_frames(tracking_df, calibration, mode, boundary_margin_px)
    data['analysis_included'] = data['frame_index'].between(first, end - 1)
    data.loc[~data['analysis_included'], ['region', 'arm_type', 'entry_region']] = 'excluded'
    per_frame, events, skipped = detect_epm_events(data, fps, min_dwell_seconds, count_initial_arm)
    selected = per_frame[per_frame['analysis_included']]
    counts = {name: int((selected['region'] == name).sum()) for name in REGION_ORDER}
    unknown = len(selected) - sum(counts.values())
    secs = _whole_seconds([counts['open_1'] + counts['open_2'],
                           counts['closed_1'] + counts['closed_2'], counts['center'], unknown], fps)
    confirmed = events.loc[events['review_state'] == 'confirmed', 'event'].value_counts() if len(events) else pd.Series(dtype=int)
    summary = pd.DataFrame([{
        'Open Arm Entries': int(confirmed.get('open_arm_entry', 0)),
        'Closed Arm Entries': int(confirmed.get('closed_arm_entry', 0)),
        'Center Entry': int(confirmed.get('center_entry', 0)),
        'Open Arm Time (whole seconds)': secs[0],
        'Closed Arm Time (whole seconds)': secs[1],
        'Center Time (whole seconds)': secs[2],
    }], columns=SUMMARY_COLUMNS)
    region_times = pd.DataFrame([{'region': k, 'frames': v, 'seconds': round(v / fps, 3)} for k, v in counts.items()] +
                                [{'region': 'unclassified', 'frames': unknown, 'seconds': round(unknown / fps, 3)}])
    statuses = selected['tracking_status']
    rejected = int((statuses == 'rejected').sum())
    review_count = int((events['review_state'] == 'requires_manual_review').sum())
    qc_dict = {
        'analysis_start_frame': first, 'analysis_end_frame_exclusive': end,
        'analysis_duration_seconds': round(len(selected) / fps, 3),
        'scored_coverage_percent': round(100 * (len(selected) - unknown) / len(selected), 2),
        'unclassified_frames': unknown, 'unclassified_seconds': round(unknown / fps, 3),
        'unclassified_whole_seconds': secs[3],
        'low_confidence_frames': int(selected['low_confidence'].fillna(False).sum()),
        'lost_frames': int((statuses == 'lost').sum()),
        'reacquiring_frames': int((statuses == 'reacquiring').sum()),
        'rejected_candidate_frames': rejected,
        'rejected_raw_jumps': int((selected['rejection_reason'] == 'raw_jump').sum()) if 'rejection_reason' in selected else 0,
        'boundary_unclassified_frames': int(selected['on_boundary'].sum()),
        'missing_head_proxy_frames': int(((statuses == 'tracked') &
            ~selected['head_estimate_source'].isin({'motion_heading', 'provided_head_proxy'})).sum()) if 'head_estimate_source' in selected else 0,
        'skipped_direct_arm_changes': skipped,
        'possible_transitions_for_review': review_count,
        'event_dwell_frames': max(1, math.ceil(min_dwell_seconds * fps)),
        'occupancy_source': 'tracked_body_centroid',
        'entry_proxy': mode,
        'review_state': 'needs_manual_review',
    }
    qc = pd.DataFrame([{'metric': k, 'value': v} for k, v in qc_dict.items()] +
                      [{'metric': f'rejection_reason_{k}', 'value': int(v)} for k, v in
                       selected.get('rejection_reason', pd.Series(dtype=str)).dropna().value_counts().items() if k])
    warnings = ['Needs manual review: operational entry proxies are NOT verified four-paw scoring.']
    warnings += [f'{unknown} frames remain unclassified; uncertain time is not invented.',
                 'Arm time uses tracked body centroid; entry events use selected operational proxy.']
    if review_count:
        warnings.append(f'{review_count} possible transitions require manual adjudication and were NOT counted.')
    if rejected:
        warnings.append(f'{rejected} frames contained rejected candidates; inspect the QC video.')
    if qc_dict['scored_coverage_percent'] < 90:
        warnings.append('Body-position coverage below 90%; inspect tracking and calibration.')
    return EPMBundle(per_frame, summary, region_times, events, qc, warnings)
