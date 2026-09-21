"""Explicit occupancy/entry contract; values are operational, not four-paw truth."""
import numpy as np
import pandas as pd
from epm import create_epm_bundle, assign_epm_frames, calibration_from_saved_settings
from tests.test_epm_tracking_v2 import maze


def rows(labels, source='motion_heading'):
    points={'center':(100,100),'open_1':(100,45),'open_2':(100,150),
            'closed_1':(50,100),'closed_2':(150,100),'missing':(np.nan,np.nan)}
    out=[]
    for i,label in enumerate(labels):
        x,y=points[label]; valid=label!='missing'
        out.append({'frame_index':i,'time_seconds':i/10,
                    'centroid_x':x,'centroid_y':y,'smoothed_x':x,'smoothed_y':y,
                    'head_shoulder_x':x,'head_shoulder_y':y,
                    'smoothed_head_shoulder_x':x,'smoothed_head_shoulder_y':y,
                    'head_estimate_source':source,
                    'tracking_status':'tracked' if valid else 'lost',
                    'low_confidence':not valid,'carried_forward':False,
                    'track_segment_id':1 if valid else np.nan,'rejection_reason':''})
    return pd.DataFrame(out)


def test_body_occupancy_survives_missing_head_orientation():
    d=rows(['center']*4+['closed_1']*4, 'centroid_fallback')
    b=create_epm_bundle(d,maze(),10,min_dwell_seconds=.2)
    assert b.region_times.set_index('region').loc['unclassified','frames']==0
    assert b.summary.iloc[0]['Closed Arm Time (whole seconds)']>=0
    assert b.summary.iloc[0]['Closed Arm Entries']==0
    assert b.qc_metrics.set_index('metric').loc['missing_head_proxy_frames','value']==8


def test_observed_transition_scores_and_gap_is_review_only():
    d=rows(['center']*3+['closed_1']*3+['center']*3+['missing']+['open_1']*3)
    b=create_epm_bundle(d,maze(),10,min_dwell_seconds=.2, count_initial_arm=False)
    assert b.events.loc[b.events.review_state=='confirmed_proxy','event'].tolist()==['closed_arm_entry','center_entry']
    assert b.events.loc[b.events.review_state=='requires_manual_review','event'].tolist()==['possible_transition']
    assert b.summary.iloc[0]['Open Arm Entries']==0


def test_same_arm_resumed_after_gap_can_later_cross():
    d=rows(['center']*3+['missing']+['center']*3+['closed_1']*3)
    b=create_epm_bundle(d,maze(),10,min_dwell_seconds=.2,count_initial_arm=False)
    assert b.summary.iloc[0]['Closed Arm Entries']==1


def test_initial_arm_only_at_start_and_interval_excludes_prior_activity():
    d=rows(['center']*10+['closed_1']*4)
    b=create_epm_bundle(d,maze(),10,min_dwell_seconds=.2,
                        analysis_start_seconds=1., analysis_end_seconds=1.4)
    assert b.summary.iloc[0]['Closed Arm Entries']==1
    assert (b.per_frame.iloc[:10].region=='excluded').all()


def test_gap_return_to_same_region_preserves_later_observed_crossing():
    d=rows(['center']*3+['missing']+['center']*3+['closed_1']*3)
    b=create_epm_bundle(d,maze(),10,mode='smoothed_centroid',min_dwell_seconds=.2,count_initial_arm=False)
    assert b.summary.iloc[0]['Closed Arm Entries']==1
    assert b.events.loc[b.events.review_state=='confirmed_proxy','event'].tolist()==['closed_arm_entry']
    assert not b.events.event.eq('possible_transition').any()


def test_short_valid_bridge_can_support_a_flagged_provisional_entry():
    d=rows(['center']*3+['missing']+['closed_1']+['missing']+['closed_1']*2)
    b=create_epm_bundle(d,maze(),10,mode='smoothed_centroid',min_dwell_seconds=.2,count_initial_arm=False)
    inferred=b.events.loc[b.events.review_state=='inferred_provisional']
    assert len(inferred)==1
    assert int(inferred.iloc[0].frame_index)==3
    assert b.summary.iloc[0]['Closed Arm Entries']==1


def test_smoothing_disagreement_uses_neighboring_observations_with_qc_flag():
    d=rows(['center']*3+['closed_2']*3)
    d.loc[3,'smoothed_x']=100
    b=create_epm_bundle(d,maze(),10,mode='smoothed_centroid',min_dwell_seconds=.2)
    assert b.per_frame.loc[3,'region']=='closed_2'
    assert b.per_frame.loc[3,'entry_region']=='closed_2'
    assert bool(b.per_frame.loc[3,'centroid_disagreement'])
    assert bool(b.per_frame.loc[3,'assignment_uncertain'])
    assert b.region_times.set_index('region').loc['unclassified','frames']==0
    assert b.summary.iloc[0]['Closed Arm Entries']==1
    assert b.events.loc[b.events.review_state=='inferred_provisional','event'].tolist()==['closed_arm_entry']


def test_short_arm_peek_is_center_time_without_arm_or_return_entry():
    d=rows(['center']*5+['open_1']*2+['center']*5)
    b=create_epm_bundle(d,maze(),10,mode='smoothed_centroid',count_initial_arm=False)
    assert b.summary.iloc[0]['Open Arm Entries']==0
    assert b.summary.iloc[0]['Center Entry']==0
    assert b.region_times.set_index('region').loc['center','frames']==12
    assert b.per_frame.loc[5:6,'peek_reassigned_to_center'].all()


def test_one_second_arm_and_point_one_second_center_entry():
    d=rows(['center']*5+['open_1']*10+['center']*2)
    b=create_epm_bundle(d,maze(),10,mode='smoothed_centroid',count_initial_arm=False)
    assert b.summary.iloc[0]['Open Arm Entries']==1
    assert b.summary.iloc[0]['Center Entry']==1
    assert b.region_times.set_index('region').loc['open_1','frames']==10


def test_missing_frames_get_position_but_impossible_diagonal_is_not_interpolated():
    d=rows(['open_1']*3+['missing']+['closed_2']*3)
    b=create_epm_bundle(d,maze(),10,mode='smoothed_centroid',count_initial_arm=False)
    assert b.per_frame.loc[3,'region']=='open_1'
    assert b.per_frame.loc[3,'assignment_method']=='held_previous'
    assert bool(b.per_frame.loc[3,'assignment_uncertain'])


def test_saved_calibration_rejects_wrong_recording():
    target=maze()
    payload={'video_signature':'video-a', 'calibration':target.to_dict()}
    assert calibration_from_saved_settings(payload,200,200,'video-a').tracking_mask().shape==(200,200)
    import pytest
    with pytest.raises(ValueError,match='different video'):
        calibration_from_saved_settings(payload,200,200,'video-b')
