"""Unit-only receiver fixtures, never experimental observations."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.xapp.telemetry.srsue_csv import parse_srsue_csv
from experiments.collection import build_live_run_records as collector
from src.controller.state import CausalStateImputer,StateSchema

HEADER='timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff;is_attached'
def row(sinr='inf',bler='2',attached='1'):
    return f'1786630000000;-63;-11;-52;{sinr};2000000;{bler};400;{attached}'

def test_legacy_parser_still_rejects_infinity():
    with pytest.raises(ValueError,match='dl_snr'): parse_srsue_csv(HEADER,row())

def test_opt_in_keeps_finite_fields_without_inventing_sinr():
    result=parse_srsue_csv(HEADER,row(),allow_unbounded_sinr=True)
    assert 'sinr_db' not in result.values
    assert result.values['sinr_positive_infinity_observed']==1
    assert result.values['ue_dl_bler']==.02
    assert result.values['ue_dl_throughput_mbps']==2
    assert result.values['receiver_attached']==1
    assert all(np.isfinite(value) for value in result.values.values())

@pytest.mark.parametrize('sinr',['nan','-inf','n/a'])
def test_other_invalid_sinr_does_not_receive_the_exception(sinr):
    with pytest.raises(ValueError): parse_srsue_csv(HEADER,row(sinr),allow_unbounded_sinr=True)

def test_missing_bler_and_detached_infinity_remain_rejected():
    with pytest.raises(ValueError,match='dl_bler'): parse_srsue_csv(HEADER,row(bler='nan'),allow_unbounded_sinr=True)
    with pytest.raises(ValueError,match='dl_snr'): parse_srsue_csv(HEADER,row(attached='0'),allow_unbounded_sinr=True)

def test_policy_substitution_is_explicit_not_measured():
    sample=parse_srsue_csv(HEADER,row(),allow_unbounded_sinr=True)
    imputer=CausalStateImputer(StateSchema(('sinr_db','bler'),np.array([10.,.1]),np.ones(2)))
    values,provenance=imputer.fill({'bler':sample.values['ue_dl_bler']},('sinr_db','bler'))
    assert values=={'sinr_db':10.,'bler':.02}
    assert provenance=={'sinr_db':'training_mean','bler':'measured'}

def test_collector_exception_requires_explicit_profile_and_exact_missing_set(monkeypatch,tmp_path:Path):
    origin=1786630000000000000
    def sample(values,raw): return collector.TimedValues(origin+1000,values,raw)
    receiver=sample({'receiver_attached':1.,'sinr_positive_infinity_observed':1.,'rsrp_dbfs':-63.,'rsrq_db':-11.,'ue_dl_bler':.02},{'dl_snr':'inf','dl_bler':'2'})
    scheduler=sample({'cqi':12.,'harq_nack_ratio':0.,'gnb_dl_throughput_mbps':2.,'prb_utilization':.3,'scheduler_queue_backlog_bytes':100.},{'scheduler':'unit'})
    rlc=sample({'rlc_drop_rate':0.,'rlc_sdu_delay_ms':2.},{'rlc':'unit'})
    sensor=sample({'clean_prb_fraction':1.,'jammer_occupancy_fraction':0.,'jammer_subband_start_norm':0.,'jammer_subband_end_norm':0.},{'sensor':'unit'})
    seen=[]
    def load_receiver(path,*,allow_unbounded_sinr=False):
        seen.append(allow_unbounded_sinr)
        return [receiver] if allow_unbounded_sinr else []
    monkeypatch.setattr(collector,'_load_srsue',load_receiver)
    monkeypatch.setattr(collector,'_load_srsue_status',lambda path:[sample({'receiver_attached':1.},{'is_attached':'1'})])
    monkeypatch.setattr(collector,'_load_ocudu',lambda path:([scheduler],[rlc]))
    monkeypatch.setattr(collector,'_load_iq_sensor',lambda path:{0:sensor})
    monkeypatch.setattr(collector,'_json_lines',lambda path:[])
    monkeypatch.setattr(collector,'_truth_clock',lambda truth,start:[(origin,0,100)])
    monkeypatch.setattr(collector,'_map_to_radio_clock',lambda samples,clock:samples)
    monkeypatch.setattr(collector,'_in_window',lambda samples,start,end:samples)
    monkeypatch.setattr(collector,'_truth_for_window',lambda *args:{})
    metadata={'run_id':'unit','scenario_id':'unit','seed':1,'evidence_class':'UNIT_ONLY','measurement_start_ns':origin,'sample_rate_hz':100,'run_length_seconds':1,'pre_jam_seconds':1,'jammer_on_seconds':0,'recovery_seconds':0}
    assert not bool(collector.build_records(tmp_path,metadata).iloc[0].evidence_complete)
    metadata['acceptance_profile']='observed_outage_and_unbounded_sinr_v2'
    result=collector.build_records(tmp_path,metadata).iloc[0]
    assert bool(result.evidence_complete) and not bool(result.complete_telemetry)
    assert not bool(result.complete_controller_state) and result.telemetry_observability=='native_positive_infinite_sinr_observed'
    assert 'sinr_db' in json.loads(result.missing_telemetry)
    assert json.loads(result.receiver_measurement_qualifiers)=={'sinr_db':'native_positive_infinity_unbounded'}
    assert result.bler==.02 and '"inf"' in result.raw_srsue_json
    rlc.values.pop('rlc_sdu_delay_ms')
    assert not bool(collector.build_records(tmp_path,metadata).iloc[0].evidence_complete)
    assert seen==[False,True,True]
