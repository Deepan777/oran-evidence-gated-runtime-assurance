"""Software unit tests only; fixtures are not radio observations or results."""
from pathlib import Path
import numpy as np
from analysis.prospective_comparison_v14 import comparison, holm_family
from src.controller.action_mask import Action
from src.controller.safety_guard import SafetyObservation, UncertaintySafetyGuard
from experiments.live import prospective_runtime_v14 as wrapper


def obs(index, throughput=1., latency=1., previous=Action.NO_OP, degraded=False):
    return SafetyObservation(index,'UNIT_ONLY','UNIT_ONLY',index,str(index),index+1,True,True,
        .2 if degraded else 0.,1. if degraded else 0.,throughput,0.,latency,previous)


def test_exploratory_reference_does_not_change_primary_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(wrapper,'OUT',tmp_path)
    primary = UncertaintySafetyGuard(); exploratory = wrapper.ExploratoryPreActionGuard()
    for i in range(10):
        assert primary.decide(obs(i)).as_dict() == exploratory.decide(obs(i)).as_dict()
    for i in range(10,13):
        primary.decide(obs(i,.2,100.,degraded=True)); exploratory.decide(obs(i,.2,100.,degraded=True))
    assert exploratory.action_reference['windows'] == [10,11,12]
    assert np.isclose(exploratory.action_reference['throughput_mbps'],.2)
    old = primary.decide(obs(13,.2,100.,Action.MCS_BACKOFF,True))
    new = exploratory.decide(obs(13,.2,100.,Action.MCS_BACKOFF,True))
    assert old.reason == 'post_action_harm_boundary_crossed'
    assert new.allow_policy
    changed = exploratory.decide(obs(14,.01,400.,Action.MCS_BACKOFF,True))
    assert changed.reason == 'post_action_harm_boundary_crossed'


def test_exploratory_still_obeys_duration_bound(tmp_path,monkeypatch):
    monkeypatch.setattr(wrapper,'OUT',tmp_path)
    guard = wrapper.ExploratoryPreActionGuard()
    for i in range(10): guard.decide(obs(i))
    for i in range(10,17):
        result = guard.decide(obs(i,.7,1.,Action.NO_OP if i <= 12 else Action.MCS_BACKOFF,True))
    assert result.allow_policy
    result = guard.decide(obs(17,.7,1.,Action.MCS_BACKOFF,True))
    assert result.reason == 'maximum_action_duration_reached'


def test_exact_binary_test_resamples_blocks_not_windows():
    rows=[]
    for i in range(4):
        for arm,value in [('guarded_policy',1),('policy',0)]:
            rows.append(dict(accepted=True,block=str(i),arm=arm,jammer='partial_band',metrics={'recovery_by_10s':value}))
    result=comparison(rows,'guarded_policy','policy','recovery_by_10s',scope='interference',seed=1)
    assert result['n_pairs']==4 and result['p_value']==.125 and result['ci95']==[1.,1.]
    holm_family([result],36)
    assert result['holm_p']==1. and result['holm_planned_family_size']==36


def test_missing_endpoints_not_imputed():
    rows=[dict(accepted=True,block='UNIT',arm=arm,jammer='none',metrics={'throughput_retention':None})
          for arm in ('guarded_policy','policy')]
    result=comparison(rows,'guarded_policy','policy','throughput_retention',scope='all',seed=1)
    assert result['n_pairs']==0 and result['p_value'] is None and result['ci95'] is None
