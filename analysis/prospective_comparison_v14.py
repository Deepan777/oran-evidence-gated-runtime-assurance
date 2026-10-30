"""Analyse actual Study V attempts; never collect, impute or alter observations."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from analysis.corrected_guard_integration_analysis import summarize_decisions, read_jsonl
from experiments.collection import build_live_run_records as collector

OUTCOMES = ('rlc_delay_outage_rate','throughput_retention','rlc_drop_rate','recovery_by_10s')
OUTAGES = {'receiver_detached_outage','right_censored_no_successful_sdu'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def load(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def native_nan_windows(evidence, metadata):
    truth = list(collector._json_lines(evidence / 'jammer_truth.jsonl'))
    clock = collector._truth_clock(truth, collector._experiment_started_ns(evidence, metadata))
    samples = collector._map_to_radio_clock(collector._load_srsue_status(evidence/'srsue_metrics.csv'), clock)
    rate = int(metadata.get('sample_rate_hz', 23040000))
    missing = []
    for index in range(45):
        found = collector._in_window(samples, index*rate, (index+1)*rate)
        if not found: continue
        try:
            if math.isnan(float(found[-1].raw['dl_snr'])): missing.append(index)
        except (KeyError, ValueError, TypeError):
            pass  # Other malformed/missing evidence is not the NaN exception.
    return missing


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    selected = values[np.isfinite(values)]
    return None if not len(selected) else float(selected.mean())


def derive_outcomes(frame, missing):
    frame = frame.sort_values('window_index')
    observed = ~frame.window_index.isin(missing)
    pre = frame[(frame.window_index < 15) & observed]
    exposure = frame[(frame.window_index >= 15) & (frame.window_index < 35) & observed]
    recovery = frame[frame.window_index >= 35]
    baseline = finite_mean(pre.dl_throughput_mbps)
    delay = exposure.rlc_sdu_delay_ms.to_numpy(dtype=float)
    classes = exposure.telemetry_observability.astype(str).to_numpy()
    violations = ~np.isfinite(delay) | (delay > 20.) | np.isin(classes, list(OUTAGES))
    goodput = finite_mean(exposure.dl_throughput_mbps)
    qualified = []
    for _, row in recovery.iterrows():
        qualified.append(bool(int(row.window_index) not in missing and baseline is not None and baseline > 0
            and math.isfinite(float(row.dl_throughput_mbps)) and float(row.dl_throughput_mbps) >= .9*baseline
            and math.isfinite(float(row.rlc_sdu_delay_ms)) and float(row.rlc_sdu_delay_ms) <= 20.
            and str(row.telemetry_observability) not in OUTAGES))
    event = next((i+3 for i in range(8) if all(qualified[i:i+3]) and len(qualified[i:i+3]) == 3), None)
    return {'rlc_delay_outage_rate': None if not len(violations) else float(violations.mean()),
            'throughput_retention': None if baseline is None or baseline <= 0 or goodput is None else goodput/baseline,
            'rlc_drop_rate': finite_mean(exposure.rlc_drop_rate),
            'recovery_by_10s': None if baseline is None or baseline <= 0 else int(event is not None), 'recovery_event_time_s': event,
            'right_censored_at_s': 10 if event is None else None,
            'recovery_by_20s': None, 'twenty_second_endpoint_status':'unavailable_frozen_followup_is_10s',
            'pre_observed_windows':len(pre), 'exposure_observed_windows':len(exposure),
            'recovery_observable_windows':int(sum(observed & (frame.window_index >= 35))),
            'native_snr_missing_windows':missing}


def audit_run(evidence, row, *, historical=False, verify_attestation=False):
    result = {**row, 'accepted':False, 'acceptance_reasons':[], 'metrics':None, 'counts':{}}
    reasons = result['acceptance_reasons']
    if not evidence.is_dir():
        reasons.append('not_attempted_or_evidence_directory_absent')
        result['status'] = 'not_attempted'
        return result
    if verify_attestation and not historical:
        attestation = load(evidence/'v14_attestation.json', {})
        if not attestation or attestation.get('run_id') != row['run_id']:
            reasons.append('missing_or_wrong_run_attestation')
        else:
            for item in attestation.get('preserved_native_evidence',[]):
                path = evidence/item['path']
                if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                    reasons.append('native_evidence_integrity_mismatch:'+item['path'])
    metadata = load(evidence/'run_metadata.json', {})
    required = ['run_metadata.json','records.parquet','radio-clock-status.json','protected-traffic-status.json',
                'protected_udp.json','srsue_metrics.csv','ocudu-metrics.jsonl','iq-spectral-estimator.jsonl',
                'jammer_truth.jsonl','gnb.full.log','srsue.full.log','client-exit-status.txt']
    if not historical: required.append('registration.log')
    elif not (evidence/'registration.log').exists():
        result['historical_provenance_gaps'] = ['registration_stdout_not_in_prior_archive; native_container_and_traffic_records_retained']
    for name in required:
        # The frozen registration script can complete quietly. Its retained
        # zero-byte stdout file is not evidence that registration failed;
        # native NR/traffic/container evidence separately verifies the run.
        if not (evidence/name).is_file() or (name != 'registration.log' and not (evidence/name).stat().st_size):
            reasons.append('missing_or_empty:'+name)
    if metadata.get('run_id') != row['run_id'] or metadata.get('seed') != row['seed']:
        reasons.append('run_identity_mismatch')
    for field, expected in [('run_length_seconds',45),('pre_jam_seconds',15),('jammer_on_seconds',20),('recovery_seconds',10)]:
        if metadata.get(field) != expected: reasons.append('frozen_phase_mismatch:'+field)
    traffic = load(evidence/'protected-traffic-status.json', {})
    if traffic.get('status') != 'PASS' or traffic.get('complete_radio_windows') != 45 or len(traffic.get('radio_windows',[])) != 45:
        reasons.append('protected_application_traffic_acceptance_failed')
    clock = load(evidence/'radio-clock-status.json', {})
    # The collector's clock status field is versioned; native covered rows
    # and collector exit status below are also required, never inferred.
    if clock.get('status') not in ('PASS','COMPLETE', 'complete', 'RADIO_TARGET_REACHED'):
        if clock.get('complete') is not True and clock.get('target_reached') is not True:
            result['radio_clock_status_document'] = clock
    exits = (evidence/'client-exit-status.txt').read_text() if (evidence/'client-exit-status.txt').exists() else ''
    if 'radio_clock_exit=0' not in exits: reasons.append('radio_clock_not_successful')
    if row.get('exit_code') not in (None,0): reasons.append('collector_exit_nonzero:'+str(row['exit_code']))
    try:
        frame = pd.read_parquet(evidence/'records.parquet')
        if len(frame) != 45 or sorted(frame.window_index.astype(int)) != list(range(45)):
            raise ValueError('not_exactly_45_unique_windows')
        if not frame.radio_window_covered.astype(bool).all(): reasons.append('radio_duration_incomplete')
        if frame.run_id.nunique() != 1 or frame.run_id.iloc[0] != row['run_id']: reasons.append('record_run_identity_mismatch')
        missing = native_nan_windows(evidence, metadata)
        complete = frame.evidence_complete.astype(bool) & ~frame.window_index.isin(missing)
        result['complete_windows'] = int(complete.sum())
        result['native_snr_nan_windows'] = missing
        bad = frame.loc[~complete, 'window_index'].astype(int).tolist()
        if historical:
            if row.get('previously_accepted') is not True: reasons.append('original_rejection_permanently_preserved')
            if not frame.evidence_complete.astype(bool).all(): reasons.append('original_strict_acceptance_incomplete')
        else:
            if int(complete.sum()) < 44: reasons.append('fewer_than_44_complete_windows')
            if any(index not in missing for index in bad): reasons.append('missing_window_not_native_snr_nan')
        result['metrics'] = derive_outcomes(frame, [] if historical else missing)
        result['observability_counts'] = dict(Counter(frame.telemetry_observability.astype(str)))
    except Exception as exc:
        reasons.append('measurement_audit:'+type(exc).__name__+':'+str(exc))
    if row['arm'] != 'no_action':
        try:
            records = read_jsonl(evidence/'live-xapp-decisions.jsonl')
            if [r['window_index'] for r in records] != list(range(45)):
                reasons.append('decision_identity_or_coverage_incomplete')
            result['counts'] = summarize_decisions(records, expected_windows=45)['counts']
            result['harm_boundary_revocations'] = sum((r.get('safety_guard') or {}).get('reason') == 'post_action_harm_boundary_crossed' for r in records)
            result['revocations'] = sum((r.get('safety_guard') or {}).get('reason') in
                ('post_action_harm_boundary_crossed','maximum_action_duration_reached') or
                (r.get('post_execution_guard') or {}).get('reason') == 'execution_uncertain_restore_baseline' for r in records)
            result['counts']['interceptions'] = result['counts'].get('intercepted_nontrivial_proposals',0)
            result['counts']['unknown_episodes'] = result['counts'].get('post_baseline_unknown_entries',0)
            result['counts']['revocation_windows'] = result['revocations']
            result['guard_reasons'] = dict(Counter((r.get('safety_guard') or {}).get('reason','unrecorded') for r in records))
            if not historical and not (evidence/'command-transport.jsonl').is_file(): reasons.append('missing_live_command_transport_records')
        except Exception as exc:
            reasons.append('decision_audit:'+type(exc).__name__+':'+str(exc))
    else:
        result['controller_records'] = 'not_applicable_controller_disabled_by_design'
    faults = read_jsonl(evidence/'fault-injection.jsonl')
    result['fault_injections'] = faults
    expected_fault = row.get('fault','none')
    if expected_fault != 'none':
        commands = read_jsonl(evidence/'command-transport.jsonl')
        actual_timeout = any(r.get('injection_triggered') and 'TimeoutError' in r.get('actual_exception','') for r in commands)
        delivered = any(r.get('fault') == expected_fault and 'proxy_error' not in r and
            (r.get('upstream_command_sent') is True and
             (r.get('actual_upstream_acknowledgement') or {}).get('cmd') == 'research_action_set' and
             (r.get('actual_upstream_acknowledgement') or {}).get('status') != 'error' and actual_timeout
             if expected_fault == 'lost_ack' else
             'actual_wire_document' in r and result['counts'].get('parameter_mismatch_executions',0)>0
             if expected_fault == 'applied_parameter_mismatch' else
             'proxy_received_command' in r and actual_timeout) for r in faults)
        result['fault_delivery'] = 'observed' if delivered else 'inconclusive_not_delivered'
        # Failed fault delivery is retained as a replication outcome, not a
        # reason to remove a measurable run or add an unplanned replacement.
    result['accepted'] = not reasons
    result['status'] = 'accepted' if result['accepted'] else 'rejected'
    return result


def comparison(rows, left, right, outcome, *, scope, seed, exploratory=False):
    values = {}
    for row in rows:
        if not row['accepted'] or row['arm'] not in (left,right): continue
        if scope == 'interference' and row['jammer'] == 'none': continue
        if scope == 'no_interference' and row['jammer'] != 'none': continue
        value = (row.get('metrics') or {}).get(outcome)
        if value is not None and math.isfinite(float(value)):
            if row['arm'] in values.setdefault(row['block'], {}): raise ValueError('duplicate accepted block-arm')
            values[row['block']][row['arm']] = float(value)
    blocks = sorted(k for k,v in values.items() if set(v) == {left,right})
    result = dict(scope=scope, left=left, right=right, outcome=outcome, n_pairs=len(blocks), blocks=blocks,
                  exploratory=exploratory, p_value=None, mean_difference=None, ci95=None,
                  bootstrap_iterations=10000, bootstrap_seed=seed)
    if not blocks: return result
    a = np.array([values[k][left] for k in blocks]); b = np.array([values[k][right] for k in blocks])
    delta = a-b
    result['mean_difference'] = float(delta.mean())
    if len(blocks) < 2:
        result['test_status'] = 'insufficient_independent_blocks'
        return result
    rng = np.random.default_rng(seed)
    estimates = delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    result['ci95'] = np.quantile(estimates,[.025,.975]).tolist()
    if outcome == 'recovery_by_10s':
        left_only = int(((a==1)&(b==0)).sum()); right_only = int(((a==0)&(b==1)).sum())
        result.update(test='exact_two_sided_McNemar', left_only=left_only, right_only=right_only,
                      both=int(((a==1)&(b==1)).sum()), neither=int(((a==0)&(b==0)).sum()),
                      p_value=1.0 if left_only+right_only == 0 else float(binomtest(left_only,left_only+right_only,.5).pvalue))
    else:
        result['test'] = 'paired_two_sided_Wilcoxon'
        result['p_value'] = 1.0 if np.all(delta == 0) else float(wilcoxon(a,b,zero_method='wilcox',alternative='two-sided').pvalue)
    return result


def holm_family(rows, family_size):
    observed = sorted((r for r in rows if r['p_value'] is not None), key=lambda r:r['p_value'])
    running = 0.
    for rank, row in enumerate(observed):
        running = max(running, min(1., (family_size-rank)*row['p_value']))
        row['holm_p'] = running
    for row in rows:
        row.setdefault('holm_p',None)
        row['holm_planned_family_size'] = family_size


def csv_write(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def analyse(study, raw_root):
    freeze = load(study/'freeze.json'); progress = load(study/'progress.json', {'completed':[]})
    observed = {r['run_id']:r for r in progress['completed']}
    rows = []
    for planned in freeze['historical_rows']+freeze['planned_rows']:
        supplied = {**planned, **observed.get(planned['run_id'],{})}
        rows.append(audit_run(raw_root/('iq-broker-live-'+planned['run_id']), supplied,
                             historical=planned.get('historical',False),verify_attestation=True))
    included = [r for r in rows if r.get('cohort') == 'primary']
    main = []
    for scope in ('all','interference','no_interference'):
        for right in ('policy','no_action','fixed_a2'):
            for outcome in OUTCOMES:
                main.append(comparison(included,'guarded_policy',right,outcome,scope=scope,seed=20261030+len(main)))
    holm_family(main,36)
    exploratory = []
    for right in ('guarded_policy','policy','no_action','fixed_a2'):
        for outcome in OUTCOMES:
            exploratory.append(comparison(rows,'gpre',right,outcome,scope='interference',
                                         seed=20262030+len(exploratory),exploratory=True))
    holm_family(exploratory,16)
    primary_ok = [r for r in included if r['accepted']]
    unique = {(r['block'],r['arm']) for r in primary_ok}
    complete_blocks = [block for block in sorted({r['block'] for r in included})
                       if {a for b,a in unique if b == block} == {'no_action','policy','guarded_policy','fixed_a2'}]
    report = {'study':'Study V prospective continuation v1.4',
              'protocol_commit':freeze['protocol_commit'], 'collection_freeze_sha256':sha(study/'freeze.json'),
              'recovery_endpoint_seconds':10, 'recovery_by_20s':'unavailable_not_imputed',
              'experimental_observations_only':True, 'complete_primary_comparison':len(unique)==48 and len(complete_blocks)==12,
              'population':{'accepted_unique_primary_rows':len(unique),'complete_matched_blocks':len(complete_blocks),
                  'recorded_attempts':sum(r['status'] != 'not_attempted' for r in rows),
                  'rejected_attempts':sum(r['status']=='rejected' for r in rows),
                  'planned_new_attempts':len(freeze['planned_rows'])},
              'all_attempts':rows,'primary_statistics':main,'exploratory_gpre_statistics':exploratory,
              'replications':[r for r in rows if r['cohort']=='replication'],
              'limitations':['Matched independent runs, not windows, determine replication.',
                  'Known original rejected attempt remains rejected; no retrospective promotion.',
                  'Native-NaN missing windows excluded from available-window metric means; original censoring labels retained.',
                  '10-second recovery follow-up cannot establish the full 20-second endpoint.',
                  'Interference is generated in the existing virtualized software-radio testbed, not over-the-air hardware.']}
    out = study/'analysis_outputs'; out.mkdir(exist_ok=True)
    (out/'prospective_comparison_v14.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    csv_write(out/'per_run_acceptance.csv',[{k:v for k,v in r.items() if k not in ('fault_injections',)} for r in rows])
    csv_write(out/'primary_matched_statistics.csv',main)
    csv_write(out/'per_run_outcomes_and_counts.csv',[{'run_id':r['run_id'],'cohort':r['cohort'],
        'arm':r['arm'],'accepted':r['accepted'],'exploratory':r.get('exploratory',False),
        'revocations':r.get('revocations'),**(r.get('metrics') or {}),**r['counts']} for r in rows])
    csv_write(out/'exploratory_gpre_matched_statistics.csv',exploratory)
    csv_write(out/'replication_counts.csv',[{'run_id':r['run_id'],'kind':r.get('replication_kind'),
              'accepted':r['accepted'],'fault_delivery':r.get('fault_delivery'),
              'revocations':r.get('revocations'),**r['counts']} for r in report['replications']])
    summary = ['# Study V prospective comparison v1.4', '',
        'Real testbed observations only. No results imputed or edited.',
        'Recovery is evaluated by **10 seconds**, as explicitly approved before collection. The full 20-second endpoint is unavailable.',
        '', 'Population: '+json.dumps(report['population']),
        'Complete 12-block primary comparison: '+str(report['complete_primary_comparison']),
        '', 'The CSV tables list every accepted, rejected and unattempted row. G-pre is exploratory only.',
        'Primary Holm family: 36 planned hypotheses across all/interference/no-interference scopes; exploratory G-pre family: 16.',
        'Intervals use 10,000 matched-block resamples. Undefined comparisons retain null tests and intervals.', '']
    for r in main:
        summary.append(f"- {r['scope']}, G-{r['right']}, {r['outcome']}: n={r['n_pairs']}, difference={r['mean_difference']}, CI={r['ci95']}, Holm p={r['holm_p']}")
    (out/'summary.md').write_text('\n'.join(summary)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True); p.add_argument('--raw-root',type=Path,required=True)
    a = p.parse_args(); print(json.dumps(analyse(a.study_root,a.raw_root)['population'],indent=2))
