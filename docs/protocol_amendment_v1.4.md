# Prospective Study V protocol amendment for release v1.4.0

Date: 30 October 2026 (Asia/Calcutta).

## Status and collection hold

This amendment is to be committed before any new radio observations are
collected. No previously rejected observation is promoted or reclassified.
The Git commit containing this document is the prospective amendment identity;
its hash must be recorded in the execution manifest before collection starts.

The user selected option 1 before collection: preserve the original 45-window
design and report the observed 10-second recovery endpoint. The full
20-second endpoint is unavailable for runs without an observed recovery
within the 10-second follow-up. No phase lengths are extended and no
unobserved follow-up is manufactured.

## Authorized measurement change and one repeat

- The rejected run is repeated once with the same seed.
- The exact rejected row is
  `cg-20261027m1-cg-b02-r2-policy-s9504`: direct policy, seed 9504,
  no interference, high offered load, and 6-dB attenuation. Its repeat has a
  new attempt identifier linked to this original row. The original rejected
  attempt and its original acceptance status remain permanently preserved.
- A window whose native SNR is NaN is treated as a missing window. A run is
  accepted if at least 44 of 45 windows are complete and all other
  prespecified acceptance checks pass.
- A missing window remains explicitly identified by its original index and
  native records. NaN is never replaced by zero, a finite cap, positive
  infinity, a generated observation, or an estimated result. Missingness is
  reported per run and phase. The previously approved native-positive-infinity
  qualifier is a distinct rule and is not extended to NaN.
- The tolerance is not a waiver for incomplete radio duration, absent traffic
  provenance, missing or duplicated decision records, missing source identity,
  or other failed prespecified acceptance checks. Controller rejection,
  lockout, unsuccessful execution, and absence of a requested fault response
  are reported outcomes, not grounds to omit an otherwise measurable run.
- There are no additional automatic retries, replacement seeds, or
  outcome-dependent stopping rules. Every attempted execution is logged and
  retained, including infrastructure failures and unsuccessful replications.

Nothing else changes: model checkpoint, schema, thresholds, conditions,
strata, seeds, outcomes, analysis plan. The additional replication study,
analysis outputs, and separately labelled exploratory G-pre arm explicitly
requested by the user are additions, not replacements for the frozen primary
G arm or a retrospective rewrite of the original analysis plan.

## Frozen design recovered from existing records

Authoritative prior records are `evaluation_freeze.json` and
`evaluation_progress.json` under
`results/corrected_guard_integration/20261027_measurement_amendment1/`,
preserved in
`revision_2026-10-23/corrected_guard_measurement_amendment1_data_20261027.tar.gz`.
The archive SHA-256 is
`1edd4e7a02b1789bee1bc3ccb756ced413eaad5b52e634b80315ff1838670e47`.

The design consists of these six specified strata, not a full Cartesian
product of all interference, load, and attenuation levels:

| Stratum | Interference | Requested JSR (dB) | Load | Attenuation (dB) | Seeds |
|---|---|---:|---|---:|---|
| 1 | none | 0 | low | 0 | 9501, 9502 |
| 2 | none | 0 | high | 6 | 9503, 9504 |
| 3 | partial-band | 0 | low | 0 | 9505, 9506 |
| 4 | partial-band | 5 | high | 0 | 9507, 9508 |
| 5 | barrage | 5 | low | 6 | 9509, 9510 |
| 6 | intermittent | 5 | high | 0 | 9511, 9512 |

Each scenario-seed block has independently executed B0 (`no_action`),
P (`policy`), G (`guarded_policy`), and A2 (`fixed_a2`). The original
condition order was randomized using Python `random.Random(block_seed)`.
Already executed rows are not reordered. Remaining primary rows retain that
recorded order, including the single repeated P row at seed 9504 followed by
the still-unexecuted A2 row in that block.

The preserved campaign records 15 attempts: 14 accepted and the rejected
seed-9504 P attempt. Therefore 34 new primary attempts are needed to finish
the intended 48 unique rows if all pass: one authorized repeat and 33
previously unexecuted rows. An incomplete or rejected new attempt remains in
the denominator of attempted runs; a complete matched cohort is not asserted
unless all required arms are accepted.

Frozen artifact SHA-256 identities:

- Controller checkpoint:
  `713b48f1a16fdae31c37cf5b6ed2a8757fa80facbd4e99de3a5e64d74621e498`.
- Ordered state schema:
  `127719f66f1b3bc3622473089c8b94206dfe65fdd9cd212a7b51a2d773946fc5`.
- Detector:
  `571e438c105729bd3bb912bb0e90ddc450614ad6d6401ffc4924c3186c59fe59`.
- gNB image identity:
  `sha256:1d7125738a48cba08ef41c8d833991ec400c807435116fb782a27c05b94eee87`.
- NR-connection-state UE image identity:
  `sha256:dae98d99363a5e3ce1888a41a804027ef18591fc719fa10dac8bfeb432a6e084`.

The numerical preprocessing and remaining implementation identities must be
verified against the source closure in the prior freeze before execution.
No retraining, controller selection, threshold tuning, or radio image
replacement is authorized by this amendment.

## Acquisition and additional studies

Use the existing real-execution OCUDU/srsUE testbed and the NR-based
connection-state exporter. This is a virtualized software-radio testbed with
ZeroMQ sample transport; it is not an over-the-air hardware experiment.
No simulated outcomes, synthetic UE response frames, retrospective result
edits, or imputed experimental results are permitted.

Every attempt retains execution, traffic and full container logs; native
receiver and gNB/RLC telemetry; protected-application per-window records;
gate decisions, commands, acknowledgements and scheduler-consumer readback;
run and source identities; acceptance reasons; and original failed records.
All conditions use identical interference cessation and recovery handling.

The requested additional replication population is separate: five real
lost-acknowledgement injections with distinct prespecified seeds and fault
windows, two live command timeouts, two live applied-parameter mismatches,
and three additional no-interference interception runs. Seeds, schedules,
fault-delivery mechanisms and required evidence must be recorded before
execution. A fault that was not actually delivered is labelled inconclusive,
not successful. Artificially editing acknowledgements or readback evidence
is prohibited.

**Exploratory G-pre only:** in the eight interference blocks, add a fifth
independent condition that changes only the post-action harm reference to
the mean of the three windows immediately before the action. The primary G
condition remains unchanged. A separate recorded randomization seed fixes
G-pre's insertion into each not-yet-executed interference block before data
collection. Exploratory G-pre evidence and outputs must be labelled
exploratory throughout, not pooled with G.

The requested additional analysis must retain an acceptance table for every
attempt, independent matched-block denominators, G-P/G-B0/G-A2 contrasts,
paired Wilcoxon and exact McNemar tests as appropriate, Holm family
adjustment, 10,000 matched-block bootstrap intervals, and separate
interference/no-interference summaries. Unobserved endpoints are not
converted to zero, recovery, or non-recovery. The original frozen analysis
is retained alongside, not overwritten by, these additional analyses.

## Recovery-horizon conflict requiring a prospective decision

The recovered Study V freeze specifies 15 pre-interference windows, 20
exposure windows, and **10 recovery windows**: 45 one-second windows total.
Study II's requested recovery endpoint observes **20 seconds after
interference cessation**, with recovery established at the end of three
consecutive qualifying windows (throughput at least 90% of the run's
pre-interference mean and finite RLC SDU delay at most 20 ms).

The existing 10-second recovery phase cannot establish a complete binary
20-second recovery endpoint. A recovery event observed within 10 seconds is
known to occur by 20 seconds, but a run without such an event is censored at
10 seconds, not demonstrated to fail recovery by 20 seconds. Existing
accepted runs cannot acquire the missing 10 seconds retrospectively.

Prospective resolution: the user selected option 1 in this chat on
30 October 2026. Preserve all 45 windows and report recovery by 10 seconds,
using the same throughput/delay thresholds, three-window persistence and
event timing as Study II. Non-events are right-censored at 10 seconds; the
full binary 20-second endpoint remains explicitly unavailable. Apply exact
McNemar tests to the observed binary 10-second endpoint, not to invented
20-second outcomes. No longer-follow-up cohort or repeat of previously
accepted rows is authorized by this choice.

## Execution schedules and additional analysis frozen before collection

- Lost-acknowledgement seeds 9601--9605 have respective fault windows
  17, 20, 23, 26 and 29 (zero-based). The live proxy forwards the command,
  records the actual upstream acknowledgement, and withholds its delivery
  until the real client socket times out.
- Command-timeout seeds 9611 and 9612 have respective fault windows 18 and
  25. A live proxy completes the actual WebSocket handshake but withholds
  the command; the real client socket times out. Lack of gNB application is
  distinguished from a lost acknowledgement after application.
- Applied-parameter-mismatch seeds 9621 and 9622 target the first nontrivial
  command at or after windows 17 and 24, respectively. One actual on-wire
  parameter differs by one valid step from the controller's request. The
  upstream acknowledgement and consumed software state remain unedited.
- These nine fault replications use the original partial-band, +5-dB JSR,
  low-load, zero-attenuation configuration. Additional no-interference
  interception runs use seeds 9631--9633, low load and zero attenuation.
- Failure to observe a scheduled command/fault is an inconclusive
  replication outcome, never a reason for an automatic replacement.
- Each exploratory G-pre insertion uses randomization seed
  `20261030 + block_seed`, preserving the relative order of the four
  primary conditions. Its mean uses the three completed consecutive
  windows before the actual command, including the just-observed window;
  the reference is held during the same nontrivial-action episode.
- The additional primary analysis uses one conservative Holm family of
  36 planned hypotheses (three contrasts, four outcomes, three scopes:
  all, interference, no interference). The separately exploratory G-pre
  analysis uses 16 hypotheses (four contrasts, four outcomes, interference
  only). Unestimable hypotheses retain null p-values and confidence
  intervals; the full planned family size is retained for adjustment.
- Native-NaN missing windows are excluded from available-window metric
  means, their indexes and phase denominators are reported, and they
  interrupt the required three-window recovery sequence. Other documented
  outage/censoring classes retain the existing Study II definitions.
  Prior accepted rows retain their original acceptance profile and are
  identified by acquisition epoch; the original rejection stays rejected.
- Bootstrap resampling uses 10,000 matched-block resamples, 95% percentile
  intervals, and recorded deterministic analysis seeds. Radio windows are
  not independent replications. All G-pre outputs are exploratory.
- Acquisition-only log correction: the existing registration helper moves
  a pre-created evidence directory. The new pilot retains registration
  stdout at a stable run-specific log path and copies it byte-for-byte
  into the final evidence directory. This prevents loss of the open log
  inode; it changes no radio, traffic, controller or phase settings. The
  original helper and all prior evidence remain unchanged. Missing
  registration stdout in prior archives is disclosed, not invented.
