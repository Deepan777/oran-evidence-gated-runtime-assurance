# Frozen controller package for the 360-run evaluation

This directory preserves the checkpoint, ordered preprocessing schema, scaler, training summary, reward/controller configuration, inference/training source, frozen run-loop source, and P-method manifest associated with the frozen 360-run matched campaign.

The freeze manifest identifies the checkpoint and schema by SHA-256 and the recovered bytes match those digests. The run-loop source included here matches the frozen digest. P rows used `LIVE_XAPP_SELECTION_MODE=policy`; consequently, this cohort evaluates learned-policy efficacy and not the later safeguarded-policy integration.

The fine-tuning summary records 228 training transitions, 47 validation transitions (all validation executions were A0), 300 gradient steps, and training-only reward normalizers. It labels the model a candidate requiring live validation and does not preserve a comparative checkpoint-selection ledger. Do not infer a missing selection criterion or use this package to claim correspondence to every earlier development run.
