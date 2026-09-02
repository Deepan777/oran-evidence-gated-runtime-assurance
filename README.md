# O-RAN evidence-gated runtime assurance code

This repository contains only the source code, configuration, experiment manifests, analysis programs, and tests needed to inspect and reproduce the O-RAN runtime-assurance implementation. Manuscripts, prompts, cover letters, figures, private cloud records, and writing drafts are intentionally excluded.

## Layout

- `src/`: detector, controller, safety guard, telemetry, jammer, xApp, and explanation modules.
- `analysis/`: evidence-gate, validation, statistical, traceability, and figure/table analysis programs.
- `experiments/`: collection and training entry points.
- `datasets/metadata/`: versioned scenario and protocol manifests (metadata only).
- `config/`, `configs/`, `ric/`, `testbed/`, `software/`: runtime configuration and deployment helpers.
- `tests/`: unit and integrity tests.

## Reproducibility boundary

The repository contains no raw cloud or software-defined-radio captures. It is intended to run against locally available evidence files or a separately supplied evidence archive. Environment-specific absolute paths have been replaced with `<project-root>` placeholders; adapt them to the checkout location before running a campaign.

## Basic checks

```bash
python -m pip install -r requirements.in
python -m pytest tests/unit
```

The recorded research results distinguish policy proposals, runtime authorization, actuator acknowledgement, and measured service outcomes. No learned checkpoint is included in this code-only release.
