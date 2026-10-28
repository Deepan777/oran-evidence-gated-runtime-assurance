# Evidence-Gated Runtime Assurance for Autonomous O-RAN Control: Code and Data

Version 1.3.0

This software separates a controller's proposed radio action from permission to execute it. The runtime checks measured spectral and service evidence, temporal persistence and action feasibility, records command acknowledgement and scheduler-consumer readback, and requests baseline restoration when execution state is uncertain or a configured boundary is crossed. Readback verifies consumed software configuration within the local trust domain; it is not an independent measurement of a physical radio effect.

## Directory layout

- `src/`: controller, guard, execution state, adapters, measurement parsers and explanation components.
- `experiments/`, `tools/`: training, collection, local control and implementation utilities.
- `frozen_controller_360_run/`: checkpoint, ordered 27-feature schema, numerical scaler, reward/controller configuration, training summary and original training/inference source.
- `analysis/`, `analysis_outputs/`: statistical and record-based analysis code and preserved JSON/CSV outputs.
- `results/`, `datasets/`, `data/`: processed measurements, fixed run manifests, actuator and corrected-loop evidence.
- `backend_sources/`, `patches/`, `config/`, `configs/`: scheduler-consumer source, patches and operator-configurable stack settings.
- `provenance/`, `reproducibility/`, `environment/`: original identities, freeze records and environment information.
- Dated output directories retain their original names where required by existing paths and hash records.

## Environment

The recorded radio configuration uses Ubuntu 24.04, eight vCPUs and 32 GB RAM, with ZeroMQ transport at 23.04 MS/s. It is an NR standalone configuration in band n3, with 20 MHz bandwidth, 15 kHz subcarrier spacing and 106 physical resource blocks. The O-CU/O-DU commit is `fa4c9fa445a3`; the UE commit is `6bcbd9e5bf86`. Offline code uses Python 3.11. These are recorded configuration details, not a claim that all historical runtime dependencies or provisioned machine identities have been independently recovered.

`requirements.in` retains the original requested dependency versions. `requirements-validation.txt` records the versions used for the local offline verification. The inherited environment lock is incomplete and is not a transitive hash lock. Native stack/container reconstruction requires Linux and the identified upstream source; prebuilt stack executables are not distributed.

## Install and validate

From a clean extraction, create a Python 3.11 environment and install the validation dependencies:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-validation.txt
python scripts/validate_release.py --check all
```

On Windows, activate with `.venv\Scripts\activate` instead. An already provisioned Python 3.11 environment can be used directly. The validation report records the actual platform and package versions; it does not claim a clean Linux deployment test.

Run all included unit tests directly with:

```bash
python -m pytest -q tests
```

The analysis checks below operate only on existing records. New output goes under `reproduction_outputs/`; distributed evidence and results are never overwritten.

```bash
python scripts/validate_release.py --check expanded
python scripts/validate_release.py --check actuator
python scripts/validate_release.py --check receiver
python scripts/validate_release.py --check transport
python scripts/validate_release.py --check benchmark
python scripts/validate_release.py --check records
python scripts/validate_release.py --check corrected-loop
```

The `expanded`, `actuator` and `receiver` checks invoke the corresponding recovery/censoring, actuator-recovery and receiver-metadata analysis modules. The `records` check invokes the timing, rollback and common-path record audit. The transport and benchmark checks validate the preserved ledgers and repetition cardinalities without sending commands or collecting new timing observations. The corrected-loop check validates existing measurement/decision identities and cohort counts; an incomplete comparison remains incomplete. The runner lists the exact underlying commands in its JSON report. `scripts/reproduce_release.py` also provides the earlier combined inference, component and recovery checks.

Optional analysis entry points expose `--help`. Legacy aggregate/generator utilities are retained as dependencies but are not invoked as new experiments. None of the offline validation commands starts a radio, a container service, a cloud machine or a new benchmark.

## Evidence boundaries and exclusions

The 360-run comparison contains 43,200 measured windows and 90 matched blocks. Its raw radio records are not included and are available from the authors on request. The separate actuator dataset has 84 accepted runs, 2,520 windows and 24 matched blocks; the command-transport ledger contains eight conditions with 40 attempts each. The corrected-loop records and rejected attempts retain their original status. These populations are distinct, and windows are not independent runs.

Credentials, subscriber-authentication values, private keys, personal communications, interactive-session logs, caches and build outputs are excluded. Public configuration copies require operator-supplied authentication settings. Some historical installation paths remain in preserved records and must be remapped by the operator. Historical per-run binary attestations and the comparative checkpoint-selection record are unavailable. A recovered frozen controller does not prove the identity of every earlier executed binary. The package does not include a self-contained radio image or establish physical-effect verification.

Immutable data identifiers and path-dependent historical labels are retained verbatim so that checksums and analysis paths remain auditable. They are not descriptive release titles. Per-file integrity is recorded in `RELEASE_MANIFEST.json` and `SHA256SUMS.txt`.

The public actuator and corrected-loop archives omit container-introspection records containing the original cloud-machine hostname, raw packet captures and ephemeral process identifiers. Packet captures are not needed for the offline analyses and are excluded to avoid exposing unparsed authentication traffic. Every retained member is byte-identical to its original. `provenance/public_data_projection.json` identifies the original archive digests, retained member hashes and omissions; the original full archives are not redistributed. Original embedded full-archive manifests describe their historical populations, not the smaller public selections.

## License

Project-authored code and data are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/. Third-party code retains its original copyright and license notices; the project license does not replace them.

## Changes from 1.2.0

Version 1.3.0 adds corrected execution-state and consumer-readback semantics, the recovered frozen controller and numerical preprocessing, explicit recovery events and right censoring, receiver-metadata sensitivity, separate actuator and transport evidence, preserved component benchmark records, corrected-loop evidence and record-based timing/rollback/common-path audits. It retains original observations and distinguishes recovered artifacts from unavailable historical attestations. See `CHANGELOG.md`.
