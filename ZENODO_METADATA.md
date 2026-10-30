# Zenodo metadata

Title: Evidence-Gated Runtime Assurance for Autonomous O-RAN Control: Code and Data

Version: 1.3.0

Resource type: Software

License: Creative Commons Attribution 4.0 International (CC BY 4.0)

Creators:

- P. Deepanramkumar; Vellore Institute of Technology
- A. Helen Sharmila; Vellore Institute of Technology

Keywords: O-RAN; xApp; runtime assurance; reinforcement learning; wireless interference; srsRAN

Related identifier: is supplemented by https://github.com/Deepan777/oran-evidence-gated-runtime-assurance

## Description

### Overview

Code and data for an evidence-gated runtime-assurance layer for autonomous O-RAN control. The layer separates a proposed action from execution authority using measured spectral/service evidence, persistence, feasibility, acknowledgement and scheduler-consumer readback. Software readback is not independent verification of a physical radio effect.

### Contents

Corrected runtime and adapters; scheduler-consumer parser and patches; unit tests; a recovered frozen controller checkpoint with ordered 27-feature preprocessing, scaler, controller/reward settings, training summary and training/inference source; recovery/censoring and receiver-metadata analyses; actuator and command-transport records; component benchmark records; corrected-loop evidence; record-based timing, rollback and common-path audits; processed measurements, manifests and SHA-256 checksums. Original frozen records retain their identities and status.

### Environment

Recorded radio configuration: Ubuntu 24.04, eight vCPUs, 32 GB RAM, ZeroMQ at 23.04 MS/s, NR standalone band n3, 20 MHz bandwidth, 15 kHz subcarrier spacing and 106 physical resource blocks. O-CU/O-DU commit fa4c9fa445a3; UE commit 6bcbd9e5bf86. Offline code uses Python 3.11. The local validation report records the actual environment used; a complete historical transitive lock is unavailable.

### Not included

Raw radio records for the 360-run comparison are available from the authors on request. Credentials, subscriber-authentication values, private keys, personal communications, interactive-session logs, caches and build outputs are excluded. Operators must provide their own authentication settings and remap historical installation paths. Historical per-run binary attestations and the comparative checkpoint-selection record are unavailable. Prebuilt radio-stack executables and container images are not supplied.

The public actuator and corrected-loop data selections omit cloud-host container introspection, raw packet captures and ephemeral process identifiers. Retained members are byte-identical to the originals, and a separate projection manifest records original digests and omissions.

### Changes from 1.2.0

Corrected execution semantics and readback; recovered frozen controller and numerical preprocessing; corrected recovery and censoring analyses; separate actuator, transport, benchmark and corrected-loop evidence; record-based timing, rollback and common-path audits. No new version-specific DOI is asserted in these prepared metadata.
