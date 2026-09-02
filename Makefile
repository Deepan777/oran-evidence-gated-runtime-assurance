.PHONY: bootstrap test start-testbed collect-data live-calibration train-detector train-live-detector prepare-live-controller live-exploration train-live-controller live-ppo-exploration train-live-ppo-controller train-controller validate freeze-confirmatory confirmatory ablations statistics traceability-audit evidence-gate reproduce-paper-results

PYTHON ?= envs/oran-action-xai/bin/python

bootstrap:
	bash software/bootstrap_ubuntu.sh

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests -ra --strict-markers

start-testbed:
	$(PYTHON) analysis/preflight.py pilot

collect-data: start-testbed
	$(PYTHON) experiments/scripts/collect_data.py

live-calibration: start-testbed
	$(PYTHON) experiments/generate_live_calibration_manifest.py
	$(PYTHON) experiments/collection/run_manifest.py --manifest datasets/metadata/manifests/live_calibration_runs.csv --output results/raw/LIVE_CALIBRATION_MANIFEST.json --gnb-image $${GNB_IMAGE:?set GNB_IMAGE to the pinned OCUDU image}

train-detector:
	$(PYTHON) experiments/training/train_detector.py

train-live-detector:
	$(PYTHON) experiments/training/train_live_detector.py

prepare-live-controller: train-live-detector
	$(PYTHON) experiments/training/prepare_live_controller.py

live-exploration: prepare-live-controller
	$(PYTHON) experiments/generate_live_exploration_manifest.py
	$(PYTHON) experiments/collection/run_manifest.py --manifest datasets/metadata/manifests/live_exploration_runs.csv --output results/raw/LIVE_EXPLORATION_MANIFEST.json --gnb-image $${GNB_IMAGE:?set GNB_IMAGE to the pinned OCUDU image}

train-live-controller:
	$(PYTHON) experiments/training/train_live_controller.py

live-ppo-exploration: prepare-live-controller
	$(PYTHON) experiments/generate_live_ppo_exploration_manifest.py
	$(PYTHON) experiments/collection/run_manifest.py --manifest datasets/metadata/manifests/live_ppo_exploration_runs.csv --output results/raw/LIVE_PPO_EXPLORATION_MANIFEST.json --gnb-image $${GNB_IMAGE:?set GNB_IMAGE to the pinned OCUDU image}

train-live-ppo-controller:
	$(PYTHON) experiments/training/train_live_ppo_controller.py

train-controller:
	$(PYTHON) experiments/training/train_controller.py

validate:
	$(PYTHON) experiments/training/validate.py

freeze-confirmatory:
	$(PYTHON) experiments/generate_confirmatory_freeze_manifest.py --gnb-image $${GNB_IMAGE:?set GNB_IMAGE to the pinned OCUDU image}

confirmatory:
	$(PYTHON) analysis/preflight.py confirmatory
	$(PYTHON) experiments/confirmatory/run.py

ablations:
	$(PYTHON) experiments/ablations/run.py

statistics:
	$(PYTHON) analysis/statistics.py

traceability-audit:
	$(PYTHON) analysis/traceability_audit.py

evidence-gate:
	$(PYTHON) analysis/preflight.py evidence
	$(PYTHON) analysis/evidence_gate.py

reproduce-paper-results:
	$(PYTHON) analysis/preflight.py results
	$(PYTHON) analysis/aggregate.py
	$(PYTHON) analysis/statistics.py
	$(PYTHON) analysis/paper_tables.py
	$(PYTHON) analysis/paper_figures.py
