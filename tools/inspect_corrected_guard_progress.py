"""Read-only compact status of the bounded corrected-guard radio study."""
from pathlib import Path
from collections import Counter
import json
import argparse

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--study-name',default='20261027_final')
parser.add_argument('--run-prefix',default='cg-20261027')
args=parser.parse_args()
study = root / "results/corrected_guard_integration" / args.study_name
for phase in ("smoke", "fault", "evaluation"):
    progress = study / f"{phase}_progress.json"
    if progress.is_file():
        data = json.loads(progress.read_text())
        print(json.dumps({"phase":phase,"status":data.get("status","RUNNING"),
                          "completed":len(data["completed"]),"expected":data["expected_runs"],
                          "incomplete_measurements":[row["run_id"] for row in data["completed"] if not row["measurements_complete"]]}))
for path in sorted((root / "results/raw").glob(f"iq-broker-live-{args.run_prefix}-*/live-xapp-decisions.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    executions = [(row.get("decision") or {}).get("execution") or {} for row in rows]
    print(json.dumps({"run":path.parent.name,"decisions":len(rows),
        "guard_reasons":dict(Counter((row.get("safety_guard") or {}).get("reason") for row in rows)),
        "verified_nontrivial":sum(e.get("executed_action") not in (None,"NO_OP") and e.get("configuration_state_status")=="VERIFIED" for e in executions),
        "unknown":sum(e.get("configuration_state_status")=="UNKNOWN" for e in executions),
        "receiver_attached":dict(Counter(row.get("receiver_attached") for row in rows))}))
