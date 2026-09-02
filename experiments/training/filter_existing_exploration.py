from __future__ import annotations
import csv, json, pathlib

def main() -> None:
    root = pathlib.Path('<project-root>')
    manifest = root / 'datasets/metadata/manifests/live_exploration_runs_recalibrated.csv'
    out = root / 'datasets/metadata/manifests/live_exploration_runs_recalibrated_complete.csv'
    with manifest.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    keep = []
    for row in rows:
        p = root / 'results/raw' / f"iq-broker-live-{row['run_id']}"
        try:
            data = json.loads((p / 'records.summary.json').read_text())
            complete = int(data.get('complete_rows', data.get('rows', 0)))
            rows_count = int(data.get('rows', complete))
        except Exception:
            continue
        if complete > 0 and complete == rows_count:
            keep.append(row)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(keep)
    print(json.dumps({'source_rows': len(rows), 'kept_rows': len(keep), 'actions': sorted({r.get('action_id') for r in keep}), 'output': str(out)}))

if __name__ == '__main__':
    main()
