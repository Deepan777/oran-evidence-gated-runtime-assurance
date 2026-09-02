from __future__ import annotations
import json, pathlib
from experiments.collection.build_live_run_records import build_records

def main() -> None:
    root = pathlib.Path('<project-root>')
    total = 0; usable = []
    for p in sorted((root/'results/raw').glob('iq-broker-live-*')):
        meta = p/'run_metadata.json'; dec = p/'live-xapp-decisions.jsonl'
        if not meta.is_file() or not dec.is_file(): continue
        try:
            md=json.loads(meta.read_text())
            if md.get('evidence_class') != 'development_live_exploration': continue
            rec=build_records(p,md)
            if not bool(rec.complete_controller_state.all()): continue
            ds=sum(1 for line in dec.read_text().splitlines() if line.strip())
            n=max(0,ds-1)
            usable.append((p.name,n,md.get('run_id'))); total += n
        except Exception as e:
            continue
    print(json.dumps({'dirs':len(usable),'transitions_approx':total,'runs':usable},indent=2))

if __name__ == '__main__': main()
