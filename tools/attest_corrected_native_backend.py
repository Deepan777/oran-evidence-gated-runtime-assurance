"""Record native software identities; this is not a physical-radio attestation."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess

root=Path(__file__).resolve().parents[1]
def output(*args):
    return subprocess.check_output(args,text=True).strip()
def digest(path):
    return hashlib.sha256((root/path).read_bytes()).hexdigest()

files=(
    'external/ocudu/include/ocudu/scheduler/research_action_control.h',
    'external/ocudu/lib/scheduler/ue_scheduling/grant_params_selector.cpp',
    'external/ocudu/lib/scheduler/ue_context/ue_link_adaptation_controller.cpp',
    'external/ocudu/lib/scheduler/slicing/inter_slice_scheduler.cpp',
    'external/srsran4g/srsue/src/metrics_csv.cc',
    'results/corrected_guard_backend/gnb',
    'results/corrected_guard_backend/ue-nr-attachment/srsue',
    'external/ocudu/build-codex-gnb/CMakeCache.txt',
    'external/srsran4g/build-codex-zmq/CMakeCache.txt',
)
images=('ocudu-gnb-zmq:consumer-readback-20261023','srsue-zmq:telemetry-nr-attachment-20261027')
report={
    'schema_version':1,'scope':'compiled software identities; no independent physical radio proof',
    'source_commits':{name:output('git','-C',str(root/'external'/name),'rev-parse','HEAD') for name in ('ocudu','srsran4g')},
    'sha256':{name:digest(name) for name in files},
    'image_ids':{name:output('sudo','docker','image','inspect',name,'--format','{{.Id}}') for name in images},
    'compiler':output('c++','--version').splitlines()[0],
    'platform':platform.platform(),
    'scheduler_readback':{'observed':'configuration actually loaded by PRB, MCS and slice scheduler consumers',
        'association':'same gNB process, complete parameters, generation matching post-command publication, fresh timestamps',
        'trust':'shared gNB process and local log collector; not authenticated remote telemetry or physical-effect verification'},
    'receiver_state_fix':{'signal':'native NR RRC CONNECTED state for NR CSV rows; LTE path unchanged',
        'not_inferred_from':['traffic success','RF level','desired connection state'],
        'guard_thresholds_changed':False},
}
destination=root/'results/corrected_guard_integration/20261027_final/native_backend_attestation.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
