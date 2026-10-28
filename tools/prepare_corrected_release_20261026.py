"""Streaming code/data packaging and integrity utilities."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def path_safe(name):
    p=PurePosixPath(name.replace('\\','/'))
    denied={'.git','.gcloud','.aws','__pycache__','build','prompts','credentials'}
    return not p.is_absolute() and '..' not in p.parts and not any(x in denied for x in p.parts) and not p.name.startswith('.env') and p.suffix.lower() not in {'.tex','.bbl','.bib','.docx','.exe','.key'}

def inventories(stage):
    stage=Path(stage)
    records=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(stage.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS.txt','RELEASE_MANIFEST.json') and not any(x in ('__pycache__','.pytest_cache','.git','reproduction_outputs','.venv') for x in p.relative_to(stage).parts)]
    (stage/'RELEASE_MANIFEST.json').write_text(json.dumps(dict(schema_version=1,hash_algorithm='SHA-256',excludes_self=True,files=records),indent=2)+'\n')
    rows=records+[dict(path='RELEASE_MANIFEST.json',sha256=digest(stage/'RELEASE_MANIFEST.json'))]
    (stage/'SHA256SUMS.txt').write_text(''.join(f"{i['sha256']}  {i['path']}\n" for i in sorted(rows,key=lambda x:x['path'])))

def zip_candidate(stage,output):
    stage,output=Path(stage),Path(output)
    if output.exists():raise FileExistsError(output)
    records=json.loads((stage/'RELEASE_MANIFEST.json').read_text())['files']
    names=[r['path'] for r in records]+['RELEASE_MANIFEST.json','SHA256SUMS.txt']
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):
            z.write(stage/name,name,compress_type=zipfile.ZIP_STORED if name.endswith('.gz') else zipfile.ZIP_DEFLATED)
    output.with_suffix('.zip.sha256').write_text(f'{digest(output)}  {output.name}\n')
