"""Archive integrity and exclusion checks, using temporary non-experimental files."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools.prepare_corrected_release_20261026 import inventories, path_safe, zip_candidate


def test_zip_streaming_preserves_destination_and_every_byte(tmp_path):
    stage=tmp_path/'stage'
    stage.mkdir()
    (stage/'source.py').write_text('VALUE = 1\n')
    nested=stage/'data'; nested.mkdir()
    (nested/'evidence.gz').write_bytes(b'archive-fixture-only'*1000)
    inventories(stage)
    archive=tmp_path/'release.zip'
    zip_candidate(stage,archive)
    manifest=json.loads((stage/'RELEASE_MANIFEST.json').read_text())
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert len(handle.namelist())==len(manifest['files'])+2
        for entry in manifest['files']:
            assert hashlib.sha256(handle.read(entry['path'])).hexdigest()==entry['sha256']
    assert archive.with_suffix('.zip.sha256').read_text().split()[0]==hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        zip_candidate(stage,archive)


@pytest.mark.parametrize('name',[
    '../outside.py','/outside.py','prompts/request.md','build/output.exe',
    'credentials/secret.key','.env','__pycache__/module.pyc',
])
def test_private_or_writing_paths_are_not_released(name):
    assert not path_safe(name)


def test_essential_source_path_is_eligible():
    assert path_safe('src/controller/safety_guard.py')
