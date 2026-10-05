"""Reports retain recoverable progress and the exact measured runtime sources."""
import errno
import hashlib
import json
from pathlib import Path

import pytest

from ragapp import evaluation


def source_report():
    return {'results': [], 'runtime_manifest': {'code': {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(evaluation.__file__).parent.glob('*.py')}}}


def test_transient_windows_access_error_retries_and_preserves_old_report_until_success(tmp_path, monkeypatch):
    target = tmp_path / 'report.json';target.write_text('{"old":true}')
    replace = Path.replace;attempts = [];waits = []
    def occupied(path, destination):
        attempts.append(path)
        if len(attempts) < 3:
            assert json.loads(target.read_text()) == {'old': True}
            raise PermissionError(errno.EACCES, 'Windows sharing lock')
        return replace(path, destination)
    monkeypatch.setattr(Path, 'replace', occupied)
    monkeypatch.setattr(evaluation.time, 'sleep', waits.append)
    evaluation.save_report(target, {'results': [{'id': '114'}]})
    assert json.loads(target.read_text()) == {'results': [{'id': '114'}]}
    assert len(attempts) == 3 and waits == [.05, .1]
    assert not list(tmp_path.glob('.report.json.checkpoint-*.json'))


def test_retry_is_bounded_and_failed_saves_keep_unique_complete_checkpoints(tmp_path, monkeypatch):
    target = tmp_path / 'report.json';target.write_text('{"results":[]}')
    old = target.read_bytes();attempts = [];waits = []
    def occupied(path, destination):
        attempts.append(path)
        raise PermissionError(errno.EACCES, 'Persistent file lock')
    monkeypatch.setattr(Path, 'replace', occupied)
    monkeypatch.setattr(evaluation.time, 'sleep', waits.append)
    for number in (114, 115):
        with pytest.raises(RuntimeError, match='Recover checkpoint'):
            evaluation.save_report(target, {'results': [{'id': str(number)}]})
    checkpoints = list(tmp_path.glob('.report.json.checkpoint-*.json'))
    assert len(attempts) == 10 and len(waits) == 8 and sum(waits) == pytest.approx(1.5)
    assert target.read_bytes() == old and len(checkpoints) == 2
    assert {json.loads(path.read_text())['results'][0]['id'] for path in checkpoints} == {'114', '115'}


def test_nontransient_replace_failure_does_not_retry(tmp_path, monkeypatch):
    waits = []
    def fail(*_):raise OSError(errno.ENOSPC, 'No space left')
    monkeypatch.setattr(Path, 'replace', fail)
    monkeypatch.setattr(evaluation.time, 'sleep', waits.append)
    with pytest.raises(RuntimeError, match='Recover checkpoint'):
        evaluation.save_report(tmp_path/'report.json', {'results': [1]})
    assert not waits
    assert len(list(tmp_path.glob('.report.json.checkpoint-*.json'))) == 1


def test_real_complete_runtime_sources_are_archived_once_and_verified(tmp_path):
    target = tmp_path / 'report.json';report = source_report()
    evaluation.save_report(target, report)
    record = report['runtime_code_snapshot'];snapshot = tmp_path / record['path']
    manifest_path = tmp_path / record['manifest_path'];manifest = json.loads(manifest_path.read_text())
    assert record['status'] == 'verified'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == record['manifest_sha256']
    assert manifest['code_digest'] == record['code_digest']
    assert manifest['files'] == {'ragapp/' + key: value for key, value in report['runtime_manifest']['code'].items()}
    stamps = {}
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected
        stamps[name] = (snapshot / name).stat().st_mtime_ns
    report['results'].append({'id': 'done'});evaluation.save_report(target, report)
    assert {name: (snapshot / name).stat().st_mtime_ns for name in manifest['files']} == stamps
    assert json.loads(target.read_text())['runtime_code_snapshot'] == record


@pytest.mark.parametrize('kind', ['wrong_hash', 'incomplete', 'unknown_file'])
def test_code_manifest_must_match_current_complete_sources_before_saving(tmp_path, kind):
    report = source_report();code = report['runtime_manifest']['code'];name = next(iter(code))
    if kind == 'wrong_hash':code[name] = '0' * 64
    elif kind == 'incomplete':code.pop(name)
    else:code['not-a-real-module.py'] = '0' * 64
    target = tmp_path/'report.json';target.write_text('{"unchanged":true}');old = target.read_bytes()
    with pytest.raises(ValueError, match='complete ragapp source'):
        evaluation.save_report(target, report)
    assert target.read_bytes() == old and not list(tmp_path.glob('.report.json.checkpoint-*'))


@pytest.mark.parametrize('kind', ['source', 'manifest'])
def test_existing_snapshot_corruption_cannot_be_silently_repaired(tmp_path, kind):
    target = tmp_path/'report.json';report = source_report();evaluation.save_report(target, report)
    snapshot = tmp_path/report['runtime_code_snapshot']['path']
    damaged = snapshot/'manifest.json' if kind == 'manifest' else next((snapshot/'ragapp').glob('*.py'))
    damaged.write_bytes(b'corrupted');old = target.read_bytes()
    with pytest.raises(ValueError, match='Archived runtime'):
        evaluation.save_report(target, report)
    assert damaged.read_bytes() == b'corrupted' and target.read_bytes() == old


def test_fake_manifest_without_code_and_frozen_external_map_are_not_mislabeled_verified(tmp_path):
    target = tmp_path/'report.json';fake = {'runtime_manifest': {'runtime_fingerprint': 'fake'}}
    evaluation.save_report(target, fake)
    assert 'runtime_code_snapshot' not in fake
    external = {'runtime_manifest': {'code': {'ragapp/config.py': 'baseline-hash'}}}
    evaluation.save_report(target, external)
    assert external['runtime_code_snapshot']['status'] == 'skipped_external_code_map'
    assert not (tmp_path/'runtime-snapshots').exists()
