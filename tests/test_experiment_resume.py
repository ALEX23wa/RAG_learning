"""Synthetic checkpoint recovery tests; no private cases or model execution."""
import hashlib
import json

import pytest

from ragapp import experiments


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    cases_path = tmp_path / 'cases.json'
    report_path = tmp_path / 'report.json'
    cases = [{'id': f'q{i}', 'question': f'Synthetic question {i}',
              'schema_version': 'quality-v2', 'split': 'dev', 'answerable': True,
              'evidence': [{'path': 'synthetic.txt', 'locator': {'line': 1},
                            'span_text': 'Synthetic evidence.'}]} for i in (1, 2)]
    cases_path.write_text(json.dumps(cases), encoding='utf-8')
    dataset_manifest = {'datasets': {'cases.json': {
        'sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest()}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(dataset_manifest), encoding='utf-8')
    calls = []

    def search(question, history, configuration):
        calls.append(question)
        item = {'id': 'chunk', 'path': 'synthetic.txt', 'locator': {'line': 1},
                'text': 'Synthetic evidence.'}
        return {'evidence': [item], 'candidates': [item], 'context': [item], 'trace': {}}

    class MemorySampler:
        peak = 123

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(experiments, 'MemorySampler', MemorySampler)
    options = {'search': search, 'manifest': {'runtime_fingerprint': 'synthetic-runtime'},
               'configuration': {'retrieval_mode': 'bm25', 'query_planning': False}}
    return cases_path, report_path, calls, options


def completed_with_stale_error(synthetic_run, monkeypatch):
    cases_path, report_path, calls, options = synthetic_run
    original_save = experiments.save_report
    failed = False

    def save_once_fails(target, report):
        nonlocal failed
        if not failed and len(report['results']) == 2 and 'blocked' not in report:
            failed = True
            raise OSError('synthetic last-case checkpoint failure')
        original_save(target, report)

    monkeypatch.setattr(experiments, 'save_report', save_once_fails)
    first = experiments.run(cases_path, report_path, **options)
    assert failed
    assert len(first['results']) == 2
    assert first['blocked']['id'] == 'q2'
    assert json.loads(report_path.read_text(encoding='utf-8')) == first
    return first


def test_resume_archives_persisted_last_case_error_without_repeating_query(synthetic_run, monkeypatch):
    cases_path, report_path, calls, options = synthetic_run
    first = completed_with_stale_error(synthetic_run, monkeypatch)
    recovered = experiments.run(cases_path, report_path, **options)
    assert calls == ['Synthetic question 1', 'Synthetic question 2']
    assert recovered['results'] == first['results']
    assert 'blocked' not in recovered
    assert recovered['status'] == 'provisional_metrics'
    assert recovered['metrics']['complete'] is True
    assert len(recovered['recovered_errors']) == 1
    archived = recovered['recovered_errors'][0]
    assert {key: archived[key] for key in first['blocked']} == first['blocked']
    assert archived['recovery_reason'] == 'completed_result_already_persisted'
    assert archived['recovered_at'] >= first['blocked']['time']
    again = experiments.run(cases_path, report_path, **options)
    assert again['recovered_errors'] == recovered['recovered_errors']
    assert again['results'] == recovered['results']
    assert len(calls) == 2


def test_resume_keeps_missing_case_block_until_query_succeeds(synthetic_run):
    cases_path, report_path, calls, options = synthetic_run
    successful_search = options['search']

    def fail_second(question, history, configuration):
        if question == 'Synthetic question 2':
            raise OSError('synthetic search failure')
        return successful_search(question, history, configuration)

    first = experiments.run(cases_path, report_path, **{**options, 'search': fail_second})
    assert [row['id'] for row in first['results']] == ['q1']
    assert first['blocked']['id'] == 'q2'
    paused = experiments.run(cases_path, report_path, max_queries=0, **options)
    assert paused['blocked'] == first['blocked']
    assert 'recovered_errors' not in paused
    assert calls == ['Synthetic question 1']
    completed = experiments.run(cases_path, report_path, **options)
    assert 'blocked' not in completed
    assert 'recovered_errors' not in completed
    assert [row['id'] for row in completed['results']] == ['q1', 'q2']
    assert calls == ['Synthetic question 1', 'Synthetic question 2']


@pytest.mark.parametrize('change, message', [
    ('dataset', 'Dataset or ablation changed'),
    ('configuration', 'Dataset or ablation changed'),
    ('dataset_manifest', 'Frozen dataset manifest changed'),
    ('runtime', 'Runtime manifest changed'),
    ('duplicate', 'duplicate or foreign case IDs'),
    ('foreign', 'duplicate or foreign case IDs'),
])
def test_stale_error_recovery_requires_unchanged_identity_and_valid_ids(
        synthetic_run, monkeypatch, change, message):
    cases_path, report_path, calls, options = synthetic_run
    completed_with_stale_error(synthetic_run, monkeypatch)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if change == 'dataset':
        report['dataset_sha256'] = 'different-dataset'
    elif change == 'configuration':
        options = {**options, 'configuration': {'retrieval_mode': 'dense'}}
    elif change == 'dataset_manifest':
        report['dataset_manifest_sha256'] = 'different-manifest'
    elif change == 'runtime':
        options = {**options, 'manifest': {'runtime_fingerprint': 'different-runtime'}}
    elif change == 'duplicate':
        report['results'].append(report['results'][-1])
    elif change == 'foreign':
        report['results'][0]['id'] = 'foreign-case'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    before = report_path.read_bytes()
    with pytest.raises(ValueError, match=message):
        experiments.run(cases_path, report_path, **options)
    assert report_path.read_bytes() == before
    assert len(calls) == 2
