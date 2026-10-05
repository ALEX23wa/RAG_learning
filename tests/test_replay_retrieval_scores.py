"""Saved-response replay uses synthetic gold; no private datasets or inference."""
import copy
import json

import pytest

from ragapp.evaluation import load_dataset
from ragapp.experiments import score_retrieval, summarize
from scripts.freeze_evidence_revision import encode, file_sha
from scripts.replay_retrieval_scores import replay_reports, score_saved_report
from test_evidence_revision import make, revision_inputs


def saved_report(inputs, *, trace=True, visible='alternative', candidates=True):
    make(inputs)
    old, old_hash, old_manifest = load_dataset(inputs['parent'] / 'dev.json')
    new, _, _ = load_dataset(inputs['output'] / 'dev.json')
    alt = new[0]['evidence_groups'][0][1]
    item = {'id': 'actual-returned-chunk', 'path': alt['path'], 'locator': alt['locator'],
            'text': alt['span_text'] if visible == 'alternative' else 'Unrelated visible passage',
            'parent': alt['span_text'], 'version': 'source-version'}
    response = {'evidence': [item], 'context': [copy.deepcopy(item)]}
    if trace:
        response['trace'] = {'lanes': []}
        if candidates:
            response['candidates'] = [{**item, 'text': alt['span_text']}]
    rows = []
    for case in old:
        answer = copy.deepcopy(response)
        rows.append({'id': case['id'], 'cluster_id': case['cluster_id'], 'category': case['category'],
            'sources': [e['path'] for e in case['evidence']],
            'group_sources': [[e['path']] for e in case['evidence']], 'elapsed': 1.25,
            'peak_rss_bytes': 12345, 'cost_cny_upper_estimate': 0,
            'gold_review_status': 'agent_source_reviewed', 'score': score_retrieval(case, answer),
            'response': answer})
    data = {'dataset_sha256': old_hash, 'dataset_manifest_sha256': old_manifest['_manifest_sha256'],
        'dataset_split': 'dev', 'dataset_case_count': len(old), 'status': 'provisional_metrics',
        'runtime_fingerprint': 'frozen-synthetic-runtime', 'runtime_manifest': {'code': {}},
        'configuration': {'retrieval_mode': 'bm25', 'query_planning': False},
        'results': rows, 'metrics': summarize(rows, len(old))}
    path = inputs['output'].parent / 'original-report.json'; path.write_bytes(encode(data))
    return path, data


def test_replay_preserves_source_timing_and_reports_scores_side_by_side(revision_inputs):
    args = revision_inputs; source, original = saved_report(args); raw = source.read_bytes()
    replay = score_saved_report(source, args['output'] / 'dev.json')
    assert source.read_bytes() == raw
    assert replay['original_v1_metrics']['span_recall_at_10']['micro'] == 0
    assert replay['evidence_revision_metrics']['span_recall_at_10']['micro'] == 1
    assert replay['results'][0]['original_v1_score'] == original['results'][0]['score']
    assert replay['results'][0]['score']['span_at_10'] == [True]
    assert replay['results'][0]['elapsed'] == 1.25
    assert replay['evidence_revision_metrics']['latency_seconds'] == original['metrics']['latency_seconds']
    assert replay['evidence_revision_metrics']['peak_rss_bytes'] == 12345
    assert replay['source_report']['sha256'] == file_sha(source)
    assert len(replay['scoring_code_sha256']) == 2
    assert replay['status'] == 'posthoc_diagnostic_only' and replay['new_model_execution'] is False
    assert 'response' not in replay['results'][0]
    assert replay['results'][0]['source_response']['response_sha256']


def test_missing_trace_stays_unknown_even_if_old_report_has_context(revision_inputs):
    args = revision_inputs; source, _ = saved_report(args, trace=False)
    replay = score_saved_report(source, args['output'] / 'dev.json')
    score = replay['results'][0]['score']
    assert score['span_at_10'] == [True]
    assert score['span_at_80'] is None and score['context_groups'] is None
    assert replay['evidence_revision_metrics']['span_recall_at_80']['micro'] is None


def test_hidden_parent_cannot_fill_visible_evidence_or_context(revision_inputs):
    args = revision_inputs; source, _ = saved_report(args, visible='unrelated')
    replay = score_saved_report(source, args['output'] / 'dev.json')
    score = replay['results'][0]['score']
    assert score['span_at_10'] == [False] and score['context_groups'] == [False]
    assert score['span_at_80'] == [True]  # Only the actual recorded candidate contains the alternative.


@pytest.mark.parametrize('mutate,match', [
    (lambda d: d.update(dataset_sha256='0' * 64), 'original dev bundle'),
    (lambda d: d.update(dataset_split='blind'), 'original dev bundle'),
    (lambda d: d['results'].append(copy.deepcopy(d['results'][0])), 'incomplete/duplicate/foreign'),
    (lambda d: d['results'][0]['score'].update(span_at_10=[True]), 'cannot be reproduced'),
    (lambda d: d.update(status='incomplete'), 'completed original'),
    (lambda d: d['metrics']['latency_seconds'].update(p95=99), 'performance summary'),
    (lambda d: d['results'][0].update(cluster_id='changed'), 'grouping'),
])
def test_invalid_original_report_cannot_be_silently_rescored(revision_inputs, mutate, match):
    args = revision_inputs; source, data = saved_report(args); mutate(data); source.write_bytes(encode(data))
    with pytest.raises(ValueError, match=match):
        score_saved_report(source, args['output'] / 'dev.json')


def test_replay_does_not_read_blind_questions_or_audit_content(revision_inputs):
    args = revision_inputs; source, _ = saved_report(args)
    for name in ('blind.json', 'blind-equivalence-audit.json', 'parent-v1/blind.json'):
        (args['output'] / name).unlink()
    replay = score_saved_report(source, args['output'] / 'dev.json')
    assert replay['verification']['blind_content_read'] is False
    assert replay['verification']['cross_split_review_repeated_by_replay'] is False
    # This scoped diagnostic is not a substitute for the full publication closure.
    from scripts.freeze_evidence_revision import validate_revision_bundle
    with pytest.raises(ValueError):
        validate_revision_bundle(args['output'])


def test_blind_dataset_and_existing_output_are_rejected(revision_inputs):
    args = revision_inputs; source, _ = saved_report(args)
    with pytest.raises(ValueError, match='dev split'):
        score_saved_report(source, args['output'] / 'blind.json')
    output = args['output'].parent / 'replay'
    records = replay_reports([source], args['output'] / 'dev.json', output)
    assert len(records) == 1 and (output / 'replay-manifest.json').is_file()
    before = (output / records[0]['report']).read_bytes()
    with pytest.raises(ValueError, match='new directory'):
        replay_reports([source], args['output'] / 'dev.json', output)
    assert (output / records[0]['report']).read_bytes() == before


def test_changed_dev_alt_after_seal_is_rejected(revision_inputs):
    args = revision_inputs; source, _ = saved_report(args)
    path = args['output'] / 'dev.json'; data = json.loads(path.read_bytes())
    data[0]['evidence_groups'][0][1]['span_text'] = 'forged'
    path.write_bytes(encode(data))
    manifest_path = args['output'] / 'manifest.json'; manifest = json.loads(manifest_path.read_bytes())
    manifest['datasets']['dev.json']['sha256'] = file_sha(path); manifest_path.write_bytes(encode(manifest))
    with pytest.raises(ValueError, match='complete reviewed alternatives'):
        score_saved_report(source, path)
