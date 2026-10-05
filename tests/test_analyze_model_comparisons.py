"""Synthetic data only: never opens local quality datasets or model reports."""
import copy
import json
import math

import pytest

from scripts import analyze_model_comparisons as analysis
from scripts import benchmark_models as benchmark


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def gold(text):
    return {'path': 'synthetic.txt', 'locator': {'line_start': 1, 'line_end': 1}, 'span_text': text}


@pytest.fixture
def bundle(tmp_path):
    dataset = tmp_path / 'dataset' / 'dev.json'
    cases = [
        {'id': 'single', 'split': 'dev', 'category': 'single', 'cluster_id': 'cluster-a',
         'answerable': True, 'question': 'Synthetic question alpha?', 'evidence_groups': [[gold('alpha proof')]]},
        {'id': 'cross', 'split': 'dev', 'category': 'cross-source', 'cluster_id': 'cluster-b',
         'answerable': True, 'question': 'Synthetic question beta and gamma?',
         'evidence_groups': [[gold('beta proof')], [gold('gamma proof')]]},
        {'id': 'negative', 'split': 'dev', 'category': 'unanswerable', 'cluster_id': 'cluster-c',
         'answerable': False, 'question': 'Synthetic absent question?'}]
    save(dataset, cases)
    save(dataset.parent / 'manifest.json', {'schema': 'synthetic-frozen-dataset-v1',
         'datasets': {'dev.json': {'sha256': analysis.downloads.sha256(dataset), 'count': len(cases)}}})
    expected, digest, dataset_manifest = benchmark.development_cases(dataset)
    corpus = [{'id': f'c{i:03}', 'path': 'synthetic.txt', 'text': f'irrelevant synthetic text {i}',
               'locator': {'line_start': 1, 'line_end': 1}} for i in range(90)]
    corpus[0]['text'] = corpus[1]['text'] = 'alpha proof'
    corpus[80]['text'] = corpus[82]['text'] = 'beta proof'
    corpus[81]['text'] = 'gamma proof'
    # A hidden parent containing the desired proof must never earn credit.
    corpus[2]['parent_text'] = 'alpha proof beta proof gamma proof'
    lexical = {'single': ['c000', 'c001'] + [f'c{i:03}' for i in range(2, 80)],
               'cross': ['c080', 'c081'] + [f'c{i:03}' for i in range(78)]}
    suite = tmp_path / 'suite'
    for name, value in {'corpus.json': corpus, 'cases.json': expected, 'lexical.json': lexical}.items():
        save(suite / name, value)
    manifest = {'schema': 'model-benchmark-v1', 'dataset_sha256': digest, 'dataset_manifest': dataset_manifest,
        'corpus_sampled': False, 'corpus_selection': 'full active corpus', 'full_active_chunks': 90,
        'chunks': 90, 'cases': len(expected), 'files': {name: analysis.downloads.sha256(suite / name)
            for name in ('corpus.json', 'cases.json', 'lexical.json')}}
    manifest['fingerprint'] = benchmark.fingerprint(manifest)
    save(suite / 'manifest.json', manifest)
    reports = []
    for name in ('e5', 'bge-m3'):
        report = {'schema': 'model-benchmark-v3', 'kind': 'embed', 'status': 'completed_development',
            'corpus_sampled': False, 'suite_fingerprint': manifest['fingerprint'],
            'model': {**benchmark.ALL_MODELS[name], 'files_sha256': 'a' * 64, 'verification': 'all_files_sha256'},
            'settings': {'batch_size': 4, 'threads': 2, 'maximum_tokens': 480, 'rss_gib': 12,
                         'reuse_index_vectors': False},
            'code_sha256': {key: 'b' * 64 for key in ('scripts/benchmark_models.py',
                          'scripts/download_candidate_models.py', 'ragapp/evaluation.py')},
            'packages': {'torch': 'synthetic', 'transformers': 'synthetic', 'sentence-transformers': 'synthetic'},
            'resources': {'elapsed_seconds': 123.0}, 'metrics': {'span_recall_at_10': .5}, 'results': []}
        for case in expected:
            # Model A wins the one-group query; B wins the two-group query.
            ids = list(range(80)) if name == 'e5' else (
                list(range(2, 82)) if case['id'] == 'single' else [80, 81] + list(range(78)))
            row = {'id': case['id'], 'query_windows': 1, 'warmup_excluded': True, 'elapsed_seconds': 1.0,
                   'ranked': [{'id': f'c{i:03}', 'score': float(80 - rank)} for rank, i in enumerate(ids)],
                   'metrics': {'eligible': True, 'span_recall_at_10': .5, 'span_recall_at_80': .5,
                               'ndcg_at_10': .5, 'corpus_span_coverage': 1.0}}
            report['results'].append(benchmark.query_commit(report, row))
        path = tmp_path / 'reports' / f'{name}.json'; save(path, report); reports.append(path)
    return {'dataset': dataset, 'suite': suite, 'reports': reports, 'output': tmp_path / 'diagnostic.json',
            'cases': expected, 'corpus': corpus, 'manifest': manifest}


def run(bundle, **kw):
    return analysis.analyze(bundle['suite'], bundle['dataset'], bundle['reports'], bundle['output'],
                            reference='e5', resamples=100, **kw)


def alter_report(bundle, mutate, recommit=True, index=0):
    path = bundle['reports'][index]; report = benchmark.read(path); mutate(report)
    if recommit:
        for row in report['results']:
            benchmark.query_commit(report, row)
    save(path, report)


def alter_suite(bundle, mutate):
    manifest = benchmark.read(bundle['suite'] / 'manifest.json'); mutate(manifest)
    manifest['fingerprint'] = benchmark.fingerprint({k: v for k, v in manifest.items() if k != 'fingerprint'})
    save(bundle['suite'] / 'manifest.json', manifest)


def test_full_group_micro_macro_and_paired_replay_are_distinct_and_read_only(bundle):
    paths = list(bundle['suite'].glob('*.json')) + list(bundle['dataset'].parent.glob('*.json')) + bundle['reports']
    before = {p: p.read_bytes() for p in paths}
    result = run(bundle)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert result['status'] == 'diagnostic_only' and not result['publication_eligible']
    e5 = result['models']['e5']['configurations']['dense']['quality']
    bge = result['models']['bge-m3']['configurations']['dense']['quality']
    assert e5['points']['group_micro_recall_at_10'] == pytest.approx(1 / 3)
    assert bge['points']['group_micro_recall_at_10'] == pytest.approx(2 / 3)
    assert e5['points']['query_macro_recall_at_10'] == bge['points']['query_macro_recall_at_10'] == .5
    assert e5['required_groups'] == 3 and e5['eligible_queries'] == 2
    assert e5['reused_evaluator_aggregates']['span_recall_at_10']['micro'] == pytest.approx(1 / 3)
    assert result['models']['e5']['dtype']['status'] == 'not_recorded_unknown'
    for metric in result['paired_model_minus_reference_ci95']['e5']['dense'].values():
        assert metric['point'] == metric['low'] == metric['high'] == 0
    delta = result['paired_model_minus_reference_ci95']['bge-m3']['dense']['group_micro_recall_at_10']
    assert delta['point'] == pytest.approx(1 / 3) and delta['clusters'] == 2
    for model in result['models'].values():
        assert set(model['configurations']) == set(analysis.CONFIGURATIONS)
        assert model['historical_resources_unadjusted'] == {'elapsed_seconds': 123.0}
        for config in model['configurations'].values():
            assert set(config['by_category']) == {'single', 'cross-source'}
            assert 'latency_seconds' not in config['quality']
            for row in config['cases']:
                assert len(row['ranked_ids']) == 80
                assert 'question' not in row and 'evidence_groups' not in row
                assert row['score']['context_groups'] is None
    assert 'Synthetic question' not in bundle['output'].read_text(encoding='utf-8')
    assert 'alpha proof' not in bundle['output'].read_text(encoding='utf-8')
    assert result['verification']['model_calls'] == result['verification']['api_calls'] == 0
    assert str(bundle['reports'][0]) in result['input_sha256']
    assert str(analysis.Path(analysis.__file__).resolve()) in result['analysis_code_sha256']


def test_rrf_fixed_80_lanes_and_stable_ties():
    dense = ['z', 'a'] + [f'd{i:03}' for i in range(100)]
    lexical = ['a', 'z'] + [f'b{i:03}' for i in range(100)]
    for k in (10, 30, 60):
        fused = analysis.fused_ranking(dense, lexical, k)
        assert len(fused) == 80 and fused[:2] == ['a', 'z']
        assert 'd078' not in fused and 'b078' not in fused


def test_strict_text_novel_gain_and_no_hidden_parent(bundle):
    case = next(c for c in bundle['cases'] if c['id'] == 'cross')
    rows = analysis.score_rows([case], bundle['corpus'], {'cross': ['c080', 'c082', 'c081']})
    assert rows[0]['score']['span_at_10'] == [True, True]
    assert rows[0]['score']['ndcg_at_10'] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    hidden = analysis.score_rows([case], bundle['corpus'], {'cross': ['c002']})
    assert hidden[0]['score']['span_at_10'] == [False, False]
    mismatch = copy.deepcopy(case); mismatch['evidence_groups'][0][0]['span_text'] = 'BETA PROOF'
    assert analysis.score_rows([mismatch], bundle['corpus'], {'cross': ['c080']})[0]['score']['span_at_10'] == [False, False]


@pytest.mark.parametrize('mutation', [
    lambda r: r['results'].pop(),
    lambda r: r['results'].append(copy.deepcopy(r['results'][0])),
    lambda r: r['results'][0]['ranked'].pop(),
    lambda r: r['results'][0]['ranked'][0].update(id='foreign'),
    lambda r: r['results'][0]['ranked'][0].update(id=r['results'][0]['ranked'][1]['id']),
    lambda r: r['results'][0]['ranked'][1].update(score=10000),
    lambda r: r['results'][0]['ranked'][0].update(score=float('nan')),
    lambda r: r['results'][0].update(query_windows=True),
    lambda r: r['results'][0].update(warmup_excluded=False),
    lambda r: r['results'][0].update(elapsed_seconds=10 ** 1000),
    lambda r: r['results'][0]['metrics'].update(span_recall_at_10=2),
    lambda r: r.update(schema='model-benchmark-v2'),
    lambda r: r.update(kind='rerank'),
    lambda r: r.update(status='running'),
    lambda r: r.update(corpus_sampled=True),
    lambda r: r.update(suite_fingerprint='c' * 64),
    lambda r: r['model'].update(revision='unpinned'),
    lambda r: r['model'].update(files_sha256='bad'),
    lambda r: r['settings'].update(rss_gib=10 ** 1000),
    lambda r: r['code_sha256'].pop('ragapp/evaluation.py'),
])
def test_invalid_reports_never_emit_diagnostic(bundle, mutation):
    alter_report(bundle, mutation)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle['output'].exists()


def test_tampered_commit_is_rejected(bundle):
    alter_report(bundle, lambda r: r['results'][0].update(elapsed_seconds=9), recommit=False)
    with pytest.raises(ValueError, match='incomplete or changed'):
        run(bundle)


def test_corpus_sampling_rejected(bundle):
    alter_suite(bundle, lambda m: m.update(corpus_sampled=True))
    with pytest.raises(ValueError, match='Sampled corpora'):
        run(bundle)


def test_query_sampling_rejected_even_with_full_corpus(bundle):
    path = bundle['suite'] / 'cases.json'; save(path, bundle['cases'][:1])
    def sample(m):
        m.update(cases=1); m['files']['cases.json'] = analysis.downloads.sha256(path)
    alter_suite(bundle, sample)
    with pytest.raises(ValueError, match='without query sampling'):
        run(bundle)


def test_frozen_dataset_hash_and_suite_byte_hash_are_checked(bundle):
    bundle['dataset'].write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen dataset file hash mismatch'):
        run(bundle)


def test_suite_byte_hash_is_checked(bundle):
    (bundle['suite'] / 'lexical.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen suite changed'):
        run(bundle)


def test_missing_explicit_dataset_freeze_is_rejected(bundle):
    save(bundle['dataset'].parent / 'manifest.json', {'schema': 'synthetic-only'})
    with pytest.raises(ValueError, match='explicit frozen content hash'):
        run(bundle)


def test_blind_case_rejected_before_analysis(bundle):
    cases = copy.deepcopy(bundle['cases']); cases[0]['split'] = 'blind'
    path = bundle['suite'] / 'cases.json'; save(path, cases)
    alter_suite(bundle, lambda m: m['files'].update({'cases.json': analysis.downloads.sha256(path)}))
    with pytest.raises(ValueError, match='Blind data'):
        run(bundle)


def test_three_models_and_v4_precision_code_epoch_differences(bundle):
    path = bundle['reports'][0]; third = benchmark.read(path)
    third['schema'] = 'model-benchmark-v4'
    third['model'] = {**benchmark.ALL_MODELS['qwen-embedding'], 'files_sha256': 'd' * 64, 'verification': 'all_files_sha256'}
    third['settings'].update(embedding_parameter_dtype='torch.float32', resource_sampling_protocol='target-pid-rss-v2')
    third['embedding_precision'] = {'requested_parameter_dtype': 'torch.float32',
        'actual_parameter_dtypes': ['torch.float32'], 'validation': 'all_parameters_checked'}
    third['code_sha256']['scripts/benchmark_models.py'] = 'e' * 64
    third['resources']['sampling_protocol'] = 'target-pid-rss-v2'
    for row in third['results']:
        benchmark.query_commit(third, row)
    path = path.parent / 'qwen.json'; save(path, third); bundle['reports'].append(path)
    result = run(bundle)
    assert set(result['models']) == {'e5', 'bge-m3', 'qwen-embedding'}
    differences = result['execution_differences_from_reference']['qwen-embedding']
    assert set(differences) == {'report_schema', 'execution_settings', 'execution_code_sha256', 'dtype', 'resource_monitoring'}
    assert result['models']['qwen-embedding']['dtype']['declared_fields']['embedding_precision.actual_parameter_dtypes'] == ['torch.float32']
    assert 'Not treated as an identical resource benchmark' in result['resource_comparison']


def test_v4_missing_actual_precision_is_rejected(bundle):
    alter_report(bundle, lambda r: r.update(schema='model-benchmark-v4'))
    with pytest.raises(ValueError, match='explicit validated CPU float32'):
        run(bundle)


def test_paired_cluster_draws_require_identical_cases_and_denominators(bundle):
    ranks = {c['id']: ['c000', 'c080', 'c081'] for c in bundle['cases']}
    rows = analysis.score_rows(bundle['cases'], bundle['corpus'], ranks)
    identical = analysis.cluster_intervals(rows, reference=copy.deepcopy(rows), resamples=100)
    assert all(x['low'] == x['high'] == 0 for x in identical.values())
    other = copy.deepcopy(rows); other[0]['cluster_id'] = 'other'
    with pytest.raises(ValueError, match='cluster or required-group'):
        analysis.cluster_intervals(rows, reference=other, resamples=100)


def test_output_overwrite_or_frozen_directory_is_rejected(bundle):
    bundle['output'].write_text('preserve', encoding='utf-8')
    with pytest.raises(ValueError, match='new artifact'):
        run(bundle)
    assert bundle['output'].read_text(encoding='utf-8') == 'preserve'
    bundle['output'] = bundle['suite'] / 'new.json'
    with pytest.raises(ValueError, match='new artifact'):
        run(bundle)


def test_duplicate_model_identity_is_rejected(bundle):
    save(bundle['reports'][1], benchmark.read(bundle['reports'][0]))
    with pytest.raises(ValueError, match='Duplicate model identities'):
        run(bundle)


def test_single_report_is_not_a_comparison(bundle):
    bundle['reports'].pop()
    with pytest.raises(ValueError, match='two or three'):
        run(bundle)


def test_input_mutation_during_analysis_prevents_output(bundle, monkeypatch):
    original = analysis.score_rows
    changed = False
    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            with bundle['dataset'].open('a', encoding='utf-8') as stream:
                stream.write(' ')
        return result
    monkeypatch.setattr(analysis, 'score_rows', mutate)
    with pytest.raises(ValueError, match='changed during comparison'):
        run(bundle)
    assert not bundle['output'].exists()


def test_corpus_hash_change_between_freeze_check_and_use_is_rejected(bundle, monkeypatch):
    original = analysis.load_full_suite
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (bundle['suite'] / 'corpus.json').open('a', encoding='utf-8') as stream:
            stream.write(' ')
        return result
    monkeypatch.setattr(analysis, 'load_full_suite', mutate)
    with pytest.raises(ValueError, match='changed during comparison'):
        run(bundle)
    assert not bundle['output'].exists()
