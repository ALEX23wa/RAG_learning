"""All inputs are synthetic fixtures; no private dataset/report is opened."""
import copy

import pytest

from scripts import analyze_reranker_comparisons as analysis
from scripts import benchmark_models as benchmark
from test_analyze_model_comparisons import bundle as embedding_bundle, save


@pytest.fixture
def bundle(embedding_bundle):
    bundle = embedding_bundle
    dense_path = bundle['reports'][0]
    dense = benchmark.read(dense_path)
    lexical = benchmark.read(bundle['suite'] / 'lexical.json')
    dense_ranks = {r['id']: [i['id'] for i in r['ranked']] for r in dense['results']}
    pools = {c['id']: analysis.common.fused_ranking(dense_ranks[c['id']], lexical[c['id']], 10)[:60]
             for c in bundle['cases']}
    pool = {'suite_fingerprint': bundle['manifest']['fingerprint'], 'kind': 'BM25+dense-RRF',
            'dense_report_sha256': analysis.common.downloads.sha256(dense_path), 'rrf_k': 10,
            'candidate_limit': 60, 'pools': pools}
    pool['fingerprint'] = benchmark.fingerprint(pool)
    pool_path = bundle['output'].parent / 'pool.json'; save(pool_path, pool)
    reports = []
    for name in ('bge-reranker', 'qwen-reranker'):
        report = copy.deepcopy(dense)
        report.update(kind='rerank', pool_fingerprint=pool['fingerprint'], results=[],
                      model={**benchmark.ALL_MODELS[name], 'files_sha256': 'e' * 64, 'verification': 'all_files_sha256'})
        report['code_sha256']['ragapp/retrieval.py'] = 'f' * 64
        for case in bundle['cases']:
            ids = list(pools[case['id']])
            # A wins the one-group query, B wins the two-group query.
            if (name == 'bge-reranker') == (case['id'] == 'single'):
                desired = {'c000', 'c001'} if case['id'] == 'single' else {'c080', 'c081'}
                ids.sort(key=lambda i: (i not in desired, i))
            else:
                desired = {'c000', 'c001'} if case['id'] == 'single' else {'c080', 'c081'}
                ids.sort(key=lambda i: (i in desired, i))
            row = {'id': case['id'], 'elapsed_seconds': 5.0, 'warmup_excluded': True,
                'ranked': [{'id': i, 'score': float(60 - rank)} for rank, i in enumerate(ids)],
                'trace': {'rerank_pairs': 60, 'silent_truncations': 0},
                'metrics': {'eligible': True, 'span_recall_at_10': .5, 'span_recall_at_80': 1.0,
                            'ndcg_at_10': .5, 'corpus_span_coverage': 1.0}}
            report['results'].append(benchmark.query_commit(report, row))
        path = dense_path.parent / f'{name}.json'; save(path, report); reports.append(path)
    bundle.update(dense_path=dense_path, pool_path=pool_path, pool=pool, reports=reports)
    return bundle


def run(bundle, **kwargs):
    return analysis.analyze(bundle['suite'], bundle['dataset'], bundle['pool_path'], bundle['dense_path'],
                            bundle['reports'], bundle['output'], resamples=100, **kwargs)


def alter_report(bundle, mutate, recommit=True, index=0):
    path = bundle['reports'][index]; report = benchmark.read(path); mutate(report)
    if recommit:
        for row in report['results']:
            benchmark.query_commit(report, row)
    save(path, report)


def alter_pool(bundle, mutate, refreeze=True):
    pool = benchmark.read(bundle['pool_path']); mutate(pool)
    if refreeze:
        pool['fingerprint'] = benchmark.fingerprint({k: v for k, v in pool.items() if k != 'fingerprint'})
    save(bundle['pool_path'], pool)


def test_complete_fixed_pool_micro_macro_and_pairing_are_read_only(bundle):
    paths = [*bundle['suite'].glob('*.json'), *bundle['dataset'].parent.glob('*.json'),
             bundle['pool_path'], bundle['dense_path'], *bundle['reports']]
    before = {p: p.read_bytes() for p in paths}
    result = run(bundle)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert result['status'] == 'diagnostic_only' and result['publication_eligible'] is False
    assert result['pool_source']['order_and_membership_reconstructed'] is True
    a = result['models']['bge-reranker']['configurations']['reranked_pool60']['quality']
    b = result['models']['qwen-reranker']['configurations']['reranked_pool60']['quality']
    assert a['points']['group_micro_recall_at_10'] == pytest.approx(1 / 3)
    assert b['points']['group_micro_recall_at_10'] == pytest.approx(2 / 3)
    assert a['points']['query_macro_recall_at_10'] == b['points']['query_macro_recall_at_10'] == .5
    assert a['points']['group_micro_pool_recall_at_60'] == b['points']['group_micro_pool_recall_at_60'] == 1
    assert a['required_groups'] == 3 and a['eligible_queries'] == 2
    for value in result['paired_model_minus_reference_ci95']['bge-reranker']['pool60'].values():
        assert value['point'] == value['low'] == value['high'] == 0
    delta = result['paired_model_minus_reference_ci95']['qwen-reranker']['pool60']
    assert delta['group_micro_recall_at_10']['point'] == pytest.approx(1 / 3)
    assert delta['group_micro_pool_recall_at_60']['point'] == delta['group_micro_pool_recall_at_60']['low'] == 0
    assert set(result['baselines']) == {'original_pool60'}
    assert result['verification']['model_calls'] == result['verification']['api_calls'] == 0
    assert result['models']['bge-reranker']['execution_identity']['precision']['status'] == 'actual_inference_dtype_not_recorded_unknown'
    text = bundle['output'].read_text(encoding='utf-8')
    assert 'alpha proof' not in text and 'Synthetic question' not in text
    assert 'span_at_80' not in text and 'recall_at_80' not in text
    assert 'latency_seconds' not in a
    assert set(result['models']['qwen-reranker']['configurations']['reranked_pool60']['by_category']) == {'single', 'cross-source'}


def test_offline20_uses_original_prefix_not_reranked_top20(bundle):
    pool = bundle['pool']['pools']['single']
    outside_prefix = pool[30]
    def rank_outside_first(report):
        row = next(r for r in report['results'] if r['id'] == 'single')
        ids = [outside_prefix] + [r['id'] for r in row['ranked'] if r['id'] != outside_prefix]
        row['ranked'] = [{'id': i, 'score': 100 - n} for n, i in enumerate(ids)]
    alter_report(bundle, rank_outside_first)
    result = run(bundle, include_prefix20=True)
    model = result['models']['bge-reranker']; full = model['configurations']['reranked_pool60']
    offline = model['configurations']['reranked_prefix20_offline']
    full_row = next(r for r in full['cases'] if r['id'] == 'single')
    offline_row = next(r for r in offline['cases'] if r['id'] == 'single')
    assert full_row['ranked_ids'][0] == outside_prefix
    assert outside_prefix not in offline_row['ranked_ids']
    assert set(offline_row['ranked_ids']) == set(pool[:20])
    assert 'group_micro_pool_recall_at_20' in offline['quality']['points']
    assert offline['new_latency_or_rss_measurement'] is False
    assert all('elapsed' not in row for row in offline['cases'])
    assert model['historical_pool60_query_times'][0]['elapsed_seconds'] == 5.0
    assert set(result['baselines']) == {'original_pool60', 'original_pool20'}


def test_multiple_prefixes_keep_original_scores_labels_ci_and_baselines(bundle):
    prefixes = [20, 30, 40, 50]
    inputs = {path: path.read_bytes() for path in [bundle['pool_path'], bundle['dense_path'], *bundle['reports']]}
    result = run(bundle, prefix_limits=prefixes)
    assert result['analysis_design']['posthoc'] is True
    assert result['analysis_design']['prefix_limits'] == prefixes
    assert result['analysis_design']['prefix_latency_measured'] is False
    assert result['analysis_design']['prefix_memory_measured'] is False
    assert set(result['baselines']) == {'original_pool60', *(f'original_pool{n}' for n in prefixes)}
    for path in bundle['reports']:
        original = benchmark.read(path)
        name = next(k for k, v in benchmark.ALL_MODELS.items() if v['repository'] == original['model']['repository'])
        model = result['models'][name]
        by_case = {r['id']: r['ranked'] for r in original['results']}
        for n in prefixes:
            config = model['configurations'][f'reranked_prefix{n}_offline']
            baseline = result['baselines'][f'original_pool{n}']
            assert config['pool_size'] == n and config['effective_evidence_k'] == 10
            assert config['posthoc_prefix_ablation'] is True and config['new_latency_or_rss_measurement'] is False
            assert config['source_scoring_pool_size'] == 60
            assert baseline['posthoc_prefix_ablation'] is True
            assert f'no measured {n}-candidate latency' in config['interpretation']
            for row in config['cases']:
                allowed = set(bundle['pool']['pools'][row['id']][:n])
                expected = [item for item in by_case[row['id']] if item['id'] in allowed]
                assert row['recorded_ranking'] == expected
                assert row['ranked_ids'] == [item['id'] for item in expected]
                assert len(row['recorded_ranking']) == n
                assert f'pool_at_{n}' in row['score'] and 'pool_at_60' not in row['score']
                assert 'elapsed_seconds' not in row
            for category in config['by_category'].values():
                assert f'group_micro_pool_recall_at_{n}' in category['points']
                assert f'pool_recall_at_{n}' in category['reused_evaluator_aggregates']
            points = config['quality']['points']
            assert points[f'group_micro_pool_recall_at_{n}'] == baseline['quality']['points'][f'group_micro_pool_recall_at_{n}']
            pair = result['paired_model_minus_reference_ci95'][name][f'prefix{n}_offline']
            assert f'group_micro_pool_recall_at_{n}' in pair
            assert pair[f'group_micro_pool_recall_at_{n}']['point'] == 0
            assert pair[f'group_micro_pool_recall_at_{n}']['low'] == pair[f'group_micro_pool_recall_at_{n}']['high'] == 0
            assert f'prefix{n}_offline' in result['paired_model_minus_original_pool_ci95'][name]
            assert f'group_micro_pool_recall_at_{n}' in config['quality']['ci95']
    assert all(path.read_bytes() == value for path, value in inputs.items())


def test_boundary_prefixes_and_legacy_switch_are_compatible(bundle):
    result = run(bundle, prefix_limits=[59, 1, 20], include_prefix20=True)
    assert result['analysis_design']['prefix_limits'] == [1, 20, 59]
    model = result['models']['bge-reranker']
    assert set(model['configurations']) == {'reranked_pool60', 'reranked_prefix1_offline',
                                          'reranked_prefix20_offline', 'reranked_prefix59_offline'}
    assert model['configurations']['reranked_prefix1_offline']['effective_evidence_k'] == 1
    assert model['configurations']['reranked_prefix59_offline']['pool_size'] == 59
    assert len(model['configurations']['reranked_prefix1_offline']['cases'][0]['ranked_ids']) == 1
    assert len(model['configurations']['reranked_prefix59_offline']['cases'][0]['ranked_ids']) == 59


@pytest.mark.parametrize('value', [[0], [-1], [60], [80], [20, 20], [True], [1.0], ['20'],
                                  [None], [[20]], {20}, '20', 20])
def test_invalid_prefixes_fail_before_reading_any_inputs(value, monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail('Invalid prefix arguments must fail before any input is opened')
    monkeypatch.setattr(analysis.common.downloads, 'sha256', unexpected_read)
    with pytest.raises(ValueError, match='Prefix limits'):
        analysis.analyze('missing', 'missing', 'missing', 'missing', [], 'missing', prefix_limits=value)


def test_prefix_normalization_is_deterministic_and_legacy_idempotent():
    assert analysis.normalize_prefix_limits() == ()
    assert analysis.normalize_prefix_limits([], True) == (20,)
    assert analysis.normalize_prefix_limits((40, 20, 30), True) == (20, 30, 40)
    with pytest.raises(ValueError, match='boolean'):
        analysis.normalize_prefix_limits([], 1)


@pytest.mark.parametrize('mutation', [
    lambda r: r['results'].pop(),
    lambda r: r['results'].append(copy.deepcopy(r['results'][0])),
    lambda r: r['results'][0]['ranked'].pop(),
    lambda r: r['results'][0]['ranked'][0].update(id='foreign'),
    lambda r: r['results'][0]['ranked'][0].update(id=r['results'][0]['ranked'][1]['id']),
    lambda r: r['results'][0]['ranked'][1].update(score=999),
    lambda r: r['results'][0]['ranked'][0].update(score=float('nan')),
    lambda r: r['results'][0].update(warmup_excluded=False),
    lambda r: r['results'][0].update(elapsed_seconds=10 ** 1000),
    lambda r: r['results'][0].pop('trace'),
    lambda r: r['results'][0]['metrics'].update(ndcg_at_10=9),
    lambda r: r.update(kind='embed'),
    lambda r: r.update(schema='model-benchmark-v2'),
    lambda r: r.update(status='running'),
    lambda r: r.update(corpus_sampled=True),
    lambda r: r.update(pool_fingerprint='f' * 64),
    lambda r: r.update(suite_fingerprint='f' * 64),
    lambda r: r['model'].update(revision='main'),
    lambda r: r['model'].update(files_sha256='missing'),
    lambda r: r['settings'].update(rss_gib=10 ** 1000),
    lambda r: r['settings'].update(reuse_index_vectors=True),
    lambda r: r['code_sha256'].pop('ragapp/retrieval.py'),
])
def test_invalid_reranker_rows_or_identity_rejected(bundle, mutation):
    alter_report(bundle, mutation)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle['output'].exists()


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(kind='BM25'),
    lambda p: p.update(candidate_limit=80),
    lambda p: p.update(rrf_k=True),
    lambda p: p.update(rrf_k=0),
    lambda p: p.update(dense_report_sha256='d' * 64),
    lambda p: p.update(suite_fingerprint='d' * 64),
    lambda p: p['pools'].pop('single'),
    lambda p: p['pools']['single'].pop(),
    lambda p: p['pools']['single'].reverse(),
    lambda p: p['pools']['single'].__setitem__(0, 'foreign'),
])
def test_pool_source_and_order_are_not_trusted_even_after_refreezing(bundle, mutation):
    alter_pool(bundle, mutation)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle['output'].exists()


def test_tampering_fixed_pool_or_commit_is_rejected(bundle):
    alter_report(bundle, lambda r: r['results'][0].update(elapsed_seconds=7), recommit=False)
    with pytest.raises(ValueError, match='incomplete or changed'):
        run(bundle)


def test_dense_source_requires_complete_commits_not_only_matching_hash(bundle):
    report = benchmark.read(bundle['dense_path']); report['results'].pop(); save(bundle['dense_path'], report)
    alter_pool(bundle, lambda p: p.update(dense_report_sha256=analysis.common.downloads.sha256(bundle['dense_path'])))
    with pytest.raises(ValueError, match='Missing completed dev query'):
        run(bundle)


def test_new_precision_and_old_unknown_are_explicit(bundle):
    def update(report):
        report['schema'] = 'model-benchmark-v4'
        report['settings'].update(reranker_parameter_dtype='torch.float32', embedding_parameter_dtype=None,
                                  resource_sampling_protocol='target-pid-rss-v2')
        report['rerank_precision'] = {'requested_parameter_dtype': 'torch.float32',
            'actual_parameter_dtypes': ['torch.float32'], 'validation': 'all_parameters_checked'}
        report['resources'].update(sampling_protocol='target-pid-rss-v2')
        for row in report['results']:
            row['trace'].update(rerank_max_pair_tokens=400, rerank_model_max_tokens=450, rerank_pair_budget_verified=True)
    alter_report(bundle, update, index=1)
    result = run(bundle)
    model = result['models']['qwen-reranker']
    assert model['execution_identity']['precision']['actual_fields'] == {'rerank_precision.actual_parameter_dtypes': ['torch.float32']}
    assert model['historical_pool60_token_observations'][0]['rerank_pair_budget_verified'] is True
    assert set(result['execution_differences_from_reference']['qwen-reranker']) == {'report_schema', 'settings', 'precision', 'resource_monitoring'}


def test_requested_precision_without_actual_measurement_is_rejected(bundle):
    alter_report(bundle, lambda r: r['settings'].update(reranker_parameter_dtype='torch.float32'))
    with pytest.raises(ValueError, match='verified actual FP32'):
        run(bundle)


@pytest.mark.parametrize('trace_update', [
    {'rerank_pair_budget_verified': False},
    {'rerank_max_pair_tokens': 481},
    {'rerank_max_pair_tokens': True},
    {'rerank_model_max_tokens': 999},
    {'rerank_max_pair_tokens': 0},
    {'silent_truncations': 1},
    {'silent_truncations': False},
])
def test_new_report_cannot_claim_verified_pair_budget_from_bad_trace(bundle, trace_update):
    def update(report):
        report['schema'] = 'model-benchmark-v4'
        report['settings']['reranker_parameter_dtype'] = 'torch.float32'
        report['rerank_precision'] = {'requested_parameter_dtype': 'torch.float32',
            'actual_parameter_dtypes': ['torch.float32'], 'validation': 'all_parameters_checked'}
        for row in report['results']:
            row['trace'].update(rerank_max_pair_tokens=400, rerank_model_max_tokens=480, rerank_pair_budget_verified=True)
        report['results'][0]['trace'].update(trace_update)
    alter_report(bundle, update)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle['output'].exists()


def test_legacy_v4_without_observed_dtype_remains_unknown(bundle):
    def update(report):
        report['schema'] = 'model-benchmark-v4'
        report['settings']['embedding_parameter_dtype'] = None
    alter_report(bundle, update)
    result = run(bundle)
    identity = result['models']['bge-reranker']['execution_identity']
    assert identity['report_schema'] == 'model-benchmark-v4'
    assert identity['precision']['status'] == 'actual_inference_dtype_not_recorded_unknown'


def test_trace_precision_tampering_invalidates_commit(bundle):
    def update(report):
        report['schema'] = 'model-benchmark-v4'
        report['settings']['reranker_parameter_dtype'] = 'torch.float32'
        report['rerank_precision'] = {'requested_parameter_dtype': 'torch.float32',
            'actual_parameter_dtypes': ['torch.float32'], 'validation': 'all_parameters_checked'}
        for row in report['results']:
            row['trace'].update(rerank_max_pair_tokens=400, rerank_model_max_tokens=480, rerank_pair_budget_verified=True)
    alter_report(bundle, update)
    alter_report(bundle, lambda r: r['results'][0]['trace'].update(rerank_max_pair_tokens=401), recommit=False)
    with pytest.raises(ValueError, match='incomplete or changed'):
        run(bundle)


def test_unknown_embedding_null_is_not_reranker_dtype():
    identity = analysis.precision_identity({'settings': {'embedding_parameter_dtype': None}})
    assert identity['actual_fields'] == identity['declared_fields'] == {}
    assert identity['status'] == 'actual_inference_dtype_not_recorded_unknown'


def test_input_mutation_during_analysis_emits_nothing(bundle, monkeypatch):
    original = analysis.common.score_rows; changed = False
    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            with bundle['pool_path'].open('a', encoding='utf-8') as stream:
                stream.write(' ')
        return result
    monkeypatch.setattr(analysis.common, 'score_rows', mutate)
    with pytest.raises(ValueError, match='changed during analysis'):
        run(bundle)
    assert not bundle['output'].exists()


def test_does_not_overwrite_artifact(bundle):
    bundle['output'].write_text('preserve', encoding='utf-8')
    with pytest.raises(ValueError, match='new file'):
        run(bundle)
    assert bundle['output'].read_text(encoding='utf-8') == 'preserve'


def test_query_sampling_rejected(bundle):
    path = bundle['suite'] / 'cases.json'; save(path, bundle['cases'][:1])
    manifest = benchmark.read(bundle['suite'] / 'manifest.json'); manifest['cases'] = 1
    manifest['files']['cases.json'] = analysis.common.downloads.sha256(path)
    manifest['fingerprint'] = benchmark.fingerprint({k: v for k, v in manifest.items() if k != 'fingerprint'})
    save(bundle['suite'] / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='without query sampling'):
        run(bundle)


def test_duplicate_models_cannot_claim_two_independent_comparisons(bundle):
    save(bundle['reports'][1], benchmark.read(bundle['reports'][0]))
    with pytest.raises(ValueError, match='Duplicate reranker'):
        run(bundle)
