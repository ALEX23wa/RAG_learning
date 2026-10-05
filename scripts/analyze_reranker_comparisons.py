"""Read-only full-dev fixed-pool reranker diagnostics; no models, API or search.

The pool source is independently reconstructed from its committed dense report
and frozen BM25 lanes. Optional original-pool prefixes are posthoc OFFLINE
quality replay, not timed experiments. Outputs never establish acceptance.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_model_comparisons as common
from scripts import benchmark_models as benchmark

require = common.require


def analyzer_code():
    return {**common.analyzer_code(), str(Path(__file__).resolve()): common.downloads.sha256(Path(__file__))}


def validate_pool(path, manifest, corpus, cases, lexical, dense_path):
    pool = benchmark.read(path)
    require(isinstance(pool, dict) and pool.get('suite_fingerprint') == manifest['fingerprint']
            and pool.get('fingerprint') == benchmark.fingerprint({k: v for k, v in pool.items() if k != 'fingerprint'}),
            'Fixed pool fingerprint/suite identity differs')
    require(pool.get('kind') == 'BM25+dense-RRF' and type(pool.get('candidate_limit')) is int
            and pool['candidate_limit'] == 60 and type(pool.get('rrf_k')) is int and pool['rrf_k'] > 0,
            'Expected a fixed BM25+dense RRF pool with 60 candidates')
    require(common.digest_string(pool.get('dense_report_sha256'))
            and common.downloads.sha256(dense_path) == pool['dense_report_sha256'], 'Dense source report hash differs')
    name, dense = common.validate_embedding_report(dense_path, manifest, corpus, cases)
    lanes = {r['id']: [i['id'] for i in r['ranked']] for r in dense['results']}
    pools = pool.get('pools'); allowed = {r['id'] for r in corpus}
    require(isinstance(pools, dict) and set(pools) == {c['id'] for c in cases}, 'Fixed pool must cover every full dev case')
    for case in cases:
        ids = pools[case['id']]
        require(isinstance(ids, list) and len(ids) == 60 and all(isinstance(i, str) for i in ids)
                and len(set(ids)) == 60 and set(ids) <= allowed, 'Each pool must contain exactly 60 unique corpus candidates')
        rebuilt = common.fused_ranking(lanes[case['id']], lexical[case['id']], pool['rrf_k'])[:60]
        require(ids == rebuilt, 'Pool order/membership differs from frozen BM25 and committed dense RRF')
    return pool, name, dense


def validate_reranker_report(path, manifest, corpus, cases, pool):
    report = benchmark.read(path)
    require(isinstance(report, dict) and report.get('schema') in benchmark.READABLE_REPORT_SCHEMAS
            and report.get('kind') == 'rerank' and report.get('status') == 'completed_development'
            and report.get('corpus_sampled') is False, 'Expected a complete unsampled reranker report')
    require(report.get('suite_fingerprint') == manifest['fingerprint']
            and report.get('pool_fingerprint') == pool['fingerprint'], 'Reranker suite/fixed-pool identity differs')
    model = report.get('model')
    require(isinstance(model, dict), 'Missing reranker identity')
    names = [name for name, spec in benchmark.ALL_MODELS.items() if spec.get('kind') == 'reranker'
             and all(model.get(k) == spec.get(k) for k in ('repository', 'revision', 'directory', 'kind'))]
    require(len(names) == 1 and common.digest_string(model.get('files_sha256'))
            and model.get('verification') in ('all_files_sha256', 'manifest_revision_and_recorded_local_hashes'),
            'Unknown/unpinned reranker or missing weight-file identity')
    settings = report.get('settings'); code = report.get('code_sha256'); packages = report.get('packages')
    require(isinstance(settings, dict) and isinstance(code, dict) and code
            and {'scripts/benchmark_models.py', 'scripts/download_candidate_models.py', 'ragapp/evaluation.py',
                 'ragapp/retrieval.py'} <= set(code)
            and all(isinstance(k, str) and common.digest_string(v) for k, v in code.items()), 'Missing execution/code identity')
    require(isinstance(packages, dict) and all(isinstance(packages.get(k), str) and packages[k]
            for k in ('torch', 'transformers', 'sentence-transformers')), 'Missing package identity')
    for key in ('batch_size', 'threads', 'maximum_tokens'):
        require(type(settings.get(key)) is int and settings[key] > 0, 'Invalid execution setting: ' + key)
    require(type(settings.get('rss_gib')) in (int, float) and 0 < settings['rss_gib'] <= 12
            and math.isfinite(settings['rss_gib']) and settings.get('reuse_index_vectors') is False,
            'Invalid reranker resource/cache setting')
    explicit_precision = settings.get('reranker_parameter_dtype') is not None or 'rerank_precision' in report
    if explicit_precision:
        require(settings.get('reranker_parameter_dtype') == 'torch.float32' and report.get('rerank_precision') == {
            'requested_parameter_dtype': 'torch.float32', 'actual_parameter_dtypes': ['torch.float32'],
            'validation': 'all_parameters_checked'}, 'New reranker precision declaration needs verified actual FP32')
    require(isinstance(report.get('results'), list), 'Missing reranker query rows')
    try:
        completed = benchmark.completed_case_ids(report, cases, [r['id'] for r in corpus], pool['pools'])
    except (TypeError, KeyError, OverflowError) as exc:
        raise ValueError('Malformed reranker query checkpoint') from exc
    require(completed == {c['id'] for c in cases}, 'Reranker report is missing full dev queries')
    for row in report['results']:
        require(isinstance(row.get('trace'), dict), 'Reranker row requires its committed trace')
        if explicit_precision:
            trace = row['trace']
            require(trace.get('rerank_pair_budget_verified') is True
                    and type(trace.get('silent_truncations')) is int and trace['silent_truncations'] == 0
                    and type(trace.get('rerank_max_pair_tokens')) is int and trace['rerank_max_pair_tokens'] > 0
                    and type(trace.get('rerank_model_max_tokens')) is int
                    and trace['rerank_max_pair_tokens'] <= trace['rerank_model_max_tokens'] <= settings['maximum_tokens'],
                    'New reranker rows require verified complete pair token budgets')
        ordered = sorted(row['ranked'], key=lambda r: (-r['score'], r['id']))
        require(row['ranked'] == ordered, 'Reranker score ties must preserve benchmark chunk-ID ordering')
    return names[0], report


def precision_identity(report):
    """Keep null/embedding-only declarations from masquerading as reranker dtype."""
    found = {}
    for section in ('model', 'settings', 'execution', 'execution_identity', 'model_loading',
                    'reranker_precision', 'rerank_precision'):
        value = report.get(section)
        if not isinstance(value, dict):
            continue
        def visit(mapping, prefix):
            for key, item in mapping.items():
                path = prefix + '.' + key
                if isinstance(item, dict):
                    visit(item, path)
                elif 'embedding' not in path.lower() and item is not None and any(
                        word in key.lower() for word in ('dtype', 'precision')):
                    found[path] = item
        visit(value, section)
    actual = {k: v for k, v in found.items() if 'actual' in k.lower() or 'observed' in k.lower()}
    return {'declared_fields': found, 'actual_fields': actual,
            'status': 'recorded_actual_declarations' if actual else 'actual_inference_dtype_not_recorded_unknown',
            'note': 'Model name, requested dtype, null fields and embedding dtype do not prove reranker inference dtype.'}


def execution_identity(report):
    resources = report.get('resources') or {}
    return {'report_schema': report['schema'], 'settings': report['settings'],
            'code_sha256': report['code_sha256'], 'packages': report['packages'],
            'precision': common.dtype_declarations(report) if report['kind'] == 'embed' else precision_identity(report),
            'resource_monitoring': {k: resources.get(k, 'not_recorded_unknown') for k in
                ('sampling_protocol', 'sample_interval_seconds', 'qdrant_discovery_interval_seconds')}}


def pool_labels(value, pool_size):
    """Internal evaluator top80 sees <=60 items; exported metric names say so."""
    names = {'span_at_80': f'pool_at_{pool_size}',
             'span_recall_at_80': f'pool_recall_at_{pool_size}',
             'group_micro_recall_at_80': f'group_micro_pool_recall_at_{pool_size}',
             'query_macro_recall_at_80': f'query_macro_pool_recall_at_{pool_size}'}
    if isinstance(value, dict):
        return {names.get(k, k): pool_labels(v, pool_size) for k, v in value.items()}
    if isinstance(value, list):
        return [pool_labels(v, pool_size) for v in value]
    return value


def configuration(rows, pool_size, resamples, seed, *, interpretation):
    categories = sorted({r['category'] for r in rows})
    result = {'pool_size': pool_size, 'effective_evidence_k': min(10, pool_size),
              'posthoc_prefix_ablation': pool_size < 60, 'source_scoring_pool_size': 60,
              'interpretation': interpretation,
              'quality': common.quality_summary(rows, resamples=resamples, seed=seed),
              'by_category': {c: common.quality_summary([r for r in rows if r['category'] == c],
                              resamples=resamples, seed=seed) for c in categories}, 'cases': rows,
              'new_latency_or_rss_measurement': False}
    return pool_labels(result, pool_size)


def normalize_prefix_limits(prefix_limits=None, include_prefix20=False):
    require(type(include_prefix20) is bool, 'include_prefix20 must be boolean')
    limits = [] if prefix_limits is None else prefix_limits
    require(isinstance(limits, (list, tuple)) and all(type(n) is int and 1 <= n <= 59 for n in limits),
            'Prefix limits must be positive integers in 1..59')
    require(len(limits) == len(set(limits)), 'Prefix limits must be unique')
    # The legacy switch is an idempotent request for 20, including when the
    # explicit list already contains it. Duplicates within that list fail.
    return tuple(sorted(set(limits) | ({20} if include_prefix20 else set())))


def analyze(suite, dataset, pool_path, dense_path, reports, output, *, reference='bge-reranker',
            include_prefix20=False, prefix_limits=None, resamples=1000, seed=1729):
    prefixes = normalize_prefix_limits(prefix_limits, include_prefix20)
    suite, dataset, pool_path, dense_path, output = map(lambda p: Path(p).resolve(),
                                                       (suite, dataset, pool_path, dense_path, output))
    paths = [Path(p).resolve() for p in reports]
    require(len(paths) >= 2 and len(set(paths)) == len(paths), 'Supply at least two distinct reranker reports')
    inputs = [suite / 'manifest.json', *[suite / n for n in ('corpus.json', 'cases.json', 'lexical.json')],
              dataset, dataset.parent / 'manifest.json', pool_path, dense_path, *paths]
    require(not output.exists() and output not in inputs and not output.is_relative_to(suite)
            and not output.is_relative_to(dataset.parent), 'Output must be a new file outside frozen input directories')
    code = analyzer_code(); hashes = {str(p): common.downloads.sha256(p) for p in inputs}
    manifest, corpus, cases, lexical = common.load_full_suite(suite, dataset)
    if manifest['dataset_manifest'].get('source_checks_sha256'):
        hashes[str(dataset.parent / 'source-checks.json')] = manifest['dataset_manifest']['source_checks_sha256']
    pool, dense_name, dense = validate_pool(pool_path, manifest, corpus, cases, lexical, dense_path)
    sizes = (60, *prefixes)
    baseline = {}; baseline_rows = {}
    for size in sizes:
        rows = common.score_rows(cases, corpus, {key: ids[:size] for key, ids in pool['pools'].items()})
        baseline_rows[size] = rows
        baseline[f'original_pool{size}'] = configuration(rows, size, resamples, seed,
            interpretation='Original frozen RRF order; no reranker and no measured baseline latency')
    models = {}; all_rows = {}
    for path in paths:
        name, report = validate_reranker_report(path, manifest, corpus, cases, pool)
        require(name not in models, 'Duplicate reranker identities cannot be merged')
        full_scored = {r['id']: r['ranked'] for r in report['results']}
        configs = {}; all_rows[name] = {}
        for size in sizes:
            prefix = {key: set(ids[:size]) for key, ids in pool['pools'].items()}
            filtered = {key: [{'id': item['id'], 'score': item['score']} for item in ranked if item['id'] in prefix[key]]
                        for key, ranked in full_scored.items()}
            ranks = {key: [item['id'] for item in ranked] for key, ranked in filtered.items()}
            rows = common.score_rows(cases, corpus, ranks)
            for row in rows:
                row['recorded_ranking'] = filtered[row['id']]
            all_rows[name][size] = rows
            configs[f'reranked_pool{size}' if size == 60 else f'reranked_prefix{size}_offline'] = configuration(
                rows, size, resamples, seed, interpretation=(
                'Actual committed ranking of the complete original 60-candidate pool' if size == 60 else
                f'Posthoc offline quality ablation: filter the same 60-candidate scores to original pool positions 1-{size}; no measured {size}-candidate latency or memory'))
        models[name] = {'report_path': str(path), 'report_sha256': hashes[str(path)], 'model': report['model'],
            'execution_identity': execution_identity(report), 'query_context_fingerprint': benchmark.query_context(report),
            'historical_pool60_resources_unadjusted': report.get('resources'),
            'historical_pool60_query_times': [{'id': r['id'], 'elapsed_seconds': r['elapsed_seconds'],
                'warmup_excluded': r['warmup_excluded']} for r in report['results']],
            'historical_pool60_token_observations': [{'id': r['id'], **{key: r['trace'].get(key, 'not_recorded_unknown')
                for key in ('rerank_pairs', 'rerank_windowed', 'silent_truncations', 'rerank_max_pair_tokens',
                            'rerank_model_max_tokens', 'rerank_pair_budget_verified')}} for r in report['results']],
            'configurations': configs,
            'identity_verification_scope': 'Recorded pinned identity and query commits; no live model weight rehash'}
    require(reference in models, 'Reference reranker is not among the complete reports')
    differences = {}; paired = {}; versus_pool = {}
    for name, model in models.items():
        identity = model['execution_identity']; ref_identity = models[reference]['execution_identity']
        differences[name] = {k: {'reference': ref_identity[k], 'model': v}
                             for k, v in identity.items() if v != ref_identity[k]}
        paired[name] = {}; versus_pool[name] = {}
        for size in sizes:
            key = f'pool{size}' if size == 60 else f'prefix{size}_offline'
            paired[name][key] = pool_labels(common.cluster_intervals(all_rows[name][size],
                reference=all_rows[reference][size], resamples=resamples, seed=seed), size)
            versus_pool[name][key] = pool_labels(common.cluster_intervals(all_rows[name][size],
                reference=baseline_rows[size], resamples=resamples, seed=seed), size)
    require(all(common.downloads.sha256(Path(p)) == digest for p, digest in hashes.items()) and analyzer_code() == code,
            'Input or scorer code changed during analysis')
    result = {'schema': 'full-dev-reranker-comparison-analysis-v1', 'status': 'diagnostic_only',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'publication_eligible': False,
        'suite_fingerprint': manifest['fingerprint'], 'dataset_sha256': manifest['dataset_sha256'],
        'dataset_manifest_sha256': manifest['dataset_manifest']['_manifest_sha256'],
        'input_sha256': hashes, 'analysis_code_sha256': code, 'pool_fingerprint': pool['fingerprint'],
        'pool_source': {'kind': pool['kind'], 'rrf_k': pool['rrf_k'], 'candidate_limit': 60,
            'dense_report_sha256': pool['dense_report_sha256'], 'dense_model': dense_name,
            'dense_execution_identity': execution_identity(dense), 'order_and_membership_reconstructed': True},
        'reference_model': reference, 'baselines': baseline, 'models': models,
        'analysis_design': {'posthoc': True, 'prefix_limits': list(prefixes),
            'purpose': 'Development diagnostics for candidate-count selection; no prospective or acceptance claim',
            'prefix_latency_measured': False, 'prefix_memory_measured': False},
        'execution_differences_from_reference': differences,
        'paired_model_minus_reference_ci95': paired, 'paired_model_minus_original_pool_ci95': versus_pool,
        'metric_contract': {'required_groups': 'All required groups retained; alternatives within a group are OR.',
            'top10': 'Strict visible span Recall@10 over at most min(10, pool_size) returned items; no parent or new candidates injected.',
            'pool_recall': 'Strict required-group recall over exactly the original pool60, or requested original prefixN; each metric is named for N, never @80.',
            'aggregation': 'Required-group micro and per-query macro reported separately; original benchmark main metric is query macro.',
            'ndcg': 'Novel-group gain from experiments.ndcg; duplicate evidence earns no repeated gain.',
            'eligibility': 'Original experiments.score_retrieval scorable rule; no coverable-only subset conditioning.',
            'prefix_ablation': 'Same recorded scores filtered by ORIGINAL pool prefixN, never the reranked topN; original score values preserved, no new model evaluation or timing.'},
        'verification': {'full_answerable_dev_cases': True, 'fixed_pool_dense_source_verified': True,
            'query_commits_verified': True, 'all_input_hashes_unchanged': True, 'model_calls': 0, 'api_calls': 0,
            'new_retrieval': False, 'blind_data_accepted': False},
        'limitations': ['Development diagnostics only; no automatic winner, promotion or acceptance conclusion.',
            'Cluster bootstrap reflects fixed development cases and does not cover all sources of uncertainty.',
            'Original timings/RSS remain 60-candidate observations; every prefix ablation has no measured latency or memory.',
            'Different schema/code/precision/monitoring epochs are recorded, not assumed to be equivalent resource benchmarks.',
            'Legacy or null precision declarations remain unknown; no actual dtype is inferred from model identity.',
            'Posthoc prefix score reuse assumes per-pair scoring; it does not measure batching, numerical or execution differences of a new run.',
            'No answers, factuality, refusal behavior or blind-set performance is evaluated.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('suite', 'dataset', 'pool', 'dense-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--reference', default='bge-reranker')
    parser.add_argument('--include-prefix20', action='store_true', help='Offline quality ablation only; no measured 20-candidate latency')
    parser.add_argument('--prefix-limits', type=int, nargs='+', help='Unique original-pool prefix sizes in 1..59; posthoc quality only')
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    result = analyze(args.suite, args.dataset, args.pool, args.dense_report, args.reports, args.output,
        reference=args.reference, include_prefix20=args.include_prefix20, prefix_limits=args.prefix_limits,
        resamples=args.resamples, seed=args.seed)
    print(json.dumps({'status': result['status'], 'models': list(result['models']), 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
