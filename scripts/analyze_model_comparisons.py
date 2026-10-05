"""Read-only, required-group analysis of completed full development embeddings.

Uses recorded dense ranks and frozen suite BM25 only. No model/API/search call.
Both the original benchmark query-macro and this group's micro metric are named.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ragapp.evaluation as evaluation
import ragapp.experiments as experiments
import ragapp.pipeline as pipeline
import scripts.benchmark_models as benchmark
import scripts.download_candidate_models as downloads

CONFIGURATIONS = ('dense', 'hybrid_rrf10', 'hybrid_rrf30', 'hybrid_rrf60')
METRICS = ('group_micro_recall_at_10', 'group_micro_recall_at_80',
           'query_macro_recall_at_10', 'query_macro_recall_at_80', 'novel_group_ndcg_at_10')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_string(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def analyzer_code():
    paths = [Path(m.__file__).resolve() for m in (evaluation, experiments, pipeline, benchmark, downloads)]
    paths.append(Path(__file__).resolve())
    return {str(p): downloads.sha256(p) for p in paths}


def load_full_suite(folder, dataset):
    """An explicit original dev dataset is necessary to detect prepare --limit."""
    folder = Path(folder).resolve(); dataset = Path(dataset).resolve()
    raw = benchmark.read(folder / 'manifest.json')
    require(raw.get('schema') == 'model-benchmark-v1'
            and set(raw.get('files', {})) == {'corpus.json', 'cases.json', 'lexical.json'}, 'Invalid full suite schema/files')
    manifest, corpus, cases, lexical = benchmark.load_suite(folder)
    require(manifest.get('corpus_sampled') is False and manifest.get('corpus_selection') == 'full active corpus',
            'Sampled corpora are not accepted')
    require(isinstance(corpus, list) and corpus and type(manifest.get('chunks')) is int
            and manifest['chunks'] == manifest.get('full_active_chunks') == len(corpus), 'Incomplete full corpus')
    expected, dataset_sha, dataset_manifest = benchmark.development_cases(dataset, limit=0)
    frozen_dataset = (dataset_manifest or {}).get('datasets', {}).get(dataset.name)
    frozen_hash = frozen_dataset.get('sha256') if isinstance(frozen_dataset, dict) else frozen_dataset
    require(frozen_hash == dataset_sha, 'Dataset must have an explicit frozen content hash')
    require(dataset_manifest and manifest.get('dataset_sha256') == dataset_sha
            and manifest.get('dataset_manifest') == dataset_manifest, 'Frozen dataset/manifest identity differs')
    require(cases == expected and len(cases) == manifest.get('cases'),
            'Suite must contain every frozen answerable dev case without query sampling')
    require(cases and all(c.get('split') in ('dev', 'development') and c.get('answerable') is True for c in cases),
            'Only answerable development cases are accepted')
    ids = [c.get('id') for c in cases]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), 'Invalid/duplicate dev case identity')
    for case in cases:
        require(isinstance(case.get('cluster_id'), str) and bool(case['cluster_id'])
                and isinstance(case.get('category'), str) and bool(case['category']), 'Case cluster/category metadata is required')
    chunk_ids = []
    for row in corpus:
        require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id']
                and isinstance(row.get('text'), str) and isinstance(row.get('path'), str)
                and isinstance(row.get('locator'), dict), 'Invalid frozen corpus evidence schema')
        chunk_ids.append(row['id'])
    require(len(chunk_ids) == len(set(chunk_ids)), 'Duplicate corpus chunk IDs')
    require(isinstance(lexical, dict) and set(lexical) == set(ids), 'BM25 must cover exactly every dev query')
    allowed = set(chunk_ids)
    for lane in lexical.values():
        require(isinstance(lane, list) and len(lane) <= 80
                and all(isinstance(i, str) for i in lane) and len(lane) == len(set(lane))
                and set(lane) <= allowed, 'BM25 lane has duplicate, foreign or over-budget candidates')
    return manifest, corpus, cases, lexical


def validate_embedding_report(path, manifest, corpus, cases):
    report = benchmark.read(path)
    require(report.get('schema') in ('model-benchmark-v3', 'model-benchmark-v4') and report.get('kind') == 'embed'
            and report.get('status') == 'completed_development' and report.get('corpus_sampled') is False,
            'Only complete unsampled v3/v4 embedding reports are accepted')
    require(report.get('suite_fingerprint') == manifest['fingerprint'], 'Embedding report suite identity differs')
    model = report.get('model')
    require(isinstance(model, dict), 'Missing recorded model identity')
    matches = [name for name, spec in benchmark.ALL_MODELS.items() if spec.get('kind') == 'embedding'
               and all(model.get(k) == spec.get(k) for k in ('repository', 'revision', 'directory', 'kind', 'dimension'))]
    require(len(matches) == 1 and type(model.get('dimension')) is int and digest_string(model.get('files_sha256'))
            and model.get('verification') in {'all_files_sha256', 'manifest_revision_and_recorded_local_hashes'},
            'Unknown/unpinned model or missing model-file identity')
    settings = report.get('settings'); code = report.get('code_sha256'); packages = report.get('packages')
    require(isinstance(settings, dict) and isinstance(code, dict) and code
            and all(isinstance(k, str) and digest_string(v) for k, v in code.items())
            and {'scripts/benchmark_models.py', 'ragapp/evaluation.py', 'scripts/download_candidate_models.py'} <= set(code),
            'Missing execution settings/code hashes')
    require(isinstance(packages, dict) and all(isinstance(packages.get(k), str) and packages[k]
            for k in ('torch', 'transformers', 'sentence-transformers')), 'Missing package identity')
    for key in ('batch_size', 'threads', 'maximum_tokens'):
        require(type(settings.get(key)) is int and settings[key] > 0, 'Invalid execution setting: ' + key)
    require(type(settings.get('rss_gib')) in (int, float) and 0 < settings['rss_gib'] <= 12
            and math.isfinite(settings['rss_gib']) and type(settings.get('reuse_index_vectors')) is bool,
            'Invalid recorded resource/cache settings')
    if report['schema'] == 'model-benchmark-v4':
        require(report.get('embedding_precision') == {
            'requested_parameter_dtype': 'torch.float32', 'actual_parameter_dtypes': ['torch.float32'],
            'validation': 'all_parameters_checked'}
            and settings.get('embedding_parameter_dtype') == 'torch.float32'
            and settings.get('resource_sampling_protocol') == 'target-pid-rss-v2',
            'v4 requires explicit validated CPU float32 and resource sampling identity')
    require(isinstance(report.get('results'), list), 'Missing query rows')
    try:
        done = benchmark.completed_case_ids(report, cases, [c['id'] for c in corpus])
    except (TypeError, KeyError, OverflowError) as exc:
        raise ValueError('Malformed query checkpoint values') from exc
    require(done == {c['id'] for c in cases}, 'Missing completed dev query rows')
    for row in report['results']:
        require(type(row.get('query_windows')) is int and row['query_windows'] > 0, 'Missing query-window identity')
    return matches[0], report


def fused_ranking(dense_ids, bm25_ids, k):
    scores = {}
    for lane in (bm25_ids[:80], dense_ids[:80]):
        for rank, identity in enumerate(lane, 1):
            scores[identity] = scores.get(identity, 0.) + 1 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))[:80]


def score_rows(cases, corpus, ranked):
    by_id = {row['id']: row for row in corpus}; rows = []
    for case in cases:
        # The matcher reads only the supplied visible text/locator, never parent.
        items = [by_id[i] for i in ranked[case['id']]][:80]
        score = experiments.score_retrieval(case, {'evidence': items[:10], 'candidates': items,
                                                   'trace': {'origin': 'recorded-ranking replay'}})
        groups = evaluation.evidence_groups(case)
        rows.append({'id': case['id'], 'cluster_id': case['cluster_id'], 'category': case['category'],
                     'sources': sorted({e['path'] for g in groups for e in g}),
                     'group_sources': [sorted({e['path'] for e in g}) for g in groups],
                     'score': score, 'ranked_ids': [i['id'] for i in items]})
    return rows


def point_metrics(rows):
    result = {}
    for k in (10, 80):
        values = [r['score'].get('span_at_' + str(k)) for r in rows]
        values = [v for v in values if v]
        result['group_micro_recall_at_' + str(k)] = sum(sum(v) for v in values) / sum(map(len, values)) if values else None
        result['query_macro_recall_at_' + str(k)] = sum(sum(v) / len(v) for v in values) / len(values) if values else None
    ndcg = [r['score']['ndcg_at_10'] for r in rows if r['score'].get('ndcg_at_10') is not None]
    result['novel_group_ndcg_at_10'] = sum(ndcg) / len(ndcg) if ndcg else None
    return result


def cluster_intervals(rows, *, reference=None, resamples=1000, seed=1729):
    """Same case/cluster draws for paired model-minus-reference differences."""
    require(type(resamples) is int and resamples >= 100 and type(seed) is int, 'Bootstrap requires >=100 resamples and integer seed')
    clusters = {}
    for row in rows:
        if row['score'].get('scorable'):
            clusters.setdefault(row['cluster_id'], []).append(row)
    if not clusters:
        return {metric: None for metric in METRICS}
    reference_rows = {r['id']: r for r in reference} if reference is not None else None
    if reference is not None:
        require(len(reference_rows) == len(reference) and {r['id'] for r in rows} == set(reference_rows),
                'Paired comparisons need identical case IDs')
        require(all(r['cluster_id'] == reference_rows[r['id']]['cluster_id']
                    and r['score'].get('scorable') == reference_rows[r['id']]['score'].get('scorable')
                    and len(r['score'].get('span_at_10') or []) ==
                        len(reference_rows[r['id']]['score'].get('span_at_10') or []) for r in rows),
                'Paired cases differ in cluster or required-group denominators')
    blocks = [clusters[k] for k in sorted(clusters)]; rng = random.Random(seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(resamples):
        selected = [row for block in rng.choices(blocks, k=len(blocks)) for row in block]
        values = point_metrics(selected)
        other = point_metrics([reference_rows[r['id']] for r in selected]) if reference_rows is not None else None
        for metric, value in values.items():
            if value is not None:
                samples[metric].append(value - other[metric] if other is not None else value)
    points = point_metrics(rows); others = point_metrics(reference) if reference is not None else None
    return {metric: {'point': points[metric] - others[metric] if others is not None else points[metric],
        'low': experiments.percentile(values, .025), 'high': experiments.percentile(values, .975),
        'confidence': .95, 'clusters': len(blocks), 'resamples': resamples, 'seed': seed,
        'method': 'paired evidence-cluster percentile bootstrap' if reference is not None else 'evidence-cluster percentile bootstrap'}
        if values else None for metric, values in samples.items()}


def quality_summary(rows, *, resamples=1000, seed=1729):
    # summarize needs timings, but synthetic zeros are only a private adapter:
    # no timing, RSS, cost or context metric is returned from this replay.
    aggregate = experiments.summarize([{**r, 'elapsed': 0} for r in rows], len(rows))
    keys = ('span_recall_at_10', 'span_recall_at_80', 'ndcg_at_10')
    return {'points': point_metrics(rows), 'eligible_queries': sum(r['score'].get('scorable', False) for r in rows),
            'required_groups': sum(len(r['score'].get('span_at_10') or []) for r in rows),
            'all_suite_queries': len(rows), 'reused_evaluator_aggregates': {k: aggregate[k] for k in keys},
            'ci95': cluster_intervals(rows, resamples=resamples, seed=seed)}


def dtype_declarations(report):
    found = {}
    def visit(value, path=''):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            current = path + '.' + key if path else key
            if any(word in key.lower() for word in ('dtype', 'precision')):
                found[current] = copy.deepcopy(item)
            elif isinstance(item, dict):
                visit(item, current)
    for key in ('model', 'settings', 'execution', 'execution_identity', 'model_loading', 'embedding_contract', 'embedding_precision'):
        visit(report.get(key), key)
    return {'declared_fields': found, 'status': 'recorded_declarations_only' if found else 'not_recorded_unknown',
            'note': 'Vector cache float32 does not prove model inference dtype; no dtype is inferred from model name.'}


def analyze(suite, dataset, reports, output, *, reference, resamples=1000, seed=1729):
    suite = Path(suite).resolve(); dataset = Path(dataset).resolve(); output = Path(output).resolve()
    paths = [Path(p).resolve() for p in reports]
    require(2 <= len(paths) <= 3 and len(set(paths)) == len(paths), 'Supply two or three distinct completed embedding reports')
    require(not output.exists() and output not in paths and not output.is_relative_to(suite)
            and not output.is_relative_to(dataset.parent), 'Output must be a new artifact outside the frozen suite/dataset')
    code = analyzer_code()
    # Hash before reading/decoding, then recheck before emitting an artifact.
    input_paths = [suite / 'manifest.json', *[suite / n for n in ('corpus.json', 'cases.json', 'lexical.json')],
                   dataset, dataset.parent / 'manifest.json', *paths]
    hashes = {str(p): downloads.sha256(p) for p in input_paths}
    manifest, corpus, cases, lexical = load_full_suite(suite, dataset)
    if manifest['dataset_manifest'].get('source_checks_sha256'):
        path = dataset.parent / 'source-checks.json'
        hashes[str(path)] = manifest['dataset_manifest']['source_checks_sha256']
    models = {}; all_rows = {}
    for path in paths:
        name, report = validate_embedding_report(path, manifest, corpus, cases)
        require(name not in models, 'Duplicate model identities cannot be silently merged')
        dense = {r['id']: [x['id'] for x in r['ranked']] for r in report['results']}
        configs = {}; all_rows[name] = {}
        for configuration in CONFIGURATIONS:
            ranks = dense if configuration == 'dense' else {
                c['id']: fused_ranking(dense[c['id']], lexical[c['id']], int(configuration.removeprefix('hybrid_rrf')))
                for c in cases}
            rows = score_rows(cases, corpus, ranks); all_rows[name][configuration] = rows
            categories = sorted({r['category'] for r in rows})
            configs[configuration] = {'quality': quality_summary(rows, resamples=resamples, seed=seed),
                'by_category': {category: quality_summary([r for r in rows if r['category'] == category],
                    resamples=resamples, seed=seed) for category in categories}, 'cases': rows}
        models[name] = {'report_path': str(path), 'report_sha256': hashes[str(path)], 'report_schema': report['schema'], 'model': report['model'],
            'execution_settings': report['settings'], 'execution_code_sha256': report['code_sha256'],
            'packages': report['packages'], 'dtype': dtype_declarations(report),
            'resource_monitoring': {'settings_protocol': report['settings'].get('resource_sampling_protocol', 'not_recorded_unknown'),
                **{k: (report.get('resources') or {}).get(k, 'not_recorded_unknown') for k in
                   ('sampling_protocol', 'sample_interval_seconds', 'qdrant_discovery_interval_seconds')}},
            'recorded_query_context_fingerprint': benchmark.query_context(report),
            'original_benchmark_metrics': report.get('metrics'),
            'original_benchmark_primary_aggregation': 'query macro, not required-group micro',
            'historical_resources_unadjusted': report.get('resources'), 'configurations': configs,
            'identity_verification_scope': 'recorded pinned model identity and query commit; model weight files are not rehashed'}
    require(reference in models, 'Declared reference model is not among completed reports')
    differences = {}; paired = {}
    ref = models[reference]
    for name, model in models.items():
        keys = ('report_schema', 'execution_settings', 'execution_code_sha256', 'packages', 'dtype', 'resource_monitoring')
        differences[name] = {key: {'equal_to_reference': model[key] == ref[key],
                                 'reference': ref[key], 'model': model[key]}
                             for key in keys if model[key] != ref[key]}
        paired[name] = {configuration: cluster_intervals(all_rows[name][configuration],
            reference=all_rows[reference][configuration], resamples=resamples, seed=seed)
            for configuration in CONFIGURATIONS}
    require(all(downloads.sha256(Path(p)) == h for p, h in hashes.items()) and analyzer_code() == code,
            'Input or analysis code changed during comparison')
    result = {'schema': 'full-dev-model-comparison-analysis-v1', 'status': 'diagnostic_only',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'publication_eligible': False,
        'suite_fingerprint': manifest['fingerprint'], 'suite_manifest': str(suite / 'manifest.json'),
        'dataset_sha256': manifest['dataset_sha256'], 'dataset_manifest_sha256': manifest['dataset_manifest']['_manifest_sha256'],
        'input_sha256': hashes, 'analysis_code_sha256': code, 'models': models,
        'reference_model': reference, 'paired_model_minus_reference_ci95': paired,
        'execution_differences_from_reference': differences,
        'resource_comparison': 'Not treated as an identical resource benchmark; dtype may be unknown and code/settings/model differ.',
        'metric_contract': {'required_groups': 'Every required group must be recalled; alternatives inside a group are OR.',
            'dense_budget': 80, 'bm25_budget': 80, 'rrf_k': [10, 30, 60], 'final_ranking_budget': 80, 'final_evidence_k': 10,
            'rrf': 'equal-weight sum(1/(k+one_based_rank)); ties by chunk ID; no reranker',
            'ndcg': 'experiments.ndcg novel-group gain; duplicate hits cannot earn repeated gain',
            'scorable': 'experiments.score_retrieval eligibility; all required groups retained, no full-pool-coverable subset conditioning',
            'historical_primary_metric': 'benchmark span_recall_at_k is query macro, not group micro'},
        'verification': {'all_full_answerable_dev_queries': True, 'query_commits_verified': True,
            'input_hashes_unchanged': True, 'model_calls': 0, 'api_calls': 0, 'new_retrieval': False,
            'blind_data_accepted': False, 'model_weight_files_rehashed': False},
        'limitations': ['Development diagnostics only; no statistical winner, model promotion or publication conclusion.',
            'Intervals resample evidence clusters and compare the same configuration/cases; they do not cover all sources of model uncertainty.',
            'RRF is replayed from saved top80 dense and frozen top80 BM25; no recall beyond those recorded lanes.',
            'No new latency/RSS is measured and no resource equality is inferred from absent dtype fields.',
            'Cross-epoch latency with different monitor protocols is descriptive only, not an equivalent resource benchmark.',
            'Source model metrics are retained as historical query-macro values; reported group micro is recomputed from visible evidence.',
            'Case output contains IDs/category/cluster/source metadata and scores, not question or gold text.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='Exact frozen full dev dataset; rejects suite query sampling')
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--reference', required=True, help='Pinned model registry name, e.g. e5')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    result = analyze(args.suite, args.dataset, args.reports, args.output,
                     reference=args.reference, resamples=args.resamples, seed=args.seed)
    print(json.dumps({'status': result['status'], 'models': list(result['models']),
                      'output': str(args.output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
