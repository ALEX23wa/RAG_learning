"""Retrieval-only ablations. No generation calls or automatic acceptance claims.

Run ``python -m ragapp.experiments --help``. Each configuration gets a separate
checkpoint and runtime fingerprint. Unreviewed gold produces provisional metrics.
"""
import argparse
import inspect
import json
import math
import os
import random
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path
from dataclasses import fields

import httpx

from .evaluation import (evidence_groups, load_dataset, located, recalled_groups,
                         save_report, span_matches, verify_runtime)

DEFAULT_CONFIGURATIONS = {
    'bm25': {'retrieval_mode': 'bm25', 'query_planning': False},
    'dense': {'retrieval_mode': 'dense', 'query_planning': False},
    'hybrid': {'retrieval_mode': 'hybrid', 'query_planning': False},
    'hybrid_rerank': {'retrieval_mode': 'hybrid_rerank', 'query_planning': False},
}


def load_settings(path=None):
    """Explicit offline settings; JSON paths are relative to its file, not cwd.

    The active pointer must not override a candidate generation selected here.
    This helper does not prepare directories, start services or load models.
    """
    from .config import Settings
    values = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    if not isinstance(values, dict):
        raise ValueError('Settings JSON must contain an object')
    unknown = set(values) - {field.name for field in fields(Settings)}
    if unknown:
        raise ValueError('Unknown Settings fields: ' + ', '.join(sorted(unknown)))
    base = Path(path).resolve().parent if path else Path.cwd()
    for name in ('root', 'corpus', 'state'):
        if name in values:
            candidate = Path(values[name]).expanduser()
            values[name] = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    values['load_active'] = False
    return Settings(**values)


def percentile(values, fraction):
    if not values:
        return None
    data = sorted(values)
    position = (len(data) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return data[left] + (data[right] - data[left]) * (position - left)


def ndcg(case, evidence, k=10):
    """Binary novel-span gain: duplicates cannot add relevance repeatedly."""
    groups = evidence_groups(case)
    if not groups:
        return None
    seen, dcg = set(), 0.0
    for rank, item in enumerate(evidence[:k], 1):
        hit = {i for i, group in enumerate(groups) if any(span_matches(g, item) for g in group)}
        if hit - seen:
            dcg += 1 / math.log2(rank + 1)
        seen |= hit
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(groups))))
    return min(1.0, dcg / ideal) if ideal else None


def score_retrieval(case, response):
    evidence = response.get('evidence', [])
    trace_available = isinstance(response.get('trace'), dict)
    candidates = response.get('candidates') if trace_available else None
    context = response.get('context') if trace_available else None
    groups = evidence_groups(case)
    scorable = (case.get('answerable', True) and case.get('scorable', True) and bool(groups)
                and all(any(e.get('span_text') or e.get('quote') for e in group) for group in groups))
    if not scorable:
        return {'scorable': False, 'reason': 'unanswerable_or_gold_requires_manual_visual_review',
                'legacy_locator_at_10': recalled_groups(case, evidence[:10], located)}
    top10 = recalled_groups(case, evidence[:10])
    top80 = recalled_groups(case, candidates[:80]) if candidates is not None else None
    covered = recalled_groups(case, context) if context is not None else None
    return {'scorable': True, 'span_at_10': top10, 'span_at_80': top80,
            'context_groups': covered, 'all_required_context': all(covered) if covered is not None else None,
            'ndcg_at_10': ndcg(case, evidence),
            'legacy_locator_at_10': recalled_groups(case, evidence[:10], located),
            'candidate_measurement_available': candidates is not None,
            'context_measurement_available': context is not None}


def bootstrap_interval(rows, key, samples=500, seed=1729):
    """95% cluster bootstrap interval; a tiny sample is reported as such."""
    clusters = defaultdict(list)
    for row in rows:
        values = row['score'].get(key)
        if values:
            clusters[row.get('cluster_id', row['id'])].append(values)
    if not clusters:
        return None
    groups = list(clusters.values())
    rng, estimates = random.Random(seed), []
    for _ in range(samples):
        draws = [rng.choice(groups) for _ in groups]
        values = [v for group in draws for row in group for v in row]
        estimates.append(sum(values) / len(values))
    return {'low': percentile(estimates, .025), 'high': percentile(estimates, .975),
            'confidence': .95, 'method': 'evidence-cluster bootstrap',
            'clusters': len(groups), 'resamples': samples, 'seed': seed}


def summarize(rows, total):
    def aggregate(key):
        values = [v for row in rows for v in (row['score'].get(key) or [])]
        per_query = [statistics.mean(row['score'][key]) for row in rows if row['score'].get(key)]
        documents = defaultdict(list)
        for row in rows:
            items = row['score'].get(key)
            if items:
                group_sources = row.get('group_sources')
                if group_sources and len(group_sources) == len(items):
                    for value, sources in zip(items, group_sources):
                        for source in sources:
                            documents[source].append(float(value))
                else:
                    for source in row.get('sources', []):
                        documents[source].append(statistics.mean(items))
        return {'micro': statistics.mean(values) if values else None,
                'query_macro': statistics.mean(per_query) if per_query else None,
                'document_macro': statistics.mean(statistics.mean(v) for v in documents.values()) if documents else None,
                'groups': len(values), 'queries': len(per_query), 'documents': len(documents),
                'ci95': bootstrap_interval(rows, key)}
    times = [r['elapsed'] for r in rows]
    ndcgs = [r['score']['ndcg_at_10'] for r in rows if r['score'].get('ndcg_at_10') is not None]
    complete_context = [r['score']['all_required_context'] for r in rows if r['score'].get('all_required_context') is not None]
    return {'completed': len(rows), 'total': total, 'complete': len(rows) == total,
        'span_recall_at_80': aggregate('span_at_80'), 'span_recall_at_10': aggregate('span_at_10'),
        'legacy_locator_recall_at_10': aggregate('legacy_locator_at_10'), 'context_coverage': aggregate('context_groups'),
        'all_required_context_rate': statistics.mean(complete_context) if complete_context else None,
        'ndcg_at_10': statistics.mean(ndcgs) if ndcgs else None,
        'latency_seconds': {'p50': percentile(times, .5), 'p95': percentile(times, .95), 'samples': len(times)},
        'peak_rss_bytes': max((r['peak_rss_bytes'] for r in rows if r.get('peak_rss_bytes') is not None), default=None),
        'rss_scope': 'runner process; HTTP server memory is not included',
        'cost_cny_upper_estimate': sum(r.get('cost_cny_upper_estimate', 0) for r in rows),
        'gold_human_verified': sum(r.get('gold_review_status') == 'human_verified' for r in rows),
        'acceptance': 'not_determined; source and answer audits required'}


class MemorySampler:
    """Sample current-process RSS during the query, including temporary peaks."""
    def __enter__(self):
        self.peak, self.stop = 0, threading.Event()
        try:
            import psutil
            process = psutil.Process()
            self.peak = process.memory_info().rss
            def sample():
                while not self.stop.is_set():
                    self.peak = max(self.peak, process.memory_info().rss)
                    self.stop.wait(.025)
            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        except ImportError:
            self.thread = None
            self.peak = None
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)


def offline_search(service):
    """Adapt the real Service.search_detailed without generation or evaluation gold."""
    def search(question, history, config):
        method = service.search_detailed
        args = {'history': history, 'options': config}
        if 'persist' in inspect.signature(method).parameters:
            args['persist'] = False
        if 'debug' in inspect.signature(method).parameters:
            args['debug'] = True
        return method(question, **args)
    return search


def run(cases_path, report_path, *, search=None, manifest=None, token=None,
        url='http://127.0.0.1:8000', configuration=None, allow_blind=False, max_queries=None):
    """Run one frozen ablation, accepting a callable or the authorized HTTP API."""
    cases, fingerprint, dataset_manifest = load_dataset(cases_path)
    if max_queries is not None and (type(max_queries) is not int or max_queries < 0):
        raise ValueError('max_queries must be a nonnegative integer')
    if any(c.get('split') == 'blind' for c in cases) and not allow_blind:
        raise ValueError('Blind set requires an explicit allow_blind final evaluation run')
    configuration = configuration or DEFAULT_CONFIGURATIONS['hybrid_rerank']
    target = Path(report_path)
    report = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {
        'dataset_sha256': fingerprint, 'configuration': configuration, 'results': [], 'status': 'incomplete'}
    if report['dataset_sha256'] != fingerprint or report['configuration'] != configuration:
        raise ValueError('Dataset or ablation changed; use a separate checkpoint')
    if report['results'] and (not report.get('runtime_fingerprint') or not report.get('runtime_manifest')):
        raise ValueError('Existing results have no frozen runtime manifest; start a new run')
    report['dataset_path'] = str(Path(cases_path).resolve())
    if dataset_manifest:
        current = dataset_manifest['_manifest_sha256']
        if report.get('dataset_manifest_sha256') not in (None, current):
            raise ValueError('Frozen dataset manifest changed; use a separate checkpoint')
        report['dataset_manifest_sha256'] = current
    strict = any(c.get('schema_version') == 'quality-v2' for c in cases)
    splits = {c.get('split') for c in cases}
    report['dataset_split'] = next(iter(splits)) if len(splits) == 1 else 'mixed'
    report['dataset_case_count'] = len(cases)
    report['execution_mode'] = 'offline' if search else 'http'
    done = {row['id'] for row in report['results']}
    if len(done) != len(report['results']) or not done <= {case['id'] for case in cases}:
        raise ValueError('Checkpoint contains duplicate or foreign case IDs')
    with httpx.Client(timeout=300, headers={'Authorization': 'Bearer ' + (token or '')}) as client:
        def check_runtime():
            if manifest:
                value = manifest() if callable(manifest) else manifest
            elif search is None:
                response = client.get(url + '/api/manifest')
                response.raise_for_status()
                value = response.json()
            else:
                raise ValueError('Offline evaluation requires a runtime manifest provider')
            verify_runtime(report, value, dataset_manifest)
        check_runtime()
        blocked = report.get('blocked')
        if isinstance(blocked, dict) and blocked.get('id') in done:
            # A per-case checkpoint can fail after append, then the final save
            # can persist both that completed result and its now-stale error.
            report.setdefault('recovered_errors', []).append({**blocked,
                'recovered_at': time.time(), 'recovery_reason': 'completed_result_already_persisted'})
            report.pop('blocked')
        save_report(target, report)
        completed_this_run = 0
        for case in cases:
            if case['id'] in done:
                continue
            if max_queries is not None and completed_this_run >= max_queries:
                break
            try:
                check_runtime()
                started = time.perf_counter()
                with MemorySampler() as memory:
                    if search:
                        answer = search(case['question'], case.get('history', []), configuration)
                    else:
                        response = client.post(url + '/api/search', json={'question': case['question'],
                            'history': case.get('history', []), 'options': configuration, 'debug': True})
                        response.raise_for_status()
                        answer = response.json()
                elapsed = time.perf_counter() - started
                check_runtime()
                if answer.get('runtime_manifest'):
                    verify_runtime(report, answer['runtime_manifest'], dataset_manifest)
                report['results'].append({'id': case['id'], 'cluster_id': case.get('cluster_id', case['id']),
                    'sources': sorted({e['path'] for group in evidence_groups(case) for e in group}),
                    'group_sources': [sorted({e['path'] for e in group}) for group in evidence_groups(case)],
                    'category': case.get('category'), 'score': score_retrieval(case, answer),
                    'elapsed': elapsed, 'peak_rss_bytes': memory.peak, 'response': answer,
                    'cost_cny_upper_estimate': answer.get('cost_cny_upper_estimate', 0),
                    'gold_review_status': case.get('review_status', 'unrecorded')})
                completed_this_run += 1
                report.pop('blocked', None)
                save_report(target, report)
            except Exception as exc:
                report['blocked'] = {'id': case['id'], 'error': str(exc), 'time': time.time()}
                break
    report['metrics'] = summarize(report['results'], len(cases))
    report['status'] = 'provisional_metrics' if len(report['results']) == len(cases) else 'incomplete'
    report['evaluation_schema'] = 'quality-v2' if strict else 'legacy_locator'
    save_report(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--mode', choices=('http', 'offline'), default='http')
    parser.add_argument('--config', choices=DEFAULT_CONFIGURATIONS, default='hybrid_rerank')
    parser.add_argument('--settings', type=Path, help='Offline Settings JSON; paths relative to this file; load_active=false')
    parser.add_argument('--rrf-k', type=int)
    parser.add_argument('--expand-context', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--query-planning', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable query planning (may call the configured model); default is disabled')
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--token-env', default='RAG_EVAL_TOKEN')
    parser.add_argument('--allow-blind', action='store_true')
    parser.add_argument('--max-queries', type=int)
    args = parser.parse_args()
    configuration = dict(DEFAULT_CONFIGURATIONS[args.config])
    configuration['query_planning'] = args.query_planning
    if args.rrf_k is not None:
        if args.rrf_k <= 0:
            parser.error('--rrf-k must be positive')
        configuration['rrf_k'] = args.rrf_k
    if args.expand_context is not None:
        configuration['expand_context'] = args.expand_context
    options = {'configuration': configuration, 'allow_blind': args.allow_blind,
               'max_queries': args.max_queries}
    if args.mode == 'offline':
        from .service import Service
        from .pipeline import runtime_manifest
        service = Service(load_settings(args.settings)).start()
        options.update(search=offline_search(service), manifest=lambda: runtime_manifest(service.settings, service.store))
    else:
        if args.settings:
            parser.error('--settings is only supported with --mode offline')
        token = os.environ.get(args.token_env)
        if not token:
            parser.error('Set the authorized RAG API token in ' + args.token_env)
        options.update(token=token, url=args.url)
    result = run(args.cases, args.report, **options)
    print(json.dumps({'status': result['status'], 'metrics': result['metrics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
