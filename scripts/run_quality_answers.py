"""Resumable, source-frozen offline Service.ask evaluation; never auto-publishes.

Defaults to the development split. Blind evaluation requires --allow-blind.
Generation uses the selected profile/stage and the service's existing budget
ledger. A retrieval-only warmup is excluded from request latency and RSS data.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragapp.audit import measured_performance
from ragapp.evaluation import answer_metrics, grade, load_dataset, save_report, verify_runtime
from ragapp.experiments import load_settings, score_retrieval


def call_ledger(store):
    """Read every locally reserved/failed/successful call without exposing keys."""
    with store.db() as db:
        return {row['id']: {key: row[key] for key in
            ('id', 'profile', 'stage', 'requested_model', 'actual_model', 'status', 'reserved', 'cost', 'usage', 'elapsed')}
            for row in db.execute('SELECT * FROM calls')}


def call_cost(calls):
    total = 0.0
    for call in calls:
        value = call['cost'] if call.get('cost') is not None else call.get('reserved')
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Call ledger has missing or invalid cost/reservation')
        total += value
    return total


def report_call_cost(report):
    """One charge per ledger call, even if a checkpoint error repeats its metadata.

    Keep attempt records intact for audit. For inconsistent snapshots of the same
    call use the conservative maximum COALESCE(cost,reserved), as campaign_cost
    does across catalogs. Nonzero anonymous costs cannot safely be deduplicated.
    """
    amounts={}
    for field in ('results','failed_attempts','checkpoint_errors'):
        for row in report.get(field,[]):
            calls=row.get('calls',[])
            if not calls and row.get('total_cost_cny_upper_estimate',0)!=0:
                raise ValueError('Nonzero report cost has no ledger calls for deduplication')
            for call in calls:
                identity=call.get('id')
                if not isinstance(identity,str) or not identity:
                    raise ValueError('Report ledger call has no stable call ID')
                amounts[identity]=max(amounts.get(identity,0),call_cost([call]))
    return sum(amounts.values())


def recover_completed_checkpoint(report, done):
    """Clear only a stale blocker backed by a complete persisted result + RSS row."""
    blocked=report.get('blocked')
    if not isinstance(blocked,dict) or blocked.get('id') not in done:return
    row=next(item for item in report['results'] if item['id']==blocked['id'])
    record=next(item for item in report['performance_measurement']['records'] if item['id']==blocked['id'])
    if (not all(isinstance(row.get(key),dict) for key in ('answer','grade','retrieval_score','resource_measurement'))
            or not isinstance(row.get('calls'),list) or not isinstance(row.get('call_ids'),list)
            or row['resource_measurement']!=record or row.get('elapsed')!=record['elapsed']):
        raise ValueError('Blocked checkpoint has no complete persisted answer/resource row')
    report.setdefault('recovered_errors',[]).append({**blocked,'recovered_at':time.time(),
        'recovery_reason':'completed_answer_and_resource_record_already_persisted'})
    report.pop('blocked')


def campaign_cost(settings):
    """Deduplicate the campaign across active/candidate catalogs, including failures."""
    ledger = {}
    if settings.campaign:
        for path in sorted(settings.state.glob('*.sqlite3')):
            with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {'calls', 'call_campaigns'} <= names:
                    continue
                for call_id, value in db.execute('''SELECT c.id, COALESCE(c.cost,c.reserved) FROM calls c
                    JOIN call_campaigns x ON x.call_id=c.id WHERE x.campaign=?''', (settings.campaign,)):
                    amount = call_cost([{'cost': value}])
                    ledger[call_id] = max(ledger.get(call_id, 0), amount)
    return {'campaign': settings.campaign, 'total_cny': sum(ledger.values()),
            'includes_failed_calls': True, 'calls': len(ledger), 'scope': 'all local generation catalogs'}


def database_pid(service):
    """Resolve the actual local Qdrant listener; never substitute client RSS."""
    import psutil
    url = urlparse(service.settings.qdrant_url)
    if url.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Combined process RSS needs a local Qdrant server')
    port = url.port or (443 if url.scheme == 'https' else 80)
    pids = {connection.pid for connection in psutil.net_connections(kind='tcp')
            if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port and connection.pid}
    if len(pids) != 1:
        raise ValueError('Cannot identify the unique local Qdrant listening process')
    pid = next(iter(pids))
    if 'qdrant' not in psutil.Process(pid).name().lower() or pid == os.getpid():
        raise ValueError('Listener is not a distinct Qdrant process')
    return pid


class ProcessRSSSampler:
    """Sample a near-simultaneous pair every 100 ms, retaining the paired peak."""
    def __init__(self, service_pid, database_pid, interval=.1):
        self.service_pid, self.database_pid, self.interval = service_pid, database_pid, interval

    def __enter__(self):
        import psutil
        self.processes = psutil.Process(self.service_pid), psutil.Process(self.database_pid)
        self.samples, self.error, self.stop = [], None, threading.Event()
        self.sample()
        def sample_loop():
            while not self.stop.wait(self.interval):
                try:
                    self.sample()
                except Exception as exc:
                    self.error = str(exc)
                    return
        self.thread = threading.Thread(target=sample_loop, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        self.samples.append({'monotonic_seconds': time.perf_counter(),
            'service_rss_bytes': self.processes[0].memory_info().rss,
            'database_rss_bytes': self.processes[1].memory_info().rss})

    def __exit__(self, *_):
        self.stop.set(); self.thread.join(timeout=2)
        try:
            self.sample()
        except Exception as exc:
            self.error = str(exc)

    def record(self, case_id, elapsed):
        if self.error:
            raise RuntimeError('Resource measurement failed: ' + self.error)
        peak = max(self.samples, key=lambda row: row['service_rss_bytes'] + row['database_rss_bytes'])
        return {'id': case_id, 'elapsed': elapsed, 'service_pid': self.service_pid,
                'database_pid': self.database_pid, **peak, 'rss_samples': self.samples,
                'sampling_interval_seconds': self.interval}


def run(cases_path, report_path, *, settings=None, service=None, profile='production', stage='development',
        max_queries=None, allow_blind=False, monitor_factory=None):
    """Run real Service.ask, preserving all trace/cost data and every checkpoint.

    ``service`` and ``monitor_factory`` allow CPU-free fakes in tests. Production
    defaults instantiate the actual service and measure this worker plus Qdrant.
    Each resumed process is separately identified and warmed up before timing.
    """
    if max_queries is not None and (type(max_queries) is not int or max_queries < 0):
        raise ValueError('max_queries must be a nonnegative integer')
    cases, fingerprint, dataset_manifest = load_dataset(cases_path)
    if any(case.get('split') == 'blind' for case in cases) and not allow_blind:
        raise ValueError('Blind set requires explicit --allow-blind')
    target = Path(report_path)
    configuration = {'profile': profile, 'stage': stage,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {
        'dataset_sha256': fingerprint, 'run_configuration': configuration, 'results': [], 'failed_attempts': []}
    if report.get('dataset_sha256') != fingerprint or report.get('run_configuration') != configuration:
        raise ValueError('Dataset or run configuration changed; start a separate run')
    if report['results'] and (not report.get('runtime_manifest') or not report.get('runtime_fingerprint')):
        raise ValueError('Existing results have no frozen runtime manifest; start a new run')
    if dataset_manifest:
        current = dataset_manifest['_manifest_sha256']
        if report.get('dataset_manifest_sha256') not in (None, current):
            raise ValueError('Frozen dataset manifest changed; start a separate run')
        report['dataset_manifest_sha256'] = current
    done = {row['id'] for row in report['results']}
    if len(done) != len(report['results']) or not done <= {case['id'] for case in cases}:
        raise ValueError('Checkpoint contains duplicate or foreign case IDs')
    if done and any(value is None for value in measured_performance(
            report.get('performance_measurement'), list(done)).values()):
        raise ValueError('Checkpoint resource records do not match completed results; start a new run')
    report.update(dataset_path=str(Path(cases_path).resolve()), dataset_case_count=len(cases),
        dataset_split=next(iter({case.get('split') for case in cases})) if len({case.get('split') for case in cases}) == 1 else 'mixed',
        evaluation_schema='quality-v2' if any(case.get('schema_version') == 'quality-v2' for case in cases) else 'legacy_locator',
        execution_mode='offline', status='incomplete')
    if service is None:
        from ragapp.service import Service
        service = Service(settings or load_settings()).start()
    settings = service.settings
    # Campaign accounting is centralized in Models.store (billing.sqlite3),
    # while Service.store still holds the selected generation's document catalog.
    billing_store = getattr(getattr(service, 'models', None), 'store', service.store)
    def check_runtime():
        verify_runtime(report, service.manifest(), dataset_manifest)
    check_runtime()
    recover_completed_checkpoint(report,done)
    save_report(target, report)
    pending = [case for case in cases if case['id'] not in done]
    if max_queries is not None:
        pending = pending[:max_queries]
    if pending:
        # Fixed query, no evaluation gold or question is used for warmup.
        service.search_detailed('检索系统预热', history=[], options={'query_planning': False}, persist=False)
        check_runtime()
        if monitor_factory is None:
            db_pid = database_pid(service)
            monitor_factory = lambda: ProcessRSSSampler(os.getpid(), db_pid)
        probe = monitor_factory()
        performance = report.setdefault('performance_measurement', {
            'scope': 'offline_service_and_database', 'measurement_method': 'psutil_process_rss',
            'warmup_excluded': True, 'service_pid': probe.service_pid, 'database_pid': probe.database_pid,
            'records': [], 'process_sessions': [], 'samples': 0})
        performance.setdefault('process_sessions', []).append({'service_pid': probe.service_pid,
            'database_pid': probe.database_pid, 'warmup_excluded': True, 'started': time.time()})
        save_report(target, report)
    for case in pending:
        before = call_ledger(billing_store)
        answer = None
        completed_row = False
        started = time.perf_counter()
        try:
            check_runtime()
            with monitor_factory() as monitor:
                started = time.perf_counter()
                answer = service.ask(case['question'], profile=profile, history=case.get('history', []), stage=stage)
                elapsed = time.perf_counter() - started
            record = monitor.record(case['id'], elapsed)
            trace = service.store.trace(answer['trace_id']) if answer.get('trace_id') else None
            if trace:
                answer['evaluation_trace'] = trace
                if trace.get('runtime_manifest'):
                    verify_runtime(report, trace['runtime_manifest'], dataset_manifest)
            check_runtime()
            calls = [value for key, value in call_ledger(billing_store).items() if key not in before]
            cost = call_cost(calls)
            row = {'id': case['id'], 'question': case['question'], 'answer': answer,
                'case_metadata': {key: case[key] for key in ('answerable', 'category', 'scorable', 'review_status') if key in case},
                'grade': grade(case, answer), 'retrieval_score': score_retrieval(case, {**answer, **(trace or {})}),
                'elapsed': elapsed, 'calls': calls, 'call_ids': [call['id'] for call in calls],
                'total_cost_cny_upper_estimate': cost, 'cost_includes_failed_calls': True,
                'cost_attribution': 'billing ledger delta during request; isolated worker required for exact per-case attribution',
                'resource_measurement': record}
            report['results'].append(row)
            report['performance_measurement']['records'].append(record)
            report['performance_measurement']['samples'] = len(report['performance_measurement']['records'])
            completed_row = True
            report.pop('blocked', None)
            report['campaign_cost'] = campaign_cost(settings)
            save_report(target, report)
        except Exception as exc:
            calls = [value for key, value in call_ledger(billing_store).items() if key not in before]
            failure = {'id': case['id'], 'error': str(exc), 'time': time.time(),
                'elapsed': time.perf_counter() - started, 'calls': calls,
                'total_cost_cny_upper_estimate': call_cost(calls), 'cost_includes_failed_calls': True}
            if answer is not None:
                failure['unscored_answer'] = answer
            # A successful request whose durable checkpoint failed is not a
            # second model attempt. Preserve its row and actual call identities.
            if completed_row:
                failure['kind']='completed_answer_checkpoint_error'
                report.setdefault('checkpoint_errors', []).append(failure)
            else:
                report.setdefault('failed_attempts', []).append(failure)
            report['blocked'] = failure
            save_report(target, report)
            break
    report['metrics'] = answer_metrics(cases, report['results'])
    report['metrics'].update(measured_performance(report.get('performance_measurement'), [row['id'] for row in report['results']]))
    report['metrics']['total_cost_cny_upper_estimate'] = report_call_cost(report)
    report['metrics']['cost_aggregation'] = 'maximum COALESCE(cost,reserved) per unique ledger call ID'
    report['campaign_cost'] = campaign_cost(settings)
    report['status'] = 'awaiting_manual_audit' if len(report['results']) == len(cases) else 'incomplete'
    save_report(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, help='Offline Settings JSON; load_active is forced false')
    parser.add_argument('--cases', type=Path, default=ROOT / '.rag/evaluation/quality-v2/dev.json')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--profile', choices=('production', 'eval_aihub_free'), default='production')
    parser.add_argument('--stage', choices=('preparation', 'development', 'acceptance'), default='development')
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--allow-blind', action='store_true')
    args = parser.parse_args()
    result = run(args.cases, args.report, settings=load_settings(args.settings), profile=args.profile,
        stage=args.stage, max_queries=args.max_queries, allow_blind=args.allow_blind)
    print(json.dumps({'status': result['status'], 'metrics': result['metrics'], 'blocked': result.get('blocked')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
