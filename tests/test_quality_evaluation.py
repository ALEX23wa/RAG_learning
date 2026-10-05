import json
import hashlib
from pathlib import Path

import pytest

from ragapp.evaluation import grade, load_dataset, span_matches, verify_runtime
from ragapp.experiments import ndcg, run, score_retrieval, summarize


def gold():
    return {'id': 'q', 'question': 'What is the limit?', 'answerable': True, 'facts': [['42']],
            'evidence': [{'path': 'source.pdf', 'locator': {'page': 2}, 'span_text': 'The limit is 42.'}]}


def item(**overrides):
    return {'id': 'c', 'path': 'source.pdf', 'version': 'v', 'locator': {'page': 2},
            'text': 'The limit is 42.', **overrides}


def test_matching_number_with_wrong_source_cannot_pass():
    evidence = item(path='unrelated.pdf')
    result = grade(gold(), {'answer': '42 [S1]', 'status': 'answered',
                           'evidence': [evidence], 'citations': [evidence]})
    assert result['citation_structurally_valid']
    assert result['citation_gold_support'] == [False]
    assert not result['pass']


def test_same_page_without_fact_is_locator_hit_but_not_span_hit():
    wrong = item(text='An unrelated sentence on the same page.')
    result = grade(gold(), {'answer': '42 [S1]', 'evidence': [wrong], 'citations': [wrong]})
    assert result['legacy_locator_recalled_at_10'] == [True]
    assert result['span_recalled_at_10'] == [False]
    assert not result['pass']


def test_parent_text_not_sent_to_model_is_not_span_evidence():
    assert not span_matches(gold()['evidence'][0], item(text='Different fragment', parent='The limit is 42.'))
    assert span_matches(gold()['evidence'][0], item(text='The limit\nis 42.'))
    assert not span_matches({**gold()['evidence'][0], 'span_text': 'Power is 42 mW.'}, item(text='Power is 42 MW.'))
    assert not span_matches({**gold()['evidence'][0], 'span_text': 'x² equals 4'}, item(text='x2 equals 4'))


def test_merged_context_requires_fact_inside_the_correct_source_span():
    text = 'Unrelated. The limit is 42.'
    evidence = item(text=text, locator={'page': 1, 'source_spans': [
        {'page': 1, 'chunk_start': 0, 'chunk_end': 11},
        {'page': 2, 'chunk_start': 11, 'chunk_end': len(text)}]})
    assert span_matches(gold()['evidence'][0], evidence)
    wrong_page = {**gold()['evidence'][0], 'locator': {'page': 1}}
    assert not span_matches(wrong_page, evidence)


def test_refusal_words_inside_an_answer_are_not_successful_refusal():
    case = {'id': 'q', 'answerable': False, 'facts': [], 'evidence': []}
    assert not grade(case, {'answer': '没有问题，答案确定是42。', 'status': 'answered'})['pass']
    assert not grade(case, {'answer': '无法直接确定，但估计为42。', 'status': 'unverified'})['pass']
    assert grade(case, {'answer': '资料不足。', 'status': 'abstained', 'citations': []})['pass']


def test_invalid_citation_version_cannot_pass():
    assert not grade(gold(), {'answer': '42', 'evidence': [item()], 'citations': [item(version='old')]})['pass']


def test_citation_number_and_numeric_substrings_cannot_supply_answer_facts():
    case = {**gold(), 'numeric': [{'value': 42, 'unit': 'kg', 'tolerance': 0}]}
    case['evidence'][0]['span_text'] = 'The limit is 42 kg.'
    evidence = item(text='The limit is 42 kg.', citation='S42')
    for text in ('The limit is 999 kg [S42].', 'The limit is 142 kg [S1].',
                 'The limit is 42.5 kg [S1].', 'The limit is 999 kg S42.'):
        result = grade(case, {'answer': text, 'status': 'answered', 'evidence': [evidence], 'citations': [evidence]})
        assert result['numeric_pass'] is False and result['pass'] is False
        assert result['facts_matched'] == [False]
    for text in ('The limit is 42 g [S1].', 'The limit is 42 kg/m² [S1].'):
        assert grade(case, {'answer': text, 'evidence': [evidence], 'citations': [evidence]})['numeric_pass'] is False
    assert grade(case, {'answer': 'The limit is 42 kg [S1].', 'evidence': [evidence], 'citations': [evidence]})['pass']


def test_missing_numeric_gold_is_unknown_and_legacy_value_only_remains_explicit():
    result = grade(gold(), {'answer': '42 [S1]', 'evidence': [item()], 'citations': [item()]})
    assert result['pass'] is True and result['numeric_pass'] is None
    assert result['numeric_evaluation'] == 'not_evaluated'
    legacy = grade({**gold(), 'numeric': [{'value': 42, 'tolerance': .01}]},
                   {'answer': '42.001 [S1]', 'evidence': [item()], 'citations': [item()]})
    assert legacy['numeric_pass'] is True and legacy['numeric_evaluation'] == 'value_only_or_partial_units'


def test_numeric_unit_case_and_conditions_must_belong_to_the_numeric_clause():
    from ragapp.evaluation import numeric_grade, scoring_text
    target = [{'value': 42, 'unit': 'mW', 'conditions': ['温度20℃']}]
    assert numeric_grade(target, scoring_text('温度20℃时为42 mW [S1]。'))[0] is True
    assert numeric_grade(target, '温度20℃时为42 MW。')[0] is False
    assert numeric_grade(target, '温度20℃时测量。另一条件为42 mW。')[0] is False
    assert numeric_grade([{'value': .001, 'unit': 'kg'}], '值为1e-3 kg。')[0] is True


def test_missing_candidate_or_context_trace_is_not_fabricated():
    result = score_retrieval(gold(), {'evidence': [item()]})
    assert result['span_at_10'] == [True]
    assert result['span_at_80'] is None
    assert result['context_groups'] is None
    assert not result['candidate_measurement_available']
    untraced = score_retrieval(gold(), {'evidence': [item()], 'candidates': [item()], 'context': [item()]})
    assert untraced['span_at_80'] is None and untraced['context_groups'] is None
    traced = score_retrieval(gold(), {'evidence': [item()], 'candidates': [item()], 'context': [item()], 'trace': {}})
    assert traced['span_at_80'] == traced['context_groups'] == [True]


def test_duplicate_hits_do_not_inflate_ndcg():
    case = gold()
    case['evidence'].append({'path': 'second.pdf', 'locator': {'page': 1}, 'span_text': 'Other fact'})
    duplicated = ndcg(case, [item(), item(id='duplicate')])
    complete = ndcg(case, [item(), item(path='second.pdf', locator={'page': 1}, text='Other fact')])
    assert duplicated < complete == 1


def test_runtime_and_source_drift_are_rejected():
    report = {}
    runtime = {'runtime_fingerprint': 'a', 'documents': {'source.pdf': {'hash': 'h'}}}
    frozen = {'sources': {'source.pdf': {'hash': 'h'}}}
    verify_runtime(report, runtime, frozen)
    with pytest.raises(ValueError, match='Runtime'):
        verify_runtime(report, {**runtime, 'runtime_fingerprint': 'b'}, frozen)
    with pytest.raises(ValueError, match='source changed'):
        verify_runtime({}, runtime, {'sources': {'source.pdf': {'hash': 'different'}}})


def test_redacted_runtime_verifies_sources_without_exposing_names():
    key = hashlib.sha256('folder/source.pdf'.encode()).hexdigest()
    manifest = {'runtime_fingerprint': 'a', 'source_fingerprints': {
        key: {'hash': 'h', 'status': 'ready', 'excluded': 0}}}
    frozen = {'sources': {'folder\\source.pdf': {'hash': 'h'}}}
    verify_runtime({}, manifest, frozen)
    manifest['source_fingerprints'][key]['excluded'] = 1
    with pytest.raises(ValueError, match='not available'):
        verify_runtime({}, manifest, frozen)


def test_quality_audit_uses_stricter_thresholds_and_requires_measurements(tmp_path):
    from ragapp.audit import finalize
    report_path, audit_path = tmp_path / 'report.json', tmp_path / 'audit.json'
    report = {'evaluation_schema': 'quality-v2', 'dataset_sha256': 'data', 'runtime_fingerprint': 'runtime',
        'performance_measurement': {'scope': 'service_and_database', 'samples': 2,
            'measurement_method': 'psutil_process_rss', 'warmup_excluded': True, 'service_pid': 1, 'database_pid': 2,
            'records': [{'id': name, 'elapsed': 4, 'service_rss_bytes': 600, 'database_rss_bytes': 400}
                        for name in ('positive', 'negative')]},
        'results': [
            {'id': 'positive', 'case_metadata': {'answerable': True}, 'answer': {
                'status': 'answered', 'evaluation_trace': {'trace': {}}},
             'retrieval_score': {'span_at_80': [True], 'context_groups': [True]},
             'grade': {'evidence_recalled': [True], 'span_recalled_at_10': [True]}},
            {'id': 'negative', 'case_metadata': {'answerable': False}, 'answer': {'status': 'abstained'},
             'grade': {'evidence_recalled': [], 'span_recalled_at_10': []}}]}
    reviews = [{'id': name, 'reason': 'Checked original evidence', 'source_checked': True, 'gold_verified': True,
        'pass': True, 'citations_valid': True, 'supported_claims': 19 if name == 'positive' else 0,
        'total_claims': 20 if name == 'positive' else 0} for name in ('positive', 'negative')]
    audit = {'reviewer': 'independent', 'dataset_sha256': 'data', 'runtime_fingerprint': 'runtime', 'reviews': reviews}
    report_path.write_text(json.dumps(report)); audit_path.write_text(json.dumps(audit))
    result = finalize(report_path, audit_path, expected=2)
    assert result['status'] == 'passed' and result['faithfulness'] == .95
    audit['reviews'][0]['supported_claims'] = 18
    audit_path.write_text(json.dumps(audit)); report_path.write_text(json.dumps(report))
    assert finalize(report_path, audit_path, expected=2)['status'] == 'failed'
    report.pop('performance_measurement')
    report_path.write_text(json.dumps(report))
    assert finalize(report_path, audit_path, expected=2)['status'] == 'awaiting_measurements'


def test_performance_requires_real_combined_samples_and_recomputes_nearest_rank():
    from ragapp.audit import measured_performance
    records = [{'id': str(i), 'elapsed': i, 'service_rss_bytes': 100 + i, 'database_rss_bytes': 200}
               for i in range(20)]
    measurement = {'scope': 'offline_service_and_database', 'measurement_method': 'psutil_process_rss',
        'warmup_excluded': True, 'samples': 20, 'service_pid': 1, 'database_pid': 2, 'records': records,
        'p95_seconds': 0, 'peak_rss_bytes': 1}
    ids = [record['id'] for record in records]
    assert measured_performance(measurement, ids) == {'p95_seconds': 18, 'peak_rss_bytes': 319}
    assert measured_performance({**measurement, 'scope': 'http_client'}, ids)['p95_seconds'] is None
    assert measured_performance({**measurement, 'warmup_excluded': False}, ids)['peak_rss_bytes'] is None
    records[-1]['rss_samples'] = [{'service_rss_bytes': 500, 'database_rss_bytes': 200}]
    assert measured_performance(measurement, ids)['peak_rss_bytes'] is None
    records[-1].pop('rss_samples'); records[-1]['elapsed'] = float('nan')
    assert measured_performance(measurement, ids)['p95_seconds'] is None


def test_explicit_settings_resolve_candidate_paths_and_ignore_active_pointer(tmp_path):
    from ragapp.experiments import load_settings
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps({'state': './isolated', 'corpus': '../originals', 'load_active': True,
                               'collection': 'candidate_01', 'rrf_k': 60}))
    settings = load_settings(path)
    assert settings.state == (tmp_path / 'isolated').resolve()
    assert settings.corpus == (tmp_path / '../originals').resolve()
    assert settings.load_active is False and settings.collection == 'candidate_01' and settings.rrf_k == 60
    path.write_text(json.dumps({'secret_key': 'unexpected'}))
    with pytest.raises(ValueError, match='Unknown Settings'):
        load_settings(path)


def test_legacy_runs_cannot_resume_unfrozen_results(tmp_path):
    from ragapp.evaluation import run as answer_run
    path, checkpoint = tmp_path / 'old.json', tmp_path / 'report.json'
    path.write_text(json.dumps([gold()]))
    checkpoint.write_text(json.dumps({'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'results': [{'id': 'q'}]}))
    with pytest.raises(ValueError, match='no frozen runtime'):
        answer_run(path, checkpoint, 'test-token')
    with pytest.raises(ValueError, match='requires a runtime'):
        run(path, tmp_path / 'fresh.json', search=lambda *_: {})


def test_offline_answer_runner_resumes_real_contract_traces_costs_and_resource_sessions(tmp_path):
    from scripts.run_quality_answers import run as answer_run
    from ragapp.store import Store
    from types import SimpleNamespace
    path, checkpoint = tmp_path / 'dev.json', tmp_path / 'answers.json'
    path.write_text(json.dumps([{**gold(), 'id': str(i), 'split': 'dev', 'schema_version': 'quality-v2'} for i in range(2)]))
    class FakeService:
        settings = SimpleNamespace(state=tmp_path, campaign='testing')
        def __init__(self):
            self.store = Store(tmp_path / 'catalog.sqlite3'); self.questions = []; self.warmups = 0
            self.models = SimpleNamespace(store=Store(tmp_path / 'billing.sqlite3'))
        def manifest(self):
            return {'runtime_fingerprint': 'stable'}
        def search_detailed(self, question, **options):
            assert question == '检索系统预热' and options['options']['query_planning'] is False
            self.warmups += 1
            return {}
        def ask(self, question, **options):
            self.questions.append((question, options))
            call_id = 'call-' + str(len(self.questions))
            with self.models.store.db() as db:
                db.execute("INSERT INTO calls(id,status,reserved,cost) VALUES(?,?,?,?)", (call_id, 'done', .5, .1))
                db.execute('INSERT INTO call_campaigns VALUES(?,?)', (call_id, 'testing'))
            trace = {'trace': {}, 'candidates': [item()], 'context': [item()], 'runtime_manifest': self.manifest()}
            return {'answer': '42 [S1]', 'status': 'answered', 'evidence': [item()], 'citations': [item()],
                    'trace_id': self.store.save_trace(trace), 'call_ids': [call_id]}
    class FakeMonitor:
        service_pid, database_pid = 10, 20
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def record(self, case_id, elapsed):
            return {'id': case_id, 'elapsed': elapsed, 'service_pid': self.service_pid, 'database_pid': self.database_pid,
                    'service_rss_bytes': 500, 'database_rss_bytes': 1000}
    service = FakeService()
    first = answer_run(path, checkpoint, service=service, monitor_factory=FakeMonitor, max_queries=1)
    assert first['status'] == 'incomplete' and first['metrics']['accuracy'] is None
    FakeMonitor.service_pid = 11
    result = answer_run(path, checkpoint, service=service, monitor_factory=FakeMonitor)
    assert result['status'] == 'awaiting_manual_audit' and len(service.questions) == 2 and service.warmups == 2
    assert result['metrics']['accuracy'] == 1 and result['metrics']['total_cost_cny_upper_estimate'] == pytest.approx(.2)
    assert result['metrics']['peak_rss_bytes'] == 1500
    assert result['campaign_cost']['total_cny'] == pytest.approx(.2)
    assert result['results'][0]['calls'][0]['cost'] == .1
    with service.store.db() as db:
        assert db.execute('SELECT COUNT(*) FROM calls').fetchone()[0] == 0
    assert result['results'][0]['retrieval_score']['span_at_80'] == [True]
    assert result['results'][0]['answer']['evaluation_trace']['trace'] == {}
    assert len(result['performance_measurement']['process_sessions']) == 2
    with pytest.raises(ValueError, match='configuration changed'):
        answer_run(path, checkpoint, service=service, monitor_factory=FakeMonitor, profile='eval_aihub_free')
    result['performance_measurement']['records'].pop()
    checkpoint.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='resource records'):
        answer_run(path, checkpoint, service=service, monitor_factory=FakeMonitor)


def test_offline_answer_failure_retains_reserved_cost_and_requires_blind_opt_in(tmp_path):
    from scripts.run_quality_answers import run as answer_run
    from ragapp.store import Store
    from types import SimpleNamespace
    path = tmp_path / 'blind.json'
    path.write_text(json.dumps([{**gold(), 'split': 'blind'}]))
    with pytest.raises(ValueError, match='allow-blind'):
        answer_run(path, tmp_path / 'report.json')
    class BrokenService:
        settings = SimpleNamespace(state=tmp_path, campaign='testing')
        store = Store(tmp_path / 'catalog.sqlite3')
        models = SimpleNamespace(store=Store(tmp_path / 'billing.sqlite3'))
        def manifest(self): return {'runtime_fingerprint': 'fixed'}
        def search_detailed(self, *_args, **_kwargs): return {}
        def ask(self, *_args, **_kwargs):
            with self.models.store.db() as db:
                db.execute("INSERT INTO calls(id,status,reserved) VALUES('failed','failed',.25)")
                db.execute("INSERT INTO call_campaigns VALUES('failed','testing')")
            raise RuntimeError('Provider stopped')
    class Monitor:
        service_pid, database_pid = 10, 20
        def __enter__(self): return self
        def __exit__(self, *_): pass
    result = answer_run(path, tmp_path / 'report.json', service=BrokenService(), monitor_factory=Monitor, allow_blind=True)
    assert result['status'] == 'incomplete' and not result['results']
    assert result['metrics']['total_cost_cny_upper_estimate'] == .25
    assert result['campaign_cost']['total_cny'] == .25
    assert len(result['failed_attempts']) == 1


def test_rss_sampler_uses_same_sample_pair_instead_of_sum_of_individual_peaks(monkeypatch):
    import psutil
    from types import SimpleNamespace
    from scripts.run_quality_answers import ProcessRSSSampler
    readings = {10: iter([200, 900, 300]), 20: iter([800, 20, 200])}
    class Process:
        def __init__(self, pid): self.pid = pid
        def memory_info(self): return SimpleNamespace(rss=next(readings[self.pid]))
    monkeypatch.setattr(psutil, 'Process', Process)
    with ProcessRSSSampler(10, 20, interval=60) as sampler:
        sampler.sample()
    record = sampler.record('q', 1.5)
    assert len(record['rss_samples']) == 3
    assert record['service_rss_bytes'] == 200 and record['database_rss_bytes'] == 800
    assert record['service_rss_bytes'] + record['database_rss_bytes'] == 1000
    assert record['elapsed'] == 1.5


def test_rss_sampler_does_not_hide_dead_database_process(monkeypatch):
    import psutil
    from types import SimpleNamespace
    from scripts.run_quality_answers import ProcessRSSSampler
    calls = {10: 0, 20: 0}
    class Process:
        def __init__(self, pid): self.pid = pid
        def memory_info(self):
            calls[self.pid] += 1
            if self.pid == 20 and calls[self.pid] > 1:
                raise psutil.NoSuchProcess(self.pid)
            return SimpleNamespace(rss=100)
    monkeypatch.setattr(psutil, 'Process', Process)
    with ProcessRSSSampler(10, 20, interval=60) as sampler:
        pass
    with pytest.raises(RuntimeError, match='Resource measurement failed'):
        sampler.record('q', 1)


def test_database_pid_resolves_qdrant_listener_and_rejects_remote_or_wrong_process(monkeypatch):
    import psutil
    from types import SimpleNamespace
    from scripts.run_quality_answers import database_pid
    service = SimpleNamespace(settings=SimpleNamespace(qdrant_url='http://127.0.0.1:6333'))
    monkeypatch.setattr(psutil, 'net_connections', lambda **kwargs: [SimpleNamespace(
        status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=6333), pid=99101)])
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(name=lambda: 'qdrant.exe'))
    assert database_pid(service) == 99101
    monkeypatch.setattr(psutil, 'Process', lambda pid: SimpleNamespace(name=lambda: 'unrelated.exe'))
    with pytest.raises(ValueError, match='not a distinct Qdrant'):
        database_pid(service)
    service.settings.qdrant_url = 'http://example.org:6333'
    with pytest.raises(ValueError, match='local Qdrant'):
        database_pid(service)


@pytest.mark.parametrize('ids', [['q', 'q'], ['foreign']])
def test_all_runners_reject_duplicate_or_foreign_resume_records(tmp_path, ids):
    from ragapp.evaluation import run as answer_run
    path, checkpoint = tmp_path / 'dev.json', tmp_path / 'report.json'
    path.write_text(json.dumps([gold()]))
    report = {'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'runtime_fingerprint': 'stable', 'runtime_manifest': {'runtime_fingerprint': 'stable'},
              'configuration': {'retrieval_mode': 'bm25'}, 'results': [{'id': value} for value in ids]}
    checkpoint.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='duplicate or foreign'):
        answer_run(path, checkpoint, 'test-token')
    with pytest.raises(ValueError, match='duplicate or foreign'):
        run(path, checkpoint, search=lambda *_: {}, manifest={'runtime_fingerprint': 'stable'},
            configuration={'retrieval_mode': 'bm25'})


def test_resume_resource_sessions_must_match_record_pids():
    from ragapp.audit import measured_performance
    record = {'id': 'q', 'elapsed': 2, 'service_rss_bytes': 500, 'database_rss_bytes': 1000,
              'service_pid': 11, 'database_pid': 20}
    measurement = {'scope': 'offline_service_and_database', 'measurement_method': 'psutil_process_rss',
        'warmup_excluded': True, 'samples': 1, 'service_pid': 10, 'database_pid': 20,
        'records': [record], 'process_sessions': [
            {'service_pid': 10, 'database_pid': 20, 'warmup_excluded': True}]}
    assert measured_performance(measurement, ['q'])['p95_seconds'] is None
    measurement['process_sessions'].append({'service_pid': 11, 'database_pid': 20, 'warmup_excluded': True})
    assert measured_performance(measurement, ['q']) == {'p95_seconds': 2, 'peak_rss_bytes': 1500}
    for key, value in [('service_pid', []), ('service_pid', True), ('elapsed', 10**1000), ('service_rss_bytes', 10**1000)]:
        saved = record[key]; record[key] = value
        assert measured_performance(measurement, ['q'])['p95_seconds'] is None
        record[key] = saved


def test_runtime_observes_unsynced_original_change_and_redacted_mismatch(tmp_path):
    from ragapp.config import Settings
    from ragapp.pipeline import runtime_manifest
    from ragapp.store import Store
    corpus = tmp_path / 'papers'; corpus.mkdir()
    source = corpus / 'source.txt'; source.write_text('original evidence')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    settings = Settings(root=tmp_path, corpus=corpus, state=tmp_path / 'state', load_active=False).prepare()
    store = Store(settings.state / settings.catalog)
    with store.db() as db:
        db.execute("INSERT INTO documents VALUES('d','source.txt',?,'v','ready',NULL,0,0)", (digest,))
    before = runtime_manifest(settings, store)
    verify_runtime({}, before, {'sources': {'source.txt': {'hash': digest}}})
    source.write_text('modified original evidence without syncing')
    changed = runtime_manifest(settings, store)
    assert changed['runtime_fingerprint'] != before['runtime_fingerprint']
    assert changed['documents']['source.txt']['hash'] == digest
    assert changed['observed_sources']['source.txt']['hash'] != digest
    with pytest.raises(ValueError, match='Original source changed'):
        verify_runtime({}, changed)
    redacted = {key: changed[key] for key in ('runtime_fingerprint', 'source_fingerprints')}
    with pytest.raises(ValueError, match='Original source changed'):
        verify_runtime({}, redacted)
    source.unlink()
    missing = runtime_manifest(settings, store)
    assert missing['observed_sources']['source.txt']['status'] == 'missing'
    assert missing['runtime_fingerprint'] != changed['runtime_fingerprint']


def test_runtime_tracks_new_supported_files_and_local_stat_cache_without_pdf_parse(tmp_path, monkeypatch):
    from ragapp.config import Settings
    from ragapp.pipeline import runtime_manifest
    from ragapp.store import Store
    corpus = tmp_path / 'papers'; corpus.mkdir()
    source = corpus / 'source.txt'; source.write_text('original evidence')
    settings = Settings(root=tmp_path, corpus=corpus, state=tmp_path / 'state', load_active=False).prepare()
    store = Store(settings.state / settings.catalog)
    before = runtime_manifest(settings, store)
    opened = []
    actual_open = Path.open
    def track_open(path, *args, **kwargs):
        if path == source:
            opened.append(path)
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', track_open)
    repeated = runtime_manifest(settings, store)
    assert repeated['runtime_fingerprint'] == before['runtime_fingerprint'] and not opened
    (corpus / 'new.pdf').write_bytes(b'This is intentionally not a parsable PDF; observation hashes bytes only.')
    (corpus / 'api_key.txt').write_text('ignored secret-like filename')
    added = runtime_manifest(settings, store)
    assert added['source_inventory']['files'] == ['new.pdf', 'source.txt']
    assert added['observed_sources']['new.pdf']['status'] == 'available'
    assert added['runtime_fingerprint'] != before['runtime_fingerprint']
    report = {}; verify_runtime(report, before)
    with pytest.raises(ValueError, match='Runtime manifest changed'):
        verify_runtime(report, added)


def test_runtime_reports_unreadable_and_outside_originals_without_writing(tmp_path, monkeypatch):
    from ragapp.pipeline import observe_sources
    corpus = tmp_path / 'papers'; corpus.mkdir()
    source = corpus / 'source.txt'; source.write_text('evidence')
    actual_open = Path.open
    def fail_open(path, *args, **kwargs):
        if path == source:
            raise PermissionError('simulated unreadable original')
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_open)
    observed, inventory = observe_sources(corpus, ['source.txt', '../outside.txt'])
    assert observed['source.txt'] == {'status': 'read_error', 'hash': None, 'error': 'PermissionError'}
    assert observed['../outside.txt']['status'] == 'outside_corpus'
    assert inventory['files'] == ['source.txt']


def test_refresh_does_not_rebuild_unreviewed_cases(tmp_path):
    from scripts.build_quality_evaluation import refresh
    output = tmp_path / 'quality'; output.mkdir()
    (tmp_path / 'papers').mkdir()
    original = tmp_path / 'papers/source.txt'; original.write_text('Original')
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    manifest = {'sources': {'source.txt': {'hash': digest}}}
    (output / 'manifest.json').write_text(json.dumps(manifest))
    (output / 'original-source-pool.json').write_text(json.dumps({digest: []}))
    for split in ('dev', 'blind'):
        categories = ['single'] * 84 + ['cross_source'] * 12 + ['multi_turn'] * 12 + ['unanswerable'] * 12
        cases = [{'id': f'{split}-{i}', 'split': split, 'schema_version': 'quality-v2', 'category': c,
                  'question': 'Preserve this authored question'} for i, c in enumerate(categories)]
        (output / f'{split}.json').write_text(json.dumps(cases))
    before = (output / 'dev.json').read_bytes()
    with pytest.raises(ValueError, match='Missing independent agent'):
        refresh(tmp_path, output)
    assert (output / 'dev.json').read_bytes() == before
    assert not (output / 'source-checks.json').exists()


def test_frozen_dataset_hash_is_enforced(tmp_path):
    path = tmp_path / 'dev.json'
    path.write_text(json.dumps([gold()]))
    (tmp_path / 'manifest.json').write_text(json.dumps({'datasets': {'dev.json': {'sha256': 'wrong'}}}))
    with pytest.raises(ValueError, match='hash mismatch'):
        load_dataset(path)


def test_offline_checkpoint_is_resumable_and_configuration_bound(tmp_path):
    path, report = tmp_path / 'dev.json', tmp_path / 'results.json'
    cases = [{**gold(), 'id': str(i), 'schema_version': 'quality-v2', 'split': 'dev'} for i in range(2)]
    path.write_text(json.dumps(cases))
    calls = []
    def search(question, history, configuration):
        calls.append(question)
        return {'evidence': [item()], 'candidates': [item()], 'context': [item()]}
    options = {'search': search, 'manifest': {'runtime_fingerprint': 'stable'},
               'configuration': {'retrieval_mode': 'bm25'}}
    first = run(path, report, max_queries=1, **options)
    assert first['status'] == 'incomplete'
    last = run(path, report, **options)
    assert len(calls) == 2 and last['status'] == 'provisional_metrics'
    assert last['metrics']['span_recall_at_10']['micro'] == 1
    assert last['metrics']['cost_cny_upper_estimate'] == 0
    with pytest.raises(ValueError, match='ablation changed'):
        run(path, report, **{**options, 'configuration': {'retrieval_mode': 'dense'}})


def test_blind_requires_explicit_final_run(tmp_path):
    path = tmp_path / 'blind.json'
    path.write_text(json.dumps([{**gold(), 'split': 'blind'}]))
    with pytest.raises(ValueError, match='allow_blind'):
        run(path, tmp_path / 'report.json', search=lambda *_: {})


def test_summary_reports_micro_macro_and_cluster_intervals():
    rows = [{'id': 'a', 'cluster_id': 'one', 'sources': ['a.pdf'], 'elapsed': 1,
             'score': {'span_at_10': [True, False], 'span_at_80': [True, True], 'context_groups': [True, False]}},
            {'id': 'b', 'cluster_id': 'two', 'sources': ['b.pdf'], 'elapsed': 3,
             'score': {'span_at_10': [True], 'span_at_80': [True], 'context_groups': [True]}}]
    result = summarize(rows, 2)
    metric = result['span_recall_at_10']
    assert metric['micro'] == pytest.approx(2 / 3)
    assert metric['query_macro'] == .75 and metric['document_macro'] == .75
    assert metric['ci95']['clusters'] == 2
    assert result['latency_seconds']['p50'] == 2


def test_generated_quality_set_contract_when_available():
    root = Path(__file__).resolve().parents[1] / '.rag/evaluation/quality-v2'
    if not (root / 'manifest.json').exists():
        pytest.skip('Private original-source dataset is not distributed with source code')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    dev, _, _ = load_dataset(root / 'dev.json')
    blind, _, _ = load_dataset(root / 'blind.json')
    assert len(dev) == len(blind) == 120
    assert manifest['covered_documents'] == 38
    assert manifest['questions_without_filename'] >= 120
    assert all(manifest['counts'][k] == 24 for k in ('cross_source', 'multi_turn', 'unanswerable'))
    dev_spans = {e['span_sha256'] for c in dev for e in c.get('evidence', []) if e.get('span_sha256')}
    blind_spans = {e['span_sha256'] for c in blind for e in c.get('evidence', []) if e.get('span_sha256')}
    assert not dev_spans & blind_spans
    assert not {c['template_group'] for c in dev} & {c['template_group'] for c in blind}
    assert manifest['human_verified_cases'] == 0
