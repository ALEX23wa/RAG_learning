"""Synthetic only: temp corpus/SQLite, real Indexer, fake vectors/models/RSS.

These tests never connect to Qdrant, read project evaluation data, load weights,
or send a paid request. Injected runner reports cannot claim actual measured RSS.
"""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragapp.config import INDEX_SETTINGS, Settings
from ragapp.indexer import Indexer
from ragapp.pipeline import index_fingerprint
from ragapp.store import Store
from scripts import rebuild_candidate as rebuild


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(len(text))), 'offset_mapping': [(i, i+1) for i in range(len(text))]}


class FixtureMonitor:
    def __init__(self, maximum, *, peak=128 * 1024**2, unavailable=False):
        self.maximum = int(maximum * 1024**3)
        self.peak = peak
        self.unavailable = unavailable

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def check(self):
        if self.peak > self.maximum:
            raise MemoryError('fixture RSS over budget')

    def report(self):
        return {'observed_peak_rss_bytes': self.peak,
                'worker_children_qdrant_peak_rss_bytes': None if self.unavailable else self.peak,
                'sampling_status': 'unavailable' if self.unavailable else 'measured',
                'sampling_error': 'fixture missing measurement' if self.unavailable else None,
                'rss_budget_bytes': self.maximum, 'sampling_protocol': 'synthetic-fixture'}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    state, corpus = tmp_path / 'state', tmp_path / 'papers'
    state.mkdir()
    corpus.mkdir()
    (state / 'generations').mkdir()
    for name, text in {'alpha.txt': 'ALPHA original source evidence.',
                       'beta.txt': 'BETA independently parsed evidence.',
                       'excluded.txt': 'Excluded original is preserved.'}.items():
        (corpus / name).write_text(text, encoding='utf-8')
    source = Settings(root=tmp_path, corpus=corpus, state=state, collection='source_generation',
                      catalog='source_generation.sqlite3', campaign='quality-v2',
                      embedding_directory='fixture-model', stable_seconds=0, load_active=False)
    source_store = Store(state / source.catalog)
    with source_store.db() as db:
        for number, path in enumerate(sorted(corpus.iterdir())):
            excluded = path.name == 'excluded.txt'
            db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',
                       (str(number), path.name, rebuild.digest(path), 'source-version',
                        'excluded' if excluded else 'ready', None, int(excluded), 1.))
            if not excluded:
                db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)',
                           ('source-' + str(number), str(number), 'source-version',
                            'FROZEN SOURCE CHUNK MUST NOT BE COPIED', '{}', None, None))
        db.executemany('INSERT INTO metadata VALUES(?,?)',
                       [('index_fingerprint', 'preserved-source-fingerprint'),
                        ('embedding_fingerprint', 'old-embedding-fingerprint'),
                        ('parsing_fingerprint', 'old-parsing-fingerprint'),
                        ('index_schema', 'old-schema'), ('index_profile', 'old-profile'),
                        ('retained_history', 'keep-this')])
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                   ('old-call', 'production', 'preparation', 'requested', 'actual', 'complete', 0., .1, '{}', None, 1., .5))
        db.execute('INSERT INTO call_campaigns VALUES(?,?)', ('old-call', 'quality-v2'))
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)', ('historical-job', 'complete', '{}', 1., 1.))
        db.execute('INSERT INTO traces VALUES(?,?,?)', ('historical-trace', 1., '{}'))
    Store(state / 'billing.sqlite3')
    rebuild.save(state / 'generations' / (source.collection + '.json'),
                 {'collection': source.collection, 'catalog': source.catalog,
                  'settings': {key: getattr(source, key) for key in INDEX_SETTINGS},
                  'index_fingerprint': 'preserved-source-fingerprint'})
    rebuild.save(state / 'active.json', {'collection': source.collection, 'catalog': source.catalog})
    model = state / 'models' / 'fixture-model'
    model.mkdir(parents=True)
    (model / 'model.safetensors').write_bytes(b'SYNTHETIC NOT MODEL WEIGHTS')
    values = asdict(source)
    values.update(collection='candidate_new', catalog='candidate_new.sqlite3', embedding_precision='float32')
    target = Settings(**values)
    settings_path, source_path = tmp_path / 'candidate-settings.json', tmp_path / 'source-settings.json'
    settings_path.write_text(json.dumps(asdict(target), default=str), encoding='utf-8')
    source_path.write_text(json.dumps(asdict(source), default=str), encoding='utf-8')
    args = SimpleNamespace(settings=settings_path, source_settings=source_path,
                           report=state / 'reports' / 'candidate.json', resume=False,
                           allow_vision=False, rss_gib=12., expected_ready=2)
    result = SimpleNamespace(args=args, target=target, source=source, state=state, corpus=corpus,
                             collections={source.collection}, vectors={}, services=[], events=[],
                             fail_upserts=0, peak=128 * 1024**2, unavailable=False,
                             environment={'code': 'fixture-code', 'packages': {'fixture': '1'}}, vision_calls=0)

    class FixtureVision:
        def vision(self, *args, **kwargs):
            result.vision_calls += 1
            return {'actual_model': 'fixture-vision', 'text': 'Synthetic cached image description.'}

    class FixtureRetrieval:
        def __init__(self):
            self.embedding = SimpleNamespace(tokenizer=CharacterTokenizer())
            self.client = SimpleNamespace(count=lambda *a, **k: SimpleNamespace(count=len(result.vectors)))

        def upsert(self, chunks):
            result.events.append(('upsert', [c['text'] for c in chunks]))
            if result.fail_upserts:
                result.fail_upserts -= 1
                raise RuntimeError('fixture interrupted document')
            result.vectors.update({c['id']: c for c in chunks})

        def remove(self, identifiers):
            for identifier in identifiers:
                result.vectors.pop(identifier, None)

    def factory(settings):
        assert args.report.exists(), 'Checkpoint must precede catalog creation'
        report = rebuild.read(args.report)
        owner = rebuild.read(state / 'candidate-builds' / (settings.collection + '.owner.json'))
        assert report['owner'] == owner
        result.events.append('service-after-claim')
        store = Store(settings.state / settings.catalog)
        retrieval = FixtureRetrieval()
        models = FixtureVision()
        service = SimpleNamespace(settings=settings, store=store, retrieval=retrieval, models=models)
        service.indexer = Indexer(settings, store, retrieval, models)

        def start():
            with store.db() as db:
                assert db.execute('SELECT value FROM metadata WHERE key=?', (rebuild.OWNER_KEY,)).fetchone()
            result.collections.add(settings.collection)
            return service

        service.start = start
        result.services.append(service)
        return service

    result.dependencies = rebuild.Dependencies(service=factory, probe=lambda settings: set(result.collections),
        assert_no_worker=lambda previous: None, worker_identity=lambda: {'pid': -1, 'create_time': 1.},
        monitor=lambda maximum: FixtureMonitor(maximum, peak=result.peak, unavailable=result.unavailable),
        environment=lambda: json.loads(json.dumps(result.environment)))
    # OCR is deliberately synthetic; no OCR engine or weights are imported.
    monkeypatch.setattr('ragapp.parsers.ocr', lambda path: 'Synthetic OCR from a fixture image')
    return result


def test_normal_sync_rechunks_and_preserves_ledger_without_publishing(harness):
    h = harness
    source_before = rebuild.source_snapshot(h.source)[1]
    active_before = (h.state / 'active.json').read_bytes()
    report = rebuild.run(h.args, h.dependencies)
    assert report['status'] == 'complete_synthetic'
    assert report['verification'] == {'expected_ready': 2, 'ready': 2, 'chunks': 2, 'vectors': 2}
    assert report['published'] is report['publication_eligible'] is report['resources_verified'] is False
    assert report['resource_measurement_kind'] == 'synthetic_fixture'
    assert rebuild.read(report['generation'])['synthetic'] is True
    assert all('FROZEN SOURCE' not in c['text'] for c in h.vectors.values())
    assert any('ALPHA original' in c['text'] for c in h.vectors.values())
    with h.services[0].store.db() as db:
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
        assert metadata['index_fingerprint'] == index_fingerprint(h.target)
        assert metadata['embedding_fingerprint'] != 'old-embedding-fingerprint'
        assert metadata['index_schema'] == h.target.index_schema
        assert metadata['retained_history'] == 'keep-this'
        assert db.execute("SELECT COUNT(*) FROM documents WHERE excluded=1").fetchone()[0] == 1
        for table in ('calls', 'call_campaigns', 'traces'):
            assert db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] == 1
        assert db.execute("SELECT id FROM jobs WHERE id='historical-job'").fetchone()
    assert rebuild.source_snapshot(h.source)[1] == source_before
    assert (h.state / 'active.json').read_bytes() == active_before
    assert h.vision_calls == 0


def test_partial_resume_reuses_generation_job_and_ready_documents_with_peak_history(harness):
    h = harness
    h.fail_upserts, h.peak = 1, 512 * 1024**2
    with pytest.raises(RuntimeError, match='failed documents'):
        rebuild.run(h.args, h.dependencies)
    failed = rebuild.read(h.args.report)
    assert failed['status'] == 'failed'
    assert failed['sync_result']['failed'] == 1
    assert not (h.state / 'generations' / (h.target.collection + '.json')).exists()
    h.args.resume, h.peak = True, 128 * 1024**2
    report = rebuild.run(h.args, h.dependencies)
    assert report['owner'] == failed['owner']
    assert report['job_id'] == failed['job_id']
    assert report['sync_result']['unchanged'] == 1
    assert report['historical_peak_rss_bytes'] == 512 * 1024**2
    assert len(report['attempts']) == 2
    assert [a['status'] for a in report['attempts']] == ['failed', 'complete_synthetic']


def test_completed_resume_is_idempotent_without_service_or_model(harness):
    h = harness
    first = rebuild.run(h.args, h.dependencies)
    h.args.resume = True
    h.dependencies.service = lambda settings: pytest.fail('Completed resume must not create a service')
    assert rebuild.run(h.args, h.dependencies) == first


@pytest.mark.parametrize('mutation', ['config', 'code', 'package', 'original', 'model', 'source', 'policy'])
def test_resume_rejects_changed_frozen_identity_before_service(harness, mutation):
    h = harness
    h.fail_upserts = 1
    with pytest.raises(RuntimeError):
        rebuild.run(h.args, h.dependencies)
    h.args.resume = True
    if mutation == 'config':
        values = rebuild.read(h.args.settings)
        values['embedding_batch'] += 1
        rebuild.save(h.args.settings, values)
    elif mutation == 'code':
        h.environment['code'] = 'different-code'
    elif mutation == 'package':
        h.environment['packages']['fixture'] = '2'
    elif mutation == 'original':
        (h.corpus / 'alpha.txt').write_text('Changed source, same document count.', encoding='utf-8')
    elif mutation == 'model':
        (h.state / 'models/fixture-model/model.safetensors').write_bytes(b'different-fixture')
    elif mutation == 'source':
        with Store(h.state / h.source.catalog).db() as db:
            db.execute("UPDATE metadata SET value='changed' WHERE key='retained_history'")
    else:
        h.args.allow_vision = True
    h.dependencies.service = lambda settings: pytest.fail('Identity rejection must precede Service')
    with pytest.raises(ValueError, match='Resume identity differs'):
        rebuild.run(h.args, h.dependencies)


@pytest.mark.parametrize('existing', ['catalog', 'collection', 'owner', 'generation'])
def test_new_build_refuses_unknown_existing_resource(harness, existing):
    h = harness
    if existing == 'catalog':
        (h.state / h.target.catalog).write_bytes(b'UNKNOWN')
    elif existing == 'collection':
        h.collections.add(h.target.collection)
    elif existing == 'owner':
        rebuild.save(h.state / 'candidate-builds' / (h.target.collection + '.owner.json'), {'someone': 'else'})
    else:
        rebuild.save(h.state / 'generations' / (h.target.collection + '.json'), {'someone': 'else'})
    with pytest.raises(ValueError, match='already exists'):
        rebuild.run(h.args, h.dependencies)
    assert not h.args.report.exists()
    assert not h.services


@pytest.mark.parametrize('protected', ['source_collection', 'source_catalog', 'default_catalog', 'billing_catalog'])
def test_source_active_default_and_billing_targets_are_protected(harness, protected):
    h = harness
    values = rebuild.read(h.args.settings)
    if protected == 'source_collection':
        values['collection'] = h.source.collection
    elif protected == 'source_catalog':
        values['catalog'] = h.source.catalog
    elif protected == 'default_catalog':
        values['catalog'] = 'catalog.sqlite3'
    else:
        values['catalog'] = 'billing.sqlite3'
    rebuild.save(h.args.settings, values)
    with pytest.raises(ValueError, match='source, active, default, or billing'):
        rebuild.run(h.args, h.dependencies)
    assert not h.args.report.exists()


def test_database_unavailable_does_not_claim_or_create_target(harness):
    h = harness
    def unavailable(settings):
        raise ConnectionError('fixture Qdrant is not running')
    h.dependencies.probe = unavailable
    with pytest.raises(ConnectionError):
        rebuild.run(h.args, h.dependencies)
    assert not h.args.report.exists()
    assert not (h.state / h.target.catalog).exists()


@pytest.mark.parametrize('lock_name', ['sync.lock', 'server.lock', 'candidate-build.lock',
    'reports/quality-v2/experiment-worker.lock', 'reports/quality-v2-models/model-worker.lock'])
def test_existing_worker_locks_refuse_before_claim(harness, lock_name):
    from filelock import FileLock, Timeout
    h = harness
    path = h.state / lock_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path), timeout=0), pytest.raises(Timeout):
        rebuild.run(h.args, h.dependencies)
    assert not h.args.report.exists()


@pytest.mark.parametrize('mode', ['zero', 'unavailable', 'over_budget'])
def test_invalid_measurements_never_produce_generation(harness, mode):
    h = harness
    if mode == 'zero':
        h.peak = 0
    elif mode == 'unavailable':
        h.unavailable = True
    else:
        h.peak = 13 * 1024**3
    with pytest.raises((RuntimeError, MemoryError)):
        rebuild.run(h.args, h.dependencies)
    failed = rebuild.read(h.args.report)
    assert failed['status'] == 'failed'
    assert failed['resources_verified'] is False
    assert not (h.state / 'generations' / (h.target.collection + '.json')).exists()
    if mode == 'over_budget':
        h.args.resume, h.peak = True, 128 * 1024**2
        with pytest.raises(MemoryError, match='prior attempt'):
            rebuild.run(h.args, h.dependencies)


def test_keyboard_interrupt_is_recorded_and_owned_target_can_resume(harness):
    h = harness
    original = h.dependencies.service
    def interrupt(settings):
        raise KeyboardInterrupt('fixture operator interruption')
    h.dependencies.service = interrupt
    with pytest.raises(KeyboardInterrupt):
        rebuild.run(h.args, h.dependencies)
    checkpoint = rebuild.read(h.args.report)
    assert checkpoint['status'] == 'failed'
    assert 'KeyboardInterrupt' in checkpoint['error']
    h.args.resume, h.dependencies.service = True, original
    assert rebuild.run(h.args, h.dependencies)['owner'] == checkpoint['owner']


def test_missing_visual_cache_is_explicit_and_never_silently_dropped(harness):
    h = harness
    (h.corpus / 'image.png').write_bytes(b'fixture image with mocked OCR')
    h.args.expected_ready = 3
    with pytest.raises(RuntimeError, match='failed documents'):
        rebuild.run(h.args, h.dependencies)
    report = rebuild.read(h.args.report)
    assert any('Visual description cache missing' in e['error'] for e in report['sync_result']['errors'])
    assert h.vision_calls == 0


def test_allow_vision_uses_existing_model_adapter_and_cache_survives_resume(harness):
    h = harness
    (h.corpus / 'image.png').write_bytes(b'fixture image with mocked OCR')
    h.args.expected_ready, h.args.allow_vision, h.fail_upserts = 3, True, 1
    with pytest.raises(RuntimeError, match='failed documents'):
        rebuild.run(h.args, h.dependencies)
    assert h.vision_calls == 1
    h.args.resume = True
    report = rebuild.run(h.args, h.dependencies)
    assert report['verification']['ready'] == 3
    assert h.vision_calls == 1
    assert any('Synthetic cached image description' in c['text'] for c in h.vectors.values())


def test_expected_ready_is_checked_against_originals_before_claim(harness):
    h = harness
    h.args.expected_ready = 38
    with pytest.raises(ValueError, match='expected-ready'):
        rebuild.run(h.args, h.dependencies)
    assert not h.args.report.exists()


def test_wrong_vector_count_never_finishes_generation(harness):
    h = harness
    original = h.dependencies.service
    def factory(settings):
        service = original(settings)
        service.retrieval.client.count = lambda *a, **k: SimpleNamespace(count=0)
        return service
    h.dependencies.service = factory
    with pytest.raises(RuntimeError, match='Vector count'):
        rebuild.run(h.args, h.dependencies)
    assert not (h.state / 'generations' / (h.target.collection + '.json')).exists()


def test_monitor_initialization_failure_is_persisted(harness):
    h = harness
    def unavailable(maximum):
        raise RuntimeError('fixture monitor cannot initialize')
    h.dependencies.monitor = unavailable
    with pytest.raises(RuntimeError, match='cannot initialize'):
        rebuild.run(h.args, h.dependencies)
    report = rebuild.read(h.args.report)
    assert report['status'] == 'failed'
    assert report['resources']['sampling_status'] == 'unavailable'
    assert not h.services


@pytest.mark.parametrize('removed', ['catalog', 'collection', 'generation'])
def test_completed_resume_refuses_missing_owned_resources(harness, removed):
    h = harness
    report = rebuild.run(h.args, h.dependencies)
    h.args.resume = True
    if removed == 'catalog':
        (h.state / h.target.catalog).unlink()
    elif removed == 'collection':
        h.collections.remove(h.target.collection)
    else:
        Path(report['generation']).unlink()
    with pytest.raises(ValueError, match='removed|lost'):
        rebuild.run(h.args, h.dependencies)


def test_hard_interruption_does_not_invent_complete_historical_rss(harness):
    h = harness
    h.fail_upserts = 1
    with pytest.raises(RuntimeError):
        rebuild.run(h.args, h.dependencies)
    report = rebuild.read(h.args.report)
    report['status'] = 'running'
    report['attempts'][-1]['status'] = 'running'
    report['attempts'][-1].pop('resources')
    rebuild.save(h.args.report, report)
    h.args.resume = True
    completed = rebuild.run(h.args, h.dependencies)
    assert completed['resource_history_complete'] is False
    assert completed['resources_verified'] is False
    assert completed['attempts'][0]['status'] == 'interrupted_without_final_measurement'
    assert rebuild.read(completed['generation'])['resource_history_complete'] is False


def test_live_checkpoint_pid_is_refused_without_process_kill(monkeypatch):
    fake = SimpleNamespace(is_running=lambda: True, create_time=lambda: 123.)
    monkeypatch.setattr('psutil.Process', lambda pid: fake)
    with pytest.raises(RuntimeError, match='live worker'):
        rebuild.assert_no_worker({'status': 'running', 'worker': {'pid': 456, 'create_time': 123.}})

