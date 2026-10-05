"""Build one explicitly owned candidate with normal parsing/chunking/sync.

No benchmark chunks, evaluation cases, rankings, or active-pointer writes are
used here. Importing this module does not open a database or load a model.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'normal-candidate-rebuild-v1'
OWNER_KEY = 'candidate_build_owner'
HISTORY_TABLES = ('calls', 'call_campaigns', 'metadata', 'jobs', 'traces')
PACKAGES = ('torch', 'transformers', 'sentence-transformers', 'tokenizers',
            'safetensors', 'numpy', 'psutil', 'filelock', 'qdrant-client',
            'httpx', 'pypdf', 'pillow', 'openpyxl', 'python-docx', 'jieba')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), default=str).encode()).hexdigest()


def save(path, value, *, exclusive=False):
    """Fsync a new claim or atomically replace an existing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')
    if exclusive:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '-',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def readonly(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Required source catalog/ledger does not exist: ' + str(path))
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        yield connection
    finally:
        connection.close()


def explicit_settings(path):
    from ragapp.experiments import load_settings
    values = read(path)
    if not isinstance(values, dict) or not {'collection', 'catalog'} <= set(values):
        raise ValueError('Use a full Settings JSON with explicit collection and catalog')
    return load_settings(path)


def environment_identity():
    paths = [*sorted((ROOT / 'ragapp').glob('*.py')),
             *sorted((ROOT / 'data_process').rglob('*.py')), Path(__file__),
             ROOT / 'scripts/benchmark_models.py', ROOT / 'scripts/download_candidate_models.py']
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'code_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
            'packages': packages,
            'python': {'executable': str(Path(sys.executable).resolve()),
                       'version': sys.version, 'executable_sha256': digest(sys.executable)}}


def model_identity(settings):
    directory = settings.state / 'models' / settings.embedding_directory
    if not directory.is_dir():
        raise ValueError('Embedding model directory is missing')
    files = {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and '.cache' not in p.relative_to(directory).parts}
    if not any(name.endswith(('.safetensors', '.bin')) for name in files):
        raise ValueError('Embedding weights are missing; this runner does not download models')
    manifest_file = directory / 'snapshot-manifest.json'
    manifest_verification = 'recorded_local_bytes'
    if manifest_file.exists():
        manifest = read(manifest_file)
        expected = {'repository': settings.embedding_model, 'revision': settings.embedding_revision,
                    'directory': settings.embedding_directory}
        if manifest.get('status') != 'complete' or any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Embedding snapshot manifest identity differs from Settings')
        for name, item in manifest.get('files', {}).items():
            if files.get(name) != item.get('sha256') or (directory / name).stat().st_size != item.get('bytes'):
                raise ValueError('Embedding snapshot member differs from its manifest: ' + name)
        if not manifest.get('files'):
            raise ValueError('Embedding snapshot manifest has no members')
        manifest_verification = 'snapshot_revision_and_all_local_bytes'
    elif settings.embedding_directory.startswith('candidate-'):
        raise ValueError('Managed candidate model requires its downloaded snapshot manifest')
    return {'model': settings.embedding_model, 'revision': settings.embedding_revision,
            'directory': str(directory.resolve()), 'files_sha256': files,
            'verification': manifest_verification}


def source_snapshot(source):
    """Copy only source ledger rows into memory, with an independent RO connection."""
    with readonly(source.state / source.catalog) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {'documents', 'metadata', 'calls', 'call_campaigns', 'jobs', 'traces'}
        if not required <= tables:
            raise ValueError('Source catalog is not a supported preserved generation')
        result = {}
        for table in ('documents', *HISTORY_TABLES):
            columns = [row[1] for row in db.execute('PRAGMA table_info(' + table + ')')]
            rows = [tuple(row) for row in db.execute('SELECT * FROM ' + table + ' ORDER BY 1')]
            result[table] = {'columns': columns, 'rows': rows}
    generation_path = source.state / 'generations' / (source.collection + '.json')
    generation = read(generation_path)
    metadata = dict(result['metadata']['rows'])
    if generation.get('collection') != source.collection or generation.get('catalog') != source.catalog:
        raise ValueError('Source generation collection/catalog mismatch')
    if not generation.get('index_fingerprint') or metadata.get('index_fingerprint') != generation['index_fingerprint']:
        raise ValueError('Source catalog fingerprint differs from its preserved generation')
    for key, value in generation.get('settings', {}).items():
        if not hasattr(source, key) or getattr(source, key) != value:
            raise ValueError('Source settings differ from the preserved generation: ' + key)
    return result, {'catalog': str((source.state / source.catalog).resolve()),
                    'generation_file': str(generation_path.resolve()),
                    'generation_sha256': digest(generation_path),
                    'index_fingerprint': generation['index_fingerprint'],
                    'tables_sha256': {name: fingerprint(value) for name, value in result.items()}}


def original_identity(settings, snapshot):
    from ragapp.pipeline import observe_sources
    docs = [dict(zip(snapshot['documents']['columns'], row)) for row in snapshot['documents']['rows']]
    _, inventory = observe_sources(settings.corpus, [])
    if inventory['status'] != 'available':
        raise ValueError('Original corpus inventory is unavailable or incomplete')
    files = {}
    for name in inventory['files']:
        path = settings.corpus / name
        before = path.stat()
        value = digest(path)  # Deliberately bypass the process-local stat/hash cache.
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Original changed while hashing: ' + name)
        files[name] = value
    excluded = sorted(d['path'].replace('\\', '/') for d in docs if d['excluded'])
    expected = sorted(set(files) - set(excluded))
    if not expected:
        raise ValueError('No non-excluded original documents to build')
    return {'files_sha256': files, 'excluded': excluded, 'expected_paths': expected}


def validate_paths(settings, source, report):
    from ragapp.config import Settings
    from urllib.parse import urlsplit
    for current in (settings, source):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,120}', current.collection):
            raise ValueError('Collection must be a simple managed name')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.sqlite3', current.catalog):
            raise ValueError('Catalog must be a simple .sqlite3 filename')
        if not current.state.is_dir() or not current.corpus.is_dir():
            raise ValueError('State and original corpus directories must already exist')
    for field in ('root', 'state', 'corpus'):
        if getattr(settings, field).resolve() != getattr(source, field).resolve():
            raise ValueError('Source and candidate must share the same ' + field)
    if settings.qdrant_url != source.qdrant_url:
        raise ValueError('Source and candidate must share Qdrant; a new endpoint could bypass ownership checks')
    endpoint = urlsplit(settings.qdrant_url)
    if endpoint.scheme != 'http' or endpoint.hostname not in {'localhost', '127.0.0.1', '::1'} or endpoint.username:
        raise ValueError('Only the existing local unauthenticated Qdrant endpoint is supported')
    defaults = Settings()
    protected_collections = {source.collection, defaults.collection}
    protected_catalogs = {source.catalog.casefold(), defaults.catalog.casefold(), 'billing.sqlite3'}
    active_path = settings.state / 'active.json'
    if active_path.exists():
        active = read(active_path)
        protected_collections.add(active['collection'])
        protected_catalogs.add(active.get('catalog', defaults.catalog).casefold())
    if settings.collection in protected_collections or settings.catalog.casefold() in protected_catalogs:
        raise ValueError('Target is a source, active, default, or billing resource')
    if not report.is_relative_to((settings.state / 'reports').resolve()) or report.suffix != '.json':
        raise ValueError('Report must be a new .json checkpoint under state/reports')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', settings.embedding_directory):
        raise ValueError('Embedding directory must be a managed model folder name')
    if not re.fullmatch('[0-9a-f]{40}', settings.embedding_revision):
        raise ValueError('Embedding revision must be an immutable 40-character commit')
    if not settings.campaign or settings.campaign != source.campaign:
        raise ValueError('Use the existing source campaign and shared billing ledger')
    if not math.isfinite(settings.campaign_limit) or not 0 < settings.campaign_limit <= min(source.campaign_limit, 100):
        raise ValueError('Candidate cannot increase the existing campaign budget above 100 CNY')
    if set(settings.campaign_stages) != set(source.campaign_stages):
        raise ValueError('Candidate campaign stages differ from the existing authorization')
    for name, limit in settings.campaign_stages.items():
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 < limit <= source.campaign_stages[name]:
            raise ValueError('Candidate cannot increase a campaign stage budget')
    with readonly(settings.state / 'billing.sqlite3') as db:
        db.execute('SELECT call_id,campaign FROM call_campaigns LIMIT 1').fetchall()


def default_probe(settings):
    """Require an already-running database; never create/start it during preflight."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=settings.qdrant_url, timeout=10)
    try:
        return {item.name for item in client.get_collections().collections}
    finally:
        client.close()


def worker_identity():
    import psutil
    return {'pid': os.getpid(), 'create_time': psutil.Process().create_time()}


def assert_no_worker(previous=None):
    import psutil
    if previous and previous.get('status') == 'running' and previous.get('worker'):
        owner = previous['worker']
        try:
            process = psutil.Process(owner['pid'])
            if process.is_running() and process.create_time() == owner['create_time']:
                raise RuntimeError('Checkpoint still belongs to a live worker: ' + str(owner['pid']))
        except psutil.NoSuchProcess:
            pass
    names = {'benchmark_models.py', 'rebuild_candidate.py', 'run-sequence.py',
             'run-embedding-sequence-v2.py'}
    for process in psutil.process_iter(['pid', 'name']):
        if process.pid == os.getpid() or 'python' not in (process.info.get('name') or '').lower():
            continue
        try:
            args = process.cmdline()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError('Cannot inspect another live Python worker') from exc
        if any(Path(arg).name in names or arg in {'ragapp.experiments', 'ragapp.server'} for arg in args):
            raise RuntimeError('Another indexing/model worker is active: ' + str(process.pid))


def default_service(settings):
    from ragapp.service import Service
    return Service(settings)


def default_monitor(maximum_gib):
    from scripts.benchmark_models import ResourceMonitor
    return ResourceMonitor(maximum_gib)


@dataclass
class Dependencies:
    """Test-only injection; reports from injected dependencies are labelled synthetic."""
    service: object = default_service
    probe: object = default_probe
    assert_no_worker: object = assert_no_worker
    monitor: object = default_monitor
    worker_identity: object = worker_identity
    environment: object = environment_identity


class CacheOnlyVision:
    def vision(self, *args, **kwargs):
        raise RuntimeError('Visual description cache missing; no image description was omitted. '
                           'This cache-only build cannot make a paid vision call. '
                           'Start with --allow-vision when campaign calls are authorized.')


def preserve_history(store, snapshot, owner):
    """Idempotent ledger migration; never import chunks or old pipeline metadata."""
    with store.db() as db:
        for table in HISTORY_TABLES:
            entry = snapshot[table]
            for row in entry['rows']:
                if table == 'metadata' and (row[0].endswith('fingerprint') or row[0] in
                        {'index_profile', 'index_schema', OWNER_KEY}):
                    continue
                db.execute('INSERT OR IGNORE INTO ' + table + '(' + ','.join(entry['columns']) + ') VALUES(' +
                           ','.join('?' for _ in row) + ')', row)
        entry = snapshot['documents']
        excluded_position = entry['columns'].index('excluded')
        for row in entry['rows']:
            if row[excluded_position]:
                db.execute('INSERT OR IGNORE INTO documents(' + ','.join(entry['columns']) + ') VALUES(' +
                           ','.join('?' for _ in row) + ')', row)
        db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (OWNER_KEY, json.dumps(owner, sort_keys=True)))


def verify_catalog_owner(path, owner):
    if not path.exists():
        return
    with readonly(path) as db:
        row = db.execute('SELECT value FROM metadata WHERE key=?', (OWNER_KEY,)).fetchone()
        if row is None:
            # A crash between Service creating its schema and the ownership write
            # is recoverable only if every mutable table is still empty.
            for table in ('documents', 'chunks', *HISTORY_TABLES):
                if db.execute('SELECT 1 FROM ' + table + ' LIMIT 1').fetchone():
                    raise ValueError('Existing candidate catalog lacks its ownership marker')
        elif json.loads(row[0]) != owner:
            raise ValueError('Candidate catalog ownership differs from the checkpoint')


def verify_resume_resources(target, collection_exists):
    if collection_exists and not target.exists():
        raise ValueError('Owned collection exists but its candidate catalog was removed; do not silently start over')
    if target.exists() and not collection_exists:
        with readonly(target) as db:
            if db.execute('SELECT 1 FROM chunks LIMIT 1').fetchone():
                raise ValueError('Owned catalog has chunks but its vector collection was removed; ordinary sync cannot repair that loss')


def merge_resources(report, resource):
    observed = resource.get('observed_peak_rss_bytes')
    if isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isfinite(observed):
        report['historical_peak_rss_bytes'] = max(report.get('historical_peak_rss_bytes', 0), observed)
    report['resources'] = resource


def resource_passed(report, budget):
    resource = report.get('resources', {})
    peak = resource.get('worker_children_qdrant_peak_rss_bytes')
    observed = resource.get('observed_peak_rss_bytes')
    return (resource.get('sampling_status') == 'measured' and resource.get('sampling_error') is None
            and type(peak) is int and 0 < peak <= budget
            and type(observed) is int and peak <= observed <= budget
            and resource.get('rss_budget_bytes') == budget
            and 0 < report.get('historical_peak_rss_bytes', 0) <= budget)


def verify_ready(service, originals, expected_count):
    expected = set(originals['expected_paths'])
    documents = service.store.documents()
    ready = {d['path'].replace('\\', '/'): d for d in documents if not d['excluded'] and d['status'] == 'ready'}
    visible = {d['path'].replace('\\', '/') for d in documents if not d['excluded'] and d['status'] != 'deleted'}
    if set(ready) != expected or visible != expected or len(ready) != expected_count:
        raise RuntimeError('Candidate is incomplete: ready document set differs from the frozen originals')
    for name, document in ready.items():
        if document['hash'] != originals['files_sha256'][name] or not document['version']:
            raise RuntimeError('Ready document has a wrong source hash/version: ' + name)
    with service.store.db() as db:
        chunk_count = db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        missing = db.execute('''SELECT d.path FROM documents d WHERE d.excluded=0 AND d.status='ready'
            AND NOT EXISTS(SELECT 1 FROM chunks c WHERE c.document_id=d.id AND c.version=d.version)''').fetchall()
        stale = db.execute('''SELECT COUNT(*) FROM chunks c LEFT JOIN documents d ON c.document_id=d.id
            WHERE d.id IS NULL OR d.excluded!=0 OR d.status!='ready' OR c.version!=d.version''').fetchone()[0]
    if not chunk_count or missing or stale:
        raise RuntimeError('Candidate chunks are missing or do not match ready document versions')
    vectors = service.retrieval.client.count(service.settings.collection, exact=True).count
    if vectors != chunk_count:
        raise RuntimeError('Vector count differs from the complete candidate chunk count')
    return {'expected_ready': expected_count, 'ready': len(ready), 'chunks': chunk_count, 'vectors': vectors}


def run(args, dependencies=None):
    """One attempt, no automatic retry, no publication. Exceptions are persisted."""
    from filelock import FileLock
    from ragapp.config import INDEX_SETTINGS
    from ragapp.pipeline import index_fingerprint

    synthetic = dependencies is not None
    dep = dependencies or Dependencies()
    settings = explicit_settings(args.settings)
    source = explicit_settings(args.source_settings)
    report_path = Path(args.report).resolve()
    if not math.isfinite(args.rss_gib) or not 0 < args.rss_gib <= 12:
        raise ValueError('RSS budget must be positive and at most 12 GiB')
    validate_paths(settings, source, report_path)
    settings_path, source_settings_path = Path(args.settings).resolve(), Path(args.source_settings).resolve()
    if report_path in {settings_path, source_settings_path}:
        raise ValueError('Report cannot overwrite a settings file')
    target = settings.state / settings.catalog
    generation_path = settings.state / 'generations' / (settings.collection + '.json')
    owner_path = settings.state / 'candidate-builds' / (settings.collection + '.owner.json')
    locks = [settings.state / 'server.lock', settings.state / 'candidate-build.lock',
             settings.state / 'reports/quality-v2/experiment-worker.lock',
             settings.state / 'reports/quality-v2-models/model-worker.lock',
             report_path.with_suffix('.lock')]
    with contextlib.ExitStack() as stack:
        for path in locks:
            path.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(FileLock(str(path), timeout=0))
        # The Indexer owns this lock during sync. Do not nest another held instance.
        with FileLock(str(settings.state / 'sync.lock'), timeout=0):
            pass
        previous = read(report_path) if report_path.exists() else None
        if bool(previous) != bool(args.resume):
            raise ValueError('Use --resume only with an existing checkpoint; never overwrite a report')
        dep.assert_no_worker(previous)
        snapshot, source_id = source_snapshot(source)
        originals = original_identity(settings, snapshot)
        expected_count = len(originals['expected_paths'])
        if args.expected_ready is not None and args.expected_ready != expected_count:
            raise ValueError('--expected-ready differs from the observed non-excluded originals')
        active = settings.state / 'active.json'
        active_hash = digest(active) if active.exists() else None
        identity = {'settings': json.loads(json.dumps(asdict(settings), default=str)),
                    'settings_file': str(settings_path), 'settings_sha256': digest(settings_path),
                    'source_settings_file': str(source_settings_path), 'source_settings_sha256': digest(source_settings_path),
                    'source': source_id, 'originals': originals, 'environment': dep.environment(),
                    'embedding': model_identity(settings), 'index_fingerprint': index_fingerprint(settings),
                    'active_pointer_sha256': active_hash, 'rss_gib': args.rss_gib,
                    'allow_vision': bool(args.allow_vision), 'expected_ready': expected_count,
                    'synthetic_dependencies': synthetic}
        collections = dep.probe(settings)
        if previous:
            if previous.get('schema') != SCHEMA or previous.get('identity') != identity:
                raise ValueError('Resume identity differs: configuration/code/packages/model/originals/source/policy changed')
            owner = previous['owner']
            if owner.get('report') != str(report_path) or owner.get('collection') != settings.collection or owner.get('catalog') != str(target.resolve()):
                raise ValueError('Checkpoint ownership does not identify this target/report')
            if owner_path.exists():
                if read(owner_path) != owner:
                    raise ValueError('Existing target claim belongs to another checkpoint')
            elif target.exists() or settings.collection in collections:
                raise ValueError('Candidate resources exist without their declared ownership claim')
            else:
                save(owner_path, owner, exclusive=True)
            verify_catalog_owner(target, owner)
            verify_resume_resources(target, settings.collection in collections)
            if generation_path.exists():
                generation = read(generation_path)
                if generation.get('build_id') != owner['build_id'] or generation.get('build_identity_sha256') != fingerprint(identity):
                    raise ValueError('Generation output belongs to another build')
            if previous.get('status') in {'complete', 'complete_synthetic'}:
                if not generation_path.exists() or not target.exists() or settings.collection not in collections:
                    raise ValueError('Completed checkpoint has lost one of its generation resources')
                return previous  # Idempotent completed checkpoint: no sync/model calls.
            report = previous
            if report.get('status') == 'running' and report.get('attempts'):
                report['attempts'][-1].update(status='interrupted_without_final_measurement',
                                              observed_interruption_at_utc=now())
        else:
            if target.exists() or settings.collection in collections or owner_path.exists() or generation_path.exists():
                raise ValueError('Target catalog/collection/claim/generation already exists without this checkpoint')
            owner = {'build_id': uuid.uuid4().hex, 'collection': settings.collection,
                     'catalog': str(target.resolve()), 'report': str(report_path),
                     'identity_sha256': fingerprint(identity)}
            report = {'schema': SCHEMA, 'status': 'claimed', 'owner': owner, 'identity': identity,
                      'job_id': str(uuid.uuid4()), 'created_at_utc': now(), 'attempts': [],
                      'published': False, 'publication_eligible': False,
                      'historical_peak_rss_bytes': 0, 'synthetic_dependencies': synthetic}
            # Both durable claims precede Service, SQLite schema, or collection creation.
            save(report_path, report, exclusive=True)
            save(owner_path, owner, exclusive=True)
        if report['historical_peak_rss_bytes'] > args.rss_gib * 1024**3:
            raise MemoryError('A prior attempt exceeded this immutable RSS budget; it cannot be erased by resume')
        attempt = {'number': len(report['attempts']) + 1, 'started_at_utc': now(),
                   'worker': dep.worker_identity(), 'status': 'running'}
        report['attempts'].append(attempt)
        report.update(status='running', worker=attempt['worker'], worker_pid=attempt['worker']['pid'],
                      collection=settings.collection, catalog=str(target.resolve()),
                      phase='initializing', updated_at_utc=now())
        save(report_path, report)
        monitor = None
        service = None
        original_job = None
        try:
            monitor = dep.monitor(args.rss_gib)
            with monitor:
                monitor.check()
                service = dep.service(settings)
                preserve_history(service.store, snapshot, owner)
                if not args.allow_vision:
                    service.indexer.models = CacheOnlyVision()  # Truthy: cached descriptions remain included.
                original_job = service.store.job

                def checkpoint_job(status='pending', progress=None, job_id=None):
                    result = original_job(status, progress, job_id)
                    report.update(phase='sync', job=service.store.get_job(result), updated_at_utc=now())
                    merge_resources(report, monitor.report())
                    save(report_path, report)
                    if status != 'failed':
                        monitor.check()
                    return result

                service.store.job = checkpoint_job
                # Preflight already required an available database and recorded ownership.
                service.start()
                monitor.check()
                result = service.indexer.sync(job_id=report['job_id'])
                report['sync_result'] = result
                if result.get('failed'):
                    raise RuntimeError('Sync left failed documents; inspect sync_result/errors and resume the same target')
                report.update(phase='verifying', updated_at_utc=now())
                report['verification'] = verify_ready(service, originals, expected_count)
                _, latest_source = source_snapshot(source)
                if latest_source != source_id or original_identity(settings, snapshot) != originals:
                    raise RuntimeError('Source catalog or originals changed during this build')
                if (digest(active) if active.exists() else None) != active_hash:
                    raise RuntimeError('Active pointer changed during this build')
                if dep.environment() != identity['environment'] or model_identity(settings) != identity['embedding']:
                    raise RuntimeError('Code/package/model identity changed during this build')
                with service.store.db() as db:
                    saved = db.execute("SELECT value FROM metadata WHERE key='index_fingerprint'").fetchone()
                if saved is None or saved[0] != identity['index_fingerprint']:
                    raise RuntimeError('Candidate pipeline fingerprint was not established by normal sync')
                monitor.check()
            merge_resources(report, monitor.report())
            if not resource_passed(report, args.rss_gib * 1024**3):
                raise RuntimeError('No valid measured RSS result; candidate completion is not certified')
            # A hard-killed attempt has an unobserved tail. Preserve that limitation
            # even when a later document-level resume succeeds below the budget.
            report['resource_history_complete'] = all(
                a.get('status') != 'interrupted_without_final_measurement' and
                a.get('resources', {}).get('sampling_status') == 'measured' and
                type(a.get('resources', {}).get('worker_children_qdrant_peak_rss_bytes')) is int and
                a['resources']['worker_children_qdrant_peak_rss_bytes'] > 0
                for a in report['attempts'][:-1])
            report['resources_verified'] = not synthetic and report['resource_history_complete']
            report['current_attempt_resources_verified'] = not synthetic
            report['resource_measurement_kind'] = 'synthetic_fixture' if synthetic else 'actual_process_rss'
            generation = {'collection': settings.collection, 'catalog': settings.catalog,
                          'settings': {key: getattr(settings, key) for key in sorted(INDEX_SETTINGS)},
                          'index_fingerprint': identity['index_fingerprint'], 'build_id': owner['build_id'],
                          'build_identity_sha256': fingerprint(identity), 'build_report': str(report_path),
                          'synthetic': synthetic, 'published': False,
                          'resource_history_complete': report['resource_history_complete']}
            save(generation_path, generation)
            report.update(status='complete_synthetic' if synthetic else 'complete', phase='complete',
                          generation=str(generation_path.resolve()), error=None, updated_at_utc=now())
            attempt.update(status=report['status'], finished_at_utc=now(), resources=report['resources'])
            save(report_path, report)
            return report
        except BaseException as exc:
            if monitor is not None:
                try:
                    merge_resources(report, monitor.report())
                except BaseException as measurement_error:
                    report['resources'] = {'sampling_status': 'unavailable', 'sampling_error': repr(measurement_error)}
            else:
                report['resources'] = {'sampling_status': 'unavailable', 'sampling_error': 'Monitor initialization failed'}
            report.update(status='failed', phase='failed', error=type(exc).__name__ + ': ' + str(exc),
                          resources_verified=False, updated_at_utc=now())
            attempt.update(status='failed', error=report['error'], finished_at_utc=now(), resources=report['resources'])
            if service is not None:
                try:
                    report['job'] = service.store.get_job(report['job_id'])
                except BaseException as job_error:
                    report['job_read_error'] = repr(job_error)
            try:
                save(report_path, report)
            except BaseException as checkpoint_error:
                emergency = report_path.with_name(report_path.stem + '.emergency-' + owner['build_id'] + '.json')
                save(emergency, {'owner': owner, 'error': report['error'],
                                 'checkpoint_error': repr(checkpoint_error), 'state': report})
            raise
        finally:
            if service is not None and original_job is not None:
                service.store.job = original_job


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--settings', type=Path, required=True, help='Explicit new candidate Settings JSON')
    result.add_argument('--source-settings', type=Path, required=True, help='Settings JSON of a preserved source generation')
    result.add_argument('--report', type=Path, required=True, help='New persistent checkpoint under state/reports')
    result.add_argument('--resume', action='store_true', help='Resume the same generation and immutable identity')
    result.add_argument('--allow-vision', action='store_true', help='Allow cache-miss vision calls under the existing campaign; default cache-only')
    result.add_argument('--rss-gib', type=float, default=12)
    result.add_argument('--expected-ready', type=int, help='Optional explicit count, e.g. 38; otherwise derive from frozen originals/exclusions')
    return result


def main():
    args = parser().parse_args()
    report = run(args)
    print(json.dumps({key: report.get(key) for key in ('status', 'owner', 'job_id', 'verification',
                     'historical_peak_rss_bytes', 'resources_verified', 'generation', 'published')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
