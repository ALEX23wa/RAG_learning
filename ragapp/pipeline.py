"""Stable provenance for caches, generations and resumable experiments. No credentials."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import re
import threading


_SOURCE_HASH_CACHE = {}
_SOURCE_HASH_LOCK = threading.Lock()


def _observed_source(root, name):
    """Observe an original, caching only unchanged local stat identities.

    Cache keys include size, nanosecond modification/change times, device and
    inode. This is a local performance optimization, not tamper-proof evidence;
    publication separately rehashes originals without this cache.
    """
    path = root / name
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            return {'status': 'outside_corpus', 'hash': None}
        if path.is_symlink():
            return {'status': 'symlink', 'hash': None}
        stat = path.stat()
        if not path.is_file():
            return {'status': 'not_a_file', 'hash': None}
        def identity(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        stamp = identity(stat)
        key = str(resolved)
        with _SOURCE_HASH_LOCK:
            cached = _SOURCE_HASH_CACHE.get(key)
        if cached and cached[0] == stamp:
            return {'status': 'available', 'hash': cached[1]}
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(part)
        if identity(path.stat()) != stamp:
            return {'status': 'changed_during_read', 'hash': None}
        value = digest.hexdigest()
        with _SOURCE_HASH_LOCK:
            if len(_SOURCE_HASH_CACHE) > 8192:
                _SOURCE_HASH_CACHE.clear()
            _SOURCE_HASH_CACHE[key] = stamp, value
        return {'status': 'available', 'hash': value}
    except FileNotFoundError:
        return {'status': 'missing', 'hash': None}
    except OSError as exc:
        return {'status': 'read_error', 'hash': None, 'error': type(exc).__name__}


def observe_sources(corpus, document_paths):
    """Read-only inventory/hash observation, without PDF parsing or indexing."""
    from .parsers import SUPPORTED
    root = Path(corpus).resolve()
    files, errors = [], []
    if not root.is_dir():
        inventory = {'status': 'unavailable', 'files': [], 'errors': []}
    else:
        def record_error(exc):
            try:
                name = Path(exc.filename).relative_to(root).as_posix() if exc.filename else '.'
            except ValueError:
                name = '.'
            errors.append({'path': name, 'error': type(exc).__name__})
        for folder, directories, names in os.walk(root, onerror=record_error, followlinks=False):
            directories[:] = [name for name in directories if not name.startswith('.')
                and name not in {'extracted', '__pycache__', 'node_modules'} and not (Path(folder) / name).is_symlink()]
            for name in names:
                path = Path(folder) / name
                if name.startswith(('.', '~', '$')) or path.is_symlink():
                    continue
                if re.search(r'(?i)(^|[._-])(secrets?|credentials?|api[_-]?key|private[_-]?key)([._-]|$)', name):
                    continue
                if path.suffix.lower() in SUPPORTED and path.resolve().is_relative_to(root):
                    files.append(path.relative_to(root).as_posix())
        inventory = {'status': 'read_error' if errors else 'available', 'files': sorted(files),
                     'errors': sorted(errors, key=lambda item: (item['path'], item['error']))}
    names = set(files) | {str(name).replace('\\', '/') for name in document_paths}
    observed = {name: _observed_source(root, name) for name in sorted(names)}
    return observed, inventory


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), default=str).encode()).hexdigest()


def parsing_fingerprint(settings):
    return fingerprint({'schema':'structure-v2', 'mode':settings.pdf_mode,
                        'overrides':settings.pdf_overrides})


def embedding_fingerprint(settings):
    value={'model':settings.embedding_model,'revision':settings.embedding_revision,
        'dimension':settings.embedding_dimension,'max_tokens':settings.embedding_max_tokens,
        'input_tokens':settings.embedding_input_tokens,'normalized':True,'adapter':'official-v1'}
    precision=getattr(settings,'embedding_precision','legacy')
    if precision not in ('legacy','float32'):
        raise ValueError('Unsupported embedding precision')
    if precision=='float32':
        value.update(parameter_dtype='torch.float32',cache_schema='verified-document-vector-v1')
    return fingerprint(value)


def index_fingerprint(settings):
    return fingerprint({'schema':settings.index_schema,'parser':parsing_fingerprint(settings),
        'chunker':'structure-v5-full-embedding-input-budget','chunk_tokens':settings.chunk_tokens,'overlap':settings.overlap,
        'embedding':embedding_fingerprint(settings)})


def embedding_text(chunk):
    locator=chunk.get('locator',{})
    if isinstance(locator,str): locator=json.loads(locator)
    return chunk.get('retrieval_text') or locator.get('retrieval_text') or chunk['text']


def runtime_manifest(settings, store):
    configuration={k:v for k,v in asdict(settings).items()
                   if k not in {'root','state','corpus','load_active','campaign_stages','campaign_limit'}}
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(Path(__file__).parent.glob('*.py'))}
    documents={d['path']:{k:d[k] for k in ('id','hash','version','status','excluded')}
               for d in store.documents()}
    observed,inventory=observe_sources(settings.corpus,documents)
    with store.db() as db:
        indexed=dict(db.execute("SELECT key,value FROM metadata WHERE key LIKE '%fingerprint' OR key='index_profile'"))
    data={'configuration':configuration,'code':code,'documents':documents,
          'observed_sources':observed,'source_inventory':inventory,
          'index':indexed,'index_fingerprint':indexed.get('index_fingerprint','legacy_unversioned'),
          'configured_index_fingerprint':index_fingerprint(settings)}
    data['source_fingerprints']={hashlib.sha256(path.replace('\\','/').encode()).hexdigest():
        {'hash':observed[path.replace('\\','/')]['hash'],'indexed_hash':doc['hash'],
         'observed_status':observed[path.replace('\\','/')]['status'],
         'status':doc['status'],'excluded':doc['excluded']} for path,doc in documents.items()}
    data['runtime_fingerprint']=fingerprint(data)
    return data
