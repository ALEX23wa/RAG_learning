"""CPU-only, immutable development-set model comparisons. Never calls generation APIs.

prepare freezes chunks and lexical pools; embed compares models on those chunks;
pool merges ONE declared embedding baseline with BM25; rerank uses that fixed pool.
Results from a sampled corpus are explicitly exploratory, never acceptance scores.
Embedding documents checkpoint after every batch (default 4, CPU). Resume the
same command with --resume; suite, model bytes, code, packages and settings must
match. --cache-dir can share validated document windows with a new identical
report. Query vectors are always inferred live, with an excluded session warmup.
--reuse-index-vectors is an optional E5-only sampled-consistency optimization;
cached/imported document timings must not be presented as encoding throughput.
v4 requires actual FP32 embedding parameters; validated index imports use a
separate storage batch (default 128). v3 reports remain readable fixed-pool inputs,
but v4 execution requires its own report/cache epoch.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragapp.evaluation import evidence_groups, load_dataset, span_matches
from ragapp.pipeline import embedding_text, fingerprint
from scripts.download_candidate_models import MODELS, sha256, verify_manifest

BASELINES = {
    'e5': {'repository': 'intfloat/multilingual-e5-base', 'revision': 'd128750597153bb5987e10b1c3493a34e5a4502a',
           'directory': 'embedding', 'kind': 'embedding', 'dimension': 768},
    'bge-reranker': {'repository': 'BAAI/bge-reranker-v2-m3', 'revision': '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
                     'directory': 'reranker', 'kind': 'reranker'},
}
ALL_MODELS = {**BASELINES, **MODELS}
REPORT_SCHEMA = 'model-benchmark-v4'
READABLE_REPORT_SCHEMAS = {'model-benchmark-v3', REPORT_SCHEMA}
EMBEDDING_DTYPE = 'torch.float32'
RESOURCE_PROTOCOL = 'target-pid-rss-v2'
RERANK_BATCHING_PROTOCOL = 'native-predict-all-pairs-forward-rss-v1'
IMPORT_BATCH_SIZE = 128


def save(path, data):
    # Suite arrays and batch markers use the same bounded, recoverable writer.
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
            prefix='.' + path.name + '.checkpoint-', suffix='.json', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
        stream.flush(); os.fsync(stream.fileno())
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if attempt == 4:
                raise RuntimeError('Checkpoint replacement blocked; recover: ' + str(temporary.resolve())) from exc
            time.sleep(.05 * 2**attempt)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_vectors(values, rows, dimension):
    """Reject malformed, nonfinite, unnormalized cache or model output."""
    import numpy as np
    array = np.asarray(values)
    if array.shape != (rows, dimension) or array.dtype.kind != 'f':
        raise ValueError('Vector shape/dtype differs from the frozen model dimension')
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1., rtol=0, atol=2e-4):
        raise ValueError('Vectors must be finite and normalized')
    return array


class DocumentVectorCache:
    """An immutable identity plus individually committed document-window batches.

    A batch JSON is the commit marker and checksums its NPY and each vector.
    An interrupted NPY without a marker is ignored; committed corruption fails
    closed. Queries never pass through this cache. Separate reports may explicitly
    reuse one cache only when their complete experiment contracts are identical.
    """
    def __init__(self, folder, contract, dimension):
        import numpy as np
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension; self.contract = contract; self.vectors = {}; self.sources = {}
        identity = {'schema': 'document-window-cache-v1', 'contract': contract,
                    'contract_fingerprint': fingerprint(contract), 'dimension': dimension}
        path = self.folder / 'manifest.json'
        if path.exists():
            if read(path) != identity:
                raise ValueError('Document cache suite/model/code/config identity changed')
        else:
            if any(self.folder.glob('batch-*')):
                raise ValueError('Document cache has batches but no identity manifest')
            save(path, identity)
        for commit in sorted(self.folder.glob('batch-*.json')):
            batch = read(commit); binary = commit.with_suffix('.npy')
            if batch.get('fingerprint') != fingerprint({k: v for k, v in batch.items() if k != 'fingerprint'}):
                raise ValueError('Document batch metadata checksum mismatch')
            if batch.get('contract_fingerprint') != identity['contract_fingerprint'] or sha256(binary) != batch.get('sha256'):
                raise ValueError('Document batch checksum/identity mismatch')
            keys = batch.get('keys')
            if not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
                raise ValueError('Invalid document batch keys')
            vectors = checked_vectors(np.load(binary, allow_pickle=False, mmap_mode='r'), len(keys), dimension)
            checksums = [hashlib.sha256(row.tobytes()).hexdigest() for row in vectors]
            if checksums != batch.get('vector_sha256') or batch.get('source') not in {'encoded', 'index_verified'}:
                raise ValueError('Document vector checksum/source mismatch')
            for key, vector in zip(keys, vectors):
                if key in self.vectors:
                    raise ValueError('Duplicate committed document window')
                self.vectors[key] = vector; self.sources[key] = batch['source']

    def key(self, document_text, window_input):
        # Includes both original retrieval text and fully templated window bytes.
        return fingerprint({'contract': fingerprint(self.contract), 'dimension': self.dimension,
            'text_sha256': hashlib.sha256(document_text.encode()).hexdigest(),
            'input_sha256': hashlib.sha256(window_input.encode()).hexdigest()})

    def add(self, keys, vectors, *, source='encoded', provenance=None):
        import numpy as np
        if not keys or len(set(keys)) != len(keys) or any(key in self.vectors for key in keys):
            raise ValueError('Document batch must contain unique uncached windows')
        if source not in {'encoded', 'index_verified'}:
            raise ValueError('Invalid document cache source')
        vectors = checked_vectors(vectors, len(keys), self.dimension)
        binary = self.folder / ('batch-' + uuid.uuid4().hex + '.npy')
        with tempfile.NamedTemporaryFile(dir=self.folder, prefix='.vector-checkpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, vectors, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(binary)
        marker = {'contract_fingerprint': fingerprint(self.contract), 'keys': keys,
            'sha256': sha256(binary), 'vector_sha256': [hashlib.sha256(v.tobytes()).hexdigest() for v in vectors],
            'source': source, 'provenance': provenance}
        marker['fingerprint'] = fingerprint(marker)
        save(binary.with_suffix('.json'), marker)
        for key, vector in zip(keys, vectors):
            self.vectors[key] = vector; self.sources[key] = source


def embedding_contract(report):
    from ragapp.config import Settings
    from ragapp.model_adapters import embedding_inputs
    settings = Settings(embedding_model=report['model']['repository'])
    return {key: report[key] for key in ('suite_fingerprint', 'model', 'settings', 'code_sha256', 'packages')} | {
        'input_templates': {'document': embedding_inputs(settings, ['{text}'], False)[0],
                            'query': embedding_inputs(settings, ['{text}'], True)[0]},
        'normalize_embeddings': True, 'window_overlap': 32, 'budget_safety_tokens': 8,
        'dtype': 'float32', 'query_cache': False,
        'embedding_precision': report.get('embedding_precision')}


def checked_embedding_precision(model, report):
    """A float32 NumPy result alone does not prove float32 computation."""
    actual = sorted({str(parameter.dtype) for parameter in model.parameters()})
    value = {'requested_parameter_dtype': EMBEDDING_DTYPE,
             'actual_parameter_dtypes': actual, 'validation': 'all_parameters_checked'}
    if actual != [EMBEDDING_DTYPE]:
        raise ValueError('CPU embedding parameter dtype differs from required torch.float32: '+str(actual))
    if report.get('embedding_precision') not in (None, value):
        raise ValueError('Resume embedding parameter precision changed')
    return value


def cached_document_matrix(cache, inputs, keys, model, batch_size, monitor, checkpoint):
    """Commit every newly inferred batch before reporting progress."""
    import numpy as np
    missing = list(dict.fromkeys(key for key in keys if key not in cache.vectors))
    texts = dict(zip(keys, inputs))
    missing_set = set(missing)
    reused = len(keys) - sum(key in missing_set for key in keys)
    encoded = 0; inference_seconds = 0.
    for start in range(0, len(missing), batch_size):
        monitor.check(); batch = missing[start:start+batch_size]
        started = time.perf_counter()
        vectors = model.encode([texts[key] for key in batch], batch_size=batch_size,
                               normalize_embeddings=True, show_progress_bar=False)
        monitor.check()
        inference_seconds += time.perf_counter() - started
        cache.add(batch, vectors)
        encoded += len(batch)
        checkpoint({'document_windows_cached': sum(key in cache.vectors for key in keys),
                    'document_windows_total': len(keys), 'new_unique_windows_encoded': encoded})
    return np.stack([cache.vectors[key] for key in keys]), {
        'cached_window_occurrences': reused, 'new_unique_windows_encoded': encoded,
        'new_document_inference_seconds': inference_seconds,
        'index_reused_window_occurrences': sum(cache.sources[key] == 'index_verified' for key in keys),
        'document_throughput_eligible': reused == 0 and not any(cache.sources[key] == 'index_verified' for key in keys)}


def reuse_index_vectors(args, suite, report, cache, documents, model, monitor, checkpoint=None):
    """Opt-in E5 reuse, with exact inputs and deterministic recomputation checks.

    Production caches do not authenticate every historic model byte. This is
    explicitly a sampled consistency check, whose proof and each imported file
    checksum are kept in the benchmark cache. It is not a throughput experiment.
    """
    import numpy as np
    from ragapp.config import Settings
    from ragapp.model_adapters import embedding_inputs
    from ragapp.pipeline import embedding_fingerprint
    if args.model != 'e5' or args.max_tokens != 512:
        raise ValueError('Index vector reuse only supports the pinned E5 model with its 512-token limit')
    spec = ALL_MODELS['e5']
    settings = Settings(state=args.state, embedding_model=spec['repository'], embedding_revision=spec['revision'],
                        embedding_directory=spec['directory'], embedding_dimension=spec['dimension'])
    expected = embedding_fingerprint(settings)
    if suite.get('index_metadata', {}).get('embedding_fingerprint') != expected:
        raise ValueError('Index embedding metadata differs from the official pinned E5 input configuration')
    directory = args.state / 'cache' / ('vectors-' + expected)
    candidates = {}
    def progress(stage, completed, total):
        monitor.check()
        if checkpoint:
            checkpoint({'stage': stage, 'completed': completed, 'total': total,
                        'eligible_unique_vectors': len(candidates)})
    for index, document in enumerate(documents):
        if index % 128 == 0: progress('index_source_validation', index, len(documents))
        if (len(document['keys']) != 1 or document['keys'][0] in cache.vectors
                or document['inputs'] != embedding_inputs(settings, [document['text']], False)):
            continue  # Boundary-stripped or multi-window inputs are not equivalent.
        key = document['keys'][0]
        path = directory / (hashlib.sha256(document['text'].encode()).hexdigest() + '.npy')
        if not path.is_file():
            continue
        before = sha256(path)
        raw = np.load(path, allow_pickle=False, mmap_mode='r')
        if raw.shape != (cache.dimension,):
            raise ValueError('Production vector dimension/shape mismatch')
        vector = checked_vectors(raw[None, :], 1, cache.dimension)[0].copy()
        if sha256(path) != before:
            raise ValueError('Production vector changed during verification')
        candidates[key] = {'input': document['inputs'][0], 'vector': vector, 'file_sha256': before,
                           'text_sha256': hashlib.sha256(document['text'].encode()).hexdigest()}
    progress('index_source_validation', len(documents), len(documents))
    sampled = sorted(candidates)[:min(8, len(candidates))]
    deltas = []
    for start in range(0, len(sampled), args.batch_size):
        monitor.check(); keys = sampled[start:start+args.batch_size]
        fresh = checked_vectors(model.encode([candidates[key]['input'] for key in keys], batch_size=args.batch_size,
            normalize_embeddings=True, show_progress_bar=False), len(keys), cache.dimension)
        monitor.check()
        saved = np.stack([candidates[key]['vector'] for key in keys])
        if not np.allclose(fresh, saved, rtol=1e-4, atol=2e-5):
            raise ValueError('Production vector subset disagrees with current pinned model; no index vectors imported')
        deltas.append(float(np.max(np.abs(fresh-saved))))
        progress('index_subset_recomputation', start+len(keys), len(sampled))
    proof = {'verification': 'exact_single_window_and_recomputed_subset', 'embedding_fingerprint': expected,
        'model_files_sha256': report['model']['files_sha256'], 'revision': spec['revision'],
        'sampled_keys': sampled, 'maximum_sample_absolute_delta': max(deltas) if deltas else None,
        'rtol': 1e-4, 'atol': 2e-5,
        'limitation': 'Sampled consistency, not authentication of every historic vector/model file'}
    ordered = sorted(candidates)
    storage_batch = getattr(args, 'import_batch_size', IMPORT_BATCH_SIZE)
    if type(storage_batch) is not int or not 1 <= storage_batch <= 1024:
        raise ValueError('Invalid index import storage batch size')
    progress('index_import_persistence', 0, len(ordered))
    for start in range(0, len(ordered), storage_batch):
        monitor.check(); keys = ordered[start:start+storage_batch]
        cache.add(keys, np.stack([candidates[key]['vector'] for key in keys]), source='index_verified',
            provenance={**proof, 'source_vectors': {key: {field: candidates[key][field]
                for field in ('file_sha256', 'text_sha256')} for key in keys}})
        progress('index_import_persistence', start+len(keys), len(ordered))
    return {**proof, 'new_unique_vectors_reused': len(ordered),
            'import_storage_batch_size': storage_batch,
            'total_committed_index_vectors': sum(source == 'index_verified' for source in cache.sources.values()),
            'document_throughput_eligible': False}


def development_cases(path, limit=0):
    # Reject by both filename and content; there is deliberately no override flag.
    if any(word in Path(path).stem.lower() for word in ('blind', 'acceptance', 'test')):
        raise ValueError('Model selection only accepts a development dataset')
    cases, digest, manifest = load_dataset(path)
    if any(c.get('split') not in (None, 'dev', 'development') for c in cases):
        raise ValueError('Holdout/blind cases cannot be used for model selection')
    if not cases or (not all(c.get('split') in ('dev', 'development') for c in cases)
                     and 'dev' not in Path(path).stem.lower()):
        raise ValueError('Dataset must explicitly identify the development split')
    selected = [c for c in cases if c.get('answerable')]
    # Diverse deterministic interleave, independent of labels and retrieved scores.
    buckets = {}
    for case in sorted(selected, key=lambda c: c['id']):
        buckets.setdefault(case.get('category', 'unknown'), []).append(case)
    selected = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                selected.append(buckets[key].pop(0))
    return (selected[:limit] if limit else selected), digest, manifest


def retrieval_question(case):
    # Isolate model comparison from LLM planning. For explicit followups retain the
    # current question FIRST and the last user turn; each is windowed, never cut.
    previous = next((h.get('content', '') for h in reversed(case.get('history', []))
                     if h.get('role') == 'user'), '')
    return case['question'] + ('\nPrevious question: ' + previous if previous else '')


def prepare(args):
    from ragapp.store import terms
    if args.catalog is None:
        active = args.state / 'active.json'
        catalog_name = read(active).get('catalog', 'catalog.sqlite3') if active.exists() else 'catalog.sqlite3'
        args.catalog = args.state / catalog_name
    if args.suite.exists() and any(args.suite.iterdir()):
        raise ValueError('Suite directory is not empty; create a new frozen suite')
    cases, dataset_hash, dataset_manifest = development_cases(args.dataset, args.limit)
    with sqlite3.connect(args.catalog.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("""SELECT c.*,d.path FROM chunks c JOIN documents d
            ON d.id=c.document_id AND c.version=d.version WHERE d.excluded=0
            AND d.status IN ('ready','update_failed') ORDER BY c.id""")]
        source_rows = [dict(r) for r in db.execute('SELECT id,path,hash,version,status,excluded FROM documents ORDER BY path')]
        lexical = {}
        active = {r['id'] for r in rows}
        for case in cases:
            tokens = list(dict.fromkeys(terms(retrieval_question(case)).split()))
            expression = ' OR '.join('"' + t.replace('"', '""') + '"' for t in tokens)
            found = db.execute('SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 2000',
                               (expression,)).fetchall() if expression else []
            lexical[case['id']] = [r[0] for r in found if r[0] in active][:80]
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
    if not rows:
        raise ValueError('No active chunks in catalog')
    for row in rows:
        row['locator'] = json.loads(row['locator']) if isinstance(row['locator'], str) else row['locator']
    if dataset_manifest and dataset_manifest.get('sources'):
        documents = {r['path'].replace('\\', '/'): r for r in source_rows}
        for path, source in dataset_manifest['sources'].items():
            actual = documents.get(path.replace('\\', '/'))
            if not actual or actual['hash'] != (source.get('hash') or source.get('sha256')):
                raise ValueError('Dataset source hash differs from catalog: ' + path)
    total = len(rows)
    if args.corpus_limit:
        # Never use gold spans to select the searchable corpus.
        chosen = list(dict.fromkeys(i for rank in range(80) for c in cases
                                   for i in lexical[c['id']][rank:rank+1]))
        already = set(chosen)
        chosen += [r['id'] for r in rows if r['id'] not in already]
        allowed = set(chosen[:args.corpus_limit])
        rows = [r for r in rows if r['id'] in allowed]
    active = {r['id'] for r in rows}
    lexical = {key: [i for i in values if i in active] for key, values in lexical.items()}
    files = {'corpus.json': rows, 'cases.json': cases, 'lexical.json': lexical}
    args.suite.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        save(args.suite / name, value)
    manifest = {'schema': 'model-benchmark-v1', 'dataset_sha256': dataset_hash,
        'dataset_manifest': dataset_manifest, 'documents': source_rows, 'index_metadata': metadata,
        'catalog': str(args.catalog.resolve()), 'full_active_chunks': total, 'chunks': len(rows),
        'cases': len(cases), 'corpus_sampled': len(rows) != total,
        'query_strategy': 'current-first plus last user turn, token-window max; no generation or gold injection',
        'corpus_selection': 'full active corpus' if len(rows) == total else 'round-robin BM25 then stable ID, no gold injection',
        'files': {name: sha256(args.suite / name) for name in files}}
    manifest['fingerprint'] = fingerprint(manifest)
    save(args.suite / 'manifest.json', manifest)
    return manifest


def load_suite(folder):
    manifest = read(folder / 'manifest.json')
    clean = {k: v for k, v in manifest.items() if k != 'fingerprint'}
    if fingerprint(clean) != manifest['fingerprint']:
        raise ValueError('Suite manifest changed')
    for name, digest in manifest['files'].items():
        if sha256(folder / name) != digest:
            raise ValueError('Frozen suite changed: ' + name)
    cases = read(folder / 'cases.json')
    if any(c.get('split') not in (None, 'dev', 'development') for c in cases):
        raise ValueError('Blind data in model comparison suite')
    return manifest, read(folder / 'corpus.json'), cases, read(folder / 'lexical.json')


def literal_coverability(case, corpus):
    all_groups = evidence_groups(case)
    groups = [g for g in all_groups if any(e.get('span_text') or e.get('quote') for e in g)]
    if not groups:
        return {'eligible': False, 'reason': 'No verified text span (legacy or visual case)'}
    available = [False] * len(groups); ideal_count = 0
    for row in corpus:
        matched = [any(span_matches(gold,row) for gold in group) for group in groups]
        available = [a or b for a,b in zip(available,matched)]
        ideal_count += any(matched)
    return {'eligible': True, 'available': available, 'ideal_count': ideal_count,
            'gold_groups': len(groups), 'groups_without_text_span': len(all_groups)-len(groups)}


class LiteralCoverageCache:
    """Full-corpus gold coverability, frozen by suite and literal scoring code."""
    def __init__(self, folder, suite_fingerprint, scoring_code, corpus):
        self.contract = {'suite_fingerprint':suite_fingerprint,'scoring_code':scoring_code,
                         'matcher':'strict_NFC_nonwhitespace_literal'}
        self.folder = Path(folder)/fingerprint(self.contract); self.corpus=corpus; self.memory={}
    def get(self,case):
        key=fingerprint(case)
        if key in self.memory:return self.memory[key]
        path=self.folder/(key+'.json')
        if path.exists():
            item=read(path)
            if (item.get('contract')!=self.contract or item.get('case_fingerprint')!=key
                    or item.get('checksum')!=fingerprint({k:v for k,v in item.items() if k!='checksum'})):
                raise ValueError('Literal coverability cache changed')
            value=item['value']
        else:
            value=literal_coverability(case,self.corpus)
            item={'contract':self.contract,'case_fingerprint':key,'value':value}
            item['checksum']=fingerprint(item);save(path,item)
        self.memory[key]=value
        return value


def span_metrics(case, ranked, corpus, coverage=None):
    coverage=literal_coverability(case,corpus) if coverage is None else coverage
    if not coverage['eligible']:return dict(coverage)
    groups=[g for g in evidence_groups(case) if any(e.get('span_text') or e.get('quote') for e in g)]
    matrix=[[any(span_matches(gold,row) for gold in group) for group in groups] for row in ranked]
    available=coverage['available']
    result = {'eligible': True, 'gold_groups': len(groups), 'groups_without_text_span': coverage['groups_without_text_span'],
              'corpus_span_coverage': sum(available) / len(groups)}
    supplemental={'publication_eligible':False,'literal_coverable_groups':sum(available),'total_literal_groups':len(groups),
                  'excluded_literal_groups':len(groups)-sum(available),
                  'bias':'Conditions on corpus literal coverability; excludes unavailable/representation-mismatched evidence; not overall quality'}
    for k in (10, 80):
        hits=[any(v[n] for v in matrix[:k]) for n in range(len(groups))]
        result['span_recall_at_' + str(k)] = sum(hits) / len(groups)
        hit_coverable=sum(hit and supported for hit,supported in zip(hits,available))
        supplemental['coverable_hits_at_'+str(k)]=hit_coverable
        supplemental['recall_at_'+str(k)+'_given_literal_coverability']=hit_coverable/sum(available) if any(available) else None
    ideal_count = coverage['ideal_count']
    ideal = sum(1 / math.log2(i+2) for i in range(min(10, ideal_count)))
    actual = sum(1 / math.log2(i+2) for i, values in enumerate(matrix[:10]) if any(values))
    result['ndcg_at_10'] = actual / ideal if ideal else 0.0
    supplemental['ndcg_at_10_given_any_literal_match']=actual/ideal if ideal else None
    result['supplemental']=supplemental
    return result


def summarize(results):
    metrics = [r['metrics'] for r in results if r['metrics']['eligible']]
    elapsed = sorted(r['elapsed_seconds'] for r in results if r.get('warmup_excluded') is True
                     and type(r.get('elapsed_seconds')) in (int,float) and math.isfinite(r['elapsed_seconds']))
    result={'completed': len(results), 'span_eligible': len(metrics),
        'warm_query_p95_seconds': elapsed[max(0, math.ceil(.95*len(elapsed))-1)] if elapsed else None,
        **{key: sum(r[key] for r in metrics) / len(metrics) if metrics else None
           for key in ('span_recall_at_10', 'span_recall_at_80', 'ndcg_at_10', 'corpus_span_coverage')}}
    supplements=[r['supplemental'] for r in metrics if 'supplemental' in r]
    covered=sum(s['literal_coverable_groups'] for s in supplements)
    total=sum(s['total_literal_groups'] for s in supplements)
    result['supplemental']={'publication_eligible':False,'literal_coverable_groups':covered,'total_literal_groups':total,
        'excluded_literal_groups':total-covered,'cases_without_literal_coverability':sum(s['literal_coverable_groups']==0 for s in supplements),
        'bias':'Group-weighted conditional diagnostic; excluded evidence can be harder or representation-mismatched; not publication or overall quality',
        **{'recall_at_'+str(k)+'_given_literal_coverability':sum(s['coverable_hits_at_'+str(k)] for s in supplements)/covered if covered else None
           for k in (10,80)}}
    result['unavailable_metrics']=[key for key in ('span_recall_at_10','span_recall_at_80','ndcg_at_10','warm_query_p95_seconds') if result[key] is None]
    return result


def windows_process_names():
    """Read a name-only OS snapshot when protected-process exe access is denied."""
    if os.name != 'nt':
        return {}
    import ctypes
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
            ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
            ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
            ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
            ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260)]
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ('Process32FirstW', 'Process32NextW'):
        function = getattr(api, name)
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        function.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        names = {}; entry = Entry(); entry.dwSize = ctypes.sizeof(entry)
        more = api.Process32FirstW(handle, ctypes.byref(entry))
        while more:
            names[entry.th32ProcessID] = entry.szExeFile
            more = api.Process32NextW(handle, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
        return names
    finally:
        api.CloseHandle(handle)


class ResourceMonitor:
    """Sample this worker, its children and Qdrant RSS; abort between batches."""
    def __init__(self, maximum_gib=12):
        self.maximum = maximum_gib * 1024**3
        self.peak = 0
        self.peak_system_used = 0
        self.stop = threading.Event()
        self.stages = {}
        self.sampling_error = None
        self.sample_lock = threading.Lock()
        self.collection_lock = threading.Lock()
        self.database_processes = {}
        self.last_discovery = None
        self.sample_interval = .2
        self.discovery_interval = 5.

    def sample(self):
        import psutil
        # Foreground budget checks and the background sampler share discovery.
        # Never preload memory_info for every unrelated process on the machine.
        with self.collection_lock:
            process = psutil.Process()
            now = time.monotonic()
            if self.last_discovery is None or now-self.last_discovery >= self.discovery_interval:
                found = {}; fallback_names = None
                for candidate in psutil.process_iter(['name']):
                    name = candidate.info.get('name')
                    if not isinstance(name, str) or not name:
                        if not candidate.is_running(): continue
                        if fallback_names is None: fallback_names = windows_process_names()
                        name = fallback_names.get(candidate.pid)
                        if not isinstance(name, str) or not name:
                            if not candidate.is_running(): continue
                            raise RuntimeError('Process discovery name unavailable')
                    if name.lower().startswith('qdrant'):
                        found[candidate.pid] = candidate
                self.database_processes = found
                self.last_discovery = now
            targets = {p.pid: p for p in process.children(recursive=True)}
            targets.update(self.database_processes)
            targets[process.pid] = process
            rss = 0
            for target in targets.values():
                try:
                    if not target.is_running():
                        if target.pid == process.pid:
                            raise RuntimeError('Worker process identity is unavailable')
                        continue  # An exited child/database no longer consumes RSS.
                    measured = target.memory_info().rss
                    if type(measured) is not int or measured <= 0:
                        raise RuntimeError('Target process RSS is unavailable')
                    rss += measured
                except psutil.NoSuchProcess:
                    if target.pid == process.pid: raise
                # AccessDenied or any other measurement error must fail closed.
            with self.sample_lock:
                self.peak = max(self.peak, rss)
                self.peak_system_used = max(self.peak_system_used, psutil.virtual_memory().used)
        return rss

    def check(self):
        if self.sampling_error is not None:
            raise RuntimeError('Resource sampling failed: '+self.sampling_error)
        try:self.sample()
        except Exception as exc:
            self.sampling_error=type(exc).__name__+': '+str(exc)
            raise RuntimeError('Resource sampling failed: '+self.sampling_error) from exc
        # Peak is latched: a transient overshoot still aborts the next boundary.
        if self.peak > self.maximum:
            raise MemoryError('Benchmark worker plus database exceeds configured RSS budget')

    def before_load(self, state, spec):
        import psutil
        folder = state / 'models' / spec['directory']
        size = sum(p.stat().st_size for p in folder.rglob('*') if p.suffix in ('.bin', '.safetensors'))
        # Qwen checkpoints are BF16 and the CPU adapter uses FP32. Leave working
        # space beyond weights; refuse to force the user's other apps into swap.
        estimated = size * (2 if 'Qwen3' in spec['repository'] else 1) + 512*1024**2
        available = psutil.virtual_memory().available
        self.preflight = {'available_bytes': available, 'estimated_load_bytes': estimated}
        if available < estimated:
            raise MemoryError(f'Insufficient system headroom: {available/1024**3:.2f} GiB available, '
                              f'estimated {estimated/1024**3:.2f} GiB needed; no model loaded')

    def __enter__(self):
        def poll():
            while not self.stop.wait(self.sample_interval):
                try:self.sample()
                except Exception as exc:
                    self.sampling_error=type(exc).__name__+': '+str(exc)
                    break
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        if not args or args[0] is None:self.check()

    @contextlib.contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0) + time.perf_counter() - started

    def report(self):
        return {'stages_seconds': self.stages, 'worker_children_qdrant_peak_rss_bytes': self.peak if self.sampling_error is None else None,
                'observed_peak_rss_bytes':self.peak,'sampling_status':'unavailable' if self.sampling_error else 'measured',
                'sampling_error':self.sampling_error,
                'system_peak_used_bytes': self.peak_system_used, 'rss_budget_bytes': self.maximum,
                'sampling_protocol': RESOURCE_PROTOCOL,
                'sample_interval_seconds': self.sample_interval,
                'qdrant_discovery_interval_seconds': self.discovery_interval,
                'rss_scope': 'benchmark worker, its children, and Qdrant; concurrent unrelated app processes excluded',
                'memory_preflight': getattr(self, 'preflight', None)}


def local_identity(state, name):
    spec = ALL_MODELS[name]
    path = state / 'models' / spec['directory']
    if name in MODELS:
        manifest = verify_manifest(path, spec)
        return {**spec, 'files_sha256': fingerprint(manifest['files']), 'verification': 'all_files_sha256'}
    baseline = read(state / 'models/manifest.json')
    if not any(r['repository'] == spec['repository'] and r['revision'] == spec['revision'] for r in baseline):
        raise ValueError('Baseline model revision differs from declared experiment')
    # Record baseline bytes too; model name alone cannot identify local weights.
    files = {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob('*'))
             if p.is_file() and '.cache' not in p.relative_to(path).parts}
    if not any(n.endswith(('.safetensors', '.bin')) for n in files):
        raise ValueError('Baseline model is incomplete')
    return {**spec, 'files_sha256': fingerprint(files), 'verification': 'manifest_revision_and_recorded_local_hashes'}


def report_base(args, manifest):
    if args.batch_size < 1 or args.threads < 1 or args.max_tokens < 64 or not 0 < args.rss_gib <= 12:
        raise ValueError('Invalid CPU batch/thread/token/RSS budget (RSS must be <=12 GiB)')
    storage_batch = getattr(args, 'import_batch_size', IMPORT_BATCH_SIZE)
    if type(storage_batch) is not int or not 1 <= storage_batch <= 1024:
        raise ValueError('Invalid index import storage batch size')
    report = {'schema': REPORT_SCHEMA, 'kind': args.command, 'status': 'running','publication_eligible':False,
        'suite_fingerprint': manifest['fingerprint'], 'corpus_sampled': manifest['corpus_sampled'],
        'model': local_identity(args.state, args.model),
        'settings': {'batch_size': args.batch_size, 'threads': args.threads, 'maximum_tokens': args.max_tokens,
                     'rss_gib': args.rss_gib, 'reuse_index_vectors': bool(getattr(args, 'reuse_index_vectors', False)),
                     'resource_sampling_protocol': RESOURCE_PROTOCOL,
                     'index_import_batch_size': storage_batch,
                     'embedding_parameter_dtype': EMBEDDING_DTYPE if args.command == 'embed' else None},
        'code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                       [*sorted((ROOT / 'ragapp').glob('*.py')), Path(__file__), ROOT / 'scripts/download_candidate_models.py']},
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'sentence-transformers')},
        'paid_api_calls': 0, 'results': [],
        'limitations': ['Development model comparison only; no answer correctness, faithfulness or blind acceptance claim.',
            'Span scores require supplied gold spans; draft gold still needs independent review.',
            'Dense embeddings only: BGE-M3 sparse and multi-vector representations are not tested.',
            'v4 pins CPU embedding FP32 and changes resource sampling/import batching. Wall times are not directly comparable with v3 epochs.',
            'Imported/cached document stage time is not model encoding throughput.']}
    if args.command == 'rerank':
        report['settings']['reranker_parameter_dtype'] = EMBEDDING_DTYPE
        report['settings']['batching_protocol'] = RERANK_BATCHING_PROTOCOL
        report['limitations'].append('Native predict receives every query pair once; RSS checks run at each model forward. '
                                    'Legacy outer-sliced rerank timings use a different batching protocol.')
    if args.output.exists():
        if not getattr(args, 'resume', False):
            raise ValueError('Output exists; use --resume with exactly matching suite/model/code/config')
        previous = read(args.output)
        fields = ('schema', 'kind', 'suite_fingerprint', 'corpus_sampled', 'model', 'settings', 'code_sha256', 'packages')
        if any(previous.get(key) != report[key] for key in fields):
            raise ValueError('Resume suite/model/code/config/package identity changed; start a separate report/cache')
        if not isinstance(previous.get('results'), list):
            raise ValueError('Invalid resume results')
        previous.setdefault('resume_history', []).append({'status': previous.get('status'),
            'resources': previous.get('resources'), 'resumed_at': time.time()})
        previous['status'] = 'running'; previous.pop('error', None)
        return previous
    if getattr(args, 'resume', False):
        raise ValueError('--resume requires an existing report checkpoint')
    return report


def query_context(report):
    keys = ('schema','kind','suite_fingerprint','model','settings',
            'code_sha256','packages','pool_fingerprint')
    if report.get('schema') == REPORT_SCHEMA:
        keys += ('embedding_precision',)
    # Presence-gated addition preserves every previously frozen v3/v4 context.
    if 'rerank_precision' in report:
        keys += ('rerank_precision',)
    return fingerprint({key:report.get(key) for key in keys})


def query_commit(report,row):
    row['commit']={'schema':'model-query-commit-v1','context_fingerprint':query_context(report),
                   'row_sha256':fingerprint({k:v for k,v in row.items() if k!='commit'})}
    return row


def validated_scores(values,count):
    import numpy as np
    scores=np.asarray(values)
    if scores.shape!=(count,) or scores.dtype.kind not in 'fi' or not np.isfinite(scores).all():
        raise ValueError('Model scores must be complete, one-dimensional and finite')
    return scores.tolist()


def validate_query_row(report,row,corpus_ids=None,pools=None):
    if report.get('schema') not in READABLE_REPORT_SCHEMAS:
        raise ValueError('Unsupported model benchmark query schema')
    if report.get('schema') == REPORT_SCHEMA and report.get('kind') == 'embed':
        expected = {'requested_parameter_dtype': EMBEDDING_DTYPE,
                    'actual_parameter_dtypes': [EMBEDDING_DTYPE], 'validation': 'all_parameters_checked'}
        if (report.get('settings', {}).get('embedding_parameter_dtype') != EMBEDDING_DTYPE
                or report.get('embedding_precision') != expected):
            raise ValueError('Embedding query has no verified FP32 parameter precision')
    if report.get('kind') == 'rerank' and ('rerank_precision' in report
            or report.get('settings', {}).get('reranker_parameter_dtype') is not None):
        expected = {'requested_parameter_dtype': EMBEDDING_DTYPE,
                    'actual_parameter_dtypes': [EMBEDDING_DTYPE], 'validation': 'all_parameters_checked'}
        if (report.get('settings', {}).get('reranker_parameter_dtype') != EMBEDDING_DTYPE
                or report.get('rerank_precision') != expected):
            raise ValueError('Reranking query has no verified FP32 parameter precision')
    commit=row.get('commit')
    if (not isinstance(commit,dict) or commit.get('schema')!='model-query-commit-v1'
            or commit.get('context_fingerprint')!=query_context(report)
            or commit.get('row_sha256')!=fingerprint({k:v for k,v in row.items() if k!='commit'})):
        raise ValueError('Query checkpoint is incomplete or changed')
    elapsed=row.get('elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0 or row.get('warmup_excluded') is not True:
        raise ValueError('Query checkpoint has no verified warm latency')
    ranked=row.get('ranked')
    if not isinstance(ranked,list) or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in ranked):
        raise ValueError('Malformed query ranking')
    ids=[r['id'] for r in ranked]
    if len(ids)!=len(set(ids)) or corpus_ids is not None and not set(ids)<=set(corpus_ids):
        raise ValueError('Query ranking contains duplicate or foreign candidates')
    scores=validated_scores([r.get('score') for r in ranked],len(ranked))
    if any(a<b for a,b in zip(scores,scores[1:])):raise ValueError('Query ranking scores are not ordered')
    if pools is not None:
        if row['id'] not in pools or set(ids)!=set(pools[row['id']]) or len(ids)!=len(pools[row['id']]):
            raise ValueError('Query checkpoint differs from its complete fixed pool')
    elif corpus_ids is not None and len(ids)!=min(80,len(corpus_ids)):
        raise ValueError('Query checkpoint has an incomplete dense ranking')
    metrics=row.get('metrics')
    if not isinstance(metrics,dict) or type(metrics.get('eligible')) is not bool:
        raise ValueError('Query checkpoint is missing metric status')
    if metrics['eligible']:
        for key in ('span_recall_at_10','span_recall_at_80','ndcg_at_10','corpus_span_coverage'):
            value=metrics.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Query checkpoint contains invalid metrics')
    elif not isinstance(metrics.get('reason'),str) or not metrics['reason'].strip():
        raise ValueError('Unavailable query metrics need an explicit reason')


def completed_case_ids(report, cases, corpus_ids=None, pools=None):
    if any(not isinstance(row,dict) or not isinstance(row.get('id'),str) for row in report['results']):
        raise ValueError('Malformed resume query results')
    ids = [row['id'] for row in report['results']]
    if len(set(ids)) != len(ids) or not set(ids) <= {case['id'] for case in cases}:
        raise ValueError('Resume contains duplicate or foreign query results')
    for row in report['results']:validate_query_row(report,row,corpus_ids,pools)
    return set(ids)


def coverage_cache(args,report,corpus):
    code={key:value for key,value in report['code_sha256'].items()
          if key in {'scripts/benchmark_models.py','ragapp/evaluation.py'}}
    return LiteralCoverageCache(args.state/'cache/model-literal-coverability',report['suite_fingerprint'],code,corpus)


def record_warmup(report,started,kind):
    report.setdefault('warmup_sessions',[]).append({'kind':kind,'elapsed_seconds':time.perf_counter()-started,
        'excluded_from_query_latency':True,'session':len(report.get('resume_history',[]))+1})


def embed(args):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from ragapp.config import Settings
    from ragapp.model_adapters import embedding_inputs
    from ragapp.textbudget import windows
    manifest, corpus, cases, _ = load_suite(args.suite)
    if ALL_MODELS[args.model]['kind'] != 'embedding':
        raise ValueError('Expected an embedding model')
    report = report_base(args, manifest)
    completed = completed_case_ids(report, cases,[row['id'] for row in corpus])
    coverage=coverage_cache(args,report,corpus)
    cache_path = getattr(args, 'cache_dir', None) or args.output.with_suffix('.vectors')
    report['document_cache'] = {'path': str(cache_path.resolve()), 'query_vectors_cached': False}
    save(args.output, report)
    torch.set_num_threads(args.threads)
    spec = ALL_MODELS[args.model]
    settings = Settings(embedding_model=spec['repository'])
    monitor = ResourceMonitor(args.rss_gib)
    try:
        with monitor:
            with monitor.stage('load_model'):
                monitor.before_load(args.state, spec)
                model = SentenceTransformer(str(args.state / 'models' / spec['directory']), device='cpu',
                    local_files_only=True, model_kwargs={'torch_dtype': torch.float32})
                model.max_seq_length = args.max_tokens
                report['embedding_precision'] = checked_embedding_precision(model, report)
                monitor.check()
            cache = DocumentVectorCache(cache_path, embedding_contract(report), spec['dimension'])
            report['document_cache']['contract_fingerprint'] = fingerprint(cache.contract)
            def checkpoint(progress):
                report['progress'] = progress
                report['resources'] = monitor.report()
                save(args.output, report)
            input_overhead = {}
            def window_inputs(text, query):
                if query not in input_overhead:
                    input_overhead[query] = len(model.tokenizer(embedding_inputs(settings, [''], query)[0])['input_ids']) + 8
                overhead = input_overhead[query]
                parts = [part for part, _, _ in windows(model.tokenizer, text, args.max_tokens-overhead, 32)]
                inputs = embedding_inputs(settings, parts, query)
                if any(len(model.tokenizer(t)['input_ids']) > args.max_tokens for t in inputs):
                    raise ValueError('Input template exceeded measured budget')
                return inputs
            with monitor.stage('document_tokenization'):
                texts, owners, keys, documents = [], [], [], []
                tokenization_started = time.perf_counter()
                def tokenization_progress(completed):
                    monitor.check()
                    checkpoint({'stage': 'document_tokenization', 'completed': completed, 'total': len(corpus),
                                'elapsed_seconds': time.perf_counter()-tokenization_started})
                tokenization_progress(0)
                for n, row in enumerate(corpus):
                    original = embedding_text(row)
                    parts = window_inputs(original, False)
                    if not parts:
                        raise ValueError('An indexed document produced no embedding windows')
                    row_keys = [cache.key(original, part) for part in parts]
                    documents.append({'text': original, 'inputs': parts, 'keys': row_keys})
                    texts.extend(parts)
                    keys.extend(row_keys)
                    owners.extend([n] * len(parts))
                    if (n+1) % 128 == 0: tokenization_progress(n+1)
                tokenization_progress(len(corpus))
            report['document_windows'] = len(texts)
            report['logical_chunks'] = len(corpus)
            report['silent_truncations'] = 0
            if getattr(args, 'reuse_index_vectors', False):
                with monitor.stage('index_vector_verification'):
                    report['index_vector_reuse'] = reuse_index_vectors(args, manifest, report, cache, documents, model, monitor, checkpoint)
                checkpoint({'stage': 'index_vector_verification', 'completed': len(documents), 'total': len(documents)})
            with monitor.stage('document_embedding'):
                matrix, reuse = cached_document_matrix(cache, texts, keys, model, args.batch_size, monitor,
                    lambda progress: checkpoint({'stage': 'document_embedding', **progress}))
                report['document_cache'].update(reuse)
                report['document_cache']['document_time_interpretation'] = (
                    'Fresh model encoding' if reuse['document_throughput_eligible'] else
                    'Includes validated reuse/cache IO; NOT a model encoding throughput measurement')
            owners = np.asarray(owners)
            if len(completed) < len(cases):
                with monitor.stage('query_warmup'):
                    monitor.check();warmup_started=time.perf_counter()
                    model.encode(window_inputs('检索性能预热', True), normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=args.batch_size)
                    monitor.check();record_warmup(report,warmup_started,'embedding')
                    save(args.output,report)
            for case in cases:
                if case['id'] in completed:
                    continue
                started = time.perf_counter()
                with monitor.stage('query_embedding'):
                    monitor.check()
                    inputs = window_inputs(retrieval_question(case), True)
                    queries = checked_vectors(model.encode(inputs, normalize_embeddings=True, show_progress_bar=False,
                        batch_size=args.batch_size), len(inputs), spec['dimension'])
                    monitor.check()
                with monitor.stage('dense_scoring'):
                    scores = np.full(len(corpus), -np.inf)
                    for q in queries:
                        np.maximum.at(scores, owners, matrix @ q)
                    ordered = np.argsort(-scores, kind='stable')[:80]
                    ranked = [corpus[int(i)] for i in ordered]
                row={'id': case['id'], 'query_windows': len(inputs),'warmup_excluded':True,
                    'elapsed_seconds': time.perf_counter()-started,
                    'ranked': [{'id': corpus[int(i)]['id'], 'score': float(scores[i])} for i in ordered],
                    'metrics': span_metrics(case, ranked, corpus,coverage.get(case))}
                query_commit(report,row);validate_query_row(report,row,[r['id'] for r in corpus])
                report['results'].append(row)
                report['resources'] = monitor.report()
                save(args.output, report)
                monitor.check()
        report['status'] = 'completed_exploratory' if manifest['corpus_sampled'] else 'completed_development'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report['resources'] = monitor.report()
    report['metrics'] = summarize(report['results'])
    save(args.output, report)
    return report


def freeze_pool(args):
    manifest, corpus, cases, lexical = load_suite(args.suite)
    if args.output.exists():
        raise ValueError('Candidate pool already exists; create a new one')
    dense = read(args.dense_report) if args.dense_report else None
    if dense and (dense.get('suite_fingerprint') != manifest['fingerprint'] or dense.get('kind') != 'embed'
                  or not dense['status'].startswith('completed')):
        raise ValueError('Dense baseline was not completed on this suite')
    if dense:
        if dense.get('schema') not in READABLE_REPORT_SCHEMAS:
            raise ValueError('Dense baseline needs complete v3/v4 query checkpoints')
        completed=completed_case_ids(dense,cases,[row['id'] for row in corpus])
        if completed!={case['id'] for case in cases}:raise ValueError('Dense baseline is missing a development query')
    dense_rows = {r['id']: r['ranked'] for r in dense['results']} if dense else {}
    pools = {}
    for case in cases:
        scores = {}
        lanes = [lexical[case['id']]]
        if dense:
            if case['id'] not in dense_rows:
                raise ValueError('Dense baseline is missing a development query')
            lanes.append([r['id'] for r in dense_rows[case['id']]])
        for lane in lanes:
            for rank, key in enumerate(lane, 1):
                scores[key] = scores.get(key, 0) + 1/(args.rrf_k+rank)
        pools[case['id']] = sorted(scores, key=lambda k: (-scores[k], k))[:args.candidates]
    result = {'suite_fingerprint': manifest['fingerprint'], 'kind': 'BM25+dense-RRF' if dense else 'BM25',
              'dense_report_sha256': sha256(args.dense_report) if dense else None,
              'rrf_k': args.rrf_k, 'candidate_limit': args.candidates, 'pools': pools}
    result['fingerprint'] = fingerprint(result)
    save(args.output, result)
    return result


@contextlib.contextmanager
def native_reranker_batches(adapter, monitor, batch_size):
    """Preserve native whole-query batching while checking every forward's RSS.

    CrossEncoder calls its own module once per batch; QwenReranker calls its
    contained causal model. No input reordering or chunking happens here. The
    original predict method and all pre-existing hooks survive every exit path.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Reranker inference batch size must be a positive integer')
    target = adapter
    if not all(callable(getattr(target, name, None)) for name in
               ('register_forward_pre_hook', 'register_forward_hook')):
        target = getattr(adapter, 'model', None)
    if not all(callable(getattr(target, name, None)) for name in
               ('register_forward_pre_hook', 'register_forward_hook')):
        raise ValueError('Reranker adapter has no forward hooks for RSS enforcement')
    original_predict = adapter.predict
    attributes = vars(adapter)
    had_override = 'predict' in attributes
    old_override = attributes.get('predict')
    checks = {'before': 0, 'after': 0}

    def before_forward(module, inputs):
        monitor.check()
        checks['before'] += 1

    def after_forward(module, inputs, output):
        monitor.check()
        checks['after'] += 1

    def predict(pairs, **kwargs):
        start = checks.copy()
        kwargs.update(batch_size=batch_size, show_progress_bar=False)
        values = original_predict(pairs, **kwargs)
        count = checks['before'] - start['before']
        if len(pairs) and (count < 1 or checks['after'] - start['after'] != count):
            raise RuntimeError('Native reranker prediction did not execute guarded forward batches')
        return validated_scores(values, len(pairs))

    def restore_predict():
        if had_override:
            adapter.predict = old_override
        else:
            del adapter.predict

    with contextlib.ExitStack() as cleanup:
        handle = target.register_forward_pre_hook(before_forward)
        cleanup.callback(handle.remove)
        # A failed forward also triggers the post check. PyTorch preserves the
        # original forward error if this always-call hook itself then fails.
        handle = target.register_forward_hook(after_forward, always_call=True)
        cleanup.callback(handle.remove)
        adapter.predict = predict
        cleanup.callback(restore_predict)
        yield checks


def rerank(args):
    import torch
    from ragapp.config import Settings
    from ragapp.retrieval import Retrieval, validated_reranker_precision
    manifest, corpus, cases, _ = load_suite(args.suite)
    pool = read(args.pool)
    if pool['suite_fingerprint'] != manifest['fingerprint'] or fingerprint({k:v for k,v in pool.items() if k != 'fingerprint'}) != pool['fingerprint']:
        raise ValueError('Candidate pool does not match the frozen suite')
    spec = ALL_MODELS[args.model]
    if spec['kind'] != 'reranker':
        raise ValueError('Expected a reranker model')
    report = report_base(args, manifest)
    if (getattr(args, 'resume', False) and report.get('pool_fingerprint') != pool['fingerprint']
            or report.get('pool_fingerprint') not in (None, pool['fingerprint'])):
        raise ValueError('Resume candidate pool changed')
    report['pool_fingerprint'] = pool['fingerprint']
    rows = {r['id']: r for r in corpus}
    pools=pool.get('pools')
    if not isinstance(pools,dict) or set(pools)!={case['id'] for case in cases}:
        raise ValueError('Fixed pool must cover exactly every frozen query')
    for ids in pools.values():
        if (not isinstance(ids,list) or not ids or any(not isinstance(key,str) for key in ids)
                or len(ids)!=len(set(ids)) or not set(ids)<=set(rows)):
            raise ValueError('Candidate pool contains duplicate, missing or unknown chunks')
    completed = completed_case_ids(report, cases,list(rows),pools)
    coverage=coverage_cache(args,report,corpus)
    save(args.output, report)
    torch.set_num_threads(args.threads)
    settings = Settings(state=args.state, reranker_model=spec['repository'], reranker_revision=spec['revision'],
        reranker_directory=spec['directory'], reranker_max_tokens=args.max_tokens)
    retrieval = Retrieval(settings, None)
    monitor = ResourceMonitor(args.rss_gib)
    try:
        with monitor, contextlib.ExitStack() as inference_guards:
            with monitor.stage('load_model'):
                monitor.before_load(args.state, spec)
                precision = validated_reranker_precision(retrieval.reranker)
                if report.get('rerank_precision') not in (None, precision):
                    raise ValueError('Resume reranker parameter precision changed')
                report['rerank_precision'] = precision
                monitor.check()
                inference_guards.enter_context(native_reranker_batches(retrieval.reranker, monitor, args.batch_size))
            if len(completed)<len(cases):
                with monitor.stage('reranker_warmup'):
                    monitor.check();warmup_started=time.perf_counter()
                    validated_scores(retrieval.rank('检索性能预热',[{'path':'warmup.txt','text':'本次为模型预热。','locator':{}}]),1)
                    monitor.check();record_warmup(report,warmup_started,'reranker')
                    save(args.output,report)
            for case in cases:
                if case['id'] in completed:
                    continue
                ids = pool['pools'][case['id']]
                if len(ids) != len(set(ids)) or any(key not in rows for key in ids):
                    raise ValueError('Candidate pool contains duplicate or unknown chunks')
                candidates = [rows[key] for key in ids]
                trace = {}
                started = time.perf_counter()
                with monitor.stage('reranking'):
                    # Production's budgeted official adapters, on a frozen pool.
                    monitor.check()
                    scores = validated_scores(retrieval.rank(retrieval_question(case), candidates, trace),len(candidates))
                    monitor.check()
                ordered = sorted(zip(candidates, scores), key=lambda r: (-r[1], r[0]['id']))
                row={'id': case['id'], 'elapsed_seconds': time.perf_counter()-started,'warmup_excluded':True,
                    'ranked': [{'id': r['id'], 'score': float(s)} for r,s in ordered],
                    'trace': trace, 'metrics': span_metrics(case, [r for r,s in ordered], corpus,coverage.get(case))}
                query_commit(report,row);validate_query_row(report,row,list(rows),pools)
                report['results'].append(row)
                report['resources'] = monitor.report()
                save(args.output, report)
                monitor.check()
        report['status'] = 'completed_exploratory' if manifest['corpus_sampled'] else 'completed_development'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report['resources'] = monitor.report()
    report['metrics'] = summarize(report['results'])
    save(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', type=Path, default=ROOT / '.rag/evaluation/quality-v2/dev.json')
    p.add_argument('--catalog', type=Path, help='Defaults to the current active generation catalog')
    p.add_argument('--state', type=Path, default=ROOT / '.rag')
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--corpus-limit', type=int, default=0)
    for name in ('embed', 'rerank'):
        p = sub.add_parser(name)
        p.add_argument('--suite', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--state', type=Path, default=ROOT / '.rag')
        p.add_argument('--model', choices=list(ALL_MODELS), required=True)
        p.add_argument('--max-tokens', type=int, default=512 if name == 'embed' else 1024)
        p.add_argument('--batch-size', type=int, default=4)
        p.add_argument('--threads', type=int, default=4)
        p.add_argument('--rss-gib', type=float, default=12)
        p.add_argument('--resume', action='store_true', help='Resume only an identical frozen experiment')
        if name == 'rerank':
            p.add_argument('--pool', type=Path, required=True)
        else:
            p.add_argument('--cache-dir', type=Path, help='Explicit reusable cache; defaults beside report as .vectors')
            p.add_argument('--reuse-index-vectors', action='store_true',
                help='E5 only: verify matching index metadata, single-window inputs and recomputed subset before reuse')
            p.add_argument('--import-batch-size', type=int, default=IMPORT_BATCH_SIZE,
                help='Validated index vector storage batch (1-1024, default 128); inference batch size is unchanged')
    p = sub.add_parser('pool')
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--dense-report', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rrf-k', type=int, choices=[10,30,60], default=10)
    p.add_argument('--candidates', type=int, default=60)
    args = parser.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if hasattr(args, 'state'):
        os.environ.setdefault('HF_HOME', str(args.state / 'huggingface'))
    runner = {'prepare': prepare, 'embed': embed, 'pool': freeze_pool, 'rerank': rerank}[args.command]
    if args.command in {'embed', 'rerank'}:
        from filelock import FileLock
        lock = args.state / 'reports/quality-v2-models/model-worker.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock), timeout=0):
            result = runner(args)
    else:
        result = runner(args)
    print(json.dumps({k: result[k] for k in ('status', 'fingerprint', 'metrics', 'resources', 'error') if k in result},
                     ensure_ascii=False, indent=2), flush=True)
    if result.get('status') == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
