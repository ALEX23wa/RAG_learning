import argparse
import json
from pathlib import Path

import pytest

from ragapp.pipeline import fingerprint
from scripts.benchmark_models import ResourceMonitor, development_cases, freeze_pool, load_suite, save, span_metrics, summarize
from scripts.download_candidate_models import sha256, target_path, verify_manifest


def test_candidate_download_cannot_replace_production_or_use_moving_revision(tmp_path):
    spec = {'directory': 'candidate-safe', 'revision': 'a' * 40}
    assert target_path(tmp_path, spec) == tmp_path / 'models/candidate-safe'
    for name in ('embedding', 'candidate-x/../../embedding', '../candidate-out'):
        with pytest.raises(ValueError):
            target_path(tmp_path, {**spec, 'directory': name})
    with pytest.raises(ValueError):
        target_path(tmp_path, {**spec, 'revision': 'main'})


def test_model_file_checksum_detects_tampering(tmp_path):
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'original')
    save(tmp_path / 'snapshot-manifest.json', {'status': 'complete', 'files': {
        weights.name: {'bytes': len(b'original'), 'sha256': sha256(weights)}}})
    assert verify_manifest(tmp_path)['status'] == 'complete'
    weights.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        verify_manifest(tmp_path)


def test_development_split_rejects_holdout_even_when_renamed(tmp_path):
    path = tmp_path / 'dev.json'
    save(path, [{'id': 'x', 'question': 'q', 'split': 'blind', 'answerable': True}])
    with pytest.raises(ValueError, match='Holdout'):
        development_cases(path)
    blind = tmp_path / 'blind.json'
    save(blind, [{'id': 'x', 'question': 'q', 'split': 'dev', 'answerable': True}])
    with pytest.raises(ValueError, match='development'):
        development_cases(blind)


def test_spans_not_location_alone_drive_model_metrics():
    case = {'evidence': [{'path': 'a.py', 'locator': {'function': 'f'}, 'span_text': 'critical=42'}]}
    incorrect = {'id': 'a', 'path': 'a.py', 'locator': {'function': 'f'}, 'text': 'other=1'}
    correct = {'id': 'b', 'path': 'a.py', 'locator': {'function': 'f'}, 'text': 'critical=42'}
    result = span_metrics(case, [incorrect], [incorrect, correct])
    assert result['span_recall_at_10'] == result['ndcg_at_10'] == 0
    assert result['corpus_span_coverage'] == 1
    result = span_metrics(case, [incorrect, correct], [incorrect, correct])
    assert result['span_recall_at_10'] == 1
    assert 0 < result['ndcg_at_10'] < 1
    assert not span_metrics({'evidence': [{'path': 'a.py'}]}, [correct], [correct])['eligible']


def make_suite(path):
    path.mkdir()
    payload = {'corpus.json': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
               'cases.json': [{'id': 'one', 'split': 'dev'}], 'lexical.json': {'one': ['a', 'b', 'c']}}
    for name, value in payload.items():
        save(path / name, value)
    manifest = {'corpus_sampled': True, 'files': {name: sha256(path / name) for name in payload}}
    manifest['fingerprint'] = fingerprint(manifest)
    save(path / 'manifest.json', manifest)
    return manifest


def test_frozen_suite_detects_chunk_change(tmp_path):
    path = tmp_path / 'suite'
    make_suite(path)
    assert load_suite(path)[0]['corpus_sampled']
    save(path / 'corpus.json', [{'id': 'modified'}])
    with pytest.raises(ValueError, match='Frozen suite changed'):
        load_suite(path)


@pytest.mark.parametrize('schema',['model-benchmark-v3','model-benchmark-v4'])
def test_fixed_pool_is_same_for_all_rerankers_and_never_injects_gold(tmp_path,schema):
    path = tmp_path / 'suite'
    manifest = make_suite(path)
    dense = tmp_path / 'dense.json'
    from scripts.benchmark_models import query_commit
    report={'schema':schema,'suite_fingerprint': manifest['fingerprint'],
            'status': 'completed_exploratory', 'kind': 'embed'}
    if schema.endswith('v4'):
        report.update(settings={'embedding_parameter_dtype':'torch.float32'},embedding_precision={
            'requested_parameter_dtype':'torch.float32','actual_parameter_dtypes':['torch.float32'],
            'validation':'all_parameters_checked'})
    row={'id':'one','warmup_excluded':True,'elapsed_seconds':1.,
         'ranked':[{'id':key,'score':float(score)} for key,score in [('c',3),('b',2),('a',1)]],
         'metrics':{'eligible':False,'reason':'No literal gold'}}
    report['results']=[query_commit(report,row)];save(dense,report)
    args = argparse.Namespace(suite=path, dense_report=dense, output=tmp_path/'pool.json', rrf_k=10, candidates=2)
    result = freeze_pool(args)
    assert result['kind'] == 'BM25+dense-RRF'
    assert len(result['pools']['one']) == 2
    assert set(result['pools']['one']) <= {'a','b','c'}
    with pytest.raises(ValueError, match='already exists'):
        freeze_pool(args)
    args.output = tmp_path/'new-pool.json'
    changed = json.loads(dense.read_text())
    changed['suite_fingerprint'] = 'other-suite'
    save(dense, changed)
    with pytest.raises(ValueError, match='this suite'):
        freeze_pool(args)


def test_no_text_span_metrics_are_reported_as_unavailable():
    result = summarize([{'metrics': {'eligible': False}}])
    assert result['span_recall_at_80'] is None
    assert result['span_eligible'] == 0


def test_memory_preflight_rejects_before_loading_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    folder = tmp_path / 'models/candidate-x'
    folder.mkdir(parents=True)
    (folder / 'model.safetensors').write_bytes(b'weights')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=1))
    monitor = ResourceMonitor()
    with pytest.raises(MemoryError, match='no model loaded'):
        monitor.before_load(tmp_path, {'directory':'candidate-x', 'repository':'Qwen/Qwen3-test'})
    assert monitor.preflight['available_bytes'] == 1


def test_window_cache_resumes_after_interruption_without_reencoding_committed_batch(tmp_path):
    import numpy as np
    from scripts.benchmark_models import DocumentVectorCache, cached_document_matrix
    cache = DocumentVectorCache(tmp_path/'cache', {'suite':'frozen', 'model_files':'sha'}, 2)
    inputs = ['passage: a','passage: b','passage: c']
    keys = [cache.key(text,text) for text in inputs]
    class Monitor:
        def check(self):pass
    class Model:
        calls = []
        fail = True
        def encode(self, texts, **kwargs):
            self.calls.append(texts)
            if texts == inputs[2:] and self.fail:raise RuntimeError('Disconnected')
            return np.tile([1.,0.], (len(texts),1))
    model = Model();progress = []
    with pytest.raises(RuntimeError, match='Disconnected'):
        cached_document_matrix(cache,inputs,keys,model,2,Monitor(),progress.append)
    assert progress[-1]['document_windows_cached'] == 2
    resumed = DocumentVectorCache(tmp_path/'cache',cache.contract,2);model.fail = False
    matrix, stats = cached_document_matrix(resumed,inputs,keys,model,2,Monitor(),progress.append)
    assert model.calls == [inputs[:2],inputs[2:],inputs[2:]]
    assert matrix.shape == (3,2) and stats['cached_window_occurrences'] == 2
    assert stats['new_unique_windows_encoded'] == 1 and not stats['document_throughput_eligible']


@pytest.mark.parametrize('vectors', [ [[float('nan'),0]], [[float('inf'),0]], [[0.,0.]], [[2.,0.]], [[1.,0.,0.]], [[1,0]] ])
def test_window_cache_rejects_invalid_model_vectors_before_commit(tmp_path,vectors):
    from scripts.benchmark_models import DocumentVectorCache
    cache = DocumentVectorCache(tmp_path/'cache',{},2)
    with pytest.raises(ValueError):cache.add(['key'],vectors)
    assert not list(cache.folder.glob('batch-*.json'))


@pytest.mark.parametrize('kind',['binary','metadata','identity'])
def test_cache_rejects_modified_committed_bytes_or_contract(tmp_path,kind):
    import numpy as np
    from scripts.benchmark_models import DocumentVectorCache
    cache = DocumentVectorCache(tmp_path/'cache',{'model_files':'original'},2)
    cache.add(['key'],np.array([[1.,0.]]))
    if kind == 'binary':next(cache.folder.glob('batch-*.npy')).write_bytes(b'corrupted')
    elif kind == 'metadata':
        path = next(cache.folder.glob('batch-*.json'));value = json.loads(path.read_text());value['keys']=['other'];save(path,value)
    with pytest.raises(ValueError, match='checksum|identity'):
        DocumentVectorCache(cache.folder,{'model_files':'changed'} if kind=='identity' else cache.contract,2)


def test_cache_key_binds_input_template_text_and_dimension_and_ignores_uncommitted_orphan(tmp_path):
    from scripts.benchmark_models import DocumentVectorCache
    cache = DocumentVectorCache(tmp_path/'cache',{'suite':'one','normalize':True},2)
    assert cache.key('body','passage: body') != cache.key('body','body')
    assert cache.key('body','passage: body') != cache.key('different body','passage: body')
    (cache.folder/'batch-orphan.npy').write_bytes(b'incomplete uncommitted batch')
    resumed = DocumentVectorCache(cache.folder,cache.contract,2)
    assert not resumed.vectors
    with pytest.raises(ValueError,match='identity'):DocumentVectorCache(cache.folder,cache.contract,3)


def benchmark_args(tmp_path, **values):
    return argparse.Namespace(command='embed',state=tmp_path,suite=tmp_path/'suite',output=tmp_path/'report.json',
        model='e5',batch_size=2,threads=1,max_tokens=512,rss_gib=12.,resume=False,
        reuse_index_vectors=False,cache_dir=None,**values)


@pytest.mark.parametrize('field',['suite_fingerprint','model','settings','code_sha256','packages'])
def test_resume_rejects_experiment_changes_before_loading_model(tmp_path,monkeypatch,field):
    from scripts import benchmark_models as bench
    monkeypatch.setattr(bench,'local_identity',lambda *_:{'repository':'fake','files_sha256':'sha'})
    monkeypatch.setattr(bench.importlib.metadata,'version',lambda *_:'fake')
    args = benchmark_args(tmp_path);manifest = {'fingerprint':'suite','corpus_sampled':True}
    report = bench.report_base(args,manifest);report[field] = 'changed';save(args.output,report)
    args.resume = True
    with pytest.raises(ValueError,match='identity changed'):bench.report_base(args,manifest)


def test_resume_requires_explicit_existing_report_and_unique_known_query_ids(tmp_path,monkeypatch):
    from scripts import benchmark_models as bench
    monkeypatch.setattr(bench,'local_identity',lambda *_:{'repository':'fake','files_sha256':'sha'})
    monkeypatch.setattr(bench.importlib.metadata,'version',lambda *_:'fake')
    args = benchmark_args(tmp_path);manifest = {'fingerprint':'suite','corpus_sampled':True}
    report = bench.report_base(args,manifest);save(args.output,report)
    with pytest.raises(ValueError,match='--resume'):bench.report_base(args,manifest)
    args.resume = True
    assert bench.report_base(args,manifest)['resume_history']
    for ids in (['x','x'],['foreign']):
        with pytest.raises(ValueError,match='duplicate or foreign'):
            bench.completed_case_ids({'results':[{'id':i} for i in ids]},[{'id':'x'}])


def index_cache_fixture(tmp_path):
    import numpy as np
    from ragapp.config import Settings
    from ragapp.pipeline import embedding_fingerprint
    from scripts.benchmark_models import DocumentVectorCache
    digest = embedding_fingerprint(Settings(state=tmp_path))
    folder = tmp_path/'cache'/('vectors-'+digest);folder.mkdir(parents=True)
    cache = DocumentVectorCache(tmp_path/'benchmark-cache',{'suite':'one'},768)
    vector = np.zeros(768,dtype=np.float32);vector[0]=1.
    documents=[]
    for text in ['first','second','third']:
        import hashlib
        np.save(folder/(hashlib.sha256(text.encode()).hexdigest()+'.npy'),vector)
        inputs = ['passage: '+text]
        documents.append({'text':text,'inputs':inputs,'keys':[cache.key(text,inputs[0])]})
    class Model:
        calls = []
        def encode(self,texts,**kwargs):
            self.calls.append(texts)
            return np.tile(vector,(len(texts),1))
    class Monitor:
        def check(self):pass
    return cache,documents,Model(),Monitor(),{'index_metadata':{'embedding_fingerprint':digest}}


def test_index_reuse_requires_exact_single_window_and_recomputes_subset(tmp_path):
    from scripts.benchmark_models import reuse_index_vectors
    cache,documents,model,monitor,suite = index_cache_fixture(tmp_path)
    documents[2]['inputs']=['passage: changed']
    result = reuse_index_vectors(benchmark_args(tmp_path),suite,{'model':{'files_sha256':'actual-weights-sha'}},
                                cache,documents,model,monitor)
    assert result['new_unique_vectors_reused']==2 and len(result['sampled_keys'])==2
    assert len(model.calls)==1 and set(model.calls[0])=={'passage: first','passage: second'}
    assert result['document_throughput_eligible'] is False
    assert set(cache.sources.values())=={'index_verified'}


def test_index_reuse_disagreement_imports_nothing(tmp_path):
    import numpy as np
    from scripts.benchmark_models import reuse_index_vectors
    cache,documents,model,monitor,suite = index_cache_fixture(tmp_path)
    def different(texts,**kwargs):
        values=np.zeros((len(texts),768),dtype=np.float32);values[:,1]=1.;return values
    model.encode=different
    with pytest.raises(ValueError,match='subset disagrees'):
        reuse_index_vectors(benchmark_args(tmp_path),suite,{'model':{'files_sha256':'sha'}},cache,documents,model,monitor)
    assert not cache.vectors and not list(cache.folder.glob('batch-*.json'))


def test_index_reuse_rejects_wrong_metadata_or_model(tmp_path):
    from scripts.benchmark_models import reuse_index_vectors
    cache,documents,model,monitor,suite = index_cache_fixture(tmp_path)
    args=benchmark_args(tmp_path)
    with pytest.raises(ValueError,match='metadata'):
        reuse_index_vectors(args,{}, {'model':{'files_sha256':'sha'}},cache,documents,model,monitor)
    args.model='qwen-embedding'
    with pytest.raises(ValueError,match='pinned E5'):
        reuse_index_vectors(args,suite,{'model':{'files_sha256':'sha'}},cache,documents,model,monitor)


def fake_embed_environment(tmp_path,monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace
    import numpy as np
    from scripts import benchmark_models as bench
    class Tokenizer:
        def __call__(self,text,**kwargs):
            return {'input_ids':list(range(len(text))), 'offset_mapping':[(i,i+1) for i in range(len(text))]}
    class Model:
        tokenizer=Tokenizer();calls=[];fail=True;dtype='torch.float32'
        def parameters(self):return iter([SimpleNamespace(dtype=self.dtype)])
        def encode(self,texts,**kwargs):
            self.calls.append(list(texts))
            if texts==['query: question'] and self.fail:raise RuntimeError('Interrupted query')
            return np.tile([1.,0.],(len(texts),1))
    model=Model()
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda *_:None,float32='torch.float32'))
    def construct(*_a,**kwargs):
        assert kwargs['model_kwargs']=={'torch_dtype':'torch.float32'}
        return model
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=construct))
    monkeypatch.setitem(bench.ALL_MODELS,'e5',{**bench.ALL_MODELS['e5'],'dimension':2})
    monkeypatch.setattr(bench,'local_identity',lambda *_:{**bench.ALL_MODELS['e5'],'files_sha256':'fake'})
    monkeypatch.setattr(bench.importlib.metadata,'version',lambda *_:'fake')
    monkeypatch.setattr(bench,'load_suite',lambda *_:({'fingerprint':'toy','corpus_sampled':True},
        [{'id':'a','text':'document','locator':{}}],[{'id':'q','question':'question','evidence':[]}],{}))
    class Monitor:
        def __init__(self,*_):pass
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def stage(self,*_):return contextlib.nullcontext()
        def before_load(self,*_):pass
        def check(self):pass
        def report(self):return {}
    monkeypatch.setattr(bench,'ResourceMonitor',Monitor)
    return bench,benchmark_args(tmp_path),model


def test_embed_resume_reuses_documents_but_reinfers_failed_query_without_query_cache(tmp_path,monkeypatch):
    bench,args,model=fake_embed_environment(tmp_path,monkeypatch)
    assert bench.embed(args)['status']=='failed'
    model.fail=False;args.resume=True
    result=bench.embed(args)
    assert result['status']=='completed_exploratory'
    assert model.calls.count(['passage: document'])==1
    assert model.calls.count(['query: question'])==2
    assert result['document_cache']['query_vectors_cached'] is False
    assert result['document_cache']['cached_window_occurrences']==1
    assert result['document_cache']['document_throughput_eligible'] is False
    assert result['schema']=='model-benchmark-v4'
    assert result['embedding_precision']['actual_parameter_dtypes']==['torch.float32']
    identity=json.loads(args.output.with_suffix('.vectors').joinpath('manifest.json').read_text())
    assert identity['contract']['embedding_precision']==result['embedding_precision']


def test_resource_guard_latches_peak_even_after_memory_falls(monkeypatch):
    monitor=ResourceMonitor(12);monitor.peak=13*1024**3
    monkeypatch.setattr(monitor,'sample',lambda:11*1024**3)
    with pytest.raises(MemoryError,match='RSS budget'):monitor.check()


def test_background_sampling_failure_is_not_silently_treated_as_measured(monkeypatch):
    import threading
    monitor=ResourceMonitor();failed=threading.Event()
    def sample():
        failed.set();raise RuntimeError('Process measurement unavailable')
    monkeypatch.setattr(monitor,'sample',sample)
    with pytest.raises(RuntimeError,match='Resource sampling failed'):
        with monitor:
            assert failed.wait(2)
            monitor.check()
    assert monitor.report()['worker_children_qdrant_peak_rss_bytes'] is None
    assert monitor.report()['sampling_status']=='unavailable'


@pytest.mark.parametrize('scores',[[],[1.],[1.,float('nan')],[1.,float('inf')],[[1.],[2.]],[True,False]])
def test_short_nonfinite_or_malformed_model_scores_fail_closed(scores):
    from scripts.benchmark_models import validated_scores
    with pytest.raises(ValueError,match='complete'):validated_scores(scores,2)


def committed_row():
    from scripts.benchmark_models import query_commit
    report={'schema':'model-benchmark-v3','kind':'rerank','pool_fingerprint':'pool'}
    row={'id':'q','warmup_excluded':True,'elapsed_seconds':1.,
         'ranked':[{'id':'a','score':2.},{'id':'b','score':1.}],
         'metrics':{'eligible':False,'reason':'No literal gold'}}
    return report,query_commit(report,row)


@pytest.mark.parametrize('kind',['id_only','altered','short','nan','duplicate','missing_metric','cold'])
def test_resume_rejects_incomplete_or_invalid_query_records(kind):
    from scripts.benchmark_models import completed_case_ids,query_commit
    report,row=committed_row()
    if kind=='id_only':row={'id':'q'}
    elif kind=='altered':row['ranked'][0]['score']=999
    elif kind=='short':row['ranked'].pop()
    elif kind=='nan':row['ranked'][0]['score']=float('nan')
    elif kind=='duplicate':row['ranked'][1]['id']='a'
    elif kind=='missing_metric':row.pop('metrics')
    elif kind=='cold':row['warmup_excluded']=False
    if kind not in {'id_only','altered'}:query_commit(report,row)
    report['results']=[row]
    with pytest.raises(ValueError):completed_case_ids(report,[{'id':'q'}],['a','b'],{'q':['a','b']})


def test_conditional_metrics_are_separate_and_cache_preserves_strict_scores(tmp_path,monkeypatch):
    from scripts import benchmark_models as bench
    case={'id':'q','evidence_groups':[[{'path':'a','span_text':'available'}],[{'path':'a','span_text':'missing'}]]}
    corpus=[{'id':'a','path':'a','text':'available','locator':{}}]
    strict=bench.span_metrics(case,corpus,corpus)
    cache=bench.LiteralCoverageCache(tmp_path,'suite',{'scoring':'v1'},corpus)
    covered=cache.get(case)
    assert strict==bench.span_metrics(case,corpus,corpus,covered)
    assert strict['span_recall_at_10']==.5 and strict['corpus_span_coverage']==.5
    assert strict['span_recall_at_80']==.5 and strict['ndcg_at_10']==1.
    extra=strict['supplemental']
    assert extra['recall_at_10_given_literal_coverability']==1.
    assert extra['publication_eligible'] is False and extra['excluded_literal_groups']==1
    monkeypatch.setattr(bench,'literal_coverability',lambda *_:(_ for _ in ()).throw(AssertionError('Must use frozen cache')))
    assert bench.LiteralCoverageCache(tmp_path,'suite',{'scoring':'v1'},corpus).get(case)==covered
    summary=bench.summarize([{'metrics':strict}])
    assert summary['span_recall_at_10']==.5
    assert summary['supplemental']['recall_at_10_given_literal_coverability']==1.
    assert summary['warm_query_p95_seconds'] is None


def test_no_coverable_gold_reports_unknown_conditional_metric_and_cache_tamper_rejects(tmp_path):
    from scripts import benchmark_models as bench
    case={'id':'q','evidence':[{'path':'a','span_text':'missing'}]}
    corpus=[{'id':'a','path':'a','text':'other','locator':{}}]
    cache=bench.LiteralCoverageCache(tmp_path,'suite',{'scoring':'v1'},corpus)
    result=bench.span_metrics(case,corpus,corpus,cache.get(case))
    assert result['span_recall_at_10']==0.
    assert result['supplemental']['recall_at_10_given_literal_coverability'] is None
    assert result['supplemental']['ndcg_at_10_given_any_literal_match'] is None
    path=next(cache.folder.glob('*.json'));data=json.loads(path.read_text());data['value']['available']=[True];save(path,data)
    with pytest.raises(ValueError,match='cache changed'):
        bench.LiteralCoverageCache(tmp_path,'suite',{'scoring':'v1'},corpus).get(case)


class FakeForwardModule:
    """Tiny hook-capable module, with no torch import or model allocation."""
    def __init__(self, action):
        self.action=action;self.pre_hooks=[];self.post_hooks=[]

    def register_forward_pre_hook(self, hook):
        from types import SimpleNamespace
        self.pre_hooks.append(hook)
        return SimpleNamespace(remove=lambda:self.pre_hooks.remove(hook) if hook in self.pre_hooks else None)

    def register_forward_hook(self, hook, *, always_call=False):
        from types import SimpleNamespace
        item=(hook,always_call);self.post_hooks.append(item)
        return SimpleNamespace(remove=lambda:self.post_hooks.remove(item) if item in self.post_hooks else None)

    def __call__(self, batch):
        try:
            for hook in list(self.pre_hooks):hook(self,(batch,))
            output=self.action(batch)
        except Exception:
            for hook,always in list(self.post_hooks):
                if always:hook(self,(batch,),None)
            raise
        for hook,_ in list(self.post_hooks):hook(self,(batch,),output)
        return output


def fake_rerank_environment(tmp_path,monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace
    from scripts import benchmark_models as bench
    import ragapp.retrieval
    model=SimpleNamespace(calls=[],fail=True,bad=None)
    model.parameters=lambda:iter([SimpleNamespace(dtype='torch.float32')])
    def forward(pairs):
        model.calls.extend(q for q,d in pairs)
        if any(q=='second' for q,d in pairs) and model.fail:raise RuntimeError('Disconnected')
        if any(q=='first' for q,d in pairs) and model.bad is not None:return model.bad
        return [1.]*len(pairs)
    module=FakeForwardModule(forward)
    model.register_forward_pre_hook=module.register_forward_pre_hook
    model.register_forward_hook=module.register_forward_hook
    model.forward_module=module
    def predict(pairs,**_):return module(pairs)
    model.predict=predict
    class Retrieval:
        def __init__(self,*_):
            # A fresh process creates a fresh predictor; do not wrap a prior session wrapper.
            model.predict=predict;self.reranker=model
        def rank(self,query,rows,trace=None):
            return self.reranker.predict([(query,row['text']) for row in rows])
    class Monitor:
        def __init__(self,*_):pass
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def stage(self,*_):return contextlib.nullcontext()
        def before_load(self,*_):pass
        def check(self):pass
        def report(self):return {}
    monkeypatch.setattr(ragapp.retrieval,'Retrieval',Retrieval)
    monkeypatch.setattr(bench,'ResourceMonitor',Monitor)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda *_:None))
    monkeypatch.setattr(bench,'local_identity',lambda *_:{'repository':'fake','files_sha256':'fake'})
    monkeypatch.setattr(bench.importlib.metadata,'version',lambda *_:'fake')
    cases=[{'id':'q1','question':'first','evidence':[]},{'id':'q2','question':'second','evidence':[]}]
    corpus=[{'id':key,'path':'a','text':key,'locator':{}} for key in ('a','b')]
    monkeypatch.setattr(bench,'load_suite',lambda *_:({'fingerprint':'suite','corpus_sampled':True},corpus,cases,{}))
    args=benchmark_args(tmp_path);args.command='rerank';args.model='bge-reranker';args.pool=tmp_path/'pool.json'
    pool={'suite_fingerprint':'suite','pools':{'q1':['a','b'],'q2':['a','b']}}
    pool['fingerprint']=fingerprint(pool);save(args.pool,pool)
    return bench,args,model,pool


def test_rerank_query_resume_skips_only_complete_rows_and_warms_each_session(tmp_path,monkeypatch):
    bench,args,model,pool=fake_rerank_environment(tmp_path,monkeypatch)
    first=bench.rerank(args)
    assert first['status']=='failed' and [r['id'] for r in first['results']]==['q1']
    assert len(first['warmup_sessions'])==1 and first['results'][0]['commit']
    args.resume=True;model.fail=False
    second=bench.rerank(args)
    assert second['status']=='completed_exploratory' and len(second['results'])==2
    assert len(second['warmup_sessions'])==2
    assert model.calls.count('检索性能预热')==2
    assert model.calls.count('first')==2  # Two candidates; this completed query was not repeated.
    assert second['metrics']['warm_query_p95_seconds'] is not None
    assert second['rerank_precision']['actual_parameter_dtypes']==['torch.float32']
    assert second['settings']['reranker_parameter_dtype']=='torch.float32'
    assert second['settings']['batching_protocol']==bench.RERANK_BATCHING_PROTOCOL
    assert not model.forward_module.pre_hooks and not model.forward_module.post_hooks


@pytest.mark.parametrize('bad',[[],[1.],[1.,float('nan')]])
def test_rerank_does_not_commit_short_or_nan_model_output(tmp_path,monkeypatch,bad):
    bench,args,model,pool=fake_rerank_environment(tmp_path,monkeypatch);model.bad=bad
    result=bench.rerank(args)
    assert result['status']=='failed' and not result['results']
    assert 'complete' in result['error']


@pytest.mark.parametrize('mutation',[
    lambda pools:pools.pop('q2'),lambda pools:pools.update(q2=[]),lambda pools:pools.update(q2=['a','a'])])
def test_rerank_missing_or_duplicate_pool_rejects_before_loading_model(tmp_path,monkeypatch,mutation):
    bench,args,model,pool=fake_rerank_environment(tmp_path,monkeypatch)
    mutation(pool['pools']);pool['fingerprint']=fingerprint({k:v for k,v in pool.items() if k!='fingerprint'});save(args.pool,pool)
    with pytest.raises(ValueError,match='pool'):bench.rerank(args)
    assert not model.calls


@pytest.mark.parametrize('dtypes',[[],['torch.bfloat16'],['torch.float32','torch.bfloat16']])
def test_embedding_precision_requires_actual_fp32_parameters(dtypes):
    from types import SimpleNamespace
    from scripts.benchmark_models import checked_embedding_precision
    model=SimpleNamespace(parameters=lambda:iter(SimpleNamespace(dtype=d) for d in dtypes))
    with pytest.raises(ValueError,match='parameter dtype'):
        checked_embedding_precision(model,{})


def test_wrong_actual_embedding_dtype_fails_before_cache_or_inference(tmp_path,monkeypatch):
    bench,args,model=fake_embed_environment(tmp_path,monkeypatch);model.dtype='torch.bfloat16'
    result=bench.embed(args)
    assert result['status']=='failed' and 'parameter dtype' in result['error']
    assert not model.calls and not args.output.with_suffix('.vectors').exists()


def test_resume_actual_precision_mismatch_is_rejected(tmp_path,monkeypatch):
    bench,args,model=fake_embed_environment(tmp_path,monkeypatch)
    result=bench.embed(args)
    result['embedding_precision']['actual_parameter_dtypes']=['torch.bfloat16'];save(args.output,result)
    args.resume=True;model.fail=False;before=len(model.calls)
    resumed=bench.embed(args)
    assert resumed['status']=='failed' and 'precision changed' in resumed['error']
    assert len(model.calls)==before


def test_float32_output_cast_does_not_relax_normalization_tolerance():
    import numpy as np
    from scripts.benchmark_models import checked_vectors
    # BF16 computation can be returned as float32 without unit-length accuracy.
    with pytest.raises(ValueError,match='normalized'):
        checked_vectors(np.array([[1.00159,0]],dtype=np.float32),1,2)


def test_embedding_precision_is_bound_to_query_and_cache_contract(tmp_path,monkeypatch):
    import copy
    from scripts import benchmark_models as bench
    monkeypatch.setattr(bench,'local_identity',lambda *_:{'repository':'fake','files_sha256':'sha'})
    monkeypatch.setattr(bench.importlib.metadata,'version',lambda *_:'fake')
    report=bench.report_base(benchmark_args(tmp_path),{'fingerprint':'suite','corpus_sampled':True})
    report['embedding_precision']={'requested_parameter_dtype':'torch.float32',
        'actual_parameter_dtypes':['torch.float32'],'validation':'all_parameters_checked'}
    original=bench.embedding_contract(report);context=bench.query_context(report)
    cache=bench.DocumentVectorCache(tmp_path/'precision-cache',original,2)
    changed=copy.deepcopy(report);changed['embedding_precision']['actual_parameter_dtypes']=['torch.bfloat16']
    assert bench.query_context(changed)!=context
    with pytest.raises(ValueError,match='identity changed'):
        bench.DocumentVectorCache(cache.folder,bench.embedding_contract(changed),2)


def test_tokenization_stage_checkpoints_precede_document_inference(tmp_path,monkeypatch):
    import copy
    bench,args,model=fake_embed_environment(tmp_path,monkeypatch)
    original=bench.save;updates=[]
    def capture(path,data):
        if path==args.output and data.get('progress'):updates.append(copy.deepcopy(data['progress']))
        original(path,data)
    monkeypatch.setattr(bench,'save',capture);bench.embed(args)
    tokenization=[r for r in updates if r['stage']=='document_tokenization']
    assert [(r['completed'],r['total']) for r in tokenization]==[(0,1),(1,1)]
    assert updates[0]['stage']=='document_tokenization'
    assert any(r['stage']=='document_embedding' for r in updates)


def test_import_storage_batch_is_independent_of_inference_and_reports_progress(tmp_path):
    from scripts.benchmark_models import reuse_index_vectors
    cache,documents,model,monitor,suite=index_cache_fixture(tmp_path)
    args=benchmark_args(tmp_path,import_batch_size=3);progress=[]
    result=reuse_index_vectors(args,suite,{'model':{'files_sha256':'sha'}},cache,documents,model,monitor,progress.append)
    assert [len(call) for call in model.calls]==[2,1]
    assert len(list(cache.folder.glob('batch-*.json')))==1
    assert result['import_storage_batch_size']==3 and not result['document_throughput_eligible']
    assert {r['stage'] for r in progress}=={'index_source_validation','index_subset_recomputation','index_import_persistence'}
    assert progress[-1]['completed']==progress[-1]['total']==3


def test_interrupted_index_import_resumes_only_remaining_vectors(tmp_path):
    from scripts.benchmark_models import DocumentVectorCache,reuse_index_vectors
    cache,documents,model,monitor,suite=index_cache_fixture(tmp_path)
    args=benchmark_args(tmp_path,import_batch_size=2);report={'model':{'files_sha256':'sha'}}
    def interrupt(progress):
        if progress['stage']=='index_import_persistence' and progress['completed']==2:
            raise RuntimeError('Disconnected after commit')
    with pytest.raises(RuntimeError,match='Disconnected'):
        reuse_index_vectors(args,suite,report,cache,documents,model,monitor,interrupt)
    resumed=DocumentVectorCache(cache.folder,cache.contract,768)
    result=reuse_index_vectors(args,suite,report,resumed,documents,model,monitor)
    assert result['new_unique_vectors_reused']==1 and result['total_committed_index_vectors']==3


def fake_process_sampling(monkeypatch):
    import psutil
    from types import SimpleNamespace
    calls=[];discoveries=[];clock=[0.]
    class Process:
        def __init__(self,pid,name,rss):self.pid=pid;self.info={'name':name};self.rss=rss;self.alive=True;self.error=None
        def children(self,recursive):assert recursive;return [child,database]
        def is_running(self):return self.alive
        def memory_info(self):
            calls.append(self.pid)
            if self.error:raise self.error
            return SimpleNamespace(rss=self.rss)
    worker=Process(1,'python.exe',100);child=Process(2,'child.exe',20)
    database=Process(3,'qdrant.exe',30);unrelated=Process(4,'other.exe',100000)
    def discover(attrs):discoveries.append(attrs);return iter([worker,child,database,unrelated])
    monkeypatch.setattr(psutil,'Process',lambda:worker)
    monkeypatch.setattr(psutil,'process_iter',discover)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(used=500))
    monkeypatch.setattr('scripts.benchmark_models.time.monotonic',lambda:clock[0])
    return worker,child,database,unrelated,calls,discoveries,clock


def test_resource_monitor_samples_only_targets_and_discovers_database_infrequently(monkeypatch):
    worker,child,database,other,calls,discoveries,clock=fake_process_sampling(monkeypatch)
    monitor=ResourceMonitor()
    assert monitor.sample()==150
    assert sorted(calls)==[1,2,3]  # Database child counted once; unrelated RSS never accessed.
    clock[0]=4.;monitor.sample();assert discoveries==[['name']]
    clock[0]=6.;monitor.sample();assert discoveries==[['name'],['name']]
    assert 4 not in calls and monitor.peak==150
    assert monitor.report()['sampling_protocol']=='target-pid-rss-v2'


def test_target_rss_access_failure_is_unknown_and_fails_closed(monkeypatch):
    import psutil
    worker,child,database,other,calls,discoveries,clock=fake_process_sampling(monkeypatch)
    child.error=psutil.AccessDenied(child.pid);monitor=ResourceMonitor()
    with pytest.raises(RuntimeError,match='sampling failed'):monitor.check()
    assert monitor.report()['worker_children_qdrant_peak_rss_bytes'] is None
    assert monitor.report()['sampling_status']=='unavailable'


def test_disappeared_target_does_not_hide_live_target_rss_or_clear_peak(monkeypatch):
    worker,child,database,other,calls,discoveries,clock=fake_process_sampling(monkeypatch)
    monitor=ResourceMonitor();monitor.check();child.alive=False;database.alive=False
    assert monitor.sample()==100 and monitor.peak==150


def test_live_process_name_discovery_failure_is_not_ignored(monkeypatch):
    worker,child,database,other,calls,discoveries,clock=fake_process_sampling(monkeypatch)
    monkeypatch.setattr('scripts.benchmark_models.windows_process_names',lambda:{})
    other.info['name']=None;monitor=ResourceMonitor()
    with pytest.raises(RuntimeError,match='discovery name unavailable'):monitor.check()


def test_protected_name_fallback_still_detects_and_measures_database(monkeypatch):
    worker,child,database,other,calls,discoveries,clock=fake_process_sampling(monkeypatch)
    database.info['name']=None;other.info['name']=None;fallback=[]
    def names():fallback.append(True);return {3:'qdrant.exe',4:'Protected System'}
    monkeypatch.setattr('scripts.benchmark_models.windows_process_names',names)
    monitor=ResourceMonitor();assert monitor.sample()==150
    assert len(fallback)==1 and 4 not in calls and 3 in monitor.database_processes


def test_name_fallback_backend_failure_fails_closed(monkeypatch):
    worker,child,database,other,calls,discoveries,clock=fake_process_sampling(monkeypatch)
    other.info['name']=None
    def unavailable():raise OSError('Snapshot unavailable')
    monkeypatch.setattr('scripts.benchmark_models.windows_process_names',unavailable)
    monitor=ResourceMonitor()
    with pytest.raises(RuntimeError,match='Snapshot unavailable'):monitor.check()
    assert monitor.report()['sampling_status']=='unavailable'


def test_new_rerank_precision_changes_query_identity_but_old_embed_context_is_unchanged():
    from scripts.benchmark_models import query_context,query_commit,validate_query_row
    base={'schema':'model-benchmark-v4','kind':'embed','embedding_precision':{'actual':'old-value'}}
    old_keys=('schema','kind','suite_fingerprint','model','settings','code_sha256','packages','pool_fingerprint','embedding_precision')
    assert query_context(base)==fingerprint({key:base.get(key) for key in old_keys})
    report,row=committed_row();previous=query_context(report)
    # Legacy v3 remains valid and has no inferred precision claim.
    validate_query_row(report,row,['a','b'],{'q':['a','b']})
    report['rerank_precision']={'requested_parameter_dtype':'torch.float32',
        'actual_parameter_dtypes':['torch.float32'],'validation':'all_parameters_checked'}
    report['settings']={'reranker_parameter_dtype':'torch.float32'}
    assert query_context(report)!=previous
    with pytest.raises(ValueError,match='checkpoint'):validate_query_row(report,row,['a','b'],{'q':['a','b']})
    query_commit(report,row);validate_query_row(report,row,['a','b'],{'q':['a','b']})


def test_new_rerank_report_cannot_claim_requested_dtype_without_actual_validation():
    from scripts.benchmark_models import query_commit,validate_query_row
    report,row=committed_row();report['schema']='model-benchmark-v4'
    report['settings']={'reranker_parameter_dtype':'torch.float32'};query_commit(report,row)
    with pytest.raises(ValueError,match='precision'):validate_query_row(report,row,['a','b'],{'q':['a','b']})


def test_rerank_wrong_actual_precision_fails_before_warmup_or_inference(tmp_path,monkeypatch):
    from types import SimpleNamespace
    bench,args,model,pool=fake_rerank_environment(tmp_path,monkeypatch)
    model.parameters=lambda:iter([SimpleNamespace(dtype='torch.bfloat16')])
    result=bench.rerank(args)
    assert result['status']=='failed' and 'FP32' in result['error']
    assert not model.calls and not result['results']


def test_resume_rerank_precision_change_rejects_before_reinference(tmp_path,monkeypatch):
    bench,args,model,pool=fake_rerank_environment(tmp_path,monkeypatch)
    first=bench.rerank(args);first['rerank_precision']['actual_parameter_dtypes']=['torch.bfloat16'];save(args.output,first)
    args.resume=True;before=len(model.calls)
    with pytest.raises(ValueError,match='precision'):bench.rerank(args)
    assert len(model.calls)==before


def native_batch_fixture(kind='bge', fail_forward=False):
    from types import SimpleNamespace
    batches=[];predict_calls=[];events=[]
    def forward(batch):
        events.append('forward');batches.append(list(batch))
        if fail_forward:raise RuntimeError('synthetic forward failed')
        return [float(q) for q,d in batch]
    if kind=='bge':
        class Adapter(FakeForwardModule):
            def predict(self,pairs,batch_size=4,**kwargs):
                predict_calls.append((pairs,batch_size,kwargs))
                ordered=sorted(range(len(pairs)),key=lambda i:-sum(map(len,pairs[i])))
                output=[None]*len(pairs)
                for start in range(0,len(ordered),batch_size):
                    indices=ordered[start:start+batch_size]
                    values=self([pairs[i] for i in indices])
                    for i,value in zip(indices,values):output[i]=value
                return output
        adapter=Adapter(forward);target=adapter
    else:
        class Adapter:
            def __init__(self):self.model=FakeForwardModule(forward)
            def predict(self,pairs,batch_size=4,**kwargs):
                predict_calls.append((pairs,batch_size,kwargs));output=[]
                for start in range(0,len(pairs),batch_size):output.extend(self.model(pairs[start:start+batch_size]))
                return output
        adapter=Adapter();target=adapter.model
    monitor=SimpleNamespace(check=lambda:events.append('check'))
    pairs=[(str(i),'x'*n) for i,n in enumerate([100,1,90,2,80,3,70,4,60])]
    return adapter,target,monitor,pairs,batches,predict_calls,events


@pytest.mark.parametrize('kind',['bge','qwen'])
def test_native_reranker_calls_predict_once_and_checks_each_forward_without_reordering(kind):
    from scripts.benchmark_models import native_reranker_batches
    adapter,target,monitor,pairs,batches,calls,events=native_batch_fixture(kind)
    original=adapter.predict;existing=[]
    pre=target.register_forward_pre_hook(lambda *_:existing.append('pre'))
    post=target.register_forward_hook(lambda *_:existing.append('post'))
    with native_reranker_batches(adapter,monitor,4) as checks:
        values=adapter.predict(pairs,batch_size=99,show_progress_bar=True)
        assert values==[float(i) for i in range(len(pairs))]
        assert len(calls)==1 and calls[0][0] is pairs and calls[0][1]==4
        assert calls[0][2]['show_progress_bar'] is False
        assert checks=={'before':3,'after':3}
    assert events==['check','forward','check']*3
    assert existing==['pre','post']*3
    assert len(target.pre_hooks)==len(target.post_hooks)==1
    assert 'predict' not in vars(adapter) and adapter.predict==original
    expected=sorted(pairs,key=lambda pair:-sum(map(len,pair))) if kind=='bge' else pairs
    assert [pair for batch in batches for pair in batch]==expected
    pre.remove();post.remove()


@pytest.mark.parametrize('kind',['bge','qwen'])
def test_forward_failure_runs_post_rss_check_and_restores_hooks_and_predict(kind):
    from scripts.benchmark_models import native_reranker_batches
    adapter,target,monitor,pairs,batches,calls,events=native_batch_fixture(kind,fail_forward=True)
    original=adapter.predict
    with pytest.raises(RuntimeError,match='synthetic forward failed'):
        with native_reranker_batches(adapter,monitor,4):adapter.predict(pairs)
    assert events==['check','forward','check']
    assert len(calls)==1 and adapter.predict==original and 'predict' not in vars(adapter)
    assert not target.pre_hooks and not target.post_hooks


@pytest.mark.parametrize('fail_at',[1,2,3,4])
def test_forward_rss_failure_aborts_and_cleans_up(fail_at):
    from scripts.benchmark_models import native_reranker_batches
    adapter,target,monitor,pairs,batches,calls,events=native_batch_fixture()
    original=adapter.predict;checks=[]
    def check():
        checks.append(len(checks)+1)
        if len(checks)==fail_at:raise MemoryError('synthetic RSS budget exceeded')
    monitor.check=check
    with pytest.raises(MemoryError,match='RSS budget'):
        with native_reranker_batches(adapter,monitor,4):adapter.predict(pairs)
    assert len(batches)<3 and adapter.predict==original
    assert not target.pre_hooks and not target.post_hooks


def test_partial_hook_registration_failure_removes_pre_hook_without_patching_predict():
    from scripts.benchmark_models import native_reranker_batches
    adapter,target,monitor,pairs,batches,calls,events=native_batch_fixture()
    original=adapter.predict
    def failure(*args,**kwargs):raise RuntimeError('synthetic hook registration failed')
    target.register_forward_hook=failure
    with pytest.raises(RuntimeError,match='registration'):
        with native_reranker_batches(adapter,monitor,4):pytest.fail('Must not enter unguarded context')
    assert not target.pre_hooks and adapter.predict==original and 'predict' not in vars(adapter)


def test_existing_predict_override_is_restored_and_no_forward_cannot_claim_guarded_inference():
    from scripts.benchmark_models import native_reranker_batches
    adapter,target,monitor,pairs,batches,calls,events=native_batch_fixture()
    override=lambda pairs,**kwargs:[1.]*len(pairs)
    adapter.predict=override
    with pytest.raises(RuntimeError,match='guarded forward'):
        with native_reranker_batches(adapter,monitor,4):adapter.predict(pairs)
    assert adapter.predict is override and not target.pre_hooks and not target.post_hooks


def test_missing_forward_hooks_fails_before_prediction():
    from types import SimpleNamespace
    from scripts.benchmark_models import native_reranker_batches
    adapter=SimpleNamespace(predict=lambda *_:pytest.fail('Unguarded prediction must not run'))
    with pytest.raises(ValueError,match='forward hooks'):
        with native_reranker_batches(adapter,SimpleNamespace(check=lambda:None),4):pass


@pytest.mark.parametrize('schema',['model-benchmark-v3','model-benchmark-v4'])
def test_legacy_rerank_query_commits_still_readable_and_new_batching_identity_is_bound(schema):
    from scripts import benchmark_models as bench
    report,row=committed_row();report['schema']=schema;report['settings']={'batch_size':4}
    keys=['schema','kind','suite_fingerprint','model','settings','code_sha256','packages','pool_fingerprint']
    if schema=='model-benchmark-v4':keys.append('embedding_precision')
    assert bench.query_context(report)==fingerprint({key:report.get(key) for key in keys})
    bench.query_commit(report,row);bench.validate_query_row(report,row,['a','b'],{'q':['a','b']})
    report['settings']['batching_protocol']=bench.RERANK_BATCHING_PROTOCOL
    with pytest.raises(ValueError,match='checkpoint'):
        bench.validate_query_row(report,row,['a','b'],{'q':['a','b']})
    bench.query_commit(report,row);bench.validate_query_row(report,row,['a','b'],{'q':['a','b']})


def test_legacy_batching_report_cannot_resume_in_new_protocol_even_with_current_code(tmp_path,monkeypatch):
    from scripts import benchmark_models as bench
    monkeypatch.setattr(bench,'local_identity',lambda *_:{'repository':'fake','files_sha256':'sha'})
    monkeypatch.setattr(bench.importlib.metadata,'version',lambda *_:'fake')
    args=benchmark_args(tmp_path);args.command='rerank';args.model='bge-reranker'
    manifest={'fingerprint':'suite','corpus_sampled':True}
    report=bench.report_base(args,manifest)
    assert report['settings'].pop('batching_protocol')==bench.RERANK_BATCHING_PROTOCOL
    save(args.output,report);args.resume=True
    with pytest.raises(ValueError,match='identity changed'):bench.report_base(args,manifest)
    assert bench.read(args.output)==report


def test_historical_declared_batching_identity_remains_readable_without_relabeling():
    from scripts import benchmark_models as bench
    report,row=committed_row()
    report['settings']={'batching_protocol':'historical-outer-sliced-experiment'}
    bench.query_commit(report,row)
    bench.validate_query_row(report,row,['a','b'],{'q':['a','b']})
    assert report['settings']['batching_protocol']=='historical-outer-sliced-experiment'


def test_new_protocol_is_reranker_only(tmp_path,monkeypatch):
    from scripts import benchmark_models as bench
    monkeypatch.setattr(bench,'local_identity',lambda *_:{'repository':'fake','files_sha256':'sha'})
    monkeypatch.setattr(bench.importlib.metadata,'version',lambda *_:'fake')
    report=bench.report_base(benchmark_args(tmp_path),{'fingerprint':'suite','corpus_sampled':True})
    assert 'batching_protocol' not in report['settings']
