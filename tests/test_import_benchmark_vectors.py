"""Generic synthetic text only. Never load real weights, cases, gold or scores."""
import copy
import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from ragapp.pipeline import embedding_fingerprint, fingerprint
from ragapp.retrieval import file_sha256, read_document_vector, document_cache_contract
from ragapp.textbudget import windows
from scripts import import_benchmark_vectors as imp


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data),encoding='utf-8')


class Tokenizer:
    def __call__(self,text,add_special_tokens=True,**kwargs):
        offsets=[(m.start(),m.end()) for m in re.finditer(r'\S',text)]
        return {'input_ids':[0]*(len(offsets)+(2 if add_special_tokens else 0)),'offset_mapping':offsets}


class Model:
    tokenizer=Tokenizer()
    def __init__(self,bad=False):self.calls=[];self.bad=bad
    def parameters(self):return [SimpleNamespace(dtype='torch.float32')]
    def encode(self,texts,**kwargs):
        self.calls.extend(texts);vectors=np.zeros((len(texts),1024),dtype=np.float32)
        vectors[:,1 if self.bad else 0]=1
        return vectors


class Monitor:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def check(self):pass
    def before_load(self,*args):pass
    def report(self):return {'peak_rss_bytes':1,'synthetic':True}


@pytest.fixture
def data(tmp_path):
    state=tmp_path/'state';suite=tmp_path/'suite';cache=tmp_path/'source-cache'
    model_dir=state/'models'/imp.SPEC['directory'];model_dir.mkdir(parents=True)
    (model_dir/'fake.bin').write_bytes(b'not-real-weights');(model_dir/'tokenizer.json').write_bytes(b'not-a-real-tokenizer')
    files={p.name:{'bytes':p.stat().st_size,'sha256':file_sha256(p)} for p in model_dir.iterdir()}
    write(model_dir/'snapshot-manifest.json',{**imp.SPEC,'files':files,'status':'complete'})
    corpus=[{'id':str(i),'text':f'Generic source text number {i}.','locator':{}} for i in range(36)]
    corpus.extend([{'id':'padded','text':'  Generic padded text.  ','locator':{}},
        {'id':'long','text':'x'*650,'locator':{}},{'id':'budget','text':'z'*490,'locator':{}}])
    write(suite/'corpus.json',corpus)
    manifest={'schema':'model-benchmark-v1','corpus_sampled':False,'chunks':len(corpus),
              'full_active_chunks':len(corpus),'files':{'corpus.json':file_sha256(suite/'corpus.json'),
                'cases.json':'deliberately-not-readable','lexical.json':'deliberately-not-readable'}}
    manifest['fingerprint']=fingerprint(manifest);write(suite/'manifest.json',manifest)
    contract={'suite_fingerprint':manifest['fingerprint'],'model':{**imp.SPEC,'files_sha256':fingerprint(files),
        'verification':'all_files_sha256'},'settings':{'maximum_tokens':512},'code_sha256':{'source.py':'c'*64},
        'packages':{'torch':'recorded-source-version'},'input_templates':{'document':'{text}','query':'{text}'},
        'normalize_embeddings':True,'window_overlap':32,'budget_safety_tokens':8,'dtype':'float32','query_cache':False}
    identity={'schema':'document-window-cache-v1','contract':contract,'contract_fingerprint':fingerprint(contract),'dimension':1024}
    write(cache/'manifest.json',identity)
    keys=[]
    for row in corpus:
        text=row['text']
        for full,_,_ in windows(Tokenizer(),text,502,32):
            key=fingerprint({'contract':fingerprint(contract),'dimension':1024,
                'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'input_sha256':hashlib.sha256(full.encode()).hexdigest()})
            if key not in keys:keys.append(key)
    vectors=np.zeros((len(keys),1024),dtype=np.float32);vectors[:,0]=1
    np.save(cache/'batch-one.npy',vectors)
    batch={'keys':keys,'contract_fingerprint':fingerprint(contract),'source':'encoded',
           'sha256':file_sha256(cache/'batch-one.npy'),'vector_sha256':[hashlib.sha256(v.tobytes()).hexdigest() for v in vectors]}
    batch['fingerprint']=fingerprint(batch);write(cache/'batch-one.json',batch)
    args=SimpleNamespace(state=state,suite=suite,source_cache=cache,output=tmp_path/'import.json',
                         sample_size=32,batch_size=4,input_tokens=480,rss_gib=12)
    return SimpleNamespace(args=args,files=files,identity=identity,batch=batch,corpus=corpus,model_dir=model_dir)


def run(data,model=None):
    return imp.import_vectors(data.args,tokenizer_factory=lambda s:Tokenizer(),
                              model_factory=lambda s:model or Model(),monitor=Monitor())


def test_import_full_input_only_and_idempotent_without_opening_cases(data,monkeypatch):
    actual_read=imp.read;opened=[]
    def limited_read(path):
        assert path.name not in {'cases.json','lexical.json'}
        opened.append(path.name);return actual_read(path)
    monkeypatch.setattr(imp,'read',limited_read)
    model=Model();result=run(data,model)
    assert len(model.calls)==32 and result['status']=='complete' and result['committed']==36
    assert result['skipped']=={'multiple_windows':1,'input_bytes_changed':1,'target_budget':1,'empty':0}
    assert not result['publication_eligible'] and result['historical_actual_dtype']=='unknown'
    assert run(data)['proof_fingerprint']==result['proof_fingerprint']
    folder=data.args.state/'cache'/('vectors-'+embedding_fingerprint(imp.candidate_settings(data.args)))
    assert len(list(folder.glob('*.npy')))==36
    proof=imp.read(folder/('import-'+result['proof_fingerprint']+'.json'))
    assert proof['source_contract']==data.identity['contract'] and len(proof['sample_records'])==32
    assert proof['historical_actual_dtype']=='unknown' and proof['sample_precision']['actual_parameter_dtypes']==['torch.float32']
    row=data.corpus[0];contract=document_cache_contract(imp.candidate_settings(data.args),fingerprint(data.files))
    assert read_document_vector(folder,row['text'],row['text'],contract)[0]==1


def test_sample_failure_creates_no_imported_vectors_and_reports_failure(data):
    with pytest.raises(ValueError,match='sample differs'):run(data,Model(bad=True))
    assert not list((data.args.state/'cache').rglob('*.npy'))
    assert imp.read(data.args.output)['status']=='failed'


@pytest.mark.parametrize('change',['hash','norm','nan','shape','dtype','duplicate','orphan','missing_window'])
def test_every_source_batch_checked_before_any_model_load(data,change):
    folder=data.args.source_cache;batch=copy.deepcopy(data.batch);array=np.load(folder/'batch-one.npy')
    if change=='hash':batch['sha256']='b'*64
    elif change=='norm':array[-1]*=2
    elif change=='nan':array[-1,0]=np.nan
    elif change=='shape':array=array[:,:10]
    elif change=='dtype':array=array.astype(np.float64)
    elif change=='duplicate':batch['keys'][-1]=batch['keys'][0]
    elif change=='orphan':np.save(folder/'batch-orphan.npy',array)
    elif change=='missing_window':array=array[:-1];batch['keys']=batch['keys'][:-1]
    if change not in {'hash','duplicate','orphan'}:
        np.save(folder/'batch-one.npy',array);batch['sha256']=file_sha256(folder/'batch-one.npy')
        batch['vector_sha256']=[hashlib.sha256(v.tobytes()).hexdigest() for v in array]
    batch['fingerprint']=fingerprint({k:v for k,v in batch.items() if k!='fingerprint'});write(folder/'batch-one.json',batch)
    with pytest.raises(ValueError):
        imp.import_vectors(data.args,tokenizer_factory=lambda s:Tokenizer(),
                           model_factory=lambda s:pytest.fail('Invalid source loaded weights'),monitor=Monitor())
    assert not list((data.args.state/'cache').rglob('*.npy'))


@pytest.mark.parametrize('change',['model_bytes','extra_file','revision','prefix','suite','missing_code','actual_dtype'])
def test_source_contract_and_pinned_model_are_verified(data,change):
    identity=copy.deepcopy(data.identity);contract=identity['contract']
    if change=='model_bytes':(data.model_dir/'fake.bin').write_bytes(b'changed')
    elif change=='extra_file':(data.model_dir/'injected.py').write_bytes(b'x')
    elif change=='revision':contract['model']['revision']='0'*40
    elif change=='prefix':contract['input_templates']['document']='passage: {text}'
    elif change=='suite':contract['suite_fingerprint']='f'*64
    elif change=='missing_code':contract['code_sha256']={}
    else:contract['embedding_precision']={'actual_parameter_dtypes':['torch.float32']}
    identity['contract_fingerprint']=fingerprint(contract);write(data.args.source_cache/'manifest.json',identity)
    with pytest.raises(ValueError):imp.load_source(data.args.suite,data.args.source_cache,imp.candidate_settings(data.args))


def test_changed_resume_configuration_rejected(data):
    run(data);data.args.sample_size=33
    with pytest.raises(ValueError,match='resume'):run(data)


def test_conflicting_existing_vector_rejects_all_new_imports(data):
    settings=imp.candidate_settings(data.args)
    folder=settings.state/'cache'/('vectors-'+embedding_fingerprint(settings));folder.mkdir(parents=True)
    key=hashlib.sha256(data.corpus[-4]['text'].encode()).hexdigest()
    value=np.zeros(1024,dtype=np.float32);value[1]=1;np.save(folder/(key+'.npy'),value)
    with pytest.raises(ValueError,match='Conflicting'):run(data)
    assert len(list(folder.glob('*.npy')))==1
    assert not list(folder.glob('import-*.json'))


@pytest.mark.parametrize('name,value',[('sample_size',31),('sample_size',True),('batch_size',5),
    ('input_tokens',513),('rss_gib',float('nan')),('rss_gib',13)])
def test_cli_budgets(data,name,value):
    setattr(data.args,name,value)
    with pytest.raises(ValueError):imp.candidate_settings(data.args)


def test_import_proof_tamper_rejected(data):
    result=run(data);folder=__import__('pathlib').Path(result['target_folder'])
    proof_path=folder/('import-'+result['proof_fingerprint']+'.json');proof=imp.read(proof_path)
    proof['historical_actual_dtype']='float32';write(proof_path,proof)
    settings=imp.candidate_settings(data.args);contract=document_cache_contract(settings,fingerprint(data.files))
    text=data.corpus[0]['text']
    with pytest.raises(ValueError,match='proof checksum'):read_document_vector(folder,text,text,contract)


def test_partial_write_is_reported_failed_then_exact_rerun_completes(data,monkeypatch):
    original=imp.write_document_vector;count=0
    def interrupted(*args,**kwargs):
        nonlocal count
        count+=1
        if count==3:raise OSError('synthetic interrupted disk write')
        return original(*args,**kwargs)
    monkeypatch.setattr(imp,'write_document_vector',interrupted)
    with pytest.raises(OSError,match='interrupted'):run(data)
    partial=imp.read(data.args.output)
    assert partial['status']=='failed' and partial['committed']==2
    monkeypatch.setattr(imp,'write_document_vector',original)
    completed=run(data)
    assert completed['status']=='complete' and completed['committed']==36
    assert completed['proof_fingerprint']==partial['proof_fingerprint']


def test_normal_service_reuses_exact_imported_text_and_encodes_new_boundaries(data):
    from ragapp.retrieval import Retrieval, validated_embedding_precision
    run(data);settings=imp.candidate_settings(data.args);model=Model()
    retrieval=Retrieval(settings,None);retrieval._embedding=model
    retrieval._document_cache_contract=document_cache_contract(settings,fingerprint(data.files))
    retrieval.embedding_precision_record=validated_embedding_precision(model)
    retrieval.vectors([data.corpus[0]['text']])
    assert model.calls==[]
    retrieval.vectors(['Generic newly formed chunk.'])
    assert model.calls==['Generic newly formed chunk.']
    retrieval.vectors([data.corpus[0]['text']],query=True)
    assert len(model.calls)==2


def test_shared_benchmark_worker_lock_blocks_import_before_model_or_source_reads(data,monkeypatch):
    from filelock import FileLock, Timeout
    path=data.args.state/'reports/quality-v2-models/model-worker.lock';path.parent.mkdir(parents=True)
    monkeypatch.setattr(imp,'load_source',lambda *a,**k:pytest.fail('Source read while another model runs'))
    with FileLock(str(path),timeout=0):
        with pytest.raises(Timeout):run(data)
    assert not data.args.output.exists()


def test_monitor_exit_failure_cannot_leave_complete_report(data):
    from filelock import FileLock, Timeout
    class ExitFailure(Monitor):
        def __exit__(self,*args):
            assert imp.read(data.args.output)['status']=='importing'
            # All exclusion locks must remain held through the final monitor check.
            for path in (data.args.state/'reports/quality-v2-models/model-worker.lock',
                         data.args.state/'server.lock',data.args.state/'sync.lock'):
                with pytest.raises(Timeout):
                    with FileLock(str(path),timeout=0):pass
            raise RuntimeError('synthetic final resource check failed')
    with pytest.raises(RuntimeError,match='final resource'):
        imp.import_vectors(data.args,tokenizer_factory=lambda s:Tokenizer(),
                           model_factory=lambda s:Model(),monitor=ExitFailure())
    report=imp.read(data.args.output)
    assert report['status']=='failed' and report['committed']==36
    assert 'completed_at_utc' not in report and 'final resource' in report['error']
    assert run(data)['status']=='complete'
