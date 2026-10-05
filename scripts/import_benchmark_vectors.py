"""Import exact single-window BGE-M3 document inputs into an opt-in FP32 cache.

No cases, gold, rankings or experiment scores are read. This does not migrate an
index or publish a model. A normal rebuild can reuse exact texts and encode new
chunk boundaries. Historical computation dtype remains unknown; a deterministic
FP32 sample checks compatibility, never certifies every old computation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ragapp.config import Settings
from ragapp.model_adapters import embedding_inputs
from ragapp.pipeline import embedding_fingerprint, embedding_text, fingerprint
from ragapp.retrieval import (Retrieval, atomic_cache_json, checked_document_vector,
    document_cache_contract, embedding_model_files, file_sha256, read_document_vector,
    validated_embedding_precision, write_document_vector)
from ragapp.textbudget import windows
from scripts.download_candidate_models import MODELS, verify_manifest

SPEC=MODELS['bge-m3']
SCHEMA='benchmark-vector-import-v1'
MINIMUM_SAMPLE=32


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition,message):
    if not condition:raise ValueError(message)


def candidate_settings(args):
    require(type(args.sample_size) is int and args.sample_size>=MINIMUM_SAMPLE,'At least 32 samples are required')
    require(type(args.batch_size) is int and 1<=args.batch_size<=4,'CPU sample batch size must be 1..4')
    require(type(args.input_tokens) is int and 64<=args.input_tokens<=512,'Input token budget must be 64..512')
    require(type(args.rss_gib) in (int,float) and 0<args.rss_gib<=12,'RSS budget must be finite and <=12 GiB')
    return Settings(state=Path(args.state),load_active=False,embedding_model=SPEC['repository'],
        embedding_revision=SPEC['revision'],embedding_directory=SPEC['directory'],embedding_dimension=1024,
        embedding_max_tokens=512,embedding_input_tokens=args.input_tokens,embedding_precision='float32',
        embedding_batch=args.batch_size)


def load_source(suite_folder, cache_folder, settings, check=lambda:None):
    """Validate the legacy contract and EVERY committed batch, without opening cases."""
    import numpy as np
    suite_folder=Path(suite_folder);cache_folder=Path(cache_folder)
    suite=read(suite_folder/'manifest.json')
    require(suite.get('schema')=='model-benchmark-v1'
            and suite.get('fingerprint')==fingerprint({k:v for k,v in suite.items() if k!='fingerprint'}),
            'Frozen corpus manifest checksum/schema mismatch')
    require(suite.get('corpus_sampled') is False and type(suite.get('chunks')) is int
            and suite['chunks']>0 and suite['chunks']==suite.get('full_active_chunks'),'A complete unsampled corpus is required')
    require(file_sha256(suite_folder/'corpus.json')==suite.get('files',{}).get('corpus.json'),'Corpus checksum mismatch')
    corpus=read(suite_folder/'corpus.json')
    require(isinstance(corpus,list) and len(corpus)==suite['chunks'] and all(isinstance(c,dict)
            and isinstance(c.get('id'),str) and c['id'] for c in corpus),'Malformed corpus')
    require(len({c['id'] for c in corpus})==len(corpus),'Duplicate corpus chunk IDs')
    identity=read(cache_folder/'manifest.json');contract=identity.get('contract',{})
    require(identity.get('schema')=='document-window-cache-v1' and identity.get('dimension')==1024
            and type(identity['dimension']) is int and identity.get('contract_fingerprint')==fingerprint(contract),
            'Source document cache identity mismatch')
    legacy_keys={'suite_fingerprint','model','settings','code_sha256','packages','input_templates',
                 'normalize_embeddings','window_overlap','budget_safety_tokens','dtype','query_cache'}
    require(isinstance(contract,dict) and set(contract)==legacy_keys,'Only the explicit legacy v3 cache contract is supported')
    require(contract['suite_fingerprint']==suite['fingerprint'],'Cache and frozen corpus differ')
    require(contract['input_templates']=={'document':'{text}','query':'{text}'}
            and contract['normalize_embeddings'] is True and contract['query_cache'] is False
            and type(contract['window_overlap']) is int and contract['window_overlap']==32
            and type(contract['budget_safety_tokens']) is int and contract['budget_safety_tokens']==8
            and contract['dtype']=='float32','Unsupported cache template/window/normalization contract')
    require(isinstance(contract['settings'],dict) and type(contract['settings'].get('maximum_tokens')) is int
            and contract['settings']['maximum_tokens']==512,'Source maximum must be the frozen 512 tokens')
    require(isinstance(contract['code_sha256'],dict) and contract['code_sha256']
            and all(isinstance(k,str) and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v)
                    for k,v in contract['code_sha256'].items()),'Source code hashes are missing')
    require(isinstance(contract['packages'],dict) and contract['packages'],'Source package versions are missing')
    model=contract['model']
    require(isinstance(model,dict) and all(model.get(k)==v for k,v in SPEC.items())
            and model.get('verification')=='all_files_sha256','Only the pinned dense BGE-M3 snapshot is supported')
    check()
    downloaded=verify_manifest(settings.state/'models'/settings.embedding_directory,SPEC)
    actual_files=embedding_model_files(settings)
    require(actual_files==downloaded['files'],'Local model contains changed, missing or unmanifested files')
    require(fingerprint(actual_files)==model.get('files_sha256'),'Actual pinned model bytes differ from the source cache')
    check()
    vectors={};origins={};batches={}
    commits=sorted(cache_folder.glob('batch-*.json'))
    require(bool(commits),'Source cache has no committed batches')
    require({p.stem for p in cache_folder.glob('batch-*.npy')}=={p.stem for p in commits},
            'Source cache contains an uncommitted or missing batch')
    for number,marker in enumerate(commits):
        if number%32==0:check()
        batch=read(marker);binary=marker.with_suffix('.npy');keys=batch.get('keys')
        require(batch.get('fingerprint')==fingerprint({k:v for k,v in batch.items() if k!='fingerprint'})
                and batch.get('contract_fingerprint')==identity['contract_fingerprint']
                and batch.get('sha256')==file_sha256(binary),'Source batch identity/checksum mismatch')
        require(isinstance(keys,list) and bool(keys) and all(isinstance(k,str)
                and re.fullmatch('[0-9a-f]{64}',k) for k in keys) and len(set(keys))==len(keys),
                'Invalid source batch keys')
        require(batch.get('source')=='encoded','BGE import requires originally encoded source vectors')
        array=np.load(binary,allow_pickle=False)
        require(array.shape==(len(keys),1024) and array.dtype==np.dtype('float32'),'Source vector shape/dtype mismatch')
        for row in array:checked_document_vector(row,1024)
        hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in array]
        require(hashes==batch.get('vector_sha256'),'Source per-vector checksum mismatch')
        for key,row,digest in zip(keys,array,hashes):
            require(key not in vectors,'Duplicate committed source cache key')
            vectors[key]=row;origins[key]={'batch':marker.name,'vector_sha256':digest}
        batches[marker.name]={'metadata_sha256':file_sha256(marker),'binary_sha256':batch['sha256']}
    return {'suite':suite,'corpus':corpus,'identity':identity,'vectors':vectors,'origins':origins,
            'batches':batches,'model_files':actual_files}


def eligible_documents(source, tokenizer, settings, check=lambda:None):
    """All corpus windows must exist; reuse only exact full-input singleton windows."""
    contract=source['identity']['contract'];maximum=contract['settings']['maximum_tokens']
    overhead=len(tokenizer('')['input_ids'])+contract['budget_safety_tokens']
    eligible={};skipped={'multiple_windows':0,'input_bytes_changed':0,'target_budget':0,'empty':0}
    for number,row in enumerate(source['corpus']):
        if number%128==0:check()
        text=embedding_text(row);require(isinstance(text,str),'Document retrieval text must be a string')
        inputs=embedding_inputs(settings,[p for p,_,_ in windows(tokenizer,text,maximum-overhead,32)],False)
        keys=[fingerprint({'contract':source['identity']['contract_fingerprint'],'dimension':1024,
                'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'input_sha256':hashlib.sha256(full.encode()).hexdigest()}) for full in inputs]
        require(all(k in source['vectors'] for k in keys),'Source cache does not cover all frozen corpus windows')
        require(all(len(tokenizer(full,add_special_tokens=True,truncation=False)['input_ids'])<=maximum for full in inputs),
                'Source window exceeds its frozen maximum')
        if not inputs:skipped['empty']+=1;continue
        if len(inputs)!=1:skipped['multiple_windows']+=1;continue
        full=embedding_inputs(settings,[text],False)[0]
        if inputs[0].encode()!=full.encode():skipped['input_bytes_changed']+=1;continue
        count=len(tokenizer(full,add_special_tokens=True,truncation=False)['input_ids'])
        if count>min(settings.embedding_input_tokens,settings.embedding_max_tokens):skipped['target_budget']+=1;continue
        key=hashlib.sha256(text.encode()).hexdigest()
        value={'text':text,'input':full,'tokens':count,'cache_key':keys[0],
               'vector_sha256':source['origins'][keys[0]]['vector_sha256']}
        require(key not in eligible or eligible[key]==value,'Conflicting duplicate document input')
        eligible[key]=value
    return eligible,skipped


def sample_consistency(eligible,source,model,sample_size,batch_size,check=lambda:None):
    import numpy as np
    require(len(eligible)>=sample_size>=MINIMUM_SAMPLE,'Not enough distinct eligible inputs for the required sample')
    precision=validated_embedding_precision(model)
    # Length-stratified deterministic choices, independent of questions/results.
    ordered=sorted(eligible,key=lambda key:(eligible[key]['tokens'],key))
    selected=[ordered[i*(len(ordered)-1)//(sample_size-1)] for i in range(sample_size)]
    records=[]
    for start in range(0,len(selected),batch_size):
        check();keys=selected[start:start+batch_size]
        values=model.encode([eligible[k]['input'] for k in keys],batch_size=batch_size,
                            normalize_embeddings=True,show_progress_bar=False)
        require(len(values)==len(keys),'Sample encoder returned an incomplete batch')
        check()
        for key,current in zip(keys,values):
            current=checked_document_vector(current,1024);old=source['vectors'][eligible[key]['cache_key']]
            difference=float(np.max(np.abs(current-old)))
            require(np.allclose(current,old,rtol=1e-4,atol=2e-5),'FP32 sample differs from historical vector; no import committed')
            records.append({'text_sha256':key,'cache_key':eligible[key]['cache_key'],
                'historical_vector_sha256':eligible[key]['vector_sha256'],
                'fp32_vector_sha256':hashlib.sha256(current.tobytes()).hexdigest(),'max_absolute_difference':difference})
    return records,precision


def import_vectors(args, *, tokenizer_factory=None, model_factory=None, monitor=None):
    """Factories are injection points for synthetic tests; CLI always uses local FP32."""
    import numpy as np
    from filelock import FileLock
    settings=candidate_settings(args)
    if monitor is None:
        from scripts.benchmark_models import ResourceMonitor
        monitor=ResourceMonitor(args.rss_gib)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():require(read(output).get('schema')==SCHEMA,'Refusing to overwrite a foreign report')
    worker_lock=settings.state/'reports/quality-v2-models/model-worker.lock'
    worker_lock.parent.mkdir(parents=True,exist_ok=True)
    # Same outer worker lock as benchmark CLI; all acquisitions fail immediately.
    with FileLock(str(worker_lock),timeout=0), FileLock(str(settings.state/'server.lock'),timeout=0), \
            FileLock(str(settings.state/'sync.lock'),timeout=0):
        report=None
        try:
            with monitor:
                source=load_source(args.suite,args.source_cache,settings,monitor.check)
                contract=document_cache_contract(settings,fingerprint(source['model_files']))
                run_identity={'source_identity':source['identity'],'source_batches':source['batches'],
                              'target_contract':contract,'sample_size':args.sample_size,'batch_size':args.batch_size,
                              'rss_gib':args.rss_gib,
                              'code':{str(p.relative_to(ROOT)):file_sha256(p) for p in
                                      [Path(__file__),ROOT/'ragapp/config.py',ROOT/'ragapp/retrieval.py',ROOT/'ragapp/pipeline.py',ROOT/'ragapp/textbudget.py',
                                       ROOT/'ragapp/model_adapters.py',ROOT/'scripts/download_candidate_models.py',ROOT/'scripts/benchmark_models.py']},
                              'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','sentence-transformers')}}
                identity=fingerprint(run_identity)
                if output.exists():require(read(output).get('run_fingerprint')==identity,'Import resume source/model/code/config changed')
                report={'schema':SCHEMA,'status':'validating','run_fingerprint':identity,'run_identity':run_identity,
                        'historical_actual_dtype':'unknown','publication_eligible':False,'documents_total':len(source['corpus']),
                        'committed':0,'limitations':['Sample consistency does not certify all historical computation dtypes.',
                            'Imported vector timing is not model encoding throughput. Ordinary rebuild and end-to-end acceptance remain required.']}
                atomic_cache_json(output,report)
                if tokenizer_factory is None:
                    from transformers import AutoTokenizer
                    tokenizer=AutoTokenizer.from_pretrained(str(settings.state/'models'/settings.embedding_directory),local_files_only=True)
                else:tokenizer=tokenizer_factory(settings)
                eligible,skipped=eligible_documents(source,tokenizer,settings,monitor.check)
                require(len(eligible)>=args.sample_size,'Not enough eligible exact inputs for a 32+ sample')
                report.update(eligible_unique=len(eligible),skipped=skipped,status='sampling');atomic_cache_json(output,report)
                monitor.before_load(settings.state,SPEC);monitor.check()
                retrieval=Retrieval(settings,None)
                model=retrieval.embedding if model_factory is None else model_factory(settings)
                # A wrapper tokenizer must reproduce the same actual document inputs.
                model_eligible,model_skipped=eligible_documents(source,model.tokenizer,settings,monitor.check)
                require(model_eligible==eligible and model_skipped==skipped,'Loaded model tokenizer differs from preflight tokenizer')
                records,precision=sample_consistency(eligible,source,model,args.sample_size,args.batch_size,monitor.check)
                require(embedding_model_files(settings)==source['model_files'],'Model bytes changed during verification')
                proof={'schema':SCHEMA,'verification':'sampled_fp32_consistency','historical_actual_dtype':'unknown',
                    'passed':True,'target_contract':contract,'sample_count':len(records),'sample_records':records,
                    'sample_precision':precision,'sample_selection':'evenly spaced over (full_input_tokens, text_sha256)',
                    'tolerance':{'rtol':1e-4,'atol':2e-5,'unit_norm_atol':2e-4},
                    'source_cache_schema':source['identity']['schema'],'source_contract':source['identity']['contract'],
                    'source_batches':source['batches'],'import_run_fingerprint':identity,
                    'eligible':{k:{field:v[field] for field in ('cache_key','vector_sha256')} for k,v in eligible.items()}}
                proof['fingerprint']=fingerprint(proof)
                folder=settings.state/'cache'/('vectors-'+embedding_fingerprint(settings))
                # Preflight every existing destination before creating any new vector.
                for key,value in eligible.items():
                    binary=folder/(key+'.npy');marker=binary.with_suffix('.json')
                    if marker.exists():
                        previous=read_document_vector(folder,value['text'],value['input'],contract)
                        require(np.array_equal(previous,source['vectors'][value['cache_key']]),'Conflicting target cache vector')
                    elif binary.exists():
                        previous=checked_document_vector(np.load(binary,allow_pickle=False),1024)
                        require(np.array_equal(previous,source['vectors'][value['cache_key']]),'Conflicting incomplete target cache vector')
                folder.mkdir(parents=True,exist_ok=True)
                proof_path=folder/('import-'+proof['fingerprint']+'.json')
                if proof_path.exists():require(read(proof_path)==proof,'Conflicting import proof')
                else:atomic_cache_json(proof_path,proof)
                report.update(status='importing',proof_fingerprint=proof['fingerprint'],target_folder=str(folder.resolve()),
                              target_embedding_settings={k:v for k,v in asdict(settings).items() if k.startswith('embedding_')})
                atomic_cache_json(output,report)
                for number,(key,value) in enumerate(eligible.items()):
                    if number%32==0:monitor.check()
                    origin={'kind':'benchmark_import','proof_fingerprint':proof['fingerprint'],
                            **{field:value[field] for field in ('cache_key','vector_sha256')}}
                    write_document_vector(folder,value['text'],value['input'],source['vectors'][value['cache_key']],contract,origin)
                    report['committed']=number+1
                    if number%128==0:atomic_cache_json(output,report)
                monitor.check()
            # Final RSS/sample checks happen in __exit__; success commits only after it.
            report.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),resources=monitor.report())
            atomic_cache_json(output,report)
            return report
        except Exception as exc:
            if report is not None:
                report.pop('completed_at_utc',None)
                report.update(status='failed',error=type(exc).__name__+': '+str(exc),resources=monitor.report())
                atomic_cache_json(output,report)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,default=ROOT/'.rag')
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--source-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New import report; identical reruns are allowed')
    parser.add_argument('--sample-size',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--input-tokens',type=int,default=480)
    parser.add_argument('--rss-gib',type=float,default=12)
    result=import_vectors(parser.parse_args())
    print(json.dumps({k:result[k] for k in ('status','documents_total','eligible_unique','committed','target_folder')}))


if __name__=='__main__':main()
