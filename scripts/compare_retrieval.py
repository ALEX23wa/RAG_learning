"""Reproducible retrieval ablations against frozen original code or current code.

No generation API calls. Baseline candidate Recall@80 is unavailable because the
original implementation returns only its final ten results; it is never inferred.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def frozen_service(snapshot, state, corpus, output):
    from scripts.freeze_baseline_assets import verify_assets, catalog_signature
    inventory=json.loads((snapshot/'snapshot-manifest.json').read_text(encoding='utf-8'))
    for entry in inventory['files']:
        path=snapshot/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Frozen baseline was modified: '+entry['path'])
    assets=verify_assets(snapshot,state)
    name='rag_frozen_baseline'
    spec=importlib.util.spec_from_file_location(name,snapshot/'ragapp/__init__.py',
        submodule_search_locations=[str(snapshot/'ragapp')])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    from importlib import import_module
    Config=import_module(name+'.config').Settings
    Service=import_module(name+'.service').Service
    # A private catalog copy prevents Store initialization from altering the snapshot.
    catalog=output.parent/'baseline-catalog.sqlite3'
    if not catalog.exists():
        import contextlib
        with contextlib.closing(sqlite3.connect((snapshot/'catalog.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as source:
            with contextlib.closing(sqlite3.connect(catalog)) as destination:
                source.backup(destination)
    catalog_signature(catalog,assets['catalog'])
    settings=Config(root=ROOT,state=state,corpus=corpus,catalog=str(catalog.resolve()),collection='papers_e5_v1')
    if (state/'active.json').exists():
        raise ValueError('Frozen baseline requires explicit isolation from an active pointer; run before publication')
    service=Service(settings).start()
    catalog_signature(catalog,assets['catalog'])
    return service,inventory


def main():
    from ragapp.experiments import run,offline_search
    from ragapp.pipeline import runtime_manifest,fingerprint
    from ragapp.config import Settings
    from ragapp.service import Service
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['baseline','p0','quality'],required=True)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'.rag/experiments/quality-v2-baseline')
    parser.add_argument('--cases',type=Path,default=ROOT/'.rag/evaluation/quality-v2/dev.json')
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--max-queries',type=int)
    parser.add_argument('--rrf-k',type=int,choices=[10,30,60],default=10)
    parser.add_argument('--no-expansion',action='store_true')
    args=parser.parse_args();args.report.parent.mkdir(parents=True,exist_ok=True)
    values=json.loads(args.settings.read_text(encoding='utf-8')) if args.settings else {}
    for key in ('root','state','corpus'):
        if key in values:values[key]=Path(values[key])
    values['load_active']=False
    if args.variant=='baseline':
        from scripts.freeze_baseline_assets import file_hash,load_manifest,verify_assets
        from ragapp.evaluation import save_report
        if args.report.exists():
            previous=json.loads(args.report.read_text(encoding='utf-8'))
            if previous.get('results') and previous.get('asset_verification',{}).get('verified') is not True:
                raise ValueError('Previous baseline rows lack completed post-run asset verification; start a separate report')
        service,inventory=frozen_service(args.snapshot,values.get('state',ROOT/'.rag'),
            values.get('corpus',ROOT/'papers'),args.report)
        assets=load_manifest(args.snapshot)
        asset_hash=file_hash(args.snapshot/'asset-manifest.json')
        verified_before=datetime.now(timezone.utc).isoformat()
        def manifest():
            if file_hash(args.snapshot/'asset-manifest.json')!=asset_hash:
                raise ValueError('Baseline asset manifest changed during the run')
            original=runtime_manifest(Settings(**values),service.store)
            original['configuration']={k:str(v) if isinstance(v,Path) else v for k,v in asdict(service.settings).items()}
            original['code']={x['path']:x['sha256'] for x in inventory['files'] if x['path'].replace('\\','/').startswith('ragapp/')}
            original['evaluation_code']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            original['scoring_code']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'ragapp/evaluation.py',ROOT/'ragapp/experiments.py',ROOT/'ragapp/sourcecheck.py',
                          ROOT/'ragapp/pipeline.py',ROOT/'ragapp/parsers.py',ROOT/'ragapp/config.py')}
            original['configured_index_fingerprint']='legacy_unversioned'
            original['baseline_assets']={'asset_manifest_sha256':asset_hash,
                'catalog_logical_sha256':assets['catalog']['sha256'],
                'models_sha256':fingerprint(assets['models']),
                'qdrant_points_sha256':assets['qdrant']['points_sha256'],
                'qdrant_configuration_sha256':assets['qdrant']['configuration_sha256'],
                'recorded_completed_utc':assets['recorded_completed_utc'],
                'original_snapshot_had_asset_hashes':False}
            original['asset_verification_code']=file_hash(ROOT/'scripts/freeze_baseline_assets.py')
            original.pop('runtime_fingerprint',None)
            original['runtime_fingerprint']=fingerprint(original)
            return original
        def search(question,history,config):
            preceding='\n'.join(str(m['content'])[:500] for m in history[-6:] if m.get('role')=='user')
            evidence=service.search((preceding+'\n'+question).strip(),10)
            return {'evidence':evidence,'context':evidence[:8],
                    'measurement_note':'Original code does not expose pre-rerank candidates'}
        configuration={'variant':'frozen_original','candidate_measurement':'unavailable','asset_manifest_sha256':asset_hash}
    else:
        if args.variant=='p0':
            values.update(lexical_candidates=40,dense_candidates=40,rerank_candidates=20,
                reranker_max_tokens=512,evidence_limit=10,expand_context=False,query_planning=False)
        service=Service(Settings(**values)).start()
        manifest=lambda:service.manifest()
        search=offline_search(service)
        configuration={'retrieval_mode':'hybrid_rerank','query_planning':False,
            'rrf_k':args.rrf_k,'expand_context':not args.no_expansion and args.variant=='quality'}
    result=run(args.cases,args.report,search=search,manifest=manifest,
               configuration=configuration,max_queries=args.max_queries)
    if args.variant=='baseline':
        try:
            if file_hash(args.snapshot/'asset-manifest.json')!=asset_hash:
                raise ValueError('Baseline asset manifest changed during the run')
            verify_assets(args.snapshot,service.settings.state,service.settings.qdrant_url,
                          working_catalog=service.store.path)
            result['asset_verification']={'before_utc':verified_before,
                'after_utc':datetime.now(timezone.utc).isoformat(),'verified':True,
                'asset_manifest_sha256':asset_hash}
        except Exception as exc:
            result['status']='asset_verification_failed'
            result['blocked']={'error':str(exc),'phase':'post_run_asset_verification'}
            result['asset_verification']={'before_utc':verified_before,'verified':False}
            save_report(args.report,result)
            raise
        save_report(args.report,result)
    print(json.dumps({'status':result['status'],'metrics':result['metrics'],
        'blocked':result.get('blocked')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
