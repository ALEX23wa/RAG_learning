import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import zipfile
import httpx
from filelock import FileLock

def backup(service,destination):
    """制作包含台账、向量快照和引用图片资产的备份包。

    Args:
        service: Service：已配置的 RAG 服务；复用其存储、检索及模型组件。
        destination: Path | str：输出 ZIP 路径，必须位于 papers 目录之外。

    Returns:
        dict：backup 为归档绝对路径，originals 提醒另存原始资料。

    Raises:
        ValueError：备份路径越界到资料目录、图片资产不受管或缺失；锁冲突和数据库/文件错误向上传递。

    Notes:
        取得同步锁后备份 SQLite、Qdrant 和受管图片并记录 SHA-256；不含原 papers、模型或 .env。
    """
    destination=Path(destination).resolve()
    if destination.is_relative_to(service.settings.corpus.resolve()): raise ValueError('Backups cannot be stored in papers')
    with FileLock(str(service.settings.state/'sync.lock'),timeout=0):
        service.start()
        with tempfile.TemporaryDirectory(dir=service.settings.state) as folder:
            folder=Path(folder)
            source=sqlite3.connect(service.store.path); target=sqlite3.connect(folder/'catalog.sqlite3')
            source.backup(target); target.close(); source.close()
            snapshot=service.retrieval.client.create_snapshot(service.settings.collection)
            url=service.settings.qdrant_url+f'/collections/{service.settings.collection}/snapshots/{snapshot.name}'
            with httpx.stream('GET',url,timeout=180) as response:
                response.raise_for_status()
                with (folder/'vectors.snapshot').open('wb') as stream:
                    for part in response.iter_bytes(): stream.write(part)
            manifest={'collection':service.settings.collection,'created':time.time(),'includes_originals':False,
                'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()},'assets':[]}
            with service.store.db() as db:
                images=[r[0] for r in db.execute('SELECT DISTINCT image FROM chunks WHERE image IS NOT NULL')]
            for image in images:
                asset=Path(image).resolve()
                if not asset.is_relative_to((service.settings.state/'cache').resolve()):
                    raise ValueError('An image is outside the managed cache; sync before backup')
                if not asset.is_file():raise ValueError('Missing image asset: '+str(asset))
                name='assets/'+asset.relative_to(service.settings.state.resolve()).as_posix()
                manifest['assets'].append({'original_path':image,'name':name,'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()})
            (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            destination.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
                for p in folder.iterdir(): archive.write(p,p.name)
                for asset in manifest['assets']:archive.write(asset['original_path'],asset['name'])
            service.retrieval.client.delete_snapshot(service.settings.collection,snapshot.name)
    return {'backup':str(destination),'originals':'Keep papers separately; it is not included'}

def restore(service,archive):
    """校验备份并恢复到新的索引集合和台账，最后切换活动指针。

    Args:
        service: Service：已配置的 RAG 服务；复用其存储、检索及模型组件。
        archive: Path | str：backup 生成的 ZIP 包。

    Returns:
        dict：restored_collection 为新集合名，restart_required=True 表示需重新启动服务加载配置。

    Raises:
        ValueError：归档结构、哈希、资产路径非法或恢复集合为空；文件锁占用时拒绝恢复。

    Notes:
        要求网页服务停止；校验归档路径和哈希，迁移图片路径并合并已发生的费用/限流状态，避免恢复重置预算。原集合保留。
    """
    with FileLock(str(service.settings.state/'server.lock'),timeout=0), FileLock(str(service.settings.state/'sync.lock'),timeout=0):
        service.start()
        with service.store.db() as db:
            current_calls=[tuple(r) for r in db.execute('SELECT * FROM calls')]
            next_free=db.execute("SELECT value FROM metadata WHERE key='aihub_next'").fetchone()
        with tempfile.TemporaryDirectory(dir=service.settings.state) as folder:
            folder=Path(folder)
            with zipfile.ZipFile(archive) as package:
                manifest=json.loads(package.read('manifest.json'))
                members={'manifest.json','catalog.sqlite3','vectors.snapshot'}
                for asset in manifest.get('assets',[]):
                    if not asset['name'].startswith('assets/cache/'):raise ValueError('Invalid asset path')
                    members.add(asset['name'])
                if set(package.namelist())!=members:raise ValueError('Invalid backup members')
                if set(manifest['files'])!={'catalog.sqlite3','vectors.snapshot'}:raise ValueError('Invalid checksum members')
                for name in members:
                    if not (folder/name).resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe archive path')
                package.extractall(folder)
            for name,checksum in manifest['files'].items():
                if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=checksum: raise ValueError('Backup checksum mismatch')
            collection=manifest['collection']+'_restored_'+str(int(time.time()))
            service.retrieval.client.create_collection(collection,vectors_config=service.retrieval.client.get_collection(service.settings.collection).config.params.vectors)
            with (folder/'vectors.snapshot').open('rb') as stream:
                response=httpx.post(service.settings.qdrant_url+f'/collections/{collection}/snapshots/upload',files={'snapshot':stream},params={'priority':'snapshot'},timeout=300)
                response.raise_for_status()
            if service.retrieval.client.count(collection,exact=True).count<1: raise ValueError('Restored vector collection is empty')
            # Restore to a new generation and preserve all already incurred costs.
            catalog=collection+'.sqlite3'
            source=sqlite3.connect(folder/'catalog.sqlite3'); target=sqlite3.connect(service.settings.state/catalog)
            source.backup(target);source.close()
            try:
                for row in current_calls:target.execute('INSERT OR REPLACE INTO calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',row)
                if next_free:
                    old=target.execute("SELECT value FROM metadata WHERE key='aihub_next'").fetchone()
                    target.execute("INSERT OR REPLACE INTO metadata VALUES('aihub_next',?)",(str(max(float(next_free[0]),float(old[0]) if old else 0)),))
                for asset in manifest.get('assets',[]):
                    extracted=folder/asset['name']
                    if hashlib.sha256(extracted.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('Asset checksum mismatch')
                    destination=(service.settings.state/Path(asset['name']).relative_to('assets')).resolve()
                    if not destination.is_relative_to((service.settings.state/'cache').resolve()):raise ValueError('Unsafe asset destination')
                    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(extracted,destination)
                    target.execute('UPDATE chunks SET image=? WHERE image=?',(str(destination),asset['original_path']))
                target.commit()
            finally:target.close()
            pointer=service.settings.state/'active.next.json'
            # 完成校验和资源写入后才原子替换活动指针；运行实例需重启读取它。
            pointer.write_text(json.dumps({'collection':collection,'catalog':catalog}),encoding='utf-8')
            pointer.replace(service.settings.state/'active.json')
    return {'restored_collection':collection,'restart_required':True}

def rebuild(service, publish=False):
    """在独立集合和台账中重建语料，全部成功后再发布。

    Args:
        service: Service：已配置的 RAG 服务；复用其存储、检索及模型组件。

    Returns:
        dict：同步统计及 published；成功时含 collection/catalog，失败时含 staging_collection 且不切换当前索引。

    Notes:
        要求网页服务停止；保留排除状态和调用台账。复用解析/向量缓存，旧集合留作回滚，失败的临时集合不对查询生效。
    """
    import copy
    import uuid
    from .store import Store
    from .retrieval import Retrieval
    from .indexer import Indexer
    with FileLock(str(service.settings.state/'server.lock'),timeout=0):
        settings=copy.copy(service.settings)
        settings.collection='papers_quality_'+uuid.uuid4().hex[:12]
        settings.catalog=settings.collection+'.sqlite3'
        staged=settings.state/(settings.collection+'.sqlite3')
        store=Store(staged)
        # Excluded source files remain excluded across rebuilds.
        with store.db() as db:
            for doc in service.store.documents():
                if doc['excluded']:
                    db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',tuple(doc[k] for k in ('id','path','hash','version','status','error','excluded','updated')))
        retrieval=Retrieval(settings,store); retrieval.start_database()
        result=Indexer(settings,store,retrieval,service.models).sync()
        if result['failed']:return {**result,'published':False,'staging_collection':settings.collection}
        # 有任何文件失败就保留现用索引；全部成功后才合并台账并切换集合。
        # Preserve billing and task history when switching document generations.
        with service.store.db() as old,store.db() as new:
            for table in ('calls','metadata','call_campaigns'):
                for row in old.execute('SELECT * FROM '+table):
                    if table=='metadata' and (row[0].endswith('fingerprint') or row[0]=='index_profile'):continue
                    new.execute(f"INSERT OR REPLACE INTO {table} VALUES({','.join('?' for _ in row)})",tuple(row))
        from .config import INDEX_SETTINGS
        from .pipeline import index_fingerprint
        pointer={'collection':settings.collection,'catalog':staged.name,
                 'settings':{key:getattr(settings,key) for key in INDEX_SETTINGS},
                 'index_fingerprint':index_fingerprint(settings)}
        folder=settings.state/'generations';folder.mkdir(exist_ok=True)
        path=folder/(settings.collection+'.json')
        path.write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding='utf-8')
        if publish:raise ValueError('Publication requires an independently audited acceptance report; use publish_generation')
        return {**result,'published':False,'generation':str(path),**pointer}


def publish_generation(service, generation_path, report_path):
    """Validate a frozen acceptance bundle, then atomically switch the active pointer.

    Publication report contract (``acceptance_schema=quality-v2-publication-v1``):

    * Retain ``evaluation.run`` / ``audit.finalize`` fields: quality-v2, blind,
      dataset_case_count=120, passed status, 120 unique results with complete
      ``independent_audit``, finite metrics, and the FULL administrative runtime
      manifest (redacted public manifests are insufficient for publication).
    * ``dataset_path`` points to the frozen blind JSON. Dataset and adjacent
      manifest SHA256 must match ``dataset_sha256`` / ``dataset_manifest_sha256``.
      Relative artifact paths are relative to the publication report directory.
    * ``audit_identity`` has nonempty ``id``, ``kind`` (human/independent_agent),
      ``independent: true``, ``run_operator_id`` and nonempty ``gold_author_ids``.
      Reviewer ID must match audit_reviewer and differ from operator/authors.
      It is a recorded attestation, not cryptographic proof of a person's identity.
    * ``performance_measurement``: scope service_and_database or
      offline_service_and_database, ``warmup_excluded: true``, ``samples: 120``;
      ``records`` contains one measured record per case: id,
      elapsed, service_rss_bytes, database_rss_bytes. RSS values are
      positive OS process measurements at the same sampled instant, not client
      RSS or an estimate. ``measurement_method=psutil_process_rss`` and positive,
      distinct ``service_pid`` / ``database_pid`` identify the measured processes.
      Resumed runs register each measured PID pair in ``process_sessions`` with
      ``warmup_excluded: true``; per-record PID pairs must belong to a registered
      session. Omitted session/record PIDs retain the single-process-run contract.
      More than one same-case memory sample is permitted via ``rss_samples``
      records with service_rss_bytes/database_rss_bytes; request peaks must match
      their maxima. p95 uses nearest rank on the 120 warm requests; peak_rss_bytes
      is the maximum simultaneous service+database RSS, including rss_samples.
    * ``campaign_cost``: nonempty campaign, finite total_cny <= 100,
      includes_failed_calls=true. Total must cover that campaign's calls across
      local generation catalogs, deduplicated by call ID with conservative
      max(COALESCE(cost,reserved)) when copies differ. Reserved/failed calls count.
    * ``legacy_regression`` references report_path, sha256, dataset_path,
      dataset_name (a 60-case entry in state/evaluation/frozen-manifest.json),
      total=completed=60, passed>=57, runtime_fingerprint. The linked report must
      have all 60 audited outcomes, the same runtime and the original dataset hash.

    Runtime configuration must match the supplied service settings with the
    generation's index settings applied. Neither model loading nor index repair
    is attempted. Rejection leaves catalogs and both active pointers unchanged;
    only ordinary lock files may be created. The old pointer is retained on success.
    The published ``runtime_settings`` is extracted only from the fully validated
    runtime manifest through ``config.RUNTIME_SETTINGS``; candidate-generation
    extra fields cannot inject runtime choices, paths or budget authorizations.
    Restarts apply these accepted runtime settings along with the index settings.
    """
    with FileLock(str(service.settings.state/'server.lock'),timeout=0), FileLock(str(service.settings.state/'sync.lock'),timeout=0):
        generation=_validate_publication(service, Path(generation_path), Path(report_path))
        path=service.settings.state/'active.json'
        if path.exists():
            previous=path.read_text(encoding='utf-8')
        else:
            previous=json.dumps({'collection':service.settings.collection,'catalog':service.settings.catalog})
        (service.settings.state/'active.previous.json').write_text(previous,encoding='utf-8')
        temporary=path.with_suffix('.next.json')
        temporary.write_text(json.dumps(generation,ensure_ascii=False),encoding='utf-8');temporary.replace(path)
    return {'published':True,'collection':generation['collection'],'restart_required':True}


def _publication_require(condition, message):
    if not condition:
        raise ValueError(message)


def _publication_number(value, name, minimum=0, maximum=None, integer=False):
    import math
    valid=type(value) in ((int,) if integer else (int,float))
    try:valid=valid and math.isfinite(value) and value>=minimum and (maximum is None or value<=maximum)
    except (OverflowError,TypeError):valid=False
    _publication_require(valid,'Invalid or unmet publication measurement: '+name)
    return value


def _publication_object(value, name):
    _publication_require(isinstance(value,dict),'Missing publication object: '+name)
    return value


def _publication_text(value, name):
    _publication_require(isinstance(value,str) and bool(value.strip()),'Missing publication identity: '+name)
    return value.strip()


def _publication_artifact(report_path, value):
    path=Path(_publication_text(value,'artifact path'))
    return path if path.is_absolute() else report_path.parent/path


def _publication_rows(report, expected, gold_ids=None, *, gold_required=False):
    rows=report.get('results')
    _publication_require(isinstance(rows,list) and len(rows)==expected,'Incomplete acceptance results')
    ids=[];passed=0
    for row in rows:
        _publication_object(row,'result')
        ids.append(_publication_text(row.get('id'),'case ID'))
        audit=_publication_object(row.get('independent_audit'),'independent_audit')
        _publication_require(type(audit.get('pass')) is bool and type(audit.get('citations_valid')) is bool,
                             'Every audit outcome must be an explicit boolean')
        _publication_require(audit.get('source_checked') is True and bool(str(audit.get('reason') or '').strip()),
                             'Every audit needs original-source verification and a reason')
        if gold_required:
            _publication_require(audit.get('gold_verified') is True,'Every blind gold item needs independent verification')
        total=_publication_number(audit.get('total_claims'),'total_claims',integer=True)
        _publication_number(audit.get('supported_claims'),'supported_claims',maximum=total,integer=True)
        passed+=audit['pass']
    _publication_require(len(set(ids))==expected and (gold_ids is None or set(ids)==set(gold_ids)),
                         'Duplicate or mismatched acceptance case IDs')
    return rows,passed


def _publication_performance(report, rows):
    import math
    from .audit import measured_performance
    metrics=report['metrics']
    _publication_require(len(rows)==120,'120 measured performance records are required')
    measured=measured_performance(report.get('performance_measurement'),[row['id'] for row in rows])
    p95=measured['p95_seconds'];peak=measured['peak_rss_bytes']
    _publication_require(p95 is not None and peak is not None,
                         'Complete warm service/database process-session measurements are required')
    _publication_number(p95,'measured p95',0,60)
    _publication_number(peak,'measured service plus database RSS',1,12*1024**3,True)
    _publication_require(math.isclose(metrics['p95_seconds'],p95,rel_tol=1e-9,abs_tol=1e-6)
        and metrics['peak_rss_bytes']==peak,'Acceptance summaries do not match measured performance samples')


def _publication_quality(report, rows, cases):
    """Recompute quality gates from audited outcomes and measured evidence groups."""
    import math
    from .evaluation import evidence_groups
    gold={c['id']:c for c in cases};span=[];candidate=[];negative=refused=correct_refusals=0
    for row in rows:
        case=gold[row['id']]
        _publication_require(type(case.get('answerable')) is bool,'Frozen case answerability is missing')
        meta=_publication_object(row.get('case_metadata'),'case metadata')
        _publication_require(type(meta.get('answerable')) is bool and meta['answerable']==case['answerable'],
                             'Result answerability differs from the frozen case')
        answer=_publication_object(row.get('answer'),'completed answer')
        _publication_text(answer.get('answer'),'answer text')
        refusal=answer.get('status')=='abstained' and not answer.get('citations')
        negative+=not case['answerable'];refused+=refusal;correct_refusals+=refusal and not case['answerable']
        if case['answerable'] and case.get('scorable',True):
            expected=len(evidence_groups(case))
            _publication_require(expected>0,'Scorable gold must contain evidence groups')
            grade=_publication_object(row.get('grade'),'grade')
            score=_publication_object(row.get('retrieval_score'),'retrieval_score')
            trace=_publication_object(answer.get('evaluation_trace'),'evaluation_trace')
            _publication_object(trace.get('trace'),'retrieval trace')
            for key,values,destination in [('span_recalled_at_10',grade.get('span_recalled_at_10'),span),
                                           ('span_at_80',score.get('span_at_80'),candidate)]:
                _publication_require(isinstance(values,list) and len(values)==expected
                    and all(type(v) is bool for v in values),'Incomplete or invalid evidence group measurements: '+key)
                destination.extend(values)
    _publication_require(bool(span) and bool(candidate) and negative>0 and refused>0,
                         'Span/candidate measurements and refusal cases are required')
    actual={'span_recall_at_10':sum(span)/len(span),'candidate_recall_at_80':sum(candidate)/len(candidate),
            'abstention_precision':correct_refusals/refused,'abstention_recall':correct_refusals/negative}
    for key,value in actual.items():
        _publication_require(math.isclose(report['metrics'][key],value,rel_tol=1e-9,abs_tol=1e-12),
                             'Quality summary differs from case measurements: '+key)


def _publication_dataset_frozen(dataset_path, manifest):
    """Content sealing and review state are separate, explicitly checked facts.

    Keep the original 'frozen' contract. The existing independently agent-reviewed
    240-case bundle is accepted only with all its sealed artifacts and split
    invariants verified; this does not promote its human-review status.
    """
    import hashlib
    import re
    from collections import Counter
    from .evaluation import load_dataset, normalized
    status=manifest.get('status')
    if manifest.get('evidence_revision') is not None or status=='evidence_revision_sealed':
        from scripts.freeze_evidence_revision import validate_revision_bundle
        return validate_revision_bundle(Path(dataset_path).parent,manifest)
    # Additional gold alternatives need the versioned review closure even if a
    # caller changes the status back to the legacy "frozen" label.
    for name in manifest.get('datasets',{}):
        candidate=Path(dataset_path).parent/name
        if candidate.is_file():
            rows,_,_=load_dataset(candidate)
            _publication_require(all(not c.get('evidence_groups') or
                c['evidence_groups']==[[e] for e in c.get('evidence',[])] for c in rows),
                'Additional evidence alternatives require a sealed revision review contract')
    if status=='frozen':return
    _publication_require(status=='agent_source_reviewed_human_pending',
                         'Dataset frozen manifest has no recognized content-freeze contract')
    root=Path(dataset_path).parent
    def checked_json(name, expected):
        _publication_require(isinstance(expected,str) and re.fullmatch(r'[0-9a-f]{64}',expected),
                             'Missing frozen artifact hash: '+name)
        path=root/name
        _publication_require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,
                             'Frozen artifact hash mismatch: '+name)
        return json.loads(path.read_text(encoding='utf-8'))
    _publication_require(manifest.get('schema_version')=='quality-v2' and manifest.get('total')==240
        and manifest.get('splits')=={'dev':120,'blind':120}
        and manifest.get('covered_documents')==38 and len(manifest.get('sources',{}))==38
        and manifest.get('agent_source_reviewed_cases')==240 and manifest.get('human_verified_cases')==0
        and isinstance(manifest.get('split_policy'),str) and bool(manifest['split_policy'].strip()),
        'Reviewed dataset lacks the fixed 240-case source/split freeze evidence')
    expected_counts={'single':168,'cross_source':24,'multi_turn':24,'unanswerable':24}
    _publication_require(manifest.get('counts')==expected_counts,'Frozen dataset category quotas changed')
    summary=checked_json('summary.json',manifest.get('summary_sha256'))
    for key in ('status','total','splits','counts','covered_documents','agent_source_reviewed_cases','human_verified_cases'):
        _publication_require(summary.get(key)==manifest.get(key),'Frozen summary disagrees with dataset manifest: '+key)
    checks=checked_json('source-checks.json',manifest.get('source_checks_sha256'))
    _publication_require(isinstance(checks,list),'Frozen source checks must be a list')
    split_cases={};all_ids=set();covered=set();cluster_sets={};template_sets={};span_sets={}
    for split in ('dev','blind'):
        name=split+'.json';entry=_publication_object(manifest.get('datasets',{}).get(name),'frozen '+split+' dataset entry')
        checked_json(name,entry.get('sha256'))
        cases,_,_=load_dataset(root/name)
        _publication_require(entry.get('count')==120 and len(cases)==120
            and Counter(c.get('category') for c in cases)==Counter(single=84,cross_source=12,multi_turn=12,unanswerable=12),
            'Frozen split size or category quotas changed: '+split)
        split_cases[split]=cases;cluster_sets[split]=set();template_sets[split]=set();span_sets[split]=set()
        for case in cases:
            _publication_require(case.get('schema_version')=='quality-v2' and case.get('split')==split
                and isinstance(case.get('id'),str) and case['id'] not in all_ids
                and isinstance(case.get('question'),str) and bool(case['question'].strip()),
                'Frozen split case identity changed')
            all_ids.add(case['id'])
            review=case.get('agent_review') or {}
            _publication_require(case.get('agent_source_reviewed') is True and review.get('source_checked') is True
                and bool(review.get('reviewer')) and bool(review.get('reason')),
                'Frozen case lacks independent agent source-review evidence')
            for key,destination in (('cluster_id',cluster_sets[split]),('template_group',template_sets[split])):
                value=case.get(key)
                _publication_require(isinstance(value,str) and bool(value),'Frozen case lacks grouping: '+key)
                destination.add(value)
            evidence=case.get('evidence',[])
            if case['category']=='unanswerable':
                _publication_require(case.get('answerable') is False and not evidence,'Frozen negative case changed')
            else:
                _publication_require(case.get('answerable') is True and bool(evidence),'Frozen positive case lacks evidence')
            if case['category']=='multi_turn':
                _publication_require(any(m.get('role')=='user' and m.get('content') for m in case.get('history',[])),
                                     'Frozen multi-turn case lost its history')
            if case['category']=='cross_source':
                _publication_require(len({e['path'] for e in evidence})>=2,'Frozen cross-source case lost distinct sources')
            for item in evidence:
                name=item['path'].replace('\\','/');source=manifest['sources'].get(name)
                _publication_require(source is not None and item.get('source_hash')==source.get('hash'),
                                     'Frozen evidence does not match its original source hash')
                covered.add(name)
                if item.get('span_text'):
                    digest=hashlib.sha256(normalized(item['span_text']).encode()).hexdigest()
                    _publication_require(item.get('span_sha256')==digest,'Frozen evidence span hash mismatch')
                    span_sets[split].add(digest)
                else:
                    _publication_require(case.get('scorable') is False,'Scorable frozen evidence lacks a text span')
        audit_name=split+'-gold-audit.json'
        artifact=checked_json(audit_name,(manifest.get('independent_review_artifacts') or {}).get(audit_name))
        _publication_require(artifact.get('dataset_sha256')==entry['sha256'] and bool(artifact.get('reviewer')),
                             'Frozen source-review artifact belongs to a different dataset')
        reviewed=artifact.get('cases')
        _publication_require(isinstance(reviewed,list) and len(reviewed)==120
            and {r.get('id') for r in reviewed}=={c['id'] for c in cases}
            and all(r.get('valid') is True and r.get('source_checked') is True for r in reviewed),
            'Frozen source-review artifact does not cover its complete split')
    _publication_require(len(all_ids)==240 and len(covered)==38 and set(c.get('case_id') for c in checks)==all_ids,
                         'Frozen source checks or source coverage are incomplete')
    for sets in (cluster_sets,template_sets,span_sets):
        _publication_require(not sets['dev'] & sets['blind'],'Frozen evidence or grouping leaks across splits')
    for item in checks:
        _publication_require(item.get('agent_source_reviewed') is True
            and (item.get('agent_review') or {}).get('source_checked') is True,
            'Frozen source-check review evidence is incomplete')
        if item.get('path'):
            source=manifest['sources'].get(item['path'].replace('\\','/'))
            _publication_require(source is not None and item.get('source_hash')==source.get('hash')
                and item.get('original_hash_verified') is True,'Frozen source check hash evidence differs')


def _validate_publication(service, generation_path, report_path):
    import contextlib
    import copy
    import math
    from .config import INDEX_SETTINGS,RUNTIME_SETTINGS,validated_runtime_settings
    from .evaluation import load_dataset
    from .indexer import Indexer,digest
    from .pipeline import index_fingerprint,runtime_manifest,embedding_fingerprint,parsing_fingerprint
    generation=_publication_object(json.loads(generation_path.read_text(encoding='utf-8')),'generation')
    report=_publication_object(json.loads(report_path.read_text(encoding='utf-8')),'report')
    _publication_require(report.get('acceptance_schema')=='quality-v2-publication-v1'
        and report.get('evaluation_schema')=='quality-v2' and report.get('dataset_split')=='blind'
        and report.get('status')=='passed','A complete quality-v2 blind publication report is required')
    _publication_number(report.get('dataset_case_count'),'blind case count',120,120,True)
    metrics=_publication_object(report.get('metrics'),'metrics')
    required={'accuracy':.9,'span_recall_at_10':.9,'evidence_recall_at_10':.9,'faithfulness':.95,
              'citation_location_validity':1.,'candidate_recall_at_80':.95,
              'abstention_precision':.9,'abstention_recall':.9}
    for key,minimum in required.items():_publication_number(metrics.get(key),key,minimum,1)
    _publication_number(metrics.get('completed'),'completed',120,120,True)
    _publication_number(metrics.get('total'),'total',120,120,True)
    _publication_number(metrics.get('p95_seconds'),'p95_seconds',0,60)
    _publication_number(metrics.get('peak_rss_bytes'),'peak_rss_bytes',1,12*1024**3,True)
    _publication_require(metrics['span_recall_at_10']==metrics['evidence_recall_at_10'],
                         'Legacy locator recall cannot substitute for span recall')
    dataset_path=_publication_artifact(report_path,report.get('dataset_path'))
    cases,dataset_hash,dataset_manifest=load_dataset(dataset_path)
    _publication_require(len(cases)==120 and all(c.get('split')=='blind' and c.get('schema_version')=='quality-v2' for c in cases),
                         'Expected exactly 120 frozen quality-v2 blind cases')
    _publication_object(dataset_manifest,'frozen dataset manifest')
    entry=_publication_object(dataset_manifest.get('datasets',{}).get(dataset_path.name),'frozen dataset entry')
    _publication_number(entry.get('count'),'manifest blind count',120,120,True)
    dataset_freeze=_publication_dataset_frozen(dataset_path,dataset_manifest) or {}
    _publication_require(dataset_manifest.get('schema_version')=='quality-v2'
        and entry.get('sha256')==dataset_hash
        and report.get('dataset_sha256')==dataset_hash
        and report.get('dataset_manifest_sha256')==dataset_manifest.get('_manifest_sha256'),
        'Acceptance dataset or its frozen manifest changed')
    rows,passed=_publication_rows(report,120,[c['id'] for c in cases],gold_required=True)
    _publication_number(metrics.get('passed'),'passed',0,120,True)
    _publication_require(passed==metrics['passed'] and math.isclose(passed/120,metrics['accuracy']),
                         'Accuracy differs from the independently audited outcomes')
    claims=sum(r['independent_audit']['total_claims'] for r in rows)
    support=sum(r['independent_audit']['supported_claims'] for r in rows)
    _publication_require(claims>0 and math.isclose(support/claims,metrics['faithfulness'])
        and all(r['independent_audit']['citations_valid'] for r in rows),'Claim support or citation audit does not meet publication gates')
    identity=_publication_object(report.get('audit_identity'),'audit_identity')
    reviewer=_publication_text(identity.get('id'),'reviewer')
    operator=_publication_text(identity.get('run_operator_id'),'run operator')
    authors=identity.get('gold_author_ids')
    _publication_require(isinstance(authors,list) and bool(authors),'Gold author identities are required')
    authors=[_publication_text(v,'gold author') for v in authors]
    _publication_require(identity.get('kind') in {'human','independent_agent'} and identity.get('independent') is True
        and reviewer==report.get('audit_reviewer') and reviewer not in [operator,*authors],
        'Acceptance reviewer must be identified and independent of the run operator and gold authors')
    recorded_gold_reviewers={c.get('agent_review',{}).get('reviewer') for c in cases if c.get('agent_review',{}).get('reviewer')}
    recorded_gold_reviewers.update(dataset_freeze.get('reviewer_ids_by_split',{}).get('blind',[]))
    if dataset_manifest.get('evidence_revision'):
        _publication_require(recorded_gold_reviewers <= set(authors),
                             'Gold author identities omit recorded blind source/equivalence reviewers')
    _publication_require(reviewer not in recorded_gold_reviewers,'Acceptance reviewer also reviewed the blind gold')
    _publication_quality(report,rows,cases)
    _publication_performance(report,rows)
    settings=copy.copy(service.settings)
    values=_publication_object(generation.get('settings'),'generation settings')
    # Old generation manifests predate opt-in precision. Never inherit the
    # publisher's candidate precision for an otherwise complete legacy manifest.
    if set(values)==set(INDEX_SETTINGS)-{'embedding_precision'}:
        values={**values,'embedding_precision':'legacy'}
    _publication_require(set(values)==set(INDEX_SETTINGS),'Generation must contain the complete index settings')
    _publication_require(values.get('embedding_precision') in ('legacy','float32'),'Invalid embedding precision')
    for key,value in values.items():
        default=getattr(settings,key)
        _publication_require(type(value) is type(default),'Invalid generation setting type: '+key)
        setattr(settings,key,value)
    settings.collection=_publication_text(generation.get('collection'),'collection')
    catalog=_publication_text(generation.get('catalog'),'catalog')
    state=service.settings.state.resolve();target=(state/catalog).resolve()
    _publication_require(target.parent==state and target.suffix=='.sqlite3' and target.is_file(),
                         'Generation catalog must be an existing direct SQLite file in state')
    settings.catalog=catalog
    expected=index_fingerprint(settings)
    _publication_require(generation.get('index_fingerprint')==expected,'Generation settings do not match the declared index fingerprint')

    class ReadOnlyStore:
        @contextlib.contextmanager
        def db(self):
            db=sqlite3.connect(target.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
            try:yield db
            finally:db.close()
        def documents(self):
            with self.db() as db:return [dict(r) for r in db.execute('SELECT * FROM documents ORDER BY path')]
    store=ReadOnlyStore()
    with store.db() as db:
        indexed=dict(db.execute('SELECT key,value FROM metadata'))
        _publication_require(indexed.get('index_fingerprint')==expected,'Target SQLite index fingerprint differs from the generation')
        _publication_require(indexed.get('embedding_fingerprint')==embedding_fingerprint(settings)
            and indexed.get('parsing_fingerprint')==parsing_fingerprint(settings)
            and indexed.get('index_profile')==settings.index_profile,'Target SQLite model/parser provenance differs from generation settings')
        _publication_require(db.execute('SELECT count(*) FROM chunks').fetchone()[0]>0,'Target generation has no indexed chunks')
    current=runtime_manifest(settings,store)
    manifest=_publication_object(report.get('runtime_manifest'),'full runtime manifest')
    _publication_require(manifest==current and report.get('runtime_fingerprint')==current['runtime_fingerprint'],
                         'Acceptance runtime differs from current code, configuration, index or source/document state')
    _publication_require(manifest.get('index_fingerprint')==expected
        and manifest.get('configured_index_fingerprint',expected)==expected,'Report is for a different index configuration')
    documents=store.documents();by_path={d['path'].replace('\\','/'):d for d in documents}
    present={p.relative_to(settings.corpus).as_posix():p for p in Indexer(settings,None,None).scan()}
    _publication_require(set(present)=={p for p,d in by_path.items() if d['status']!='deleted'},
                         'Source inventory changed since acceptance')
    live={d['path'].replace('\\','/'):d for d in service.store.documents()}
    for name,doc in by_path.items():
        _publication_require(type(doc['excluded']) is int and doc['excluded'] in (0,1),'Invalid document exclusion state')
        if name in live:
            _publication_require(live[name]['excluded']==doc['excluded'],'Source exclusion state changed since the generation was built')
        if doc['status']=='deleted':continue
        _publication_require(doc['excluded'] or doc['status']=='ready','Target has unready or failed source documents')
        _publication_require(digest(present[name])==doc['hash'],'Original source bytes changed since acceptance')
    frozen_sources=dataset_manifest.get('sources',{})
    _publication_require(isinstance(frozen_sources,dict) and bool(frozen_sources),'Frozen dataset source hashes are required')
    for name,source in frozen_sources.items():
        doc=by_path.get(name.replace('\\','/'))
        _publication_require(doc is not None and doc['status']=='ready' and not doc['excluded']
            and doc['hash']==(source.get('hash') or source.get('sha256')),'Frozen gold source is changed, excluded or unavailable')
    cost=_publication_object(report.get('campaign_cost'),'campaign_cost')
    campaign=_publication_text(cost.get('campaign'),'campaign')
    amount=_publication_number(cost.get('total_cny'),'campaign cost',0,100)
    _publication_require(cost.get('includes_failed_calls') is True and manifest['configuration'].get('campaign')==campaign,
                         'Campaign cost must include failures and match the evaluated campaign')
    ledger={}
    for database in sorted(state.glob('*.sqlite3')):
        with contextlib.closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            names={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'calls','call_campaigns'}<=names:continue
            for call_id,spent in db.execute('''SELECT c.id,COALESCE(c.cost,c.reserved) FROM calls c
                JOIN call_campaigns x ON x.call_id=c.id WHERE x.campaign=?''',(campaign,)):
                spent=_publication_number(spent,'ledger campaign cost')
                ledger[call_id]=max(ledger.get(call_id,0),spent)
    _publication_require(sum(ledger.values())<=amount+1e-9,'Campaign cost omits recorded calls, failures or reservations')
    legacy=_publication_object(report.get('legacy_regression'),'legacy_regression')
    for key in ('completed','total'):_publication_number(legacy.get(key),'legacy '+key,60,60,True)
    _publication_number(legacy.get('passed'),'legacy passed',57,60,True)
    _publication_require(legacy.get('runtime_fingerprint')==current['runtime_fingerprint'],'Legacy regression was run on a different runtime')
    legacy_path=_publication_artifact(report_path,legacy.get('report_path'))
    _publication_require(digest(legacy_path)==legacy.get('sha256'),'Legacy regression artifact hash changed')
    legacy_report=_publication_object(json.loads(legacy_path.read_text(encoding='utf-8')),'legacy report')
    legacy_dataset=_publication_artifact(report_path,legacy.get('dataset_path'))
    raw=json.loads(legacy_dataset.read_text(encoding='utf-8'));legacy_cases=raw.get('cases') if isinstance(raw,dict) else raw
    _publication_require(isinstance(legacy_cases,list) and len(legacy_cases)==60,'Legacy dataset must have 60 original cases')
    legacy_hash=digest(legacy_dataset)
    frozen=json.loads((state/'evaluation/frozen-manifest.json').read_text(encoding='utf-8'))
    reference=_publication_object(frozen.get(legacy.get('dataset_name')),'original legacy dataset entry')
    _publication_require(reference.get('count')==60 and reference.get('sha256')==legacy_hash
        and legacy_report.get('dataset_sha256')==legacy_hash,'Legacy regression is not the original frozen 60-case dataset')
    _,legacy_passed=_publication_rows(legacy_report,60,[c['id'] for c in legacy_cases])
    _publication_require(legacy_report.get('status')=='passed' and legacy_report.get('runtime_fingerprint')==current['runtime_fingerprint']
        and legacy_passed==legacy['passed'] and legacy_passed>=57,'Legacy 60-case regression did not pass on this runtime')
    # Do not write the untrusted input generation wholesale: only this validated
    # manifest can supply settings that control reranking or answer verification.
    runtime=validated_runtime_settings({key:manifest['configuration'][key] for key in RUNTIME_SETTINGS},complete=True)
    return {'collection':settings.collection,'catalog':settings.catalog,
            'settings':{key:values[key] for key in INDEX_SETTINGS},'index_fingerprint':expected,
            'runtime_settings':runtime}
