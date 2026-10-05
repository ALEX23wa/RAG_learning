"""Publication rejects stale/incomplete measurements before touching active pointers."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from filelock import FileLock, Timeout
import pytest

from ragapp.config import Settings, INDEX_SETTINGS, RUNTIME_SETTINGS
from ragapp.maintenance import publish_generation
from ragapp.pipeline import embedding_fingerprint, index_fingerprint, parsing_fingerprint, runtime_manifest
from ragapp.store import Store


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    state=tmp_path/'.rag';state.mkdir();corpus=tmp_path/'papers';corpus.mkdir()
    source=corpus/'source.txt';source.write_text('Original source evidence.',encoding='utf-8')
    evaluation=state/'evaluation';evaluation.mkdir();quality=evaluation/'quality-v2';quality.mkdir()
    settings=Settings(root=tmp_path,state=state,corpus=corpus,campaign='quality-test',load_active=False)
    current=Store(state/'catalog.sqlite3')
    with current.db() as db:
        db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',('doc','source.txt',sha(source),'old-version','ready',None,0,1))
    staged=copy.copy(settings);staged.catalog='papers_quality_test.sqlite3';staged.collection='papers_quality_test'
    store=Store(state/staged.catalog)
    with store.db() as db:
        db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',('new-doc','source.txt',sha(source),'new-version','ready',None,0,1))
        db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)',('chunk','new-doc','new-version','Original source evidence.','{}','',None))
        db.executemany('INSERT INTO metadata VALUES(?,?)',[
            ('index_fingerprint',index_fingerprint(staged)),('embedding_fingerprint',embedding_fingerprint(staged)),
            ('parsing_fingerprint',parsing_fingerprint(staged)),('index_profile',staged.index_profile)])
    generation={'collection':staged.collection,'catalog':staged.catalog,
        'settings':{k:getattr(staged,k) for k in INDEX_SETTINGS},'index_fingerprint':index_fingerprint(staged)}
    gp=state/'generation.json';write(gp,generation)
    cases=[{'id':f'blind-{i}','schema_version':'quality-v2','split':'blind','answerable':i<108,
            'evidence':[{'path':'source.txt','span_text':'Original source evidence.'}] if i<108 else []} for i in range(120)]
    dataset=quality/'blind.json';write(dataset,cases)
    write(quality/'manifest.json',{'schema_version':'quality-v2','status':'frozen',
        'datasets':{'blind.json':{'count':120,'sha256':sha(dataset)}},
        'sources':{'source.txt':{'hash':sha(source)}}})
    manifest=runtime_manifest(staged,store)
    def audited(case_id):
        return {'id':case_id,'independent_audit':{'pass':True,'citations_valid':True,'gold_verified':True,
            'source_checked':True,'reason':'Original source checked independently','total_claims':1,'supported_claims':1}}
    old_cases=[{'id':f'old-{i}'} for i in range(60)]
    old_dataset=evaluation/'original-acceptance.json';write(old_dataset,old_cases)
    write(evaluation/'frozen-manifest.json',{'original-acceptance':{'count':60,'sha256':sha(old_dataset)}})
    old_report=state/'legacy-report.json';write(old_report,{'status':'passed','dataset_sha256':sha(old_dataset),
        'runtime_fingerprint':manifest['runtime_fingerprint'],'results':[audited(c['id']) for c in old_cases]})
    report={'acceptance_schema':'quality-v2-publication-v1','evaluation_schema':'quality-v2','dataset_split':'blind',
        'dataset_case_count':120,'status':'passed','dataset_path':str(dataset),'dataset_sha256':sha(dataset),
        'dataset_manifest_sha256':sha(quality/'manifest.json'),'runtime_manifest':manifest,
        'runtime_fingerprint':manifest['runtime_fingerprint'],'audit_reviewer':'reviewer-independent',
        'audit_identity':{'id':'reviewer-independent','kind':'independent_agent','independent':True,
            'run_operator_id':'run-operator','gold_author_ids':['gold-author']},
        'metrics':{'completed':120,'total':120,'passed':120,'accuracy':1.0,'span_recall_at_10':1.0,
            'evidence_recall_at_10':1.0,'faithfulness':1.0,'citation_location_validity':1.0,
            'candidate_recall_at_80':1.0,'abstention_precision':1.0,'abstention_recall':1.0,
            'p95_seconds':4.0,'peak_rss_bytes':3000},
        'results':[audited(c['id']) for c in cases],
        'performance_measurement':{'scope':'service_and_database','measurement_method':'psutil_process_rss',
            'warmup_excluded':True,'samples':120,'service_pid':100,'database_pid':101,
            'records':[{'id':c['id'],'elapsed':4.0,'service_rss_bytes':2000,'database_rss_bytes':1000} for c in cases]},
        'campaign_cost':{'campaign':'quality-test','total_cny':0.,'includes_failed_calls':True},
        'legacy_regression':{'report_path':str(old_report),'sha256':sha(old_report),'dataset_path':str(old_dataset),
            'dataset_name':'original-acceptance','completed':60,'total':60,'passed':60,
            'runtime_fingerprint':manifest['runtime_fingerprint']}}
    for row,case in zip(report['results'],cases):
        row.update(case_metadata={'answerable':case['answerable']},
            answer={'answer':'Source-backed answer' if case['answerable'] else 'Insufficient source information',
                    'status':'answered' if case['answerable'] else 'abstained','citations':[],
                    'evaluation_trace':{'trace':{}}},
            grade={'span_recalled_at_10':[True] if case['answerable'] else []},
            retrieval_score={'span_at_80':[True] if case['answerable'] else []})
    rp=state/'publication.json';write(rp,report)
    write(state/'active.json',{'collection':'old','catalog':'catalog.sqlite3'})
    write(state/'active.previous.json',{'collection':'older','catalog':'older.sqlite3'})
    return SimpleNamespace(service=SimpleNamespace(settings=settings,store=current),settings=staged,store=store,
        state=state,source=source,generation=generation,generation_path=gp,report=report,report_path=rp,
        dataset=dataset,legacy_report=old_report,legacy_dataset=old_dataset)


def reject_unchanged(bundle, match=None):
    before={p.name:p.read_bytes() for p in bundle.state.glob('active*')}
    with pytest.raises(ValueError,match=match):
        publish_generation(bundle.service,bundle.generation_path,bundle.report_path)
    after={p.name:p.read_bytes() for p in bundle.state.glob('active*')}
    assert after==before


def update_report(bundle):
    write(bundle.report_path,bundle.report)


def test_complete_audited_bundle_publishes_and_preserves_rollback(bundle):
    old=(bundle.state/'active.json').read_bytes()
    result=publish_generation(bundle.service,bundle.generation_path,bundle.report_path)
    assert result['published'] is True
    assert json.loads((bundle.state/'active.json').read_text())=={
        **bundle.generation,'runtime_settings':{key:getattr(bundle.settings,key) for key in RUNTIME_SETTINGS}}
    assert (bundle.state/'active.previous.json').read_bytes()==old


def replace_measured_runtime(bundle, **changes):
    """A synthetic newly measured run, including its matching legacy regression."""
    for key,value in changes.items():
        setattr(bundle.settings,key,value)
        setattr(bundle.service.settings,key,value)
    manifest=runtime_manifest(bundle.settings,bundle.store)
    bundle.report.update(runtime_manifest=manifest,runtime_fingerprint=manifest['runtime_fingerprint'])
    old=json.loads(bundle.legacy_report.read_text())
    old['runtime_fingerprint']=manifest['runtime_fingerprint'];write(bundle.legacy_report,old)
    bundle.report['legacy_regression'].update(runtime_fingerprint=manifest['runtime_fingerprint'],sha256=sha(bundle.legacy_report))
    update_report(bundle)


def test_accepted_runtime_survives_restart_without_authorizing_campaign_or_paths(bundle):
    changes={'reranker_model':'Qwen/Qwen3-Reranker-0.6B','reranker_revision':'a'*40,
        'reranker_directory':'candidate-qwen3-reranker-0.6b','reranker_max_tokens':1024,
        'lexical_candidates':80,'dense_candidates':80,'rerank_candidates':60,'evidence_limit':12,
        'rrf_k':30,'expand_context':True,'query_planning':True,'verify_answers':True}
    replace_measured_runtime(bundle,**changes)
    publish_generation(bundle.service,bundle.generation_path,bundle.report_path)
    pointer=json.loads((bundle.state/'active.json').read_text())
    assert set(pointer['runtime_settings'])==RUNTIME_SETTINGS
    restored=Settings(root=bundle.settings.root,state=bundle.state,corpus=bundle.settings.corpus,
                      campaign='operator-authorized',campaign_limit=7.,qdrant_url='http://local:9999').prepare()
    assert {key:getattr(restored,key) for key in changes}==changes
    assert (restored.collection,restored.catalog)==(bundle.settings.collection,bundle.settings.catalog)
    assert restored.campaign=='operator-authorized' and restored.campaign_limit==7.
    assert restored.qdrant_url=='http://local:9999'


def test_generation_cannot_inject_runtime_or_arbitrary_pointer_fields(bundle):
    bundle.generation.update(runtime_settings={'rrf_k':777,'campaign':'unauthorized','state':'elsewhere'},
                             campaign_limit=999999,unverified_field={'anything':True})
    write(bundle.generation_path,bundle.generation)
    publish_generation(bundle.service,bundle.generation_path,bundle.report_path)
    pointer=json.loads((bundle.state/'active.json').read_text())
    assert set(pointer)=={'collection','catalog','settings','index_fingerprint','runtime_settings'}
    assert pointer['runtime_settings']['rrf_k']==bundle.settings.rrf_k
    assert not {'campaign','state','campaign_limit'} & pointer['runtime_settings'].keys()


def test_unmeasured_runtime_change_cannot_be_published(bundle):
    bundle.service.settings.rrf_k=30
    reject_unchanged(bundle,'Acceptance runtime differs')


def test_invalid_measured_runtime_rejects_before_pointer_writes(bundle):
    replace_measured_runtime(bundle,reranker_directory='../external-model')
    reject_unchanged(bundle,'managed model folder')


def test_registered_resume_process_pair_can_publish(bundle):
    performance=bundle.report['performance_measurement']
    performance['process_sessions']=[{'service_pid':s,'database_pid':d,'warmup_excluded':True}
                                    for s,d in [(100,101),(200,201)]]
    for record in performance['records'][60:]:record.update(service_pid=200,database_pid=201)
    update_report(bundle)
    assert publish_generation(bundle.service,bundle.generation_path,bundle.report_path)['published']


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(process_sessions=[]),
    lambda p:p.update(process_sessions=[{'service_pid':100,'database_pid':101,'warmup_excluded':False}]),
    lambda p:p.update(process_sessions=[{'service_pid':200,'database_pid':201,'warmup_excluded':True}]),
    lambda p:p['records'][0].update(service_pid=200,database_pid=201),
    lambda p:p['records'][0].update(service_pid=101,database_pid=100),
    lambda p:p['records'][0].update(service_pid=True),
    lambda p:p['records'][0].update(service_pid=[]),
    lambda p:p['records'][0].update(elapsed=10**1000),
])
def test_unregistered_or_invalid_performance_process_records_reject_without_mutation(bundle,mutation):
    mutation(bundle.report['performance_measurement']);update_report(bundle)
    reject_unchanged(bundle,'process-session')


def test_manifest_must_be_frozen_not_only_hash_consistent(bundle):
    path=bundle.dataset.parent/'manifest.json';manifest=json.loads(path.read_text())
    manifest['status']='draft';write(path,manifest)
    bundle.report['dataset_manifest_sha256']=sha(path);update_report(bundle)
    reject_unchanged(bundle,'frozen manifest')


@pytest.mark.parametrize('value',[None,True,False,float('nan'),float('inf'),-float('inf'),'1',-1,1.1])
@pytest.mark.parametrize('field',['accuracy','span_recall_at_10','evidence_recall_at_10','faithfulness',
                                  'citation_location_validity','candidate_recall_at_80','abstention_precision','abstention_recall'])
def test_invalid_quality_metric_rejects_without_mutation(bundle,value,field):
    bundle.report['metrics'][field]=value;update_report(bundle)
    reject_unchanged(bundle,'measurement')


@pytest.mark.parametrize('field,value',[('p95_seconds',None),('p95_seconds',True),('p95_seconds',float('nan')),
    ('p95_seconds',float('inf')),('p95_seconds',61),('peak_rss_bytes',True),('peak_rss_bytes',0),
    ('peak_rss_bytes',float('nan')),('peak_rss_bytes',13*1024**3),('completed',True),('total',119)])
def test_invalid_performance_or_completeness_scalar_rejects(bundle,field,value):
    bundle.report['metrics'][field]=value;update_report(bundle);reject_unchanged(bundle,'measurement')


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(evaluation_schema='legacy_locator'),
    lambda r:r.update(dataset_split='dev'),
    lambda r:r.update(dataset_case_count=True),
    lambda r:r.update(status='awaiting_manual_audit'),
    lambda r:r['results'].pop(),
    lambda r:r['results'].__setitem__(1,r['results'][0]),
    lambda r:r['results'][0].pop('independent_audit'),
    lambda r:r['results'][0]['independent_audit'].update(gold_verified=False),
    lambda r:r['audit_identity'].update(id='run-operator'),
    lambda r:r['audit_identity'].update(independent=False),
    lambda r:r['audit_identity'].update(gold_author_ids=['reviewer-independent']),
    lambda r:r.update(dataset_sha256='changed'),
    lambda r:r.update(dataset_manifest_sha256='changed'),
    lambda r:r['performance_measurement'].update(scope='runner_process'),
    lambda r:r['performance_measurement'].update(warmup_excluded=False),
    lambda r:r['performance_measurement'].update(database_pid=100),
    lambda r:r['performance_measurement']['records'][0].update(database_rss_bytes=None),
    lambda r:r['performance_measurement']['records'][0].update(id='unknown'),
    lambda r:r['performance_measurement']['records'][0].update(service_rss_bytes=12000),
    lambda r:r['performance_measurement']['records'][0].update(rss_samples=[]),
    lambda r:r['campaign_cost'].update(total_cny=float('nan')),
    lambda r:r['campaign_cost'].update(total_cny=True),
    lambda r:r['campaign_cost'].update(total_cny=100.01),
    lambda r:r['campaign_cost'].update(includes_failed_calls=False),
    lambda r:r['legacy_regression'].update(passed=56),
    lambda r:r['legacy_regression'].update(runtime_fingerprint='old-runtime'),
    lambda r:r['legacy_regression'].update(sha256='changed'),
    lambda r:r['runtime_manifest'].update(code={}),
    lambda r:r['runtime_manifest']['documents']['source.txt'].update(excluded=1),
    lambda r:r['runtime_manifest']['configuration'].update(rrf_k=999),
])
def test_incomplete_or_misrepresented_publication_artifacts_reject(bundle,mutation):
    mutation(bundle.report);update_report(bundle);reject_unchanged(bundle)


def test_generation_settings_and_declared_fingerprint_must_agree(bundle):
    bundle.generation['settings']['chunk_tokens']+=1;write(bundle.generation_path,bundle.generation)
    reject_unchanged(bundle,'Generation settings')


def test_missing_index_settings_rejects(bundle):
    bundle.generation['settings'].pop('pdf_mode');write(bundle.generation_path,bundle.generation)
    reject_unchanged(bundle,'complete index settings')


def test_legacy_generation_missing_precision_ignores_publisher_candidate_precision(bundle):
    bundle.generation['settings'].pop('embedding_precision')
    write(bundle.generation_path,bundle.generation)
    bundle.service.settings.embedding_precision='float32'
    assert publish_generation(bundle.service,bundle.generation_path,bundle.report_path)['published']
    pointer=json.loads((bundle.service.settings.state/'active.json').read_text())
    assert pointer['settings']['embedding_precision']=='legacy'


@pytest.mark.parametrize('value',[None,True,[],'bfloat16'])
def test_malformed_precision_cannot_publish(bundle,value):
    bundle.generation['settings']['embedding_precision']=value;write(bundle.generation_path,bundle.generation)
    reject_unchanged(bundle,'embedding precision')


def test_actual_target_sqlite_fingerprint_checked(bundle):
    with bundle.store.db() as db:db.execute("UPDATE metadata SET value='other' WHERE key='index_fingerprint'")
    reject_unchanged(bundle,'SQLite index fingerprint')


@pytest.mark.parametrize('kind',['changed','new','deleted','exclusion'])
def test_source_bytes_inventory_and_current_exclusions_checked(bundle,kind):
    if kind=='changed':bundle.source.write_text('Changed after evaluation')
    elif kind=='new':(bundle.source.parent/'new.txt').write_text('Not in evaluated generation')
    elif kind=='deleted':bundle.source.unlink()
    else:
        with bundle.service.store.db() as db:db.execute('UPDATE documents SET excluded=1')
    reject_unchanged(bundle,'source|Source')


def test_failed_or_reserved_calls_from_another_catalog_count_towards_campaign(bundle):
    old=Store(bundle.state/'other.sqlite3')
    with old.db() as db:
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            ('failed-call','production','development','m',None,'failed',3.,None,None,'failed',1.,1.))
        db.execute('INSERT INTO call_campaigns VALUES(?,?)',('failed-call','quality-test'))
    reject_unchanged(bundle,'omits recorded')
    bundle.report['campaign_cost']['total_cny']=3;update_report(bundle)
    assert publish_generation(bundle.service,bundle.generation_path,bundle.report_path)['published']


def test_legacy_claim_of_57_passes_cannot_hide_56_audited_results(bundle):
    old=json.loads(bundle.legacy_report.read_text())
    for row in old['results'][:4]:row['independent_audit']['pass']=False
    write(bundle.legacy_report,old)
    bundle.report['legacy_regression'].update(sha256=sha(bundle.legacy_report),passed=57)
    update_report(bundle);reject_unchanged(bundle,'Legacy 60-case')


def test_catalog_path_cannot_escape_state(bundle,tmp_path):
    bundle.generation['catalog']='../elsewhere.sqlite3';write(bundle.generation_path,bundle.generation)
    reject_unchanged(bundle,'direct SQLite')


def test_running_server_blocks_publish_before_validation_or_pointer_writes(bundle):
    old=(bundle.state/'active.json').read_bytes()
    with FileLock(str(bundle.state/'server.lock'),timeout=0):
        with pytest.raises(Timeout):publish_generation(bundle.service,bundle.generation_path,bundle.report_path)
    assert (bundle.state/'active.json').read_bytes()==old


@pytest.mark.parametrize('mutation',[
    lambda r:r['results'][0]['grade'].update(span_recalled_at_10=[False]),
    lambda r:r['results'][0]['grade'].update(span_recalled_at_10=[True,True]),
    lambda r:r['results'][0]['retrieval_score'].update(span_at_80=[None]),
    lambda r:r['results'][0]['retrieval_score'].update(span_at_80=[False]),
    lambda r:r['results'][0]['case_metadata'].update(answerable=False),
    lambda r:r['results'][0]['answer'].update(status='abstained'),
    lambda r:r['results'][110]['answer'].update(status='answered'),
    lambda r:r['results'][0]['answer'].pop('evaluation_trace'),
    lambda r:r['results'][0]['answer'].update(answer=''),
])
def test_quality_summary_cannot_hide_missing_or_wrong_case_measurements(bundle,mutation):
    mutation(bundle.report);update_report(bundle);reject_unchanged(bundle)


@pytest.fixture
def reviewed_bundle(tmp_path):
    """Synthetic 240-case sealing evidence; no private gold or human review."""
    root=tmp_path/'reviewed';root.mkdir()
    sources={f'source-{i}.txt':{'hash':hashlib.sha256(str(i).encode()).hexdigest()} for i in range(38)}
    source_names=list(sources);checks=[];datasets={};artifacts={}
    for split in ('dev','blind'):
        cases=[]
        for i in range(120):
            category='single' if i<84 else 'cross_source' if i<96 else 'multi_turn' if i<108 else 'unanswerable'
            identity=f'{split}-{i}'
            review={'reviewer':split+'-source-reviewer','source_checked':True,'reason':'Synthetic original checked'}
            case={'id':identity,'schema_version':'quality-v2','split':split,'question':'Explain case '+identity,
                  'answerable':category!='unanswerable','category':category,'cluster_id':identity,
                  'template_group':split+'-template','agent_source_reviewed':True,'agent_review':review,'evidence':[]}
            if category=='multi_turn':case['history']=[{'role':'user','content':'Earlier context '+identity}]
            for j in range(2 if category=='cross_source' else 0 if category=='unanswerable' else 1):
                name=source_names[(i+j)%38];text=identity+f'-span-{j}'
                evidence={'path':name,'source_hash':sources[name]['hash'],'span_text':text,
                          'span_sha256':hashlib.sha256(text.encode()).hexdigest()}
                case['evidence'].append(evidence)
                checks.append({'case_id':identity,**evidence,'original_hash_verified':True,
                               'agent_source_reviewed':True,'agent_review':review})
            if category=='unanswerable':
                checks.append({'case_id':identity,'agent_source_reviewed':True,'agent_review':review})
            cases.append(case)
        dataset=root/(split+'.json');write(dataset,cases)
        datasets[dataset.name]={'count':120,'sha256':sha(dataset)}
        artifact=root/(split+'-gold-audit.json')
        write(artifact,{'dataset_sha256':sha(dataset),'reviewer':split+'-source-reviewer',
            'cases':[{'id':case['id'],'valid':True,'source_checked':True} for case in cases]})
        artifacts[artifact.name]=sha(artifact)
    write(root/'source-checks.json',checks)
    summary={'status':'agent_source_reviewed_human_pending','total':240,'splits':{'dev':120,'blind':120},
        'counts':{'single':168,'cross_source':24,'multi_turn':24,'unanswerable':24},'covered_documents':38,
        'agent_source_reviewed_cases':240,'human_verified_cases':0}
    write(root/'summary.json',summary)
    manifest={**summary,'schema_version':'quality-v2','sources':sources,'split_policy':'fixed evidence clusters/templates',
        'datasets':datasets,'source_checks_sha256':sha(root/'source-checks.json'),
        'summary_sha256':sha(root/'summary.json'),'independent_review_artifacts':artifacts}
    write(root/'manifest.json',manifest)
    return root,manifest


def test_review_pending_bundle_is_content_frozen_without_claiming_human_review(reviewed_bundle):
    from ragapp.maintenance import _publication_dataset_frozen
    root,manifest=reviewed_bundle
    before={p.name:p.read_bytes() for p in root.iterdir()}
    _publication_dataset_frozen(root/'blind.json',manifest)
    assert manifest['human_verified_cases']==0
    assert {p.name:p.read_bytes() for p in root.iterdir()}==before


@pytest.mark.parametrize('field',['source_checks_sha256','summary_sha256','independent_review_artifacts','split_policy'])
def test_review_pending_requires_all_original_freeze_evidence(reviewed_bundle,field):
    from ragapp.maintenance import _publication_dataset_frozen
    root,manifest=reviewed_bundle;manifest.pop(field)
    with pytest.raises(ValueError):_publication_dataset_frozen(root/'blind.json',manifest)


@pytest.mark.parametrize('artifact',['dev.json','blind.json','source-checks.json','summary.json','blind-gold-audit.json'])
def test_review_pending_rejects_any_sealed_artifact_mismatch(reviewed_bundle,artifact):
    from ragapp.maintenance import _publication_dataset_frozen
    root,manifest=reviewed_bundle
    with (root/artifact).open('a',encoding='utf-8') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='hash mismatch'):_publication_dataset_frozen(root/'blind.json',manifest)


def test_rehashed_review_pending_cannot_hide_cross_split_group_leak(reviewed_bundle):
    from ragapp.maintenance import _publication_dataset_frozen
    root,manifest=reviewed_bundle
    path=root/'blind.json';cases=json.loads(path.read_text());cases[0]['cluster_id']='dev-0';write(path,cases)
    manifest['datasets']['blind.json']['sha256']=sha(path)
    audit_path=root/'blind-gold-audit.json';audit=json.loads(audit_path.read_text());audit['dataset_sha256']=sha(path)
    write(audit_path,audit);manifest['independent_review_artifacts'][audit_path.name]=sha(audit_path)
    write(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='leaks across splits'):_publication_dataset_frozen(path,manifest)


@pytest.mark.parametrize('status',['draft','pending_human_review','unverified',None])
def test_arbitrary_pending_status_is_not_a_freeze_contract(reviewed_bundle,status):
    from ragapp.maintenance import _publication_dataset_frozen
    root,manifest=reviewed_bundle;manifest['status']=status
    with pytest.raises(ValueError,match='recognized content-freeze'):_publication_dataset_frozen(root/'blind.json',manifest)


def test_content_freeze_does_not_replace_independent_answer_review(bundle):
    bundle.report['results'][0]['independent_audit']['gold_verified']=False
    update_report(bundle);reject_unchanged(bundle,'gold item')


@pytest.mark.parametrize('include_author,reviewer_is_author,allowed',[
    (False,False,False),(True,True,False),(True,False,True),
])
def test_revision_closure_reviewers_cannot_be_hidden_from_answer_independence(
        bundle,monkeypatch,include_author,reviewer_is_author,allowed):
    """The structural closure is separately tested with real synthetic bundles."""
    import ragapp.maintenance as maintenance
    path=bundle.dataset.parent/'manifest.json';manifest=json.loads(path.read_bytes())
    manifest['evidence_revision']={'synthetic_identity_gate_test':True};write(path,manifest)
    bundle.report['dataset_manifest_sha256']=sha(path)
    monkeypatch.setattr(maintenance,'_publication_dataset_frozen',lambda *args:
        {'reviewer_ids_by_split':{'blind':['equivalence-author']}})
    if include_author:bundle.report['audit_identity']['gold_author_ids'].append('equivalence-author')
    if reviewer_is_author:
        bundle.report['audit_reviewer']='equivalence-author'
        bundle.report['audit_identity']['id']='equivalence-author'
    update_report(bundle)
    if allowed:
        assert publish_generation(bundle.service,bundle.generation_path,bundle.report_path)['published']
    else:
        reject_unchanged(bundle,'Gold author identities omit|independent')
