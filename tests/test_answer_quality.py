import copy
import json
from types import SimpleNamespace
import pytest
from ragapp.service import Service
from ragapp.config import Settings
from ragapp.indexer import digest


class Tokenizer:
    def __call__(self,text,**kwargs):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}


class Models:
    def __init__(self,answers,store):self.answers=iter(answers);self.sent=[];self.store=store
    def complete(self,messages,*a,**kw):
        self.sent.append(messages)
        answer=next(self.answers)
        return {'text':answer if isinstance(answer,str) else json.dumps(answer),'call_id':str(len(self.sent)),
                'cost_cny_upper_estimate':.01,'profile':'production'}


def setup(tmp_path,answers,text='Fact: 42',path='a.txt',question='What value?'):
    corpus=tmp_path/'papers';corpus.mkdir();source=corpus/path;source.write_text(text)
    s=Service(Settings(root=tmp_path,corpus=corpus,state=tmp_path/'state',query_planning=False,verify_answers=True))
    with s.store.db() as db:
        db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',('d',path,digest(source),'v','ready',None,0,0))
        db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)',('c','d','v',text,'{}',text,None))
    row={'id':'c','document_id':'d','version':'v','path':path,'text':text,'parent':text,'locator':{'line_start':1,'line_end':1},'image':None,'score':1}
    calls=[]
    def search(query,**kwargs):
        calls.append((query,kwargs.get('document_ids')))
        kwargs['trace'].update(candidates=[copy.deepcopy(row)],elapsed=.01)
        return [copy.deepcopy(row)]
    s.retrieval=SimpleNamespace(search=search,rank=lambda q,rows,trace:[1.0]*len(rows),
                               embedding=SimpleNamespace(tokenizer=Tokenizer()))
    s.models=Models(answers,s.store)
    return s,calls


def sufficient(passage='S1:P1'):
    return {'sufficient':True,'coverage':[{'requirement_index':0,'supported':True,
        'passage_ids':[passage]}]}


def supported():return {'supported':True,'claims':[{'claim':'42','citations':['S1'],'supported':True}]}


def test_verified_answer_records_all_call_costs(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',supported()])
    answer=s.ask('What value?')
    assert answer['status']=='answered' and len(s.models.sent)==3
    assert answer['total_cost_cny_upper_estimate']==pytest.approx(.03)
    assert answer['trace_id'] and s.store.trace(answer['trace_id'])


def test_unsupported_answer_has_one_repair_only(tmp_path):
    rejected={'supported':False,'claims':[{'claim':'wrong','supported':False,'citations':['S1']}]}
    s,_=setup(tmp_path,[sufficient(),'wrong [S1]',rejected,'still wrong [S1]',rejected])
    answer=s.ask('What value?')
    assert answer['status']=='verification_failed' and len(s.models.sent)==5
    assert 'still wrong' not in answer['answer']


def test_invalid_sufficiency_fails_closed(tmp_path):
    s,_=setup(tmp_path,[{'sufficient':'true'}])
    assert s.ask('What value?')['status']=='verification_failed'
    assert len(s.models.sent)==1


def test_positive_sufficiency_without_grounded_quote_fails_closed(tmp_path):
    s,_=setup(tmp_path,[sufficient('S999:P1')])
    result=s.ask('What value?')
    assert result['status']=='verification_failed' and len(s.models.sent)==1
    assert result['total_cost_cny_upper_estimate']==pytest.approx(.01)
    assert s.store.trace(result['trace_id'])['trace']['sufficiency_coverage_errors']==[
        'invalid_coverage_passages']
    assert s.store.trace(result['trace_id'])['trace']['sufficiency_checks'][0]['coverage'][0]['passage_ids']==['S999:P1']


def test_malformed_judgment_still_counts_paid_call(tmp_path):
    s,_=setup(tmp_path,['not JSON'])
    answer=s.ask('What value?')
    assert answer['status']=='verification_failed'
    assert answer['total_cost_cny_upper_estimate']==pytest.approx(.01)
    assert answer['call_ids']==['1']


def test_malformed_planner_still_counts_paid_call(tmp_path):
    from ragapp.query import plan_query
    s,_=setup(tmp_path,['not JSON'])
    plan=plan_query('比较两种方法',models=s.models)
    assert plan.queries==['比较两种方法']
    assert plan.call['call_id']=='1' and plan.call['cost_cny_upper_estimate']==.01


@pytest.mark.parametrize('content',['not JSON',{'standalone':'独立问题','subqueries':[],'requirements':[]}])
def test_retrieval_response_exposes_successful_or_malformed_planner_cost(tmp_path,content):
    s,_=setup(tmp_path,[content]);s.settings.query_planning=True
    original=s.models.complete
    def complete(*args,**kwargs):
        return {**original(*args,**kwargs),'api_key':'do-not-return','headers':{'Authorization':'secret'}}
    s.models.complete=complete
    response=s.search_detailed('比较两种方法')
    assert response['cost_cny_upper_estimate']==pytest.approx(.01)
    assert response['call_ids']==['1'] and len(response['planner_calls'])==1
    assert response['planner_calls'][0]['cost_cny_upper_estimate']==pytest.approx(.01)
    assert 'api_key' not in response['trace']['plan']['call']
    assert 'do-not-return' not in json.dumps(response) and 'Authorization' not in json.dumps(response)
    assert s.store.trace(response['trace_id'])['cost_cny_upper_estimate']==pytest.approx(.01)


def test_retrieval_response_includes_failed_planner_call_cost(tmp_path):
    s,_=setup(tmp_path,[]);s.settings.query_planning=True
    def fail(*args,**kwargs):
        exc=RuntimeError('unavailable')
        exc.call={'call_id':'failed-1','status':'failed','cost_cny_upper_estimate':.03,
                  'requested_model':'planner','api_key':'secret'}
        raise exc
    s.models.complete=fail
    response=s.search_detailed('比较两种方法')
    assert response['trace']['plan']['planning']=='current_question_fallback'
    assert response['cost_cny_upper_estimate']==pytest.approx(.03)
    assert response['call_ids']==['failed-1']
    assert response['planner_calls'][0]['status']=='failed'
    assert 'secret' not in json.dumps(response)


def test_direct_retrieval_has_no_planner_cost_or_call(tmp_path):
    s,_=setup(tmp_path,[])
    response=s.search_detailed('What value?')
    assert response['cost_cny_upper_estimate']==0
    assert response['planner_calls']==response['call_ids']==[] and not s.models.sent


@pytest.mark.parametrize('cost',[None,float('nan'),float('inf'),-1,True,'0.01'])
def test_retrieval_does_not_report_unknown_planner_cost_as_zero(tmp_path,cost):
    s,_=setup(tmp_path,[]);s.settings.query_planning=True
    s.models.complete=lambda *a,**kw:{'text':'not JSON','call_id':'charged','cost_cny_upper_estimate':cost}
    with pytest.raises(ValueError,match='Planner call cost'):
        s.search_detailed('比较两种方法')


@pytest.mark.parametrize('scores',[[],[1.,2.],[float('nan')],[[1.]],['1.0']])
def test_service_rejects_malformed_rank_adapter_result(tmp_path,scores):
    s,_=setup(tmp_path,[]);s.retrieval.rank=lambda *args:scores
    with pytest.raises(ValueError,match='Reranker'):
        s.search_detailed('What value?')


def test_source_changed_without_sync_cannot_return_valid_citation(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',supported()])
    complete=s.models.complete
    def mutate(*args,**kwargs):
        value=complete(*args,**kwargs)
        if len(s.models.sent)==3:(s.settings.corpus/'a.txt').write_text('Fact: 999')
        return value
    s.models.complete=mutate
    answer=s.ask('What value?')
    assert answer['status']=='stale' and not answer['citations'] and not answer['evidence']


def test_incomplete_table_filter_cannot_fall_back_to_model_arithmetic(tmp_path):
    s,_=setup(tmp_path,[],text='date,value\n2024-01-01,1\n2024-01-02,2\n',path='data.csv')
    answer=s.ask('data.csv 在2024-01-01且 value 大于0的平均值')
    assert answer['status']=='abstained' and not s.models.sent
    assert not answer['calculations']


def test_retry_rebuilds_calculation_context(tmp_path):
    s,_=setup(tmp_path,[{'sufficient':False,'queries':['complete table']},sufficient(),
        '总和3 [S1]',supported()],text='value\n1\n2\n',path='data.csv')
    answer=s.ask('data.csv 的 value 列总和')
    assert answer['status']=='answered'
    second=json.loads(s.models.sent[1][1]['content'])
    assert 'Deterministic calculation' in second['untrusted_evidence'][0]['passages'][0]['text']
    assert second['deterministic_calculations'][0]['value']==3
    assert answer['citations'][0]['locator']['kind']=='calculation'


def test_large_original_aggregate_survives_generation_budget_and_keeps_audit_rows(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'平均值为2 [S1]',supported()],
        text='value\n'+'2\n'*3500,path='data.csv')
    # This fake retriever returns the whole file in one chunk and counts chars
    # as tokens. Admit that fixture; keep the real generation budget unchanged.
    s.settings.context_tokens=16000
    result=s.ask('data.csv 全部 value 列的平均值')
    assert result['status']=='answered'
    assert result['calculations'][0]['included_rows']==list(range(2,3502))
    sent=json.loads(s.models.sent[1][1]['content'])
    calculation=sent['deterministic_calculations'][0]
    assert calculation['value']==2 and calculation['included_rows_ranges']==[[2,3501]]
    assert 'included_rows' not in calculation
    evidence=json.loads(sent['untrusted_evidence'][0]['text'].split('\n',1)[1])
    assert evidence['included_rows_ranges']==[[2,3501]]
    assert s.store.trace(result['trace_id'])['trace']['generation_input_upper_tokens']<=s.settings.generation_context_tokens


def test_physical_row_lookup_supplies_target_and_row_identity_to_judges(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'传感器的原值为7 [S1]',supported()],
        text='设备,原值\n电话,2\n传感器,7\n',path='data.csv')
    result=s.ask('data.csv 第3行的传感器原值是多少？')
    assert result['status']=='answered'
    for messages in s.models.sent[:2]:
        value=json.loads(messages[1]['content'])['deterministic_calculations'][0]
        assert value['value']=='7' and value['range']=='B3:B3'
        assert value['row_context']['cells']['A3']['value']=='传感器'
    assert result['citations'][0]['locator']['row_context']['range']=='A3:B3'


def test_original_code_locations_reach_generation_after_pooling_and_expansion(tmp_path):
    code='# config\n\nSIZE_DAYS = 365\nNEXT_VALUE = 12\n'
    s,_=setup(tmp_path,[sufficient(),'SIZE_DAYS为365，原文件第3行 [S1]',supported()],text=code,path='code.py')
    search=s.retrieval.search
    def expanded(*a,**kw):
        rows=search(*a,**kw)
        rows[0]['locator'].update(line_start=4,line_end=4,context_expanded=True,
            source_spans=[{'line_start':1,'line_end':4}])
        return rows
    s.retrieval.search=expanded
    result=s.ask('code.py SIZE_DAYS初始值是多少，给出代码位置')
    assert result['status']=='answered'
    for messages in s.models.sent[:2]:
        loc=json.loads(messages[1]['content'])['untrusted_evidence'][0]['locator']
        assert loc['matched_lines']=={'SIZE_DAYS':3}
    assert result['citations'][0]['locator']['matched_lines']=={'SIZE_DAYS':3}


def test_coordinate_header_lookup_reaches_generation_with_physical_header_citation(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'温度 [S1]',
        {'supported':True,'claims':[{'claim':'温度','citations':['S1'],'supported':True}]}],
        text='设备,温度\na,25\n',path='sensors.csv')
    result=s.ask('sensors.csv 中 B2 所在列的表头是什么？')
    assert result['status']=='answered'
    assert result['calculations'][0]['value']=='温度'
    assert result['calculations'][0]['requested_cell']=='B2'
    cite=result['citations'][0]['locator']
    assert cite['operation']=='header_lookup' and cite['range']=='B1:B1'
    assert cite['requested_cell']=='B2' and cite['column_letter']=='B'
    assert cite['row_start']==cite['row_end']==1
    assert '温度' in json.loads(s.models.sent[1][1]['content'])['untrusted_evidence'][0]['text']


def test_original_document_scope_survives_rewrite_and_retry(tmp_path,monkeypatch):
    from ragapp.query import QueryPlan
    s,calls=setup(tmp_path,[{'sufficient':False,'queries':['no filename here']},{'sufficient':False,'missing':['fact']}])
    monkeypatch.setattr('ragapp.service.plan_query',lambda question,history,**kw:
                        QueryPlan(question,[question,'rewritten without filename'],kw['document_ids']))
    s.ask('根据a.txt说明答案')
    assert len(calls)==3 and all(ids==['d'] for _,ids in calls)


def test_multiple_queries_share_one_rerank_budget(tmp_path,monkeypatch):
    from ragapp.query import QueryPlan
    s,_=setup(tmp_path,[])
    original=s.retrieval.search;rank_calls=[];search_options=[]
    def recall(question,**kwargs):
        search_options.append(kwargs['options'])
        return original(question,**kwargs)
    def rank(question,rows,trace):
        rank_calls.append((question,len(rows)));return [1.0]*len(rows)
    s.retrieval.search=recall;s.retrieval.rank=rank
    monkeypatch.setattr('ragapp.service.plan_query',lambda q,h,**kw:
        QueryPlan(q,[q,'standalone','object A','object B'],kw['document_ids']))
    response=s.search_detailed('current question')
    assert len(rank_calls)==1 and len(search_options)==4
    assert all(o['retrieval_mode']=='hybrid' for o in search_options)
    assert rank_calls[0][0]=='current question'  # A subquery is not the complete rerank intent.
    assert response['trace']['pooled_rerank']['candidates']<=s.settings.rerank_candidates


def test_supplemental_queries_also_share_one_rerank(tmp_path):
    s,_=setup(tmp_path,[{'sufficient':False,'queries':['A','B','C']},
        sufficient(),'42 [S1]',supported()])
    calls=[]
    def rank(q,rows,trace):calls.append(len(rows));return [1.0]*len(rows)
    s.retrieval.rank=rank
    answer=s.ask('What value?')
    assert answer['status']=='answered' and len(calls)==2


def test_campaign_budget_is_shared_across_catalogs(tmp_path,monkeypatch):
    from ragapp.llm import Models as RealModels
    from ragapp.store import Store
    settings=Settings(root=tmp_path,state=tmp_path/'state',campaign='quality-v2').prepare()
    one=RealModels(settings,Store(settings.state/'first.sqlite3'))
    two=RealModels(settings,Store(settings.state/'second.sqlite3'))
    two.keys={'DEEPSEEK_API_KEY':'fake'}
    with one.store.db() as db:
        db.execute("INSERT INTO calls VALUES('spent','production','development','m','m','failed',100,NULL,NULL,NULL,0,0)")
        db.execute("INSERT INTO call_campaigns VALUES('spent','quality-v2')")
    monkeypatch.setattr('httpx.post',lambda *a,**k:pytest.fail('Network after exhausted budget'))
    with pytest.raises(RuntimeError,match='budget'):two.complete([{'role':'user','content':'test'}])


def test_resolved_followup_reaches_reranker_and_generation(tmp_path,monkeypatch):
    from ragapp.query import QueryPlan
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',supported()])
    monkeypatch.setattr('ragapp.service.plan_query',lambda q,h,**kw:
        QueryPlan(q,[q,'sensor calibration'],kw['document_ids'],standalone='sensor calibration'))
    rank_queries=[]
    s.retrieval.rank=lambda q,rows,trace:rank_queries.append(q) or [1.]*len(rows)
    answer=s.ask('这些怎么校准？',history=[{'role':'user','content':'sensors?'}])
    assert answer['status']=='answered' and rank_queries==['sensor calibration']
    for messages in s.models.sent:
        payload=json.loads(messages[1]['content'])
        assert payload['question']=='sensor calibration'
        assert payload['original_question']=='这些怎么校准？'


def test_uncited_claim_produces_explicit_repair_feedback(tmp_path):
    judgment={'supported':True,'unsupported_claims':[],
              'claims':[{'claim':'uncited extra condition','supported':True,'citations':[]}]}
    s,_=setup(tmp_path,[sufficient(),'42 [S1]. Extra condition.',judgment,'42 [S1]',supported()])
    answer=s.ask('What value?')
    assert answer['status']=='answered'
    repair=json.loads(s.models.sent[3][1]['content'])
    assert repair['issues']['validation_errors']==[
        {'claim':'uncited extra condition','reason':'missing_or_invalid_citations'}]
    trace=s.store.trace(answer['trace_id'])['trace']
    assert trace['answer_checks'][0]['validation_errors']==repair['issues']['validation_errors']


@pytest.mark.parametrize('claims',[None,[],['wrong'],[{'claim':'42','supported':True,'citations':'S1'}]])
def test_malformed_claim_list_fails_closed_after_one_repair(tmp_path,claims):
    rejected={'supported':True,'claims':claims}
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',rejected,'42 [S1]',rejected])
    answer=s.ask('What value?')
    assert answer['status']=='verification_failed' and len(s.models.sent)==5
    assert json.loads(s.models.sent[3][1]['content'])['issues']['validation_errors']


def test_claim_judge_has_separate_bounded_output_budget(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',supported()])
    complete=s.models.complete;budgets=[]
    def record(*args,**kwargs):
        budgets.append(kwargs.get('max_tokens'));return complete(*args,**kwargs)
    s.models.complete=record
    assert s.ask('What value?')['status']=='answered'
    assert budgets==[1600,None,2400]


def test_overall_supported_cannot_override_reported_unsupported_claim(tmp_path):
    contradictory={**supported(),'unsupported_claims':['The answer omits a necessary condition.']}
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',contradictory,'42 [S1]',supported()])
    assert s.ask('What value?')['status']=='answered'
    assert len(s.models.sent)==5
    repair=json.loads(s.models.sent[3][1]['content'])
    assert repair['issues']['validation_errors']==[{'reason':'reported_unsupported_claims'}]


def test_exact_model_inputs_and_raw_invalid_output_are_admin_only(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'missing citations','still no citations'])
    result=s.ask('What value?')
    assert result['status']=='invalid_citation'
    trace=s.store.trace(result['trace_id'])['trace']
    assert [step['step'] for step in trace['model_steps']]==['sufficiency','generation','answer_repair']
    assert trace['model_steps'][1]['output']=='missing citations'
    assert trace['model_steps'][0]['messages']==s.models.sent[0]
    assert 'model_steps' not in result and 'missing citations' not in result['answer']


@pytest.mark.parametrize('initial',['42','42 [S999]'])
def test_missing_or_unknown_citation_uses_one_repair_then_semantic_verification(tmp_path,initial):
    s,_=setup(tmp_path,[sufficient(),initial,'42 [S1]',supported()])
    result=s.ask('What value?')
    assert result['status']=='answered' and len(s.models.sent)==4
    repair=json.loads(s.models.sent[2][1]['content'])
    assert repair['issues']['allowed_citations']==['S1']
    assert result['total_cost_cny_upper_estimate']==pytest.approx(.04)


def test_citation_repair_cannot_trigger_second_semantic_repair(tmp_path):
    rejected={'supported':False,'claims':[{'claim':'42','citations':['S1'],'supported':False}]}
    s,_=setup(tmp_path,[sufficient(),'42 [S999]','42 [S1]',rejected])
    result=s.ask('What value?')
    assert result['status']=='verification_failed' and len(s.models.sent)==4


def test_model_trace_is_immutable_and_does_not_copy_provider_secrets(tmp_path):
    s,_=setup(tmp_path,[]);trace={};messages=[{'role':'user','content':'original'}]
    def complete(sent,*a,**kw):
        sent[0]['content']='mutated'
        return {'text':'output','call_id':'1','api_key':'SECRET','headers':{'Authorization':'SECRET'}}
    s.models.complete=complete
    s._complete_traced(messages,'production','development',trace=trace)
    assert trace['model_steps'][0]['messages'][0]['content']=='original'
    assert 'SECRET' not in json.dumps(trace)


def test_malformed_paid_judge_output_is_preserved_in_trace(tmp_path):
    s,_=setup(tmp_path,['{truncated'])
    result=s.ask('What value?');trace=s.store.trace(result['trace_id'])['trace']
    assert result['status']=='verification_failed'
    assert trace['model_steps'][0]['output']=='{truncated'
    assert trace['model_steps'][0]['call']['call_id'] in result['call_ids']


def test_failed_provider_step_is_recorded_without_exception_secrets(tmp_path):
    s,_=setup(tmp_path,[])
    def fail(*a,**kw):
        exc=RuntimeError('SECRET');exc.call={'call_id':'failed','cost_cny_upper_estimate':.02,'api_key':'SECRET'}
        raise exc
    s.models.complete=fail
    result=s.ask('What value?');trace=s.store.trace(result['trace_id'])['trace']
    assert trace['model_steps'][0]['status']=='failed'
    assert 'SECRET' not in json.dumps(trace['model_steps'])
    assert result['total_cost_cny_upper_estimate']==.02


def test_experimental_reasoning_only_changes_production_judges_and_is_traced(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',supported()])
    s.settings.judge_reasoning_effort='low'
    result=s.ask('What value?');steps=s.store.trace(result['trace_id'])['trace']['model_steps']
    assert result['status']=='answered'
    assert [step['options'].get('reasoning_effort') for step in steps]==['low',None,'low']
    assert [step['options'].get('max_tokens') for step in steps]==[6144,None,6144]


def test_free_judge_never_inherits_paid_thinking_configuration(tmp_path):
    s,_=setup(tmp_path,[{'supported':True}]);s.settings.judge_reasoning_effort='low';trace={}
    s._judge('judge',{},'eval_aihub_free','development',max_tokens=1000,trace=trace)
    assert trace['model_steps'][0]['options']=={'max_tokens':1000}


def test_sufficiency_only_thinking_does_not_change_claim_judge(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',supported()])
    s.settings.judge_reasoning_effort='low';s.settings.judge_reasoning_scope='sufficiency'
    result=s.ask('What value?');steps=s.store.trace(result['trace_id'])['trace']['model_steps']
    assert [step['options'].get('reasoning_effort') for step in steps]==['low',None,None]
    assert steps[-1]['options']['max_tokens']==2400


def test_evidence_outline_reaches_generation_and_preserves_full_context(tmp_path):
    decision=sufficient();decision['coverage'][0]['reason']='Do not copy judge speculation'
    s,_=setup(tmp_path,[decision,'42 [S1]',supported()]);s.settings.use_evidence_outline=True
    result=s.ask('What value?');data=json.loads(s.models.sent[1][1]['content'])
    assert data['request_evidence'][0]['request']=='What value?'
    assert data['request_evidence'][0]['passages'][0]['text']=='Fact: 42'
    assert data['request_evidence'][0]['passages'][0]['source']=='a.txt'
    assert data['untrusted_evidence'][0]['text']=='Fact: 42'
    assert 'judge speculation' not in json.dumps(data)
    assert s.store.trace(result['trace_id'])['trace']['evidence_outline']['status']=='included'


def test_over_budget_outline_is_explicitly_omitted_without_cutting_original_evidence(tmp_path):
    s,_=setup(tmp_path,[sufficient(),'42 [S1]',supported(),sufficient(),'42 [S1]',supported()],text='x'*2000)
    baseline=s.ask('What value?');base=s.store.trace(baseline['trace_id'])['trace']['generation_input_upper_tokens']
    s.settings.use_evidence_outline=True;s.settings.generation_context_tokens=base+100
    result=s.ask('What value?');trace=s.store.trace(result['trace_id'])['trace']
    assert trace['evidence_outline']['status']=='omitted_over_budget'
    data=json.loads(s.models.sent[4][1]['content'])
    assert 'request_evidence' not in data and data['untrusted_evidence'][0]['text']=='x'*2000
