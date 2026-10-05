import json
from types import SimpleNamespace
import numpy as np
import pytest
from ragapp.config import Settings
from ragapp.store import Store
from ragapp.retrieval import Retrieval
from ragapp.query import plan_query
from ragapp.context import build_context
from ragapp.textbudget import windows


class Tokenizer:
    def __call__(self,text,text_pair=None,**kwargs):
        length=len(text)+(len(text_pair)+4 if text_pair is not None else 0)
        return {'offset_mapping':[(i,i+1) for i in range(len(text))],'input_ids':list(range(length))}


class Encoder:
    tokenizer=Tokenizer()
    def __init__(self):self.inputs=[]
    def encode(self,texts,**kwargs):
        self.inputs.extend(texts);return np.ones((len(texts),3))


class Ranker:
    tokenizer=Tokenizer()
    def __init__(self):self.pairs=[]
    def predict(self,pairs,**kwargs):
        self.pairs.extend(pairs);return [10 if 'correct' in d else 0 for _,d in pairs]


def retrieval(tmp_path):
    settings=Settings(root=tmp_path,state=tmp_path/'state',query_tokens=80).prepare()
    store=Store(settings.state/'catalog.sqlite3');r=Retrieval(settings,store)
    r._embedding=Encoder();r._reranker=Ranker()
    return r


def put(r,identity,text='correct',path='general.txt',document='d'):
    with r.store.db() as db:
        db.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?,?,?)',(document,path,'h','v','ready',None,0,0))
        db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?)',(identity,document,'v',text,'{}',text,None))


def test_soft_alias_does_not_restrict_general_question(tmp_path):
    r=retrieval(tmp_path);put(r,'a',path='LSTM_notes.pdf');put(r,'b',path='course.pdf',document='other')
    assert r.store.focused('LSTM 是什么？')[1]=={'d'}
    assert r.store.source_scope('LSTM 是什么？') is None
    assert r.store.source_scope('LSTM_notes.pdf 的结论')=={'d'}
    assert r.store.source_scope('什么是LSTM',[])==set()


def test_named_chinese_source_is_only_a_soft_hint(tmp_path):
    r=retrieval(tmp_path)
    put(r,'intro',path='folder/1.土壤概述.pdf',document='intro')
    put(r,'sensors',path='folder/2.传感器操作.pdf',document='sensors')
    question='传感器操作课件如何测温？另一份概述课件怎样定义湿度？'
    hints=r.store.source_hints(question)
    assert {h['document_id'] for h in hints}=={'intro','sensors'}
    assert r.store.focused(question)[1]=={'intro','sensors'}
    assert r.store.source_scope(question) is None
    assert r.store.source_hints('湿度是什么意思？')==[]
    assert r.store.source_hints('概述这个方法')==[]


def test_source_hints_use_longest_named_suffix_and_active_versions(tmp_path):
    r=retrieval(tmp_path)
    put(r,'a',path='土壤概述.pdf',document='a')
    put(r,'b',path='大气概述.pdf',document='b')
    put(r,'c',path='水质概述.pdf',document='c')
    assert {h['document_id'] for h in r.store.source_hints('土壤概述课件如何定义湿度？')}=={'a'}
    r.store.exclude('c')
    assert {h['document_id'] for h in r.store.source_hints('概述课件')}=={'a','b'}


def test_soft_source_dense_lane_does_not_remove_global_candidates(tmp_path,monkeypatch):
    r=retrieval(tmp_path);put(r,'global','correct','general.txt','global-doc')
    put(r,'hint','other','hint.txt','hint-doc')
    monkeypatch.setattr(r.store,'focused',lambda q:([],{'hint-doc'}))
    monkeypatch.setattr(r.store,'lexical',lambda *a:[])
    filters=[]
    def query(*a,**kwargs):
        filters.append(kwargs['query_filter'])
        key='global' if kwargs['query_filter'] is None else 'hint'
        return SimpleNamespace(points=[SimpleNamespace(id=key)])
    r.client=SimpleNamespace(query_points=query)
    trace={};result=r.search('question',trace=trace)
    assert len(filters)==2 and filters[0] is None
    assert filters[1].must[0].match.any==['hint-doc']
    assert {row['id'] for row in result}=={'global','hint'}
    assert 'hint_dense' in next(row for row in trace['candidates'] if row['id']=='hint')['ranks']


def test_legacy_vectors_reject_changed_model_revision(tmp_path):
    r=retrieval(tmp_path);put(r,'a')
    r.settings.embedding_revision='another-revision'
    with pytest.raises(RuntimeError,match='original pinned E5'):
        r.search('LSTM')


def test_legacy_vectors_cannot_be_queried_with_candidate_precision(tmp_path):
    r=retrieval(tmp_path);put(r,'a')
    r.settings.embedding_precision='float32'
    with pytest.raises(RuntimeError,match='original pinned E5'):
        r.search('LSTM')


def test_bm25_budgeting_does_not_load_embedding_weights(tmp_path,monkeypatch):
    r=retrieval(tmp_path);put(r,'a');r._embedding=None;r._tokenizer=Tokenizer()
    monkeypatch.setattr(r.store,'lexical',lambda *args:['a'])
    monkeypatch.setattr(Retrieval,'embedding',property(lambda self:pytest.fail('BM25 loaded embedding weights')))
    trace={}
    assert r.search('correct',options={'retrieval_mode':'bm25'},trace=trace)[0]['id']=='a'
    assert trace['query_input_tokens']==7 and trace['query_window_budget']==80


def test_focused_does_not_starve_best_hybrid_candidate(tmp_path,monkeypatch):
    r=retrieval(tmp_path);put(r,'answer')
    focused=[f'f{i}' for i in range(25)]
    for i in focused:put(r,i,'noise')
    monkeypatch.setattr(r.store,'focused',lambda q:(focused,{'d'}))
    monkeypatch.setattr(r.store,'lexical',lambda *a:['answer'])
    r.client=SimpleNamespace(query_points=lambda *a,**k:SimpleNamespace(points=[SimpleNamespace(id='answer')]))
    trace={};result=r.search('question',trace=trace,options={'rerank_candidates':20})
    assert result[0]['id']=='answer'
    assert any('correct' in d for _,d in r._reranker.pairs)


def test_long_question_retrieves_tail_without_history_contamination(tmp_path,monkeypatch):
    r=retrieval(tmp_path);put(r,'a')
    monkeypatch.setattr(r.store,'lexical',lambda *a:['a'])
    r.client=SimpleNamespace(query_points=lambda *a,**k:SimpleNamespace(points=[SimpleNamespace(id='a')]))
    question='unrelated '*100+'CURRENT_UNIQUE'
    plan=plan_query(question,[{'role':'user','content':'OLD_FILENAME.py '*300}])
    trace={};r.search(plan.queries[0],trace=trace)
    assert any('CURRENT_UNIQUE' in text for text in r._embedding.inputs)
    assert all('OLD_FILENAME' not in text for text in r._embedding.inputs)
    assert trace['query_windows']>1


def test_deleted_candidates_trigger_bounded_refill(tmp_path,monkeypatch):
    r=retrieval(tmp_path);put(r,'a');put(r,'b')
    monkeypatch.setattr(r.store,'lexical',lambda *a:[])
    ids=['stale1','stale2','stale3','stale4','a','b'];requested=[]
    def query(*a,**kwargs):
        requested.append(kwargs['limit']);return SimpleNamespace(points=[SimpleNamespace(id=i) for i in ids[:kwargs['limit']]])
    r.client=SimpleNamespace(query_points=query)
    result=r.search('question',options={'dense_candidates':2})
    assert requested==[4,8] and {x['id'] for x in result}=={'a','b'}


def test_reranking_includes_document_tail(tmp_path):
    r=retrieval(tmp_path);r.settings.reranker_max_tokens=160
    score=r.rank('question',[{'path':'a.txt','text':'noise '*100+'correct at tail','locator':{}}])
    assert score==[10] and len(r._reranker.pairs)>1
    assert all(len(q)+len(d)+8<=160 for q,d in r._reranker.pairs)


@pytest.mark.parametrize('predictions', [[], [1.0], [1.0,2.0,3.0],
    [float('nan'),1.0], [float('inf'),1.0], [float('-inf'),1.0],
    [[1.0],[2.0]], np.array([[1.0],[2.0]]), [np.array(1.0),2.0],
    ['1.0',2.0], [True,2.0], [1+2j,2.0], '12', {'a':1.0,'b':2.0}, None])
def test_reranking_rejects_invalid_model_output(tmp_path,predictions):
    r=retrieval(tmp_path)
    r._reranker=SimpleNamespace(predict=lambda *a,**kw:predictions)
    rows=[{'path':'a.txt','text':'first'},{'path':'b.txt','text':'second'}]
    with pytest.raises(ValueError,match='Reranker'):
        r.rank('question',rows)


def test_reranking_validates_window_count_not_just_candidate_count(tmp_path):
    r=retrieval(tmp_path);r.settings.reranker_max_tokens=160
    r._reranker.predict=lambda pairs,**kw:[1.0]
    with pytest.raises(ValueError,match='score count mismatch'):
        r.rank('question',[{'path':'a.txt','text':'long text '*100}])


def test_reranking_accepts_finite_numpy_scalars_without_changing_order(tmp_path):
    r=retrieval(tmp_path)
    r._reranker=SimpleNamespace(predict=lambda *a,**kw:np.array([-2.5,1.25],dtype=np.float32))
    rows=[{'path':'a.txt','text':'first'},{'path':'b.txt','text':'second'}]
    assert r.rank('question',rows)==[-2.5,1.25]


def test_context_deduplicates_parent_and_respects_serialized_budget():
    rows=[{'id':str(i),'document_id':'d','version':'v','path':'a.txt','citation':f'S{i+1}',
           'text':'child','parent':'full parent evidence','locator':{'parent_id':'p',
           'source_spans':[{'line_start':1,'line_end':1}],
           'parent_source_spans':[{'line_start':1,'line_end':3}]}} for i in range(2)]
    selected,trace=build_context(rows,Tokenizer(),1000)
    assert len(selected)==1 and selected[0]['text']=='full parent evidence'
    assert selected[0]['locator']['source_spans'][0]['line_end']==3
    assert trace['context_tokens']<=1000


def test_failed_rewrite_preserves_original_question():
    class Models:
        def complete(self,*a,**k):return {'text':'not json'}
    question='它和第二个方法有何区别？'
    plan=plan_query(question,[{'role':'user','content':'prior'}],models=Models())
    assert plan.queries==[question] and plan.planning=='current_question_fallback'


def test_windows_preserve_source_whitespace():
    source='def f():\n    x = 1\n    return x\n'
    for text,a,b in windows(Tokenizer(),source,16,2):assert text==source[a:b]


def test_bge_loader_requests_fp32_and_rejects_unverified_parameters(tmp_path,monkeypatch):
    import sys
    calls=[];model=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(dtype='torch.bfloat16')]))
    def construct(*args,**kwargs):calls.append(kwargs);return model
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(float32='torch.float32'))
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(CrossEncoder=construct))
    r=Retrieval(Settings(state=tmp_path),None)
    with pytest.raises(ValueError,match='FP32'):r.reranker
    assert r._reranker is None  # A failed check must not leave a reusable invalid model.
    assert calls[0]['model_kwargs']=={'torch_dtype':'torch.float32'}
    model.parameters=lambda:iter([SimpleNamespace(dtype='torch.float32')])
    assert r.reranker is model and len(calls)==2


@pytest.mark.parametrize('dtypes',[[],['torch.bfloat16'],['torch.float32','torch.bfloat16']])
def test_reranker_precision_checks_every_parameter_and_qwen_wrapper(dtypes):
    from ragapp.retrieval import validated_reranker_precision
    model=SimpleNamespace(parameters=lambda:iter(SimpleNamespace(dtype=d) for d in dtypes))
    with pytest.raises(ValueError,match='FP32'):validated_reranker_precision(SimpleNamespace(model=model))
    model.parameters=lambda:iter([SimpleNamespace(dtype='torch.float32')])
    assert validated_reranker_precision(SimpleNamespace(model=model))['actual_parameter_dtypes']==['torch.float32']


def test_reranker_pair_length_includes_special_tokens_and_checks_before_predict(tmp_path):
    class SpecialTokenizer(Tokenizer):
        def __call__(self,text,text_pair=None,**kwargs):
            result=super().__call__(text,text_pair,**kwargs)
            if text_pair is not None:
                assert kwargs['truncation'] is False and kwargs['add_special_tokens'] is True
                result['input_ids']+=[0]*1000
            return result
    r=retrieval(tmp_path);r.settings.reranker_max_tokens=160;r._reranker.tokenizer=SpecialTokenizer();trace={}
    with pytest.raises(ValueError,match='before inference'):
        r.rank('query',[{'path':'a.txt','text':'document'}],trace)
    assert not r._reranker.pairs
    assert trace['rerank_max_pair_tokens']>trace['rerank_model_max_tokens']==160
    assert trace['silent_truncations'] is None and trace['rerank_pair_budget_verified'] is False


def test_reranker_budget_trace_measures_pairs_and_preserves_long_question_and_document_tails(tmp_path):
    r=retrieval(tmp_path);r.settings.reranker_max_tokens=160;trace={}
    r.rank('question '*30+'QUERY_TAIL',[{'path':'a.txt','text':'sentence '*60+'DOCUMENT_TAIL'}],trace)
    pairs=r._reranker.pairs
    assert any('QUERY_TAIL' in q for q,d in pairs) and any('DOCUMENT_TAIL' in d for q,d in pairs)
    assert trace['rerank_max_pair_tokens']==max(len(q)+len(d)+4 for q,d in pairs)
    assert trace['rerank_max_pair_tokens']<=160 and trace['rerank_model_max_tokens']==160
    assert trace['rerank_pair_budget_verified'] is True and trace['silent_truncations']==0


def test_qwen_budget_trace_counts_full_instruction_and_chat_template(tmp_path):
    from ragapp.model_adapters import TASK
    r=retrieval(tmp_path);r.settings.reranker_model='Qwen/Qwen3-Reranker-0.6B';r.settings.reranker_max_tokens=1024
    r._reranker.prefix='CHAT_PREFIX\n';r._reranker.suffix='CHAT_SUFFIX\n';r._reranker.maximum=800;trace={}
    r.rank('query',[{'path':'a.txt','text':'document'}],trace)
    q,d=r._reranker.pairs[0]
    expected=len(r._reranker.prefix+f'<Instruct>: {TASK}\n<Query>: {q}\n<Document>: {d}'+r._reranker.suffix)
    assert trace['rerank_max_pair_tokens']==expected and trace['rerank_model_max_tokens']==800
    assert trace['rerank_pair_budget_verified'] is True


def test_adapter_without_tokenizer_cannot_claim_verified_zero_truncations(tmp_path):
    r=retrieval(tmp_path);r._reranker=SimpleNamespace(predict=lambda *a,**kw:[1.]);trace={}
    r.rank('query',[{'path':'a.txt','text':'document'}],trace)
    assert trace['silent_truncations'] is None and trace['rerank_pair_budget_verified'] is False
