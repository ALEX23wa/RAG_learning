"""Bounded query planning. History assists reference resolution, never supplies facts."""
from dataclasses import dataclass, field, asdict
import json
import re


def json_object(text):
    text=text.strip()
    if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
    value=json.loads(text)
    if not isinstance(value,dict):raise ValueError('Expected JSON object')
    return value


@dataclass
class QueryPlan:
    original: str
    queries: list[str]
    document_ids: list[str] | None = None
    requirements: list[str] = field(default_factory=list)
    planning: str = 'direct'
    call: dict | None = None
    standalone: str | None = None

    def asdict(self):return asdict(self)


def plan_query(question, history=None, *, document_ids=None, models=None, profile='production',stage='development'):
    # Mandatory requirements are verbatim user requests, never planner inventions.
    clauses=[part.strip() for part in re.findall(r'[^?？;；]+[?？;；]?',question) if part.strip()]
    # Formatting instructions still remain in the complete user question, but
    # are not separate factual requirements or retrieval lanes.
    format_only=re.compile(r'(?:请)?(?:分别|各自|逐项|两部分|两者|每部分)?(?:各)?'
        r'(?:(?:给出|给|提供|标注|附上|附)?(?:出处|来源|引用)|'
        r'(?:依据|根据)(?:两份|各自的|对应的|所引的)?(?:课件|材料|文档|资料)回答)[。！!]?')
    if len(clauses)>1:
        factual=[part for part in clauses if not format_only.fullmatch(part)]
        if factual:clauses=factual
    clauses=clauses if clauses and all(len(part)>=4 for part in clauses) else [question]
    plan=QueryPlan(question,[question],document_ids,requirements=clauses,standalone=question)
    followup=bool(re.search(r'它|上述|前者|后者|当时|这[个份篇种项些]|那[些个]|该[文模型方法函数]|其(?:中|他|结果)|\b(?:it|that|these|those|then|former|latter)\b',question,re.I))
    comparison=bool(re.search(r'比较|对比|差异|区别|联系|结合|分别|相比|compare|versus|difference|\bvs\b',question,re.I))
    # Bounded relevant user history is sent only to the planner, not to the retriever.
    previous=[str(m.get('content',''))[-1500:] for m in (history or []) if m.get('role')=='user'][-3:] if followup else []
    followup=followup and bool(previous)
    if len(clauses)>1 and not followup:
        # Explicit independent questions already supply source/object boundaries.
        # Rewriting these can silently move the second request to the first source.
        plan.queries=list(dict.fromkeys([question]+clauses))[:4]
        plan.planning='explicit_clauses'
        return plan
    if not (followup or comparison or len(clauses)>1):return plan
    if models is None:
        plan.planning='current_question_fallback';return plan
    prompt=('将当前问题转成独立检索问题，仅用历史消解指代，不能回答问题或添加事实。'
            '返回JSON：{"standalone":"完整独立问题","subqueries":[最多3个检索子问题]}。'
            '逐个保留对象与其来源、时间和所问维度的对应关系，不能将第二个来源改成第一个。'
            '只消解指代或拆分明确请求，不能添加公式、数值、实现细节等用户未要求的条件。'
            '保留当前问题的全部限制；不要继承已换话题的来源。')
    try:
        call=models.complete([{'role':'system','content':prompt},{'role':'user','content':json.dumps(
            {'question':question,'previous_user_questions':previous},ensure_ascii=False)}],profile,stage,max_tokens=700)
        plan.call={k:v for k,v in call.items() if k!='text'}
        obj=json_object(call['text'])
        standalone=obj.get('standalone','')
        subs=obj.get('subqueries',[])
        if not isinstance(standalone,str) or not 0<len(standalone.strip())<=2000 or not isinstance(subs,list):
            raise ValueError('Invalid query plan')
        # A complete question must not be replaced by an object-specific subquery.
        # For followups, the standalone is a proposed reference resolution, not evidence.
        if followup:plan.standalone=standalone.strip()
        generated=([plan.standalone] if followup else [])+(clauses if len(clauses)>1 else [])+subs
        generated=[q.strip() for q in generated if isinstance(q,str) and 0<len(q.strip())<=2000]
        # The original lane always preserves every clause; at most three extra lanes.
        plan.queries=list(dict.fromkeys([question]+generated))[:4]
        plan.planning='rewritten';plan.call={k:v for k,v in call.items() if k!='text'}
    except Exception as exc:
        plan.planning='current_question_fallback'
        plan.call={**(plan.call or getattr(exc,'call',{})), 'error':type(exc).__name__}
        if getattr(exc,'call_id',None):plan.call['call_id']=exc.call_id
    return plan
