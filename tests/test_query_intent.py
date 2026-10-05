import json

import pytest

from ragapp.query import plan_query


class Planner:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def complete(self, messages, *args, **kwargs):
        self.messages.append(messages)
        return {'text': json.dumps(self.response)}


def test_model_cannot_add_mandatory_conditions_or_replace_complete_question():
    question = '比较松树与杉树的生长条件'
    model = Planner({'standalone': '松树的数值模型是什么？',
                     'subqueries': ['松树生长条件', '杉树生长条件'],
                     'requirements': ['必须给出微分方程和数值边界']})
    plan = plan_query(question, models=model)
    assert plan.requirements == [question]
    assert plan.standalone == question
    assert plan.queries[0] == question


def test_two_explicit_requests_keep_their_own_sources_before_model_queries():
    question = '甲园记录中的温度是多少？乙园记录中的湿度是多少？'
    model = Planner({'standalone': question, 'subqueries': ['甲园温度与湿度']})
    plan = plan_query(question, models=model)
    assert not model.messages  # Already independent clauses need no model rewrite.
    assert plan.queries[:3] == [question, '甲园记录中的温度是多少？', '乙园记录中的湿度是多少？']
    assert plan.requirements == plan.queries[1:3]
    assert len(plan.queries) <= 4
    assert plan.planning == 'explicit_clauses'


@pytest.mark.parametrize('question', ['这些仪器如何校准？', '当时的采样频率是多少？',
                                      'How were these instruments calibrated?'])
def test_demonstrative_followup_resolves_history_without_using_it_as_evidence(question):
    standalone = '甲园传感器的校准或采样方式是什么？'
    model = Planner({'standalone': standalone, 'subqueries': []})
    history = [{'role': 'user', 'content': '甲园部署了哪些传感器？'},
               {'role': 'assistant', 'content': 'UNVERIFIED ANSWER'}]
    plan = plan_query(question, history, models=model)
    assert plan.standalone == standalone
    assert plan.queries == [question, standalone]
    sent = json.loads(model.messages[0][1]['content'])
    assert sent['previous_user_questions'] == [history[0]['content']]
    assert 'UNVERIFIED ANSWER' not in json.dumps(sent)


def test_unrelated_history_does_not_rewrite_complete_question():
    model = Planner({'standalone': 'old topic', 'subqueries': []})
    question = '氧气的化学符号是什么？'
    plan = plan_query(question, [{'role': 'user', 'content': 'OLD TOPIC'}], models=model)
    assert plan.standalone == question and plan.queries == [question]
    assert not model.messages


def test_empty_followup_rewrite_falls_back_to_current_question():
    model = Planner({'standalone': '', 'subqueries': ['irrelevant']})
    question = '这些仪器如何校准？'
    plan = plan_query(question, [{'role': 'user', 'content': 'prior'}], models=model)
    assert plan.standalone == question and plan.queries == [question]
    assert plan.planning == 'current_question_fallback'


@pytest.mark.parametrize('instruction',['请分别引用。','两部分各给出处。','分别依据两份课件回答。'])
def test_citation_format_is_not_a_factual_requirement(instruction):
    first='甲园的湿度是多少？';second='乙园的温度是多少？'
    plan=plan_query(first+second+instruction)
    assert plan.requirements==[first,second]
    assert plan.queries==[plan.original,first,second]
    assert instruction in plan.original


def test_trailing_substantive_request_is_not_dropped():
    plan=plan_query('甲园的湿度是多少？同时总结季节变化。')
    assert plan.requirements==['甲园的湿度是多少？','同时总结季节变化。']
