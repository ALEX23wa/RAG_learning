"""Synthetic answer checkpoints only: no model, network or project evaluation data."""
import copy
import json
from types import SimpleNamespace

import pytest

from ragapp.store import Store
from scripts import run_quality_answers as runner


class Monitor:
    service_pid,database_pid=10,20
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def record(self,case_id,elapsed):
        return {'id':case_id,'elapsed':elapsed,'service_pid':self.service_pid,'database_pid':self.database_pid,
                'service_rss_bytes':1000,'database_rss_bytes':2000}


class Service:
    def __init__(self,root):
        self.settings=SimpleNamespace(state=root,campaign='synthetic')
        self.store=Store(root/'catalog.sqlite3');self.models=SimpleNamespace(store=Store(root/'billing.sqlite3'))
        self.calls=[];self.warmups=0;self.fail=False
    def manifest(self):return {'runtime_fingerprint':'synthetic-fixed-runtime'}
    def search_detailed(self,*args,**kwargs):self.warmups+=1;return {}
    def ask(self,question,**kwargs):
        self.calls.append(question);identity='call-'+str(len(self.calls))
        with self.models.store.db() as db:
            db.execute('INSERT INTO calls(id,status,cost,reserved) VALUES(?,?,?,?)',
                       (identity,'failed' if self.fail else 'complete',None if self.fail else .1,.25))
            db.execute('INSERT INTO call_campaigns VALUES(?,?)',(identity,'synthetic'))
        if self.fail:raise RuntimeError('Synthetic provider interruption')
        return {'answer':'No supporting source.','status':'abstained','evidence':[],'citations':[]}


@pytest.fixture
def fixture(tmp_path):
    dataset=tmp_path/'synthetic.json';dataset.write_text(json.dumps([
        {'id':'q1','question':'Generic one','split':'dev','answerable':False,'evidence':[]},
        {'id':'q2','question':'Generic two','split':'dev','answerable':False,'evidence':[]}]))
    return SimpleNamespace(dataset=dataset,report=tmp_path/'report.json',service=Service(tmp_path))


def run(fixture,**kwargs):
    return runner.run(fixture.dataset,fixture.report,service=fixture.service,monitor_factory=Monitor,**kwargs)


@pytest.mark.parametrize('fail_after_rows',[1,2])
def test_save_failure_recovery_does_not_repeat_completed_answers_or_double_charge(fixture,monkeypatch,fail_after_rows):
    real=runner.save_report;failed=False
    def save(path,report):
        nonlocal failed
        if len(report.get('results',[]))==fail_after_rows and not failed:
            failed=True;raise PermissionError('Synthetic checkpoint replacement denied')
        return real(path,report)
    monkeypatch.setattr(runner,'save_report',save)
    partial=run(fixture)
    assert partial['blocked']['id']=='q'+str(fail_after_rows)
    assert len(partial['results'])==fail_after_rows and not partial['failed_attempts']
    assert len(partial['checkpoint_errors'])==1
    assert partial['metrics']['total_cost_cny_upper_estimate']==pytest.approx(.1*fail_after_rows)
    assert len(fixture.service.calls)==fail_after_rows
    warmups=fixture.service.warmups
    resumed=run(fixture)
    assert len(fixture.service.calls)==2 and 'blocked' not in resumed
    assert resumed['metrics']['total_cost_cny_upper_estimate']==pytest.approx(.2)
    assert len(resumed['recovered_errors'])==1 and len(resumed['checkpoint_errors'])==1
    assert len(resumed['performance_measurement']['records'])==2
    assert fixture.service.warmups==warmups+(fail_after_rows==1)
    assert resumed['campaign_cost']['total_cny']==pytest.approx(.2)


def test_legacy_stale_blocker_preserves_failed_history_and_deduplicates_calls(fixture):
    result=run(fixture)
    row=result['results'][-1]
    old_failure={'id':row['id'],'error':'Old post-append save failure','calls':copy.deepcopy(row['calls']),
                 'total_cost_cny_upper_estimate':row['total_cost_cny_upper_estimate']}
    # Emulate the old runner, which recorded the same call as result and failure.
    result['failed_attempts'].append(old_failure);result['blocked']=old_failure
    fixture.report.write_text(json.dumps(result))
    warmups=fixture.service.warmups
    resumed=run(fixture)
    assert resumed['failed_attempts']==[old_failure] and 'blocked' not in resumed
    assert resumed['metrics']['total_cost_cny_upper_estimate']==pytest.approx(.2)
    assert len(fixture.service.calls)==2 and fixture.service.warmups==warmups


def test_genuine_provider_failure_remains_chargeable_and_retries_only_pending_case(fixture):
    first=run(fixture,max_queries=1)
    fixture.service.fail=True
    failure=run(fixture)
    assert failure['blocked']['id']=='q2' and len(failure['results'])==1
    assert failure['metrics']['total_cost_cny_upper_estimate']==pytest.approx(.35)
    assert len(failure['failed_attempts'])==1 and not failure.get('recovered_errors')
    fixture.service.fail=False
    resumed=run(fixture)
    assert fixture.service.calls==['Generic one','Generic two','Generic two']
    assert len(resumed['failed_attempts'])==1 and len(resumed['results'])==2
    assert resumed['metrics']['total_cost_cny_upper_estimate']==pytest.approx(.45)
    assert resumed['campaign_cost']['total_cny']==pytest.approx(.45)


def test_missing_resource_or_incomplete_answer_never_clears_stale_blocker(fixture):
    result=run(fixture)
    result['blocked']={'id':'q2','error':'Persisted error'}
    result['results'][-1].pop('answer')
    fixture.report.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='complete persisted'):run(fixture)
    result['performance_measurement']['records'].pop()
    fixture.report.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='resource records'):run(fixture)
    assert len(fixture.service.calls)==2


def test_cost_deduplication_retains_conservative_maximum_and_rejects_unknown_ids():
    report={'results':[{'calls':[{'id':'same','cost':.1}]}],
            'failed_attempts':[{'calls':[{'id':'same','cost':None,'reserved':.25},{'id':'other','cost':.3}]}],
            'checkpoint_errors':[{'calls':[{'id':'same','cost':.1}]}]}
    assert runner.report_call_cost(report)==pytest.approx(.55)
    with pytest.raises(ValueError,match='stable call ID'):
        runner.report_call_cost({'results':[{'calls':[{'cost':.1}]}]})
    with pytest.raises(ValueError,match='no ledger calls'):
        runner.report_call_cost({'results':[{'calls':[],'total_cost_cny_upper_estimate':.1}]})
