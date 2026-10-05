"""Production tuning is explicit and accepted runtime settings survive restart."""
import json
from pathlib import Path

import pytest

from ragapp.config import Settings, INDEX_SETTINGS, RUNTIME_SETTINGS, validated_runtime_settings


def local_settings(tmp_path, **values):
    return Settings(root=tmp_path,corpus=tmp_path/'papers',state=tmp_path/'state',**values)


def pointer(config, **values):
    config.state.mkdir(parents=True,exist_ok=True)
    (config.state/'active.json').write_text(json.dumps({
        'collection':'accepted-collection','catalog':'accepted.sqlite3',**values}),encoding='utf-8')


def test_defaults_retain_baseline_tuning_and_p0_budgets():
    settings=Settings()
    assert (settings.lexical_candidates,settings.dense_candidates,settings.rerank_candidates)==(40,40,20)
    assert (settings.reranker_max_tokens,settings.evidence_limit,settings.rrf_k)==(512,10,10)
    assert not settings.expand_context and not settings.query_planning and not settings.verify_answers
    assert settings.embedding_input_tokens==480 and settings.embedding_max_tokens==512
    assert settings.query_tokens==256
    assert settings.judge_reasoning_effort=='disabled'


def test_quality_preset_explicitly_enables_candidate_tuning():
    values=json.loads((Path(__file__).resolve().parents[1]/'config/quality-v2.json').read_text(encoding='utf-8'))
    assert values['load_active'] is False
    assert [values[k] for k in ('lexical_candidates','dense_candidates','rerank_candidates')]==[80,80,60]
    assert values['reranker_max_tokens']==1024 and values['evidence_limit']==12
    assert all(values[k] is True for k in ('expand_context','query_planning','verify_answers'))


def test_old_pointer_preserves_baseline_and_explicit_local_runtime(tmp_path):
    settings=local_settings(tmp_path,query_planning=True,rrf_k=30)
    pointer(settings,settings={'chunk_tokens':384,'campaign_limit':999})
    settings.prepare()
    assert (settings.collection,settings.catalog)==('accepted-collection','accepted.sqlite3')
    assert settings.query_planning and settings.rrf_k==30
    assert settings.reranker_max_tokens==512 and settings.rerank_candidates==20
    assert settings.campaign_limit==100.


def test_complete_published_runtime_overrides_constructor_defaults_only_in_allowlist(tmp_path):
    settings=local_settings(tmp_path,campaign='local-only',campaign_limit=9.)
    accepted={key:getattr(Settings(),key) for key in RUNTIME_SETTINGS}
    accepted.update(rrf_k=30,reranker_directory='candidate-reranker',reranker_model='test/reranker',
                    reranker_revision='a'*40,verify_answers=True,reranker_max_tokens=1024)
    pointer(settings,runtime_settings=accepted)
    settings.prepare()
    assert {key:getattr(settings,key) for key in RUNTIME_SETTINGS}==accepted
    assert settings.campaign=='local-only' and settings.campaign_limit==9.
    assert settings.state==tmp_path/'state' and settings.corpus==tmp_path/'papers'


def test_explicit_experiment_bypasses_active_runtime_and_index(tmp_path):
    settings=local_settings(tmp_path,load_active=False,rrf_k=60)
    pointer(settings,runtime_settings={'rrf_k':30},settings={'chunk_tokens':999})
    settings.prepare()
    assert settings.rrf_k==60 and settings.chunk_tokens==384
    assert settings.collection=='papers_e5_v1'


def test_runtime_allowlist_excludes_cost_authorization_endpoints_paths_and_index():
    assert not RUNTIME_SETTINGS & INDEX_SETTINGS
    assert not RUNTIME_SETTINGS & {'root','state','corpus','campaign','campaign_limit','campaign_stages',
                                   'qdrant_url','collection','catalog','load_active','stable_seconds'}


@pytest.mark.parametrize('values',[
    None,[],{'campaign':'unauthorized'},{'state':'other'},{'qdrant_url':'other'},
    {'rrf_k':True},{'rrf_k':None},{'rrf_k':-1},{'rrf_k':float('nan')},
    {'context_tokens':0},{'rerank_candidates':1.5},{'verify_answers':1},
    {'retrieval_mode':'unsafe'},{'reranker_model':''},{'reranker_revision':'main'},
    {'reranker_directory':'../external'},{'reranker_directory':'C:\\models'},
    {'reranker_directory':'/external'},{'reranker_directory':'..'},
])
def test_malformed_runtime_pointer_is_rejected(tmp_path,values):
    settings=local_settings(tmp_path)
    pointer(settings,runtime_settings=values)
    with pytest.raises(ValueError):settings.prepare()


def test_new_publication_requires_complete_allowlist_but_old_partial_pointers_load(tmp_path):
    with pytest.raises(ValueError,match='complete'):validated_runtime_settings({'rrf_k':30},complete=True)
    assert validated_runtime_settings({'rrf_k':0})=={'rrf_k':0}
    settings=local_settings(tmp_path);pointer(settings,runtime_settings={'rrf_k':30});settings.prepare()
    assert settings.rrf_k==30 and settings.reranker_max_tokens==512


def test_legacy_pointer_explicitly_resets_candidate_precision(tmp_path):
    settings=local_settings(tmp_path,embedding_precision='float32')
    pointer(settings,settings={'embedding_model':settings.embedding_model})
    settings.prepare()
    assert settings.embedding_precision=='legacy'
    pointer(settings,settings={'embedding_precision':'float32'})
    settings.prepare()
    assert settings.embedding_precision=='float32'


@pytest.mark.parametrize('value',[None,[],True,'fp16','auto'])
def test_unknown_embedding_precision_rejected(tmp_path,value):
    with pytest.raises(ValueError,match='embedding_precision'):
        local_settings(tmp_path,embedding_precision=value).prepare()


@pytest.mark.parametrize('values',[
    {'judge_reasoning_effort':'auto'},{'judge_reasoning_effort':None},
    {'judge_reasoning_scope':'unknown'},{'use_evidence_outline':'yes'},
    {'judge_max_tokens':0},{'judge_max_tokens':16385},{'judge_max_tokens':True}])
def test_judge_experiment_settings_fail_closed(tmp_path,values):
    with pytest.raises(ValueError):local_settings(tmp_path,**values).prepare()
