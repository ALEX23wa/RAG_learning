"""Synthetic precision/cache checks, with no local model weights or evaluation data."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ragapp.config import Settings
from ragapp.pipeline import embedding_fingerprint, fingerprint
from ragapp.retrieval import (Retrieval, checked_document_vector, document_cache_contract,
    read_document_vector, validated_embedding_precision, write_document_vector)


def unit(dimension=3):
    value=np.zeros(dimension,dtype=np.float32);value[0]=1;return value


def test_legacy_fingerprint_is_exact_historical_payload_and_fp32_isolated():
    settings=Settings()
    old={'model':settings.embedding_model,'revision':settings.embedding_revision,
         'dimension':settings.embedding_dimension,'max_tokens':settings.embedding_max_tokens,
         'input_tokens':settings.embedding_input_tokens,'normalized':True,'adapter':'official-v1'}
    assert embedding_fingerprint(settings)==fingerprint(old)
    settings.embedding_precision='float32'
    assert embedding_fingerprint(settings)!=fingerprint(old)


@pytest.mark.parametrize('dtypes',[[],['torch.bfloat16'],['torch.float32','torch.float16']])
def test_embedding_actual_parameter_precision_rejects_cast_output_or_mixed_parameters(dtypes):
    model=SimpleNamespace(parameters=lambda:[SimpleNamespace(dtype=d) for d in dtypes])
    with pytest.raises(ValueError,match='FP32'):validated_embedding_precision(model)


@pytest.mark.parametrize('precision',['legacy','float32'])
def test_cpu_constructor_precision_is_opt_in_and_recorded(tmp_path,monkeypatch,precision):
    from ragapp import retrieval
    captured={}
    model=SimpleNamespace(parameters=lambda:[SimpleNamespace(dtype='torch.float32')])
    def constructor(*args,**kwargs):captured.update(kwargs);return model
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=constructor))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(float32='torch.float32',set_num_threads=lambda n:None))
    monkeypatch.setattr(retrieval,'embedding_model_files',lambda settings:{'weights':{'sha256':'a'*64,'bytes':1}})
    r=Retrieval(Settings(state=tmp_path,embedding_precision=precision),None)
    assert r.embedding is model and r.embedding is model
    assert captured['device']=='cpu' and captured['local_files_only'] is True
    if precision=='legacy':assert 'model_kwargs' not in captured and r.embedding_precision_record is None
    else:
        assert captured['model_kwargs']=={'torch_dtype':'torch.float32'}
        assert r.embedding_precision_record['actual_parameter_dtypes']==['torch.float32']


def test_failed_precision_is_not_cached(tmp_path,monkeypatch):
    model=SimpleNamespace(parameters=lambda:[SimpleNamespace(dtype='torch.bfloat16')])
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=lambda *a,**k:model))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(float32='torch.float32',set_num_threads=lambda n:None))
    r=Retrieval(Settings(state=tmp_path,embedding_precision='float32'),None)
    with pytest.raises(ValueError,match='FP32'):r.embedding
    assert r._embedding is None


def cache(tmp_path):
    settings=Settings(state=tmp_path,embedding_precision='float32',embedding_dimension=3)
    contract=document_cache_contract(settings,'f'*64)
    origin={'kind':'encoded','actual_parameter_dtypes':['torch.float32']}
    return contract,origin


def test_verified_cache_roundtrip_idempotence_and_conflict(tmp_path):
    contract,origin=cache(tmp_path)
    assert write_document_vector(tmp_path,'body','passage: body',unit(),contract,origin)
    assert not write_document_vector(tmp_path,'body','passage: body',unit(),contract,origin)
    assert np.array_equal(read_document_vector(tmp_path,'body','passage: body',contract),unit())
    changed=np.roll(unit(),1)
    with pytest.raises(ValueError,match='Conflicting'):
        write_document_vector(tmp_path,'body','passage: body',changed,contract,origin)
    for bad_input,bad_contract in [('body',contract),('passage: body',{**contract,'model_files_sha256':'a'*64})]:
        with pytest.raises(ValueError,match='identity'):
            read_document_vector(tmp_path,'body',bad_input,bad_contract)


def test_binary_checksum_and_missing_marker_fail_closed(tmp_path):
    contract,origin=cache(tmp_path);write_document_vector(tmp_path,'body','body',unit(),contract,origin)
    binary=next(tmp_path.glob('*.npy'));binary.write_bytes(binary.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):read_document_vector(tmp_path,'body','body',contract)
    binary.with_suffix('.json').unlink()
    with pytest.raises(ValueError,match='Incomplete'):read_document_vector(tmp_path,'body','body',contract)


def test_exact_interrupted_binary_can_finish_but_conflicts_cannot(tmp_path):
    contract,origin=cache(tmp_path);key=hashlib.sha256(b'body').hexdigest()
    np.save(tmp_path/(key+'.npy'),unit())
    assert write_document_vector(tmp_path,'body','body',unit(),contract,origin)
    assert read_document_vector(tmp_path,'body','body',contract) is not None


@pytest.mark.parametrize('value',[np.zeros(3,dtype=np.float32),np.array([float('nan'),0,1],dtype=np.float32),
    np.ones(4,dtype=np.float32),np.array([1,0,0],dtype=np.float64)])
def test_verified_vectors_reject_bad_shape_finite_norm_and_dtype(value):
    with pytest.raises(ValueError):checked_document_vector(value,3)


def test_fp32_budget_checked_before_encode_and_query_never_cached(tmp_path):
    settings=Settings(state=tmp_path,embedding_precision='float32',embedding_dimension=3,
                      embedding_model='BAAI/bge-m3',embedding_input_tokens=12,embedding_max_tokens=16)
    calls=[]
    def encode(texts,**kwargs):calls.append(list(texts));return np.stack([unit() for _ in texts])
    model=SimpleNamespace(tokenizer=lambda text,**kw:{'input_ids':[0]*(len(text)+2)},encode=encode)
    r=Retrieval(settings,None);r._embedding=model
    r._document_cache_contract=document_cache_contract(settings,'f'*64)
    r.embedding_precision_record={'actual_parameter_dtypes':['torch.float32']}
    r.vectors(['short']);r.vectors(['short'])
    assert len(calls)==1
    r.vectors(['short'],True);r.vectors(['short'],True)
    assert len(calls)==3
    with pytest.raises(ValueError,match='silent truncation'):r.vectors(['x'*12])
    assert len(calls)==3


@pytest.mark.parametrize('warm_cache',[False,True])
def test_fp32_repeated_documents_encode_once_and_restore_order(tmp_path,warm_cache):
    settings=Settings(state=tmp_path,embedding_precision='float32',embedding_dimension=3,
                      embedding_model='BAAI/bge-m3',embedding_batch=2)
    calls=[]
    base={'repeat':unit(),'cached':np.roll(unit(),1),'tail':np.roll(unit(),2)}
    def encode(texts,**kwargs):
        calls.append(list(texts));seen={};result=[]
        for text in texts:
            vector=base[text].copy()
            # Repeated inference can differ with padding/batch composition.
            count=seen.get(text,0);seen[text]=count+1
            vector[(int(vector.argmax())+1)%3]+=count*1e-7
            result.append(vector)
        return np.stack(result)
    model=SimpleNamespace(tokenizer=lambda text,**kw:{'input_ids':[0]*3},encode=encode)
    r=Retrieval(settings,None);r._embedding=model
    r._document_cache_contract=document_cache_contract(settings,'f'*64)
    r.embedding_precision_record={'actual_parameter_dtypes':['torch.float32']}
    if warm_cache:r.vectors(['cached']);calls.clear()
    texts=['repeat','cached','repeat','tail','cached','repeat']
    result=r.vectors(texts)
    assert calls==[['repeat','tail'] if warm_cache else ['repeat','cached','tail']]
    assert np.array_equal(result,np.stack([base[text] for text in texts]))
    assert np.array_equal(r.vectors(list(reversed(texts))),result[::-1])
    assert len(calls)==1
