import json
from types import SimpleNamespace

import pytest

from ragapp.config import Settings
from ragapp.indexer import Indexer
from ragapp.pipeline import index_fingerprint, parsing_fingerprint, embedding_fingerprint
from ragapp.store import Store


class Tokenizer:
    def __call__(self,text,**kwargs):
        return {'offset_mapping':[(i,i+1) for i in range(len(text))]}


class Vectors:
    embedding=SimpleNamespace(tokenizer=Tokenizer())
    def __init__(self):self.chunks=[]
    def upsert(self,chunks):self.chunks.extend(chunks)
    def remove(self,ids):pass


def fixture(tmp_path):
    corpus=tmp_path/'papers';corpus.mkdir()
    settings=Settings(root=tmp_path,corpus=corpus,state=tmp_path/'state',stable_seconds=0).prepare()
    store=Store(settings.state/settings.catalog);vectors=Vectors()
    return settings,store,vectors,Indexer(settings,store,vectors)


def test_fingerprint_initialization_and_model_change_requires_rebuild(tmp_path):
    settings,store,vectors,index=fixture(tmp_path)
    (settings.corpus/'a.txt').write_text('source evidence')
    assert index.sync()['completed']==1
    with store.db() as db:metadata=dict(db.execute('SELECT key,value FROM metadata'))
    assert metadata['index_fingerprint']==index_fingerprint(settings)
    assert metadata['parsing_fingerprint']==parsing_fingerprint(settings)
    assert metadata['embedding_fingerprint']==embedding_fingerprint(settings)
    ids=[c['id'] for c in vectors.chunks]
    settings.embedding_revision='different-model-revision'
    with pytest.raises(RuntimeError,match='fingerprint changed'):
        index.sync()
    assert ids==[c['id'] for c in vectors.chunks]
    assert store.active(ids)
    # A rejected attempt must release both locks.
    settings.embedding_revision='d128750597153bb5987e10b1c3493a34e5a4502a'
    assert index.sync()['unchanged']==1


def test_populated_legacy_catalog_is_read_only_until_rebuild(tmp_path):
    settings,store,vectors,index=fixture(tmp_path)
    with store.db() as db:
        db.execute("INSERT INTO documents VALUES('d','old.txt','h','v','ready',NULL,0,0)")
        db.execute("INSERT INTO chunks VALUES('c','d','v','old','{}','old',NULL)")
    with pytest.raises(RuntimeError,match='Legacy index'):
        index.sync()
    assert store.active(['c']) and not vectors.chunks
    with store.db() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==0


def test_new_parsing_cache_does_not_reuse_legacy_or_tampered_blocks(tmp_path):
    from ragapp.indexer import digest
    settings,store,vectors,index=fixture(tmp_path)
    source=settings.corpus/'a.txt';source.write_text('the actual source')
    legacy=settings.state/'cache'/digest(source);legacy.mkdir()
    (legacy/'parsed-v1.json').write_text(json.dumps([{'text':'wrong cached source','locator':{}}]))
    assert index.sync()['completed']==1
    assert vectors.chunks[0]['text']=='the actual source'
    parsed=next(legacy.rglob('parsed.json'))
    saved=json.loads(parsed.read_text());saved['blocks'][0]['text']='tampered'
    parsed.write_text(json.dumps(saved))
    settings.collection='new-generation'
    new_store=Store(settings.state/'new.sqlite3');new_vectors=Vectors()
    assert Indexer(settings,new_store,new_vectors).sync()['completed']==1
    assert new_vectors.chunks[0]['text']=='the actual source'


def test_metadata_decoration_cannot_truncate_source(tmp_path):
    settings,store,vectors,index=fixture(tmp_path)
    settings.embedding_input_tokens=30
    item={'text':'correct source body','locator':{},'retrieval_text':'very long title '*20+'\ncorrect source body'}
    actual=index._retrieval_text(item,Tokenizer())
    assert len(actual)<=30 and actual.endswith(item['text'])


class SpecialTokenizer(Tokenizer):
    def __call__(self,text,add_special_tokens=False,**kwargs):
        encoded=super().__call__(text,**kwargs)
        encoded['input_ids']=list(range(len(text)+(2 if add_special_tokens else 0)))
        return encoded


@pytest.mark.parametrize('model,query',[
    ('intfloat/multilingual-e5-base',False),('intfloat/multilingual-e5-base',True),
    ('BAAI/bge-m3',False),('Qwen/Qwen3-Embedding-0.6B',False),('Qwen/Qwen3-Embedding-0.6B',True)])
def test_full_template_and_special_tokens_counted_for_all_adapters(tmp_path,model,query):
    from ragapp.model_adapters import embedding_inputs
    settings,_,_,index=fixture(tmp_path)
    settings.embedding_model=model
    settings.embedding_input_tokens=170
    body='original m³/s ＣＤ body'
    item={'text':body,'locator':{},'retrieval_text':'overlong section '*40+'\n'+body}
    tokenizer=SpecialTokenizer()
    fitted=index._retrieval_text(item,tokenizer,query=query)
    official=embedding_inputs(settings,[fitted],query)[0]
    assert len(tokenizer(official,add_special_tokens=True)['input_ids'])<=170
    assert fitted.endswith(body) and item['text']==body


def test_body_requires_smaller_source_windows_not_silent_truncation(tmp_path):
    settings,_,_,index=fixture(tmp_path)
    settings.chunk_tokens=30;settings.overlap=3;settings.embedding_input_tokens=24
    tokenizer=SpecialTokenizer();source='  alpha beta\n  gamma delta\n  theta omega'
    blocks=[{'text':source,'locator':{'line_start':1,'document_char_start':0}}]
    chunks=list(index._chunks(blocks,tokenizer))
    assert len(chunks)>1
    covered=set()
    for item in chunks:
        fitted=index._retrieval_text(item,tokenizer)
        assert index._embedding_tokens(fitted,tokenizer)<=24
        for span in item['locator']['source_spans']:
            a,b=span['document_char_start'],span['document_char_end']
            assert source[a:b]==item['text'][span['chunk_start']:span['chunk_end']]
            covered.update(range(a,b))
    assert all(i in covered for i,c in enumerate(source) if not c.isspace())
    with pytest.raises(ValueError,match='rechunk first'):
        index._retrieval_text({'text':source,'locator':{}},tokenizer)


def test_too_small_budget_fails_instead_of_emitting_oversize_input(tmp_path):
    settings,_,_,index=fixture(tmp_path)
    settings.embedding_input_tokens=4
    with pytest.raises(ValueError,match='One source token'):
        list(index._chunks([{'text':'evidence','locator':{}}],SpecialTokenizer()))


def test_default_body_windows_are_unchanged_when_full_input_fits(tmp_path):
    from ragapp.structure import structural_chunks
    settings,_,_,index=fixture(tmp_path)
    blocks=[{'text':'Original statement. '*50,'locator':{'page':1}}]
    tokenizer=SpecialTokenizer()
    old=list(structural_chunks(blocks,tokenizer,settings.chunk_tokens,settings.overlap))
    assert list(index._chunks(blocks,tokenizer))==old


def test_real_e5_tokenizer_preserves_body_and_fits_complete_input(tmp_path):
    from pathlib import Path
    from transformers import AutoTokenizer
    path=Path(__file__).resolve().parents[1]/'.rag/models/embedding'
    if not (path/'tokenizer.json').exists():pytest.skip('Local tokenizer unavailable')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    settings,_,_,index=fixture(tmp_path)
    body='原始证据必须保留：mW ≠ MW，x² 与 x2，温度 21.4 °C。'
    item={'text':body,'locator':{},'retrieval_text':'附表1-研究中心专职人员统计表 '*150+'\n'+body}
    fitted=index._retrieval_text(item,tokenizer)
    assert fitted.endswith(body)
    assert index._embedding_tokens(fitted,tokenizer)<=480
