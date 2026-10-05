import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragapp.parsers import chunk, parse, pdf


class CharacterTokenizer:
    def __call__(self,text,**kwargs):
        return {'offset_mapping':[(i,i+1) for i in range(len(text))]}


def test_short_paragraphs_merge_only_within_section():
    blocks=[{'text':'Definition.','locator':{'paragraph':1,'section_path':['Alpha']}},
            {'text':'Uses the definition.','locator':{'paragraph':2,'section_path':['Alpha']}},
            {'text':'Different subject.','locator':{'paragraph':3,'section_path':['Beta']}}]
    chunks=list(chunk(blocks,CharacterTokenizer(),64,8))
    assert len(chunks)==2
    assert chunks[0]['text']=='Definition.\n\nUses the definition.'
    assert [s['paragraph'] for s in chunks[0]['locator']['source_spans']]==[1,2]
    assert chunks[0]['locator']['parent_id']!=chunks[1]['locator']['parent_id']
    assert chunks[0]['retrieval_text'].startswith('Alpha\n')
    assert not chunks[0]['text'].startswith('Alpha')


def test_pdf_heading_and_body_share_bounded_parent_with_exact_spans():
    title = '3. Incident Light'
    body = 'CCD sensor is sensitive to infrared light. It degrades resolution and PRNU.'
    blocks = [
        {'text': title, 'locator': {'page': 13, 'block': 1, 'type': 'paragraph_title', 'section_path': [title]}},
        {'text': body, 'locator': {'page': 13, 'block': 2, 'type': 'text', 'section_path': [title]}},
    ]
    chunks = list(chunk(blocks, CharacterTokenizer(), 64, 8))
    assert len(chunks) >= 2
    assert len({c['locator']['parent_id'] for c in chunks}) == 1
    assert all(c['parent'] == title + '\n\n' + body for c in chunks)
    for c in chunks:
        assert len(c['text']) <= 64 and len(c['parent']) <= 128
        parent_spans = c['locator']['parent_source_spans']
        assert [(s['page'], s['block'], s['type']) for s in parent_spans] == [
            (13, 1, 'paragraph_title'), (13, 2, 'text')]
        for span, original in zip(parent_spans, (title, body)):
            assert c['parent'][span['chunk_start']:span['chunk_end']] == original
        for span in c['locator']['source_spans']:
            original = blocks[span['block'] - 1]['text']
            assert c['text'][span['chunk_start']:span['chunk_end']] == original[span['char_start']:span['char_end']]


@pytest.mark.parametrize('boundary', ['equation', 'table', 'image', 'chart'])
def test_pdf_heading_merge_does_not_cross_non_prose_boundaries(boundary):
    blocks = [
        {'text': 'Heading', 'locator': {'page': 1, 'type': 'heading', 'section_path': ['A']}},
        {'text': 'First body.', 'locator': {'page': 1, 'type': 'text', 'section_path': ['A']}},
        {'text': 'separate object', 'locator': {'page': 1, 'type': boundary, 'section_path': ['A']}},
        {'text': 'Second body.', 'locator': {'page': 1, 'type': 'text', 'section_path': ['A']}},
        {'text': 'Other section.', 'locator': {'page': 1, 'type': 'text', 'section_path': ['B']}},
        {'text': 'Next page.', 'locator': {'page': 2, 'type': 'text', 'section_path': ['B']}},
    ]
    chunks = list(chunk(blocks, CharacterTokenizer(), 100, 8))
    assert [c['text'] for c in chunks] == [
        'Heading\n\nFirst body.', 'separate object', 'Second body.', 'Other section.', 'Next page.']
    assert len({c['locator']['parent_id'] for c in chunks}) == 5


def test_pdf_adjacent_equations_remain_separate_parents():
    blocks = [{'text': text, 'locator': {'page': 1, 'type': 'equation', 'section_path': ['A']}}
              for text in ('x=1', 'y=2')]
    chunks = list(chunk(blocks, CharacterTokenizer(), 100, 8))
    assert len(chunks) == 2
    assert chunks[0]['locator']['parent_id'] != chunks[1]['locator']['parent_id']


def test_chunker_v4_changes_generation_without_changing_parser_cache(tmp_path):
    from ragapp.config import Settings
    from ragapp.pipeline import parsing_fingerprint, embedding_fingerprint, index_fingerprint, fingerprint
    settings = Settings(root=tmp_path, state=tmp_path/'state')
    parser = fingerprint({'schema': 'structure-v2', 'mode': settings.pdf_mode, 'overrides': settings.pdf_overrides})
    assert parsing_fingerprint(settings) == parser
    old = fingerprint({'schema': settings.index_schema, 'parser': parser,
        'chunker': 'structure-v3-ast-statements', 'chunk_tokens': settings.chunk_tokens,
        'overlap': settings.overlap, 'embedding': embedding_fingerprint(settings)})
    assert index_fingerprint(settings) != old


def test_ast_chunks_preserve_characters_and_exact_lines(tmp_path):
    source='import os\n\n@decorator\ndef function():\n    x = "ＣＤ m³/s β"\n    return x\n'
    path=tmp_path/'example.py';path.write_bytes(source.encode('utf-8'))
    chunks=list(chunk(parse(path,tmp_path/'cache'),CharacterTokenizer(),24,4))
    assert any(c['locator'].get('symbol')=='function' for c in chunks)
    coverage=set()
    for c in chunks:
        assert len(c['text'])<=24
        assert len(c['parent'])<=48
        for span in c['locator']['source_spans']:
            a,b=span['document_char_start'],span['document_char_end']
            assert source[a:b]==c['text'][span['chunk_start']:span['chunk_end']]
            assert span['line_start']==source[:a].count('\n')+1
            assert span['line_end']==source[:b-1].count('\n')+1
            coverage.update(range(a,b))
    assert all(i in coverage for i,c in enumerate(source) if not c.isspace())


def test_crlf_and_indentation_are_preserved(tmp_path):
    source='def f():\r\n\tvalue = 1\r\n\treturn value\r\n'
    path=tmp_path/'crlf.py';path.write_bytes(source.encode())
    chunks=list(chunk(parse(path,tmp_path/'cache'),CharacterTokenizer(),100,8))
    assert chunks[0]['text']==source
    span=chunks[0]['locator']['source_spans'][0]
    assert span['line_start']==1 and span['line_end']==3
    assert span['document_char_end']==len(source)


def test_csv_row_groups_include_schema_and_precise_sources(tmp_path):
    path=tmp_path/'values.csv';path.write_text('day,value\na,10\nb,20\nc,30\n')
    blocks=parse(path,tmp_path/'cache')
    chunks=list(chunk(blocks,CharacterTokenizer(),80,8))
    assert chunks[0]['locator']['kind']=='table_schema'
    assert chunks[0]['text']=='day | value'
    rows=[c for c in chunks if c['locator']['kind']=='table_row']
    assert len(rows)==1
    assert [s['row'] for s in rows[0]['locator']['source_spans']]==[2,3,4]
    assert rows[0]['locator']['row_start']==2 and rows[0]['locator']['row_end']==4
    assert 'day | value' in rows[0]['retrieval_text']


def test_csv_extra_fields_are_not_silently_discarded(tmp_path):
    path=tmp_path/'bad.csv';path.write_text('a,b\n1,2,3\n')
    with pytest.raises(ValueError,match='3 fields'):
        parse(path,tmp_path/'cache')


def test_table_groups_keep_normal_rows_whole(tmp_path):
    path=tmp_path/'table.csv';path.write_text('label,value\nalpha,111\nbeta,222\ngamma,333\ndelta,444\n')
    parsed=parse(path,tmp_path/'cache')
    chunks=list(chunk(parsed,CharacterTokenizer(),50,8))
    for c in chunks:
        if c['locator']['kind']=='table_schema':continue
        for span in c['locator']['source_spans']:
            expected=parsed[span['row']-2]['text']
            assert c['text'][span['chunk_start']:span['chunk_end']]==expected


def test_sentence_boundary_and_validation():
    text='First sentence. Second sentence. Third sentence.'
    chunks=list(chunk([{'text':text,'locator':{}}],CharacterTokenizer(),30,4))
    assert chunks[0]['text']=='First sentence. '
    assert all(len(c['text'])<=30 for c in chunks)
    for maximum,overlap in [(0,0),(10,10),(10,-1)]:
        with pytest.raises(ValueError):list(chunk([],CharacterTokenizer(),maximum,overlap))


def test_real_tokenizer_never_decodes_source_text():
    model=Path(__file__).resolve().parents[1]/'.rag/models/embedding'
    if not (model/'tokenizer.json').exists():pytest.skip('Local tokenizer unavailable')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    text=('    value = "ＣＤ m³/s β"\n\treturn value  # 原始代码\n')*30
    chunks=list(chunk([{'text':text,'locator':{'line_start':1}}],tokenizer,48,8))
    for c in chunks:
        assert len(tokenizer(c['text'],add_special_tokens=False)['input_ids'])<=48
        assert c['text'] in text


def _pdf_fakes(monkeypatch, texts, calls):
    import pypdf
    from data_process import data_md
    class Page:
        mediabox=SimpleNamespace(width=600)
        def __init__(self,text):self.text=text
        def extract_text(self,**kwargs):return self.text
    class Reader:
        is_encrypted=False
        def __init__(self,path):self.pages=[Page(t) for t in texts]
    class Writer:
        def __init__(self):self.pages=[]
        def add_page(self,page):self.pages.append(page)
        def write(self,stream):stream.write(json.dumps([p.text for p in self.pages]).encode())
    def convert(source,destination,**kwargs):
        payload=json.loads(source.read_text())
        calls.append((payload,destination))
        data={'pages':[{'page_idx':i,'blocks':[{'type':'text','bbox':[0,0,1,1],
                    'content':f'layout {i}'}]} for i in range(len(payload))]}
        (destination/'structured_content.json').write_text(json.dumps(data))
        return destination/'markdown.md'
    monkeypatch.setattr(pypdf,'PdfReader',Reader)
    monkeypatch.setattr(pypdf,'PdfWriter',Writer)
    monkeypatch.setattr(data_md,'pdf_to_markdown_mineru',convert)


def test_pdf_auto_layout_provenance_and_original_page_mapping(tmp_path,monkeypatch):
    calls=[];texts=['normal prose '*20,'Column A    Column B\n'*8,'more prose '*20]
    _pdf_fakes(monkeypatch,texts,calls)
    source=tmp_path/'source.pdf';source.write_bytes(b'original source')
    settings=SimpleNamespace(pdf_mode='auto',pdf_overrides={},corpus=tmp_path)
    cache=tmp_path/'cache'
    first=pdf(source,cache,settings)
    assert len(calls)==1 and len(calls[0][0])==1
    assert [b['locator']['page'] for b in first]==[1,2,3]
    assert first[1]['locator']['method']=='mineru'
    assert first[1]['locator']['bbox']==[0,0,1,1]
    manifest=json.loads(next(cache.rglob('provenance.json')).read_text())
    assert manifest['input']['original_pages']==[2]
    assert len(manifest['input']['source_sha256'])==64
    assert pdf(source,cache,settings)==first and len(calls)==1
    source.write_bytes(b'changed original source')
    pdf(source,cache,settings)
    assert len(calls)==2


def test_explicit_pdf_text_mode_and_no_unproven_cache_reuse(tmp_path,monkeypatch):
    calls=[];_pdf_fakes(monkeypatch,['x'],calls)
    source=tmp_path/'source.pdf';source.write_bytes(b'source')
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'page-1.json').write_text(json.dumps([{'text':'stale unverified'}]))
    settings=SimpleNamespace(pdf_mode='auto',pdf_overrides={'source.pdf':'text'},corpus=tmp_path)
    result=pdf(source,cache,settings)
    assert result[0]['text']=='x' and not calls
    assert result[0]['locator']['quality_flags']==['sparse_text']


def test_explicit_layout_reruns_a_tampered_page_checkpoint(tmp_path,monkeypatch):
    calls=[];_pdf_fakes(monkeypatch,['ordinary prose '*20],calls)
    source=tmp_path/'source.pdf';source.write_bytes(b'source')
    settings=SimpleNamespace(pdf_mode='layout',pdf_overrides={},corpus=tmp_path)
    cache=tmp_path/'cache';result=pdf(source,cache,settings)
    assert result[0]['locator']['method']=='mineru' and len(calls)==1
    checkpoint=next(cache.rglob('page-1.json'))
    saved=json.loads(checkpoint.read_text());saved['blocks'][0]['text']='not the original parse'
    checkpoint.write_text(json.dumps(saved))
    # Valid batch provenance may reconstruct an invalid page checkpoint, but
    # its tampered text must never be published.
    assert pdf(source,cache,settings)[0]['text']=='layout 0'
    assert len(calls)==1


def test_page_override_reuses_only_identical_effective_routes(tmp_path,monkeypatch):
    calls=[];_pdf_fakes(monkeypatch,['scanned body',' '],calls)
    source=tmp_path/'source.pdf';source.write_bytes(b'source')
    settings=SimpleNamespace(pdf_mode='auto',pdf_overrides={},corpus=tmp_path)
    cache=tmp_path/'cache'
    first=pdf(source,cache,settings)
    assert len(calls)==1 and first[1]['locator']['method']=='mineru'
    settings.pdf_overrides={'source.pdf':{'default':'auto','2':'text'}}
    reviewed=pdf(source,cache,settings)
    assert len(calls)==1
    assert reviewed[0]==first[0]
    assert reviewed[1]['locator']['method']=='pdf_text'
    assert reviewed[1]['locator']['parse_mode']=='text' and not reviewed[1]['text'].strip()
    # A changed source or changed route cannot borrow an old page checkpoint.
    settings.pdf_overrides={'source.pdf':{'default':'auto','1':'text','2':'layout'}}
    final=pdf(source,cache,settings)
    assert len(calls)==2 and final[0]['locator']['method']=='pdf_text'


def test_reviewed_pdf_override_is_bound_to_source_hash(tmp_path,monkeypatch):
    calls=[];_pdf_fakes(monkeypatch,[' '],calls)
    source=tmp_path/'source.pdf';source.write_bytes(b'changed document')
    settings=SimpleNamespace(pdf_mode='auto',pdf_overrides={'source.pdf':{
        '1':'text','source_sha256':'original-reviewed-hash'}},corpus=tmp_path)
    with pytest.raises(ValueError,match='override does not match'):
        pdf(source,tmp_path/'cache',settings)


def test_ast_multiline_statement_boundary_preferred(tmp_path):
    source='import math\n\ndef f():\n    x = (\n        1 +\n        2\n    )\n    y = 5\n    return x + y\n'
    path=tmp_path/'source.py';path.write_bytes(source.encode())
    blocks=parse(path,tmp_path/'cache')
    assert blocks[0]['locator']['imports']==['import math']
    rows=list(chunk(blocks,CharacterTokenizer(),51,0))
    function=[r for r in rows if r['locator'].get('symbol')=='f']
    assert function[0]['text'].endswith('    )\n')
    assert all(r['locator']['structure']=='python_ast' for r in function)
