import hashlib
import json
import sqlite3

import pytest

from scripts.audit_corpus_coverage import audit


def fixture(tmp_path, expected, chunks):
    cases = [{'id': 'dev-1', 'split': 'dev', 'category': 'single', 'answerable': True,
              'evidence': [{'path': 'a.pdf', 'source_hash': 'h', 'locator': {'page': 2}, 'span_text': expected}]}]
    path = tmp_path / 'dev.json'
    path.write_text(json.dumps(cases), encoding='utf-8')
    (tmp_path / 'manifest.json').write_text(json.dumps({'datasets': {'dev.json': hashlib.sha256(path.read_bytes()).hexdigest()}}), encoding='utf-8')
    catalog = tmp_path / 'catalog.sqlite3'
    with sqlite3.connect(catalog) as db:
        db.executescript('CREATE TABLE documents (id,path,hash,version,status,excluded); CREATE TABLE chunks (id,document_id,version,text,locator,parent);')
        db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)', ('d', 'a.pdf', 'h', 'current', 'ready', 0))
        for item in chunks:
            db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?)', (item['id'], 'd', item.get('version', 'current'), item['text'], json.dumps(item['locator']), item.get('parent', '')))
    return catalog, path


def test_only_active_versions_count_and_catalog_stays_unchanged(tmp_path):
    catalog, cases = fixture(tmp_path, 'correct evidence', [
        {'id': 'old', 'version': 'old', 'text': 'correct evidence', 'locator': {'page': 2}},
        {'id': 'active', 'text': 'other content', 'locator': {'page': 2}}])
    before = catalog.read_bytes()
    report = audit(catalog, cases)
    assert report['active_chunks'] == 1
    assert report['summary']['child_span']['covered'] == 0
    assert report['evidence_diagnostics'][0]['reason'].startswith('quote_absent')
    assert catalog.read_bytes() == before


def test_parent_requires_mapping_and_wrong_page_cannot_help(tmp_path):
    catalog, cases = fixture(tmp_path, 'correct evidence', [
        {'id': 'a', 'text': 'correct', 'locator': {'page': 2}, 'parent': 'correct evidence'},
        {'id': 'b', 'text': 'evidence', 'locator': {'page': 2, 'parent_source_spans': [{'page': 3, 'chunk_start': 0, 'chunk_end': 16}]}, 'parent': 'correct evidence'}])
    result = audit(catalog, cases)['evidence_diagnostics'][0]
    assert not result['parent_span_covered']
    assert result['quote_present_in_unmapped_parent']
    with sqlite3.connect(catalog) as db:
        loc = {'page': 2, 'parent_source_spans': [{'page': 2, 'chunk_start': 0, 'chunk_end': 16}]}
        db.execute('UPDATE chunks SET locator=? WHERE id=?', (json.dumps(loc), 'b'))
    result = audit(catalog, cases)['evidence_diagnostics'][0]
    assert result['parent_span_covered']
    assert result['reason'] == 'covered_only_by_mapped_parent'


def test_cross_chunk_reconstruction_is_not_coverage(tmp_path):
    catalog, cases = fixture(tmp_path, 'alpha beta gamma', [
        {'id': 'one', 'text': 'alpha beta', 'locator': {'page': 2, 'token_start': 0}},
        {'id': 'two', 'text': 'beta gamma', 'locator': {'page': 2, 'token_start': 2}}])
    result = audit(catalog, cases)['evidence_diagnostics'][0]
    assert not result['child_span_covered']
    assert result['reason'] == 'quote_split_across_children_without_usable_parent'


def test_display_adjacent_spans_without_source_adjacency_remain_diagnostic(tmp_path):
    catalog, cases = fixture(tmp_path, 'alpha beta', [
        {'id': 'one', 'text': 'alpha\n\nbeta', 'locator': {'page': 2, 'source_spans': [
            {'page': 2, 'chunk_start': 0, 'chunk_end': 5}, {'page': 2, 'chunk_start': 7, 'chunk_end': 11}]}}])
    result = audit(catalog, cases)['evidence_diagnostics'][0]
    assert not result['child_span_covered']
    assert result['mapped_child_covered']
    assert result['reason'] == 'mapped_text_found_but_source_adjacency_unverified'


def replace_cases(path, cases):
    path.write_text(json.dumps(cases), encoding='utf-8')
    (path.parent/'manifest.json').write_text(json.dumps({
        'datasets': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}), encoding='utf-8')


@pytest.mark.parametrize('original', ['missing original', 'actual content'])
def test_alternatives_count_once_for_summary_source_and_type(tmp_path, original):
    catalog, path = fixture(tmp_path, original, [
        {'id':'one','text':'actual content','locator':{'page':2}}])
    cases=json.loads(path.read_text('utf-8')); old=cases[0]['evidence'][0]
    cases[0]['evidence_groups']=[[old,{**old,'span_text':'actual content'}]]
    replace_cases(path,cases)
    report=audit(catalog,path)
    assert report['schema_version']=='corpus-coverage-v2'
    for summary in [report['summary'],report['by_source']['a.pdf'],report['by_type']['.pdf']]:
        assert summary['required_groups']==1
        assert summary['counting_unit']=='required_evidence_group'
        assert summary['child_span']=={'covered':1,'total':1,'fraction':1.0}
        assert sum(summary['reasons'].values())==1
    assert report['alternative_summary']['alternative_representations']==2
    assert report['alternative_summary']['child_span']['covered']==(1 if original.startswith('missing') else 2)
    assert report['evidence_diagnostics_unit']=='alternative_representation'
    assert len(report['evidence_diagnostics'])==2 and len(report['group_diagnostics'])==1
    assert report['case_coverage']['either_span_covered']=={'covered':1,'total':1}


def test_covered_alternative_cannot_satisfy_another_required_group(tmp_path):
    catalog,path=fixture(tmp_path,'original',[
        {'id':'one','text':'alternate','locator':{'page':2}}])
    cases=json.loads(path.read_text('utf-8')); old=cases[0]['evidence'][0]
    cases[0]['evidence_groups']=[[old,{**old,'span_text':'alternate'}],[{**old,'span_text':'second required fact'}]]
    replace_cases(path,cases)
    report=audit(catalog,path)
    assert report['summary']['child_or_parent_span']=={'covered':1,'total':2,'fraction':.5}
    assert report['by_type']['.pdf']['required_groups']==2
    assert report['case_coverage']['either_span_covered']=={'covered':0,'total':1}
    assert report['missing_case_ids']==['dev-1']


def test_source_type_buckets_only_count_their_own_alternatives(tmp_path):
    catalog,path=fixture(tmp_path,'missing',[])
    cases=json.loads(path.read_text('utf-8')); old=cases[0]['evidence'][0]
    alternative={**old,'path':'b.txt','locator':{'line_start':1,'line_end':1},'span_text':'actual'}
    cases[0]['evidence_groups']=[[old,alternative,dict(alternative)]]
    replace_cases(path,cases)
    with sqlite3.connect(catalog) as db:
        db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)',('other','b.txt','h','current','ready',0))
        db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?)',('other','other','current','actual',
            json.dumps({'line_start':1,'line_end':1}),''))
    report=audit(catalog,path)
    assert report['summary']['child_span']=={'covered':1,'total':1,'fraction':1.0}
    assert report['by_source']['a.pdf']['child_span']['covered']==0
    assert report['by_source']['b.txt']['child_span']=={'covered':1,'total':1,'fraction':1.0}
    assert report['by_type']['.pdf']['child_span']['covered']==0
    assert report['by_type']['.txt']['required_groups']==1


def test_strict_coverage_accepts_explicitly_adjacent_physical_spans(tmp_path):
    spans=[{'page':2,'block':1,'char_start':0,'char_end':5,'chunk_start':0,'chunk_end':5},
           {'page':2,'block':2,'char_start':0,'char_end':4,'chunk_start':7,'chunk_end':11}]
    catalog,path=fixture(tmp_path,'alpha beta',[
        {'id':'one','text':'alpha\n\nbeta','locator':{'page':2,'source_spans':spans}}])
    report=audit(catalog,path)
    assert report['summary']['child_span']=={'covered':1,'total':1,'fraction':1.0}
    assert report['evidence_diagnostics'][0]['reason']=='covered_by_child'


def test_strict_parent_coverage_accepts_explicitly_adjacent_physical_spans(tmp_path):
    spans=[{'page':2,'block':1,'char_start':0,'char_end':5,'chunk_start':0,'chunk_end':5},
           {'page':2,'block':2,'char_start':0,'char_end':4,'chunk_start':7,'chunk_end':11}]
    catalog,path=fixture(tmp_path,'alpha beta',[
        {'id':'one','text':'alpha','parent':'alpha\n\nbeta','locator':{'page':2,'parent_source_spans':spans}}])
    report=audit(catalog,path)
    assert report['summary']['child_span']['covered']==0
    assert report['summary']['child_or_parent_span']['covered']==1
    assert report['group_diagnostics'][0]['reason']=='covered_only_by_mapped_parent'


def test_blind_and_modified_gold_refused(tmp_path):
    with pytest.raises(ValueError, match='Blind'):
        audit(tmp_path / 'missing.sqlite3', tmp_path / 'blind.json')
    catalog, cases = fixture(tmp_path, 'mW', [{'id': 'a', 'text': 'MW', 'locator': {'page': 2}}])
    assert audit(catalog, cases)['summary']['child_span']['covered'] == 0
    cases.write_text(cases.read_text() + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit(catalog, cases)
