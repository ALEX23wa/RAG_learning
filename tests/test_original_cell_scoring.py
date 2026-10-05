import copy
import json
import pytest
from ragapp.evaluation import grade, original_cell_matches, span_matches


def fixture():
    expected={'path':'inventory.xlsx','extraction':'original_xlsx_values',
              'locator':{'sheet':'Inventory','range':'A2:E2'},'span_text':'B2=Sensor name'}
    data={'operation':'header_lookup','sheet':'Inventory','range':'B2:B2','header_row':2,
          'row_start':2,'row_end':2,'value':'Sensor name','column':'Sensor name',
          'cells':{'B2':{'value':'Sensor name','column':'Sensor name','unit':None}}}
    evidence={'id':'row','path':'inventory.xlsx','version':'v1','citation':'S1',
              'locator':{'kind':'calculation',**{k:copy.deepcopy(data[k]) for k in
                         ('operation','sheet','range','header_row','row_start','row_end','cells')}}}
    evidence['text']='[Deterministic calculation from original table]\n'+json.dumps(data)
    return expected,evidence,data


def test_exact_original_cell_support_does_not_change_literal_recall():
    expected,evidence,_=fixture()
    assert original_cell_matches(expected,evidence)
    assert not span_matches(expected,evidence)
    case={'answerable':True,'facts':[['Sensor name']],'evidence':[expected]}
    result=grade(case,{'answer':'Sensor name [S1]','status':'answered',
                       'citations':[evidence],'evidence':[evidence]})
    assert result['pass']
    assert result['span_recalled_at_10']==[False]
    assert result['citation_literal_gold_support']==[False]
    assert result['citation_original_cell_support']==[True]


@pytest.mark.parametrize('field,value',[
    ('path','different.xlsx'),('extraction','original_pdf_text'),
    ('span_text','C2=Sensor name'),('span_text','B2=Sensor names'),
    ('span_text','B2=Sensor name\nC2=Cost'),('span_text','XFE2=Sensor name')])
def test_wrong_source_coordinate_or_exact_value_cannot_pass(field,value):
    expected,evidence,_=fixture();expected[field]=value
    assert not original_cell_matches(expected,evidence)


@pytest.mark.parametrize('range_value',['C2:E2','A3:E3','E2:A2','A0:E2','A2:E2:F2','bogus'])
def test_gold_range_must_contain_the_actual_cell(range_value):
    expected,evidence,_=fixture();expected['locator']['range']=range_value
    assert not original_cell_matches(expected,evidence)


@pytest.mark.parametrize('field,value',[
    ('operation','sum'),('sheet','Other'),('range','A2:E2'),('header_row',3),
    ('row_start',True),('row_end',3),('cells',{'C2':{'value':'Sensor name'}})])
def test_both_visible_and_physical_result_must_be_exact_header_cell(field,value):
    expected,evidence,data=fixture();data[field]=value;evidence['locator'][field]=copy.deepcopy(value)
    evidence['text']='[Deterministic calculation from original table]\n'+json.dumps(data)
    assert not original_cell_matches(expected,evidence)


def test_claim_text_and_detached_cell_metadata_are_not_source_evidence():
    expected,evidence,data=fixture()
    evidence['text']='B2=Sensor name'
    assert not original_cell_matches(expected,evidence)
    data['cells']['B2']['value']='Other sensor'
    evidence['text']='[Deterministic calculation from original table]\n'+json.dumps(data)
    assert not original_cell_matches(expected,evidence)


def test_extra_gold_constraints_are_not_ignored():
    expected,evidence,_=fixture();expected['locator']['table']='Second table'
    assert not original_cell_matches(expected,evidence)
