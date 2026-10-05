import copy
import pytest
from ragapp.sufficiency import evidence_outline,passage_payload


def data():
    return passage_payload({'requirements':['Describe storage','Describe transport'],
        'untrusted_evidence':[{'citation':'S1','source':'storage.md','version':'v1','text':'Store at 8 C.'},
                              {'citation':'S2','source':'transport.md','version':'v2','text':'Transport sealed.'}]})


def decision():
    return {'sufficient':True,'coverage':[
        {'requirement_index':1,'supported':True,'passage_ids':['S2:P1']},
        {'requirement_index':0,'supported':True,'passage_ids':['S1:P1']}], 'missing':[]}


def test_outline_uses_real_source_slices_and_preserves_input():
    p=data();d=decision();before=copy.deepcopy(d)
    d['coverage'][0]['evidence_quotes']=[{'text':'invented claim'}]
    result=evidence_outline(d,p)
    assert [r['requirement_index'] for r in result]==[0,1]
    assert [r['passages'][0]['text'] for r in result]==['Store at 8 C.','Transport sealed.']
    assert result[1]['passages'][0]['source']=='transport.md'
    assert d['coverage'][0]['evidence_quotes']==[{'text':'invented claim'}]
    assert evidence_outline({'sufficient':False},p)==[]


@pytest.mark.parametrize('change',['unknown_id','missing_requirement','unsupported'])
def test_invalid_outline_never_reaches_generation(change):
    d=decision()
    if change=='unknown_id':d['coverage'][0]['passage_ids']=['S1:P99']
    if change=='missing_requirement':d['coverage'].pop()
    if change=='unsupported':d['coverage'][0]['supported']=False
    with pytest.raises(ValueError):evidence_outline(d,data())
