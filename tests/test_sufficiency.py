import copy
import pytest

from ragapp.sufficiency import coverage_errors
from ragapp.sufficiency import passage_payload, passage_coverage_errors


def fixture():
    payload={'requirements':['What concentration?'],
             'untrusted_evidence':[{'citation':'S1','text':'Concentration is 4.2 mg/L; not 42 mg/L.'}]}
    judgment={'sufficient':True,'missing':[],'coverage':[{'requirement_index':0,'supported':True,
              'citations':['S1'],'quote':'Concentration is 4.2 mg/L'}]}
    return judgment,payload


def test_positive_decision_requires_existing_quote_for_each_request():
    judgment,payload=fixture()
    assert coverage_errors(judgment,payload)==[]
    payload['requirements'].append('When was it sampled?')
    assert 'incomplete_or_duplicate_requirements' in coverage_errors(judgment,payload)


@pytest.mark.parametrize('quote',['Concentration is 42 mg/L','Concentration is 4.2 g/L',
                                 '浓度为4.2 mg/L','',None])
def test_numbers_units_and_model_translation_are_not_source_quotes(quote):
    judgment,payload=fixture();judgment['coverage'][0]['quote']=quote
    assert coverage_errors(judgment,payload)


def test_only_whitespace_and_unicode_compatibility_are_normalized():
    judgment,payload=fixture();judgment['coverage'][0]['quote']='Concentration\nis\u00a0４.２ mg/L'
    assert coverage_errors(judgment,payload)==[]


@pytest.mark.parametrize('refs',[[],['S2'],'S1',[{}]])
def test_nonexistent_or_malformed_citations_rejected(refs):
    judgment,payload=fixture();judgment['coverage'][0]['citations']=refs
    assert 'invalid_coverage_citations' in coverage_errors(judgment,payload)


def test_missing_duplicate_and_contradictory_coverage_is_rejected():
    judgment,payload=fixture();del judgment['coverage']
    assert coverage_errors(judgment,payload)==['missing_coverage']
    judgment,payload=fixture();judgment['coverage']*=2
    assert 'incomplete_or_duplicate_requirements' in coverage_errors(judgment,payload)
    judgment,payload=fixture();judgment['missing']=['concentration']
    assert 'contradictory_missing_requirements' in coverage_errors(judgment,payload)


def test_visual_quote_requires_current_evidence_citation():
    judgment,payload=fixture();judgment['coverage'][0]['quote']='The indicator is blue.'
    payload['visual_verification']=[{'citation':'S1','text':'The indicator is blue.'}]
    assert coverage_errors(judgment,payload)==[]
    payload['visual_verification'][0]['citation']='S2'
    assert coverage_errors(judgment,payload)


def test_negative_decision_does_not_require_invented_quote():
    assert coverage_errors({'sufficient':False},fixture()[1])==[]


def test_passage_ids_extract_original_text_without_model_transcription():
    _,payload=fixture();source=payload['untrusted_evidence'][0]['text']
    prepared=passage_payload(payload)
    decision={'sufficient':True,'coverage':[{'requirement_index':0,'supported':True,'passage_ids':['S1:P1']}]}
    assert passage_coverage_errors(decision,prepared)==[]
    assert decision['coverage'][0]['evidence_quotes'][0]['text']==source
    assert payload['untrusted_evidence'][0]['text']==source
    assert 'text' not in prepared['untrusted_evidence'][0]


def test_passage_slices_preserve_whitespace_and_offsets_for_long_text():
    _,payload=fixture();source='First paragraph.\n\n'+('    long formula × 4.2 mg/L\n'*100)+'\nTail.'
    payload['untrusted_evidence'][0]['text']=source
    prepared=passage_payload(payload);parts=prepared['untrusted_evidence'][0]['passages']
    assert ''.join(p['text'] for p in parts)==source
    assert len({p['id'] for p in parts})==len(parts)
    for p in parts:assert p['text']==source[p['context_char_start']:p['context_char_end']]


@pytest.mark.parametrize('refs',[[],['S1:P999'],['S2:P1'],'S1:P1',[{}]])
def test_unknown_or_malformed_passage_ids_fail_closed(refs):
    prepared=passage_payload(fixture()[1])
    decision={'sufficient':True,'coverage':[{'requirement_index':0,'supported':True,'passage_ids':refs}]}
    assert passage_coverage_errors(decision,prepared)==['invalid_coverage_passages']


def test_passage_coverage_requires_each_request_once_and_consistent_decision():
    prepared=passage_payload(fixture()[1]);prepared['requirements'].append('When?')
    entry={'requirement_index':0,'supported':True,'passage_ids':['S1:P1']}
    decision={'sufficient':True,'coverage':[entry,copy.deepcopy(entry)]}
    assert 'incomplete_or_duplicate_requirements' in passage_coverage_errors(decision,prepared)
    decision['coverage'][1]['requirement_index']=1;decision['coverage'][1]['supported']=False
    assert 'unsupported_requirement' in passage_coverage_errors(decision,prepared)


def test_visual_passage_namespace_is_separate_and_requires_active_citation():
    _,payload=fixture();payload['visual_verification']=[{'citation':'S1','text':'Indicator is blue.'}]
    prepared=passage_payload(payload)
    decision={'sufficient':True,'coverage':[{'requirement_index':0,'supported':True,'passage_ids':['S1:V1:P1']}]}
    assert passage_coverage_errors(decision,prepared)==[]
    assert decision['coverage'][0]['evidence_quotes'][0]['text']=='Indicator is blue.'
