"""Synthetic revision bundles only: never read private dev/blind questions."""
import copy
from datetime import datetime, timezone
import hashlib
import json

import pytest

from ragapp.evaluation import load_dataset, normalized
from ragapp.maintenance import _publication_dataset_frozen
from scripts.freeze_evidence_revision import (encode, file_sha, freeze_revision,
                                             validate_revision_bundle, validate_split_audit)


def write(path, value):
    path.write_bytes(encode(value))


def review(split, when):
    return {'reviewer_id': split + '-source-reviewer', 'reviewed_at_utc': when,
            'source_checked': True, 'physical_page_verified': True,
            'semantic_invariants_preserved': True, 'model_outputs_used': False,
            'method': 'Synthetic original physical page checked',
            'rule_id': 'fullwidth_punctuation_only',
            'reason': 'Synthetic punctuation change; units, math, negation and conditions preserved.'}


@pytest.fixture
def revision_inputs(tmp_path):
    parent = tmp_path / 'v1'; parent.mkdir()
    corpus = tmp_path / 'papers'; corpus.mkdir()
    when = '2026-01-01T00:00:00+00:00'
    datasets = {}; sources = {}; audit_paths = {}; cases_by_split = {}
    for split in ('dev', 'blind'):
        name = split + '.pdf'
        (corpus / name).write_bytes(b'Synthetic original source ' + split.encode())
        sources[name] = {'hash': file_sha(corpus / name)}
        span = split + ': -42 kg; x^2 remains negative.'
        evidence = {'path': name, 'source_hash': sources[name]['hash'], 'locator': {'page': 1},
                    'span_text': span, 'span_sha256': hashlib.sha256(normalized(span).encode()).hexdigest()}
        case = {'id': split + '-positive', 'schema_version': 'quality-v2', 'split': split,
                'category': 'single', 'question': 'What is ' + split + ' mass?', 'facts': [['-42 kg']],
                'answerable': True, 'cluster_id': split + '-cluster', 'template_group': split + '-template',
                'evidence': [evidence], 'opaque_nested_field': {'unchanged': [True, 12.3, '中文']}}
        negative = {**copy.deepcopy(case), 'id': split + '-negative', 'answerable': False,
                    'question': 'What is absent in ' + split + '?', 'evidence': [], 'category': 'unanswerable'}
        cases_by_split[split] = [case, negative]
        write(parent / (split + '.json'), [case, negative])
        datasets[split + '.json'] = {'count': 2, 'sha256': file_sha(parent / (split + '.json'))}
    manifest = {'schema_version': 'quality-v2', 'status': 'frozen', 'datasets': datasets,
                'sources': sources, 'human_verified_cases': 0, 'split_policy': 'Fixed synthetic clusters/templates'}
    write(parent / 'manifest.json', manifest)
    for split in ('dev', 'blind'):
        alt = copy.deepcopy(cases_by_split[split][0]['evidence'][0]); alt['span_text'] = alt['span_text'].replace(':', '：')
        alt['span_sha256'] = hashlib.sha256(normalized(alt['span_text']).encode()).hexdigest()
        alt['span_literal_sha256'] = hashlib.sha256(alt['span_text'].encode()).hexdigest()
        alt['review'] = review(split, when)
        audit = {'schema_version': 'evidence-equivalence-audit-v1', 'split': split,
                 'parent_manifest_sha256': file_sha(parent / 'manifest.json'),
                 'parent_dataset_sha256': datasets[split + '.json']['sha256'],
                 'reviewer_id': split + '-source-reviewer', 'reviewed_at_utc': when,
                 'post_hoc_dev_representation_repair': True, 'blind_model_outputs_seen': False,
                 'model_outputs_used': False,
                 'records': [{'case_id': split + '-positive', 'group_index': 0,
                              'status': 'equivalent_added', 'alternatives': [alt], 'review': review(split, when)}]}
        path = tmp_path / (split + '-audit.json'); write(path, audit); audit_paths[split] = path
    return {'parent': parent, 'corpus': corpus, 'audits': audit_paths, 'output': tmp_path / 'v2',
            'declaration': {'declared_by': 'synthetic-operator', 'declared_at_utc': when,
                            'blind_model_runs_started': False}}


def make(inputs, **overrides):
    return freeze_revision(**{**inputs, 'seal': True, **overrides})


def mutate_audit(inputs, split, mutate):
    path = inputs['audits'][split]; data = json.loads(path.read_bytes()); mutate(data); write(path, data)


def test_sealed_revision_preserves_parent_and_loads_with_existing_evaluator(revision_inputs):
    args = revision_inputs
    original = {p.name: p.read_bytes() for p in args['parent'].iterdir()}
    manifest = make(args)
    for split in ('dev', 'blind'):
        old = json.loads(original[split + '.json'])
        new, fingerprint, loaded = load_dataset(args['output'] / (split + '.json'))
        assert fingerprint == manifest['datasets'][split + '.json']['sha256']
        assert loaded['status'] == 'evidence_revision_sealed'
        for before, after in zip(old, new):
            assert {k: v for k, v in after.items() if k != 'evidence_groups'} == before
            if before['evidence']:
                assert after['evidence_groups'][0][0] == before['evidence'][0]
                assert len(after['evidence_groups'][0]) == 2
        assert (args['output'] / 'parent-v1' / (split + '.json')).read_bytes() == original[split + '.json']
    assert {p.name: p.read_bytes() for p in args['parent'].iterdir()} == original
    assert manifest['human_verified_cases'] == 0
    assert validate_revision_bundle(args['output'], corpus=args['corpus']) == {
        'valid': True, 'alternatives_added': 2, 'semantic_equivalence_machine_proved': False,
        'reviewer_ids_by_split': {'dev': ['dev-source-reviewer'], 'blind': ['blind-source-reviewer']}}
    closure = _publication_dataset_frozen(args['output'] / 'blind.json', manifest)
    assert closure['reviewer_ids_by_split']['blind'] == ['blind-source-reviewer']


@pytest.mark.parametrize('mutation,match', [
    (lambda a: a['records'][0]['alternatives'][0].update(span_literal_sha256='0' * 64), 'literal hash'),
    (lambda a: a['records'][0]['alternatives'][0].update(span_sha256='0' * 64), 'normalized hash'),
    (lambda a: a['records'][0]['alternatives'][0].update(locator={'page': 2}), 'physical locator'),
    (lambda a: a['records'][0]['alternatives'][0].update(source_hash='0' * 64), 'physical locator'),
    (lambda a: a['records'][0]['alternatives'][0]['review'].pop('physical_page_verified'), 'physical-page'),
    (lambda a: a['records'][0]['alternatives'][0]['review'].update(semantic_invariants_preserved=False), 'invariants'),
    (lambda a: a['records'][0]['alternatives'][0]['review'].update(model_outputs_used=True), 'model outputs'),
    (lambda a: a['records'][0]['review'].update(source_checked=False), 'source_checked'),
    (lambda a: a['records'][0]['review'].pop('reason'), 'reason'),
    (lambda a: a['records'][0]['review'].update(reviewed_at_utc='2026-01-01'), 'timezone'),
    (lambda a: a['records'][0]['review'].update(reviewed_at_utc='2999-01-01T00:00:00Z'), 'future'),
    (lambda a: a.update(model_outputs_used=True), 'model outputs'),
    (lambda a: a.update(parent_manifest_sha256='0' * 64), 'different parent'),
    (lambda a: a['records'].clear(), 'every original evidence group'),
    (lambda a: a['records'].append(copy.deepcopy(a['records'][0])), 'Duplicate'),
    (lambda a: a['records'][0].update(group_index=True), 'identity'),
    (lambda a: a['records'][0].update(alternatives=a['records'][0]['alternatives'] * 4), 'Too many'),
])
def test_invalid_review_never_produces_sealed_output(revision_inputs, mutation, match):
    mutate_audit(revision_inputs, 'dev', mutation)
    with pytest.raises(ValueError, match=match):
        make(revision_inputs)
    assert not revision_inputs['output'].exists()


def test_unresolved_can_exist_as_draft_but_cannot_seal_or_publish(revision_inputs):
    mutate_audit(revision_inputs, 'dev', lambda a: a['records'][0].update(status='unresolved', alternatives=[]))
    with pytest.raises(ValueError, match='Unresolved'):
        make(revision_inputs)
    manifest = make(revision_inputs, seal=False)
    assert manifest['status'] == 'evidence_revision_draft'
    with pytest.raises(ValueError, match='not sealed'):
        _publication_dataset_frozen(revision_inputs['output'] / 'blind.json', manifest)


def test_blind_exposure_or_missing_operator_declaration_blocks_seal(revision_inputs):
    with pytest.raises(ValueError, match='before-blind-model'):
        make(revision_inputs, declaration=None)
    mutate_audit(revision_inputs, 'blind', lambda a: a.update(blind_model_outputs_seen=True))
    with pytest.raises(ValueError, match='model-output exposure'):
        make(revision_inputs)


def test_changed_original_source_blocks_creation_and_later_verification(revision_inputs):
    args = revision_inputs; source = args['corpus'] / 'dev.pdf'; before = source.read_bytes()
    source.write_bytes(before + b'changed')
    with pytest.raises(ValueError, match='Original source hash'):
        make(args)
    source.write_bytes(before); make(args)
    source.write_bytes(before + b'changed')
    with pytest.raises(ValueError, match='Original source hash'):
        validate_revision_bundle(args['output'], corpus=args['corpus'])


def test_existing_output_cannot_be_overwritten(revision_inputs):
    make(revision_inputs); path = revision_inputs['output'] / 'manifest.json'; original = path.read_bytes()
    with pytest.raises(ValueError, match='new independent directory'):
        make(revision_inputs)
    assert path.read_bytes() == original


def test_rehashing_changed_question_does_not_bypass_preservation(revision_inputs):
    manifest = make(revision_inputs); root = revision_inputs['output']; path = root / 'dev.json'
    cases = json.loads(path.read_bytes()); cases[0]['question'] = 'Changed question'; write(path, cases)
    manifest['datasets']['dev.json']['sha256'] = file_sha(path); write(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='preserve parent fields'):
        _publication_dataset_frozen(path, manifest)


def test_rehashing_missing_alternative_review_cannot_bypass_publication(revision_inputs):
    manifest = make(revision_inputs); root = revision_inputs['output']; path = root / 'blind-equivalence-audit.json'
    audit = json.loads(path.read_bytes()); audit['records'][0]['alternatives'][0]['review'].pop('reason'); write(path, audit)
    manifest['evidence_revision']['audits']['blind']['sha256'] = file_sha(path); write(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='reason'):
        _publication_dataset_frozen(root / 'blind.json', manifest)


def test_legacy_frozen_status_cannot_bypass_extra_alternative_audit(revision_inputs):
    manifest = make(revision_inputs); root = revision_inputs['output']
    manifest.pop('evidence_revision'); manifest['status'] = 'frozen'; write(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Additional evidence alternatives'):
        _publication_dataset_frozen(root / 'blind.json', manifest)


def test_cross_split_alternative_literal_collision_is_rejected(revision_inputs):
    def collide(a):
        alt = a['records'][0]['alternatives'][0]; alt['span_text'] = 'same literal across splits'
        alt['span_sha256'] = hashlib.sha256(normalized(alt['span_text']).encode()).hexdigest()
        alt['span_literal_sha256'] = hashlib.sha256(alt['span_text'].encode()).hexdigest()
    for split in ('dev', 'blind'):
        mutate_audit(revision_inputs, split, collide)
    with pytest.raises(ValueError, match='Cross-split evidence leakage'):
        make(revision_inputs)


def test_single_split_validation_never_reads_other_split_or_seals(revision_inputs):
    args = revision_inputs
    (args['parent'] / 'blind.json').unlink()
    args['audits']['blind'].unlink()
    result = validate_split_audit(args['parent'], 'dev', args['audits']['dev'], args['corpus'])
    assert result['valid_split_structure_and_source_hashes']
    assert result['alternatives_added'] == 1
    assert result['sealed'] is False and result['cross_split_checked'] is False
    assert not args['output'].exists()
