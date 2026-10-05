"""Seal an independently source-reviewed evidence revision without changing v1.

No model, parser or retrieval output is consulted. Equivalence is a reviewer's
explicit assertion, not a semantic fact proved by this structural validator.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragapp.evaluation import evidence_groups, normalized

SCHEMA = 'evidence-equivalence-audit-v1'
SEALED = 'evidence_revision_sealed'
MAX_ALTERNATIVES = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_path(root, name):
    require(isinstance(name, str) and bool(name), 'Missing relative artifact/source path')
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path != Path(root).resolve(),
            'Artifact/source path escapes its root')
    return path


def checked(root, name, expected):
    require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected),
            'Missing artifact hash: ' + name)
    path = safe_path(root, name)
    require(path.is_file() and file_sha(path) == expected, 'Artifact hash mismatch: ' + name)
    return read_json(path)


def timestamp(value, name):
    require(isinstance(value, str), 'Missing review timestamp: ' + name)
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('Invalid review timestamp: ' + name) from None
    require(result.tzinfo is not None, 'Review timestamp requires a timezone: ' + name)
    require(result <= datetime.now(timezone.utc), 'Review timestamp is in the future: ' + name)
    return result


def reviewer(review):
    return review.get('reviewer_id') or review.get('reviewer')


def review_record(review, *, physical=False):
    require(isinstance(review, dict), 'Missing source review')
    for key in ('source_checked', 'semantic_invariants_preserved'):
        require(review.get(key) is True, 'Review must assert ' + key)
    require(review.get('model_outputs_used') is False, 'Review must exclude model outputs')
    require(isinstance(reviewer(review), str) and bool(reviewer(review).strip()), 'Missing reviewer identity')
    for key in ('method', 'rule_id', 'reason'):
        require(isinstance(review.get(key), str) and bool(review[key].strip()), 'Missing review ' + key)
    timestamp(review.get('reviewed_at_utc'), 'reviewed_at_utc')
    if physical:
        require(review.get('visual_checked') is True or review.get('physical_page_verified') is True,
                'Alternative lacks original physical-page verification')
    return review


def load_parent(parent):
    parent = Path(parent)
    manifest = read_json(parent / 'manifest.json')
    require(manifest.get('schema_version') == 'quality-v2'
            and manifest.get('status') in {'frozen', 'agent_source_reviewed_human_pending'}
            and not manifest.get('evidence_revision'), 'Parent must be the original frozen v1 bundle')
    from ragapp.maintenance import _publication_dataset_frozen
    _publication_dataset_frozen(parent / 'dev.json', manifest)
    files = {'manifest.json': (parent / 'manifest.json').read_bytes()}
    datasets = {}
    for split in ('dev', 'blind'):
        name = split + '.json'
        entry = manifest.get('datasets', {}).get(name, {})
        require(isinstance(entry, dict), 'Parent dataset entry must include count/hash')
        cases = checked(parent, name, entry.get('sha256'))
        require(isinstance(cases, list) and len(cases) == entry.get('count'), 'Parent split count mismatch')
        datasets[split] = cases
        files[name] = (parent / name).read_bytes()
    artifacts = {'source-checks.json': manifest.get('source_checks_sha256'),
                 'summary.json': manifest.get('summary_sha256'),
                 **manifest.get('independent_review_artifacts', {})}
    for name, digest in artifacts.items():
        if digest:
            checked(parent, name, digest)
            files[name] = safe_path(parent, name).read_bytes()
    require(isinstance(manifest.get('sources'), dict) and bool(manifest['sources']), 'Parent sources are missing')
    return manifest, datasets, files


def check_sources(manifest, corpus):
    verified = {}
    for name, source in manifest['sources'].items():
        expected = source.get('hash') or source.get('sha256')
        require(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'Invalid frozen source hash')
        path = safe_path(corpus, name)
        require(path.is_file() and file_sha(path) == expected, 'Original source hash mismatch: ' + name)
        verified[name] = expected
    return verified


def apply_audits(manifest, datasets, audits, *, seal, only_split=None):
    """Pure structural validation, suitable for synthetic tests and publication."""
    result = {}; changes = []; unresolved = []; identities = set()
    split_spans = {}; split_clusters = {}; split_templates = {}
    parent_manifest_sha = audits['_parent_manifest_sha256']
    require(only_split in (None, 'dev', 'blind') and not (seal and only_split),
            'A single-split check cannot seal a revision')
    selected_splits = (only_split,) if only_split else ('dev', 'blind')
    for split in selected_splits:
        audit = audits[split]
        require(audit.get('schema_version') == SCHEMA and audit.get('split') == split, 'Wrong equivalence audit schema/split')
        require(audit.get('parent_manifest_sha256') == parent_manifest_sha
                and audit.get('parent_dataset_sha256') == manifest['datasets'][split + '.json']['sha256'],
                'Equivalence audit belongs to a different parent')
        require(audit.get('model_outputs_used') is False, 'Audit must exclude model outputs')
        require(isinstance(reviewer(audit), str) and bool(reviewer(audit)), 'Missing audit reviewer')
        timestamp(audit.get('reviewed_at_utc'), 'audit reviewed_at_utc')
        if split == 'dev':
            require(audit.get('post_hoc_dev_representation_repair') is True,
                    'Observed dev repair must be declared post hoc')
        else:
            require(audit.get('blind_model_outputs_seen') is False, 'Blind audit must precede model-output exposure')
        records = audit.get('records')
        require(isinstance(records, list), 'Audit records are missing')
        lookup = {}
        for record in records:
            require(isinstance(record, dict) and isinstance(record.get('case_id'), str)
                    and type(record.get('group_index')) is int and record['group_index'] >= 0, 'Invalid audit record identity')
            key = (record['case_id'], record['group_index'])
            require(key not in lookup, 'Duplicate audit record')
            lookup[key] = record
        expected_keys = {(case['id'], gi) for case in datasets[split]
                         for gi, _ in enumerate(evidence_groups(case))}
        require(set(lookup) == expected_keys, 'Audit does not cover exactly every original evidence group')
        result[split] = []; split_spans[split] = set(); split_clusters[split] = set(); split_templates[split] = set()
        for case in datasets[split]:
            require(case.get('split') == split and case.get('id') not in identities, 'Parent case identity/split collision')
            identities.add(case['id'])
            for key, target in (('cluster_id', split_clusters[split]), ('template_group', split_templates[split])):
                require(isinstance(case.get(key), str) and bool(case[key]), 'Missing parent grouping: ' + key)
                target.add(case[key])
            copy_case = copy.deepcopy(case); groups = []
            for gi, original_group in enumerate(evidence_groups(case)):
                require(isinstance(original_group, list) and bool(original_group), 'Empty original evidence group')
                record = lookup[(case['id'], gi)]; status = record.get('status')
                require(status in {'unchanged', 'equivalent_added', 'unresolved'}, 'Invalid equivalence review status')
                alternatives = record.get('alternatives', [])
                require(isinstance(alternatives, list) and len(alternatives) <= MAX_ALTERNATIVES, 'Too many alternatives')
                require((status == 'equivalent_added') == bool(alternatives), 'Review status/alternatives disagree')
                if status == 'unresolved':
                    unresolved.append({'case_id': case['id'], 'group_index': gi})
                    require(not seal, 'Unresolved review cannot be sealed')
                else:
                    group_review = review_record(record.get('review', record))
                    require(reviewer(group_review) == reviewer(audit), 'Group reviewer differs from split audit reviewer')
                    require(timestamp(group_review['reviewed_at_utc'], 'group time') <=
                            timestamp(audit['reviewed_at_utc'], 'audit time'), 'Group review postdates audit')
                group = copy.deepcopy(original_group); originals = []
                for evidence in original_group:
                    name = evidence.get('path', '').replace('\\', '/')
                    require(name in manifest['sources'] and evidence.get('source_hash') ==
                            (manifest['sources'][name].get('hash') or manifest['sources'][name].get('sha256')),
                            'Original evidence source differs from parent manifest')
                    originals.append(evidence)
                seen = {normalized(e.get('span_text', '')) for e in originals if e.get('span_text')}
                for index, alt in enumerate(alternatives):
                    require(isinstance(alt, dict), 'Invalid alternative')
                    require(any(all(alt.get(k) == e.get(k) for k in ('path', 'source_hash', 'locator')) for e in originals),
                            'Alternative changed source/path/physical locator')
                    text = alt.get('span_text')
                    require(isinstance(text, str) and bool(normalized(text)), 'Alternative needs a literal text span')
                    require(alt.get('span_literal_sha256') == sha(text.encode('utf-8')), 'Alternative literal hash mismatch')
                    require(alt.get('span_sha256') == sha(normalized(text).encode('utf-8')), 'Alternative normalized hash mismatch')
                    require(normalized(text) not in seen, 'Duplicate equivalent evidence alternative')
                    seen.add(normalized(text))
                    review = review_record(alt.get('review', alt), physical=True)
                    require(reviewer(review) == reviewer(audit), 'Alternative reviewer differs from split audit reviewer')
                    require(timestamp(review['reviewed_at_utc'], 'alternative time') <=
                            timestamp(audit['reviewed_at_utc'], 'audit time'), 'Alternative review postdates audit')
                    clean = {key: copy.deepcopy(alt[key]) for key in
                             ('path', 'source_hash', 'locator', 'span_text', 'span_sha256', 'span_literal_sha256')}
                    if 'extraction' in alt:
                        clean['extraction'] = copy.deepcopy(alt['extraction'])
                    group.append(clean)
                for evidence in group:
                    if evidence.get('span_text'):
                        split_spans[split].add(sha(normalized(evidence['span_text']).encode('utf-8')))
                groups.append(group)
                changes.append({'split': split, 'case_id': case['id'], 'group_index': gi, 'status': status,
                                'original_group_sha256': sha(encode(original_group)),
                                'new_group_sha256': sha(encode(group)), 'alternatives_added': len(alternatives)})
            # No other case field is changed, including evidence/facts/question.
            copy_case['evidence_groups'] = groups
            result[split].append(copy_case)
    if only_split is None:
        for label, sets in [('evidence', split_spans), ('cluster', split_clusters), ('template', split_templates)]:
            require(not sets['dev'] & sets['blind'], 'Cross-split ' + label + ' leakage')
    return result, changes, unresolved


def validate_split_audit(parent, split, audit_path, corpus):
    """Read only one split's questions; does not validate/seal the whole bundle."""
    require(split in ('dev', 'blind'), 'Invalid split')
    parent = Path(parent)
    manifest = read_json(parent / 'manifest.json')
    require(manifest.get('schema_version') == 'quality-v2'
            and manifest.get('status') in {'frozen', 'agent_source_reviewed_human_pending'}
            and not manifest.get('evidence_revision'), 'Parent must be frozen v1')
    entry = manifest.get('datasets', {}).get(split + '.json', {})
    cases = checked(parent, split + '.json', entry.get('sha256'))
    require(isinstance(cases, list) and len(cases) == entry.get('count'), 'Parent split count mismatch')
    audit = read_json(audit_path)
    _, changes, unresolved = apply_audits(manifest, {split: cases},
        {split: audit, '_parent_manifest_sha256': file_sha(parent / 'manifest.json')},
        seal=False, only_split=split)
    check_sources(manifest, corpus)
    return {'valid_split_structure_and_source_hashes': True, 'split': split,
            'groups_reviewed': len(changes), 'alternatives_added': sum(c['alternatives_added'] for c in changes),
            'unresolved': unresolved, 'cross_split_checked': False, 'sealed': False,
            'semantic_equivalence_machine_proved': False}


def seal_declaration(declaration, sealed_at):
    require(isinstance(declaration, dict) and declaration.get('blind_model_runs_started') is False,
            'Sealing requires an explicit before-blind-model declaration')
    require(isinstance(declaration.get('declared_by'), str) and bool(declaration['declared_by']), 'Missing seal declarant')
    require(timestamp(declaration.get('declared_at_utc'), 'seal declaration') <= timestamp(sealed_at, 'sealed_at_utc'),
            'Seal declaration postdates sealing')


def freeze_revision(parent, audits, output, corpus, *, seal=False, declaration=None):
    parent, output, corpus = Path(parent).resolve(), Path(output).resolve(), Path(corpus).resolve()
    require(not output.exists() and not output.is_relative_to(parent) and not parent.is_relative_to(output),
            'Output must be a new independent directory; never overwrite v1 or an existing revision')
    require(not output.is_relative_to(corpus), 'Evaluation output cannot be inside the source corpus')
    manifest, datasets, original_files = load_parent(parent)
    audit_bytes = {split: Path(audits[split]).read_bytes() for split in ('dev', 'blind')}
    audit_values = {split: json.loads(raw) for split, raw in audit_bytes.items()}
    audit_values['_parent_manifest_sha256'] = sha(original_files['manifest.json'])
    new, changes, unresolved = apply_audits(manifest, datasets, audit_values, seal=seal)
    verified = check_sources(manifest, corpus)
    sealed_at = datetime.now(timezone.utc).isoformat()
    if seal:
        seal_declaration(declaration, sealed_at)
    artifacts = {f'parent-v1/{name}': raw for name, raw in original_files.items()}
    artifacts.update({f'{split}-equivalence-audit.json': raw for split, raw in audit_bytes.items()})
    artifacts['change-audit.json'] = encode(changes)
    checks = {'verified_utc': sealed_at, 'method': 'full_sha256_original_file_bytes', 'sources': verified,
              'semantic_equivalence': 'independent review assertion; not machine proved'}
    artifacts['source-checks.json'] = encode(checks)
    for split, cases in new.items():
        artifacts[split + '.json'] = encode(cases)
    summary = {'status': SEALED if seal else 'evidence_revision_draft', 'unresolved': unresolved,
               'alternatives_added': sum(c['alternatives_added'] for c in changes),
               'groups_reviewed': len(changes), 'post_hoc_dev_representation_repair': True,
               'human_verified_claim_added': False, 'automatic_promotion': False}
    artifacts['summary.json'] = encode(summary)
    revision = {'schema_version': 'evidence-revision-v1', 'parent_directory': 'parent-v1',
                'parent_manifest_sha256': sha(original_files['manifest.json']),
                'parent_dataset_sha256': {s: manifest['datasets'][s + '.json']['sha256'] for s in ('dev', 'blind')},
                'audits': {s: {'path': s + '-equivalence-audit.json', 'sha256': sha(audit_bytes[s])} for s in ('dev', 'blind')},
                'change_audit_sha256': sha(artifacts['change-audit.json']),
                'post_hoc_dev_representation_repair': True,
                'sealed_at_utc': sealed_at if seal else None, 'blind_seal_declaration': declaration if seal else None,
                'sealing_claim_basis': 'explicit operator and source-reviewer declarations, not proof of absence of earlier runs',
                'source_corpus': str(corpus), 'automatic_promotion': False,
                'semantic_equivalence': 'review assertion; original physical source remains authoritative'}
    updated = copy.deepcopy(manifest)
    updated.update(status=summary['status'], evidence_revision=revision,
                   datasets={s + '.json': {'sha256': sha(artifacts[s + '.json']), 'count': len(new[s])} for s in new},
                   source_checks_sha256=sha(artifacts['source-checks.json']), summary_sha256=sha(artifacts['summary.json']))
    # Parent audits apply to the preserved parent bytes, not the revised dataset.
    updated.pop('independent_review_artifacts', None)
    for name, raw in original_files.items():
        require(safe_path(parent, name).read_bytes() == raw, 'Parent changed during validation')
    for split, raw in audit_bytes.items():
        require(Path(audits[split]).read_bytes() == raw, 'Audit changed during validation')
    check_sources(manifest, corpus)
    output.mkdir(parents=True)
    for name, raw in artifacts.items():
        path = safe_path(output, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    # Manifest is written last; interrupted output is never a sealed bundle.
    (output / 'manifest.json').write_bytes(encode(updated))
    if seal:
        try:
            validate_revision_bundle(output, updated, corpus=corpus)
        except Exception:
            updated['status'] = 'evidence_revision_validation_failed'
            (output / 'manifest.json').write_bytes(encode(updated))
            raise
    return updated


def validate_revision_bundle(root, manifest=None, *, corpus=None):
    """Rebuild every alternate from sealed audits; used again by publication."""
    root = Path(root)
    disk = read_json(root / 'manifest.json')
    if manifest is not None:
        require({k: v for k, v in manifest.items() if not k.startswith('_')} == disk, 'Revision manifest view differs from sealed bytes')
    manifest = disk
    require(manifest.get('schema_version') == 'quality-v2' and manifest.get('status') == SEALED, 'Evidence revision is not sealed')
    rev = manifest.get('evidence_revision')
    require(isinstance(rev, dict) and rev.get('schema_version') == 'evidence-revision-v1'
            and rev.get('post_hoc_dev_representation_repair') is True and rev.get('automatic_promotion') is False,
            'Missing evidence revision freeze contract')
    require(rev.get('parent_directory') == 'parent-v1', 'Invalid parent snapshot location')
    checked(root / 'parent-v1', 'manifest.json', rev.get('parent_manifest_sha256'))
    parent, datasets, _ = load_parent(root / 'parent-v1')
    changed_metadata = {'status', 'datasets', 'source_checks_sha256', 'summary_sha256', 'independent_review_artifacts'}
    for key, value in parent.items():
        if key not in changed_metadata:
            require(manifest.get(key) == value, 'Revision changed parent metadata: ' + key)
    require(manifest.get('sources') == parent['sources'], 'Revision changed frozen sources')
    require(rev.get('parent_dataset_sha256') == {s: parent['datasets'][s + '.json']['sha256'] for s in ('dev', 'blind')},
            'Revision parent dataset references changed')
    seal_declaration(rev.get('blind_seal_declaration'), rev.get('sealed_at_utc'))
    audits = {'_parent_manifest_sha256': rev['parent_manifest_sha256']}
    for split in ('dev', 'blind'):
        entry = rev.get('audits', {}).get(split, {})
        require(entry.get('path') == split + '-equivalence-audit.json', 'Wrong split audit artifact')
        audits[split] = checked(root, entry['path'], entry.get('sha256'))
        require(timestamp(audits[split].get('reviewed_at_utc'), 'audit time') <= timestamp(rev['sealed_at_utc'], 'seal time'),
                'Audit postdates dataset sealing')
    expected, changes, unresolved = apply_audits(parent, datasets, audits, seal=True)
    require(not unresolved, 'Sealed revision has unresolved review')
    for split in ('dev', 'blind'):
        entry = manifest.get('datasets', {}).get(split + '.json', {})
        actual = checked(root, split + '.json', entry.get('sha256'))
        require(actual == expected[split] and entry.get('count') == len(expected[split]),
                'Revised cases do not exactly preserve parent fields and reviewed alternatives')
    require(checked(root, 'change-audit.json', rev.get('change_audit_sha256')) == changes, 'Change audit mismatch')
    summary = checked(root, 'summary.json', manifest.get('summary_sha256'))
    require(summary.get('status') == SEALED and summary.get('unresolved') == []
            and summary.get('groups_reviewed') == len(changes)
            and summary.get('alternatives_added') == sum(c['alternatives_added'] for c in changes), 'Revision summary mismatch')
    sources = checked(root, 'source-checks.json', manifest.get('source_checks_sha256'))
    expected_hashes = {n: s.get('hash') or s.get('sha256') for n, s in parent['sources'].items()}
    require(sources.get('method') == 'full_sha256_original_file_bytes' and sources.get('sources') == expected_hashes,
            'Revision full source-hash checks missing')
    if corpus is not None:
        check_sources(parent, corpus)
    return {'valid': True, 'alternatives_added': summary['alternatives_added'],
            'semantic_equivalence_machine_proved': False,
            'reviewer_ids_by_split': {split: [reviewer(audits[split])] for split in ('dev', 'blind')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--dev-audit', type=Path)
    parser.add_argument('--blind-audit', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--check-split', choices=('dev', 'blind'),
                        help='Read-only structural/source-hash check of just this split; never seals')
    parser.add_argument('--before-blind-declaration', type=Path,
                        help='JSON with declared_by, declared_at_utc, blind_model_runs_started:false')
    args = parser.parse_args()
    if args.check_split:
        if args.seal or args.output or args.before_blind_declaration:
            parser.error('--check-split cannot seal or write a revision')
        audit = getattr(args, args.check_split + '_audit')
        if audit is None:
            parser.error('The selected split audit is required')
        print(json.dumps(validate_split_audit(args.parent, args.check_split, audit, args.corpus), ensure_ascii=False))
        return
    if not (args.dev_audit and args.blind_audit and args.output):
        parser.error('Both split audits and a new output directory are required')
    result = freeze_revision(args.parent, {'dev': args.dev_audit, 'blind': args.blind_audit},
                             args.output, args.corpus, seal=args.seal,
                             declaration=read_json(args.before_blind_declaration) if args.before_blind_declaration else None)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()),
                      'automatic_promotion': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
