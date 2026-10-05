"""Rescore saved dev retrieval responses against a sealed evidence revision.

No search, model, parser or API call. Historical responses/timing are immutable;
the output is a post-hoc representation diagnostic, never a new quality run.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ragapp.evaluation as evaluation
import ragapp.experiments as experiments
from scripts.freeze_evidence_revision import (SEALED, apply_audits, checked, encode,
    file_sha, read_json, require, seal_declaration, timestamp)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def dev_revision(dataset_path):
    """Validate dev lineage without reading blind questions/answers/audit content.

    Full cross-split semantic review remains the seal/publication validator's job.
    Here the immutable manifest binds the other split, but it is not opened.
    """
    dataset_path = Path(dataset_path).resolve()
    require(dataset_path.name == 'dev.json', 'Replay only accepts the sealed dev split')
    cases, dataset_hash, manifest = evaluation.load_dataset(dataset_path)
    require(manifest and manifest.get('status') == SEALED, 'Replay requires a sealed evidence revision')
    rev = manifest.get('evidence_revision') or {}; root = dataset_path.parent
    require(rev.get('schema_version') == 'evidence-revision-v1'
            and rev.get('post_hoc_dev_representation_repair') is True
            and rev.get('automatic_promotion') is False and rev.get('parent_directory') == 'parent-v1',
            'Invalid evidence revision contract')
    seal_declaration(rev.get('blind_seal_declaration'), rev.get('sealed_at_utc'))
    parent = checked(root / 'parent-v1', 'manifest.json', rev.get('parent_manifest_sha256'))
    require(parent.get('schema_version') == 'quality-v2' and parent.get('status') in
            {'frozen', 'agent_source_reviewed_human_pending'} and not parent.get('evidence_revision'),
            'Replay parent must be frozen original v1')
    entry = parent.get('datasets', {}).get('dev.json', {})
    original = checked(root / 'parent-v1', 'dev.json', entry.get('sha256'))
    require(rev.get('parent_dataset_sha256', {}).get('dev') == entry.get('sha256'), 'Parent dev reference mismatch')
    audit_entry = rev.get('audits', {}).get('dev', {})
    require(audit_entry.get('path') == 'dev-equivalence-audit.json', 'Wrong dev audit artifact')
    audit = checked(root, audit_entry['path'], audit_entry.get('sha256'))
    require(timestamp(audit.get('reviewed_at_utc'), 'audit time') <= timestamp(rev.get('sealed_at_utc'), 'seal time'),
            'Dev review postdates sealing')
    expected, changes, unresolved = apply_audits(parent, {'dev': original},
        {'dev': audit, '_parent_manifest_sha256': rev['parent_manifest_sha256']}, seal=False, only_split='dev')
    require(not unresolved and cases == expected['dev'], 'Dev revision differs from complete reviewed alternatives')
    require(all(c.get('split') == 'dev' for c in cases), 'Replay cannot read blind cases')
    require(len(cases) == manifest['datasets']['dev.json'].get('count') == entry.get('count'), 'Dev case count mismatch')
    change_audit = checked(root, 'change-audit.json', rev.get('change_audit_sha256'))
    require([r for r in change_audit if r.get('split') == 'dev'] == changes, 'Dev change audit mismatch')
    summary = checked(root, 'summary.json', manifest.get('summary_sha256'))
    require(summary.get('status') == SEALED and not summary.get('unresolved'), 'Revision summary is not sealed')
    require(manifest.get('sources') == parent.get('sources'), 'Revision changed original source hashes')
    source_checks = checked(root, 'source-checks.json', manifest.get('source_checks_sha256'))
    require(source_checks.get('method') == 'full_sha256_original_file_bytes' and source_checks.get('sources') ==
            {name: source.get('hash') or source.get('sha256') for name, source in parent['sources'].items()},
            'Sealed full source-hash checks missing')
    return cases, original, dataset_hash, manifest


def scoring_files():
    return {str(Path(module.__file__).resolve()): file_sha(module.__file__)
            for module in (evaluation, experiments)}


def score_saved_report(source_path, dataset_path):
    """Return a new small report referencing original response bytes by hash."""
    source_path = Path(source_path).resolve(); dataset_path = Path(dataset_path).resolve()
    source_hash = file_sha(source_path); code = scoring_files()
    cases, originals, dataset_hash, manifest = dev_revision(dataset_path)
    report = read_json(source_path); rev = manifest['evidence_revision']; original_map = {c['id']: c for c in originals}
    by_id = {c['id']: c for c in cases}; source_rows = report.get('results')
    require(report.get('dataset_split') == 'dev' and report.get('dataset_sha256') == rev['parent_dataset_sha256']['dev']
            and report.get('dataset_manifest_sha256') == rev['parent_manifest_sha256'],
            'Saved response report is not from this original dev bundle')
    require(isinstance(source_rows, list) and len(source_rows) == len(cases)
            and len({r.get('id') for r in source_rows}) == len(cases)
            and {r.get('id') for r in source_rows} == set(by_id), 'Saved report has incomplete/duplicate/foreign cases')
    require(report.get('status') == 'provisional_metrics' and report.get('metrics', {}).get('complete') is True,
            'Only completed original retrieval reports can be replayed')
    require(bool(report.get('runtime_fingerprint')) and isinstance(report.get('runtime_manifest'), dict),
            'Original report lacks runtime provenance')
    rows = []; original_rows = []
    for position, row in enumerate(source_rows):
        response = row.get('response')
        require(isinstance(response, dict) and isinstance(response.get('evidence'), list), 'Missing actual saved evidence')
        for key in ('candidates', 'context'):
            require(key not in response or response[key] is None or isinstance(response[key], list),
                    'Invalid saved ' + key)
        previous = experiments.score_retrieval(original_map[row['id']], response)
        require(previous == row.get('score'), 'Original v1 score cannot be reproduced from saved response: ' + row['id'])
        case = by_id[row['id']]
        require(row.get('cluster_id') == case.get('cluster_id', case['id'])
                and row.get('category') == case.get('category'), 'Saved result grouping differs from frozen case')
        score = experiments.score_retrieval(case, response)
        fields = ('id', 'cluster_id', 'sources', 'group_sources', 'category', 'elapsed', 'peak_rss_bytes',
                  'cost_cny_upper_estimate', 'gold_review_status')
        new_row = {k: copy.deepcopy(row[k]) for k in fields if k in row}
        require(type(new_row.get('elapsed')) in (int, float) and new_row['elapsed'] >= 0,
                'Missing historical retrieval timing')
        new_row.update(score=score, original_v1_score=copy.deepcopy(row['score']),
            source_response={'source_result_index': position, 'case_id': row['id'],
                             'response_sha256': object_sha(response),
                             'hash_encoding': 'UTF8 JSON sort_keys=true ensure_ascii=false separators=comma/colon'},
            elapsed_origin='unchanged original retrieval run; replay execution time is not measured here')
        rows.append(new_row)
        original_rows.append({**new_row, 'score': previous})
    metrics = experiments.summarize(rows, len(cases)); original_metrics = experiments.summarize(original_rows, len(cases))
    for name in ('latency_seconds', 'peak_rss_bytes', 'rss_scope', 'cost_cny_upper_estimate'):
        require(metrics[name] == original_metrics[name], 'Replay changed historical performance: ' + name)
        require(original_metrics[name] == report['metrics'][name], 'Original performance summary does not match its records: ' + name)
    require(file_sha(source_path) == source_hash and scoring_files() == code, 'Source report or scorer changed during replay')
    require(file_sha(dataset_path) == dataset_hash and file_sha(dataset_path.parent / 'manifest.json') == manifest['_manifest_sha256'],
            'Revised suite changed during replay')
    return {'schema_version': 'retrieval-evidence-replay-v1', 'status': 'posthoc_diagnostic_only',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset_split': 'dev',
        'dataset_path': str(dataset_path), 'dataset_sha256': dataset_hash,
        'dataset_manifest_sha256': manifest['_manifest_sha256'], 'dataset_case_count': len(cases),
        'post_hoc_dev_representation_repair': True, 'new_model_execution': False,
        'source_report': {'path': str(source_path), 'sha256': source_hash,
                          'dataset_sha256': report['dataset_sha256'],
                          'dataset_manifest_sha256': report['dataset_manifest_sha256'],
                          'runtime_fingerprint': report['runtime_fingerprint'],
                          'configuration': report.get('configuration'), 'status': report['status']},
        'scoring_code_sha256': code, 'replay_script_sha256': file_sha(__file__),
        'original_v1_metrics': copy.deepcopy(report['metrics']), 'evidence_revision_metrics': metrics,
        'results': rows, 'verification': {'v1_scores_reproduced': True, 'dev_lineage_rebuilt_from_review': True,
            'blind_content_read': False, 'cross_split_review_repeated_by_replay': False,
            'source_responses_unchanged': True, 'historical_timing_preserved': True},
        'limitations': ['Post-hoc dev representation repair, not prospective model or retrieval improvement.',
            'Only actual saved evidence/candidates/context are scored; hidden parent text is never substituted.',
            'Missing original candidate/context trace remains unknown; no retrieval or generation is performed.',
            'Historical RSS/timing/cost are copied, not remeasured; replay adds no performance evidence.',
            'Dev lineage is rebuilt here; complete cross-split freeze and independent answer audit remain separate publication gates.']}


def replay_reports(report_paths, dataset_path, output_directory):
    output_directory = Path(output_directory).resolve()
    paths = [Path(p).resolve() for p in report_paths]
    require(paths and len(paths) == len(set(paths)) and len({p.stem for p in paths}) == len(paths), 'Duplicate/empty source reports')
    require(not output_directory.exists() and not any(p.is_relative_to(output_directory) for p in paths)
            and not Path(dataset_path).resolve().is_relative_to(output_directory),
            'Replay output must be a new directory, separate from input files')
    output_directory.mkdir(parents=True)
    index = []
    for path in paths:
        report = score_saved_report(path, dataset_path)
        target = output_directory / (path.stem + '-evidence-v2.json')
        target.write_bytes(encode(report))
        index.append({'source_report': str(path), 'source_sha256': report['source_report']['sha256'],
                      'report': target.name, 'sha256': file_sha(target),
                      'v1_span10': report['original_v1_metrics']['span_recall_at_10'],
                      'evidence_v2_span10': report['evidence_revision_metrics']['span_recall_at_10']})
    (output_directory / 'replay-manifest.json').write_bytes(encode({'status': 'posthoc_diagnostic_only',
        'reports': index, 'new_model_execution': False, 'source_reports_modified': False}))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = replay_reports(args.reports, args.dataset, args.output)
    print(json.dumps({'reports': len(result), 'output': str(args.output.resolve()),
                      'status': 'posthoc_diagnostic_only'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
