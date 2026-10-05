"""Read-only dev evidence coverage in the entire active index, never Recall@K.

No search, ranking, embedding or answer model runs. Parent text is counted only
when parent_source_spans map its actual characters to the expected location.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragapp.evaluation import evidence_groups, load_dataset, located, normalized, span_matches
from ragapp.evidence_text import canonicalize_display


def _path(value):
    return str(value).replace('\\', '/')


def _pieces(expected, item):
    """Only text actually mapped to a matching physical location may be used."""
    if _path(expected['path']) != _path(item['path']):
        return []
    text, loc = item['text'], item['locator']
    spans = loc.get('source_spans') or [loc]
    result = []
    for span in spans:
        if not located(expected, {'path': item['path'], 'locator': span}):
            continue
        start, end = span.get('chunk_start', 0), span.get('chunk_end', len(text))
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(text):
            continue
        result.append((start, end, text[start:end], span))
    return sorted(result, key=lambda piece: (piece[0], piece[1]))


def _mapped_match(expected, item):
    """Display-adjacent mapped text, without proving original-source adjacency.

    The strict scorer already accepts explicitly contiguous physical source spans.
    This wider diagnostic can explain missing adjacency metadata; it cannot turn
    unknown physical joins into verified coverage. Neither accepts a wrong page.
    """
    quote = normalized(expected.get('span_text') or expected.get('quote') or '')
    if not quote:
        return False
    text = item['text']
    merged, last = '', None
    for start, end, piece, _ in _pieces(expected, item):
        if last is None or start > last and text[last:start].strip():
            merged = normalized(piece)
        elif start >= last:
            merged += normalized(piece)
        else:
            merged += normalized(text[max(start, last):end])
        last = max(last or 0, end)
        if quote in merged:
            return True
    return False


def _display_match(expected, item):
    """PDF representation sensitivity only, never a replacement for span recall.

    Inspect one already mapped physical piece at a time. Preserve mathematical
    symbols, cell boundaries, case and all actual text; unknown joins stay unknown.
    Code and other source types retain their literal scoring contract.
    """
    if Path(expected['path']).suffix.lower() != '.pdf':
        return False
    quote = canonicalize_display(expected.get('span_text') or expected.get('quote') or '').text
    if not quote:
        return False
    for start, end, piece, span in _pieces(expected, item):
        display = canonicalize_display(piece, locator=span)
        display.require_full_coverage([(0, end - start)])
        if quote in display.text:
            return True
    return False


def _order(item, span):
    loc = item['locator']
    values = [span.get(key, loc.get(key, 0)) for key in
              ('page', 'paragraph', 'row', 'line_start', 'block_order', 'block', 'char_start', 'token_start', 'order')]
    return tuple((0, value) if isinstance(value, (int, float)) else (1, str(value)) for value in values)


def _stitch(fragments):
    """Diagnostic reconstruction in physical/token order; not a retrievable item."""
    joined = ''
    for text in fragments:
        if not text or text in joined:
            continue
        overlap = 0
        for size in range(min(len(joined), len(text)), 0, -1):
            if joined.endswith(text[:size]):
                overlap = size
                break
        joined += text[overlap:]
    return joined


def _inspect(expected, document, children, parents, unmapped_parents):
    quote = normalized(expected.get('span_text') or expected.get('quote') or '')
    base = {'path': _path(expected['path']), 'locator': expected.get('locator', {}),
            'child_span_covered': False, 'parent_span_covered': False,
            'mapped_child_covered': False, 'mapped_parent_covered': False}
    if not document:
        return {**base, 'reason': 'source_missing_from_catalog'}
    if document['excluded'] or document['status'] not in {'ready', 'update_failed'}:
        return {**base, 'reason': 'source_not_active', 'document_status': document['status']}
    if expected.get('source_hash') and expected['source_hash'] != document['hash']:
        return {**base, 'reason': 'source_hash_mismatch'}
    if not quote:
        return {**base, 'reason': 'no_text_gold_span'}
    child_hits = [x['id'] for x in children if span_matches(expected, x)]
    parent_hits = [x['id'] for x in parents if span_matches(expected, x)]
    mapped_child = [x['id'] for x in children if _mapped_match(expected, x)]
    mapped_parent = [x['id'] for x in parents if _mapped_match(expected, x)]
    result = {**base, 'child_span_covered': bool(child_hits), 'parent_span_covered': bool(parent_hits),
              'mapped_child_covered': bool(mapped_child), 'mapped_parent_covered': bool(mapped_parent),
              'child_ids': child_hits[:5], 'parent_ids': parent_hits[:5],
              'mapped_child_ids': mapped_child[:5], 'mapped_parent_ids': mapped_parent[:5]}
    matching = [x for x in children + parents if located(expected, x)]
    result['pdf_display_single_span_text_present'] = (
        any(_display_match(expected, item) for item in matching)
        if Path(expected['path']).suffix.lower() == '.pdf' else None)
    result['matching_location_items'] = len(matching)
    result['quote_present_in_unmapped_parent'] = any(quote in normalized(x) for x in unmapped_parents)
    result['quote_present_elsewhere_in_source'] = any(quote in normalized(x['text']) for x in children + parents)
    if child_hits:
        reason = 'covered_by_child'
    elif parent_hits:
        reason = 'covered_only_by_mapped_parent'
    elif mapped_child or mapped_parent:
        reason = 'mapped_text_found_but_source_adjacency_unverified'
    else:
        fragments = []
        for item in children:
            for _, _, text, span in _pieces(expected, item):
                fragments.append((_order(item, span), normalized(text)))
        combined = _stitch(text for _, text in sorted(fragments, key=lambda x: x[0]))
        split_found = bool(combined) and quote in combined
        result['cross_chunk_reconstruction_contains_quote'] = split_found
        if split_found:
            reason = 'quote_split_across_children_without_usable_parent'
        elif result['quote_present_in_unmapped_parent']:
            reason = 'parent_text_present_but_parent_provenance_missing'
        elif result['quote_present_elsewhere_in_source']:
            reason = 'quote_present_but_physical_mapping_does_not_cover_it'
        elif not children:
            reason = 'active_source_has_no_chunks'
        elif not matching:
            reason = 'physical_location_missing_from_index'
        else:
            reason = 'quote_absent_at_matching_location_possible_extraction_change'
        # Short fragments help debug parsing without dumping all gold answers.
        result['gold_excerpt'] = (expected.get('span_text') or expected.get('quote') or '')[:160]
        if fragments:
            result['indexed_excerpt'] = fragments[0][1][:240]
            characters = set(combined)
            result['gold_characters_present_fraction'] = round(sum(ch in characters for ch in quote) / len(quote), 4)
    result['reason'] = reason
    return result


def _group_result(rows):
    """OR interchangeable representations once for each required evidence group."""
    if not rows:
        raise ValueError('Required evidence group has no alternatives')
    result = {field: any(row[field] for row in rows) for field in (
        'child_span_covered', 'parent_span_covered', 'mapped_child_covered',
        'mapped_parent_covered', 'either_span_covered', 'either_mapped_covered')}
    if result['child_span_covered']:
        reason = 'covered_by_child'
    elif result['parent_span_covered']:
        reason = 'covered_only_by_mapped_parent'
    elif result['either_mapped_covered']:
        reason = 'mapped_text_found_but_source_adjacency_unverified'
    else:
        reasons = {row['reason'] for row in rows}
        reason = next(iter(reasons)) if len(reasons) == 1 else 'no_alternative_covered'
    display = [row['pdf_display_single_span_text_present'] for row in rows
               if row.get('pdf_display_single_span_text_present') is not None]
    return {**result, 'case_id': rows[0]['case_id'], 'group': rows[0]['group'],
            'sources': sorted({row['path'] for row in rows}),
            'types': sorted({row['type'] for row in rows}), 'reason': reason,
            'alternative_count': len(rows),
            'covered_alternative_indices': [row['alternative_index'] for row in rows if row['either_span_covered']],
            'alternative_reasons': dict(Counter(row['reason'] for row in rows)),
            'pdf_display_single_span_text_present': any(display) if display else None}


def _summary(rows, *, unit='required_evidence_group'):
    total = len(rows)
    def rate(field):
        numerator = sum(bool(x.get(field)) for x in rows)
        return {'covered': numerator, 'total': total, 'fraction': numerator / total if total else None}
    return {'counting_unit': unit, 'total': total,
            'required_groups' if unit == 'required_evidence_group' else 'alternative_representations': total,
            # Historical consumers used this count. Its unit is now explicit;
            # primary summaries count groups, not individual alternative spans.
            'evidence_references': total, 'child_span': rate('child_span_covered'),
            'child_or_parent_span': rate('either_span_covered'),
            'mapped_child_or_parent': rate('either_mapped_covered'),
            'reasons': dict(Counter(x['reason'] for x in rows)),
            'pdf_display_sensitivity_not_recall': {
                'text_present': sum(x.get('pdf_display_single_span_text_present') is True for x in rows),
                'pdf_references_checked': sum(x.get('pdf_display_single_span_text_present') is not None for x in rows),
                'counting_unit': unit,
                'meaning': 'Typographic display equivalence in one physical span; excludes unknown table/cross-span joins; never used for acceptance.'}}


def audit(catalog, cases_path):
    cases_path = Path(cases_path)
    if 'blind' in cases_path.name.casefold():
        raise ValueError('Blind cases are prohibited by this development diagnostic')
    cases, dataset_hash, manifest = load_dataset(cases_path)
    if not manifest or cases_path.name not in manifest.get('datasets', {}):
        raise ValueError('A frozen dataset manifest and matching dataset hash are required')
    if any(c.get('split') != 'dev' for c in cases):
        raise ValueError('Only dev cases are permitted; no blind report was produced')
    catalog = Path(catalog).resolve()
    wanted = {_path(e['path']) for c in cases for group in evidence_groups(c) for e in group}
    docs, children, parents, unmapped = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    parent_seen = set()
    snapshot = hashlib.sha256()
    active_count = 0
    with sqlite3.connect(catalog.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        for row in db.execute('SELECT id,path,hash,version,status,excluded FROM documents ORDER BY id'):
            doc = dict(row)
            doc['path'] = _path(doc['path'])
            docs[doc['path']] = doc
            snapshot.update(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode())
        query = '''SELECT c.id,c.document_id,c.version,c.text,c.locator,c.parent,d.path
                   FROM chunks c JOIN documents d ON c.document_id=d.id AND c.version=d.version
                   WHERE d.excluded=0 AND d.status IN ('ready','update_failed') ORDER BY c.id'''
        for row in db.execute(query):
            active_count += 1
            item = dict(row)
            snapshot.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode())
            path = item['path'] = _path(item['path'])
            if path not in wanted:
                continue
            item['locator'] = json.loads(item['locator'])
            children[path].append(item)
            parent = item.get('parent')
            if not parent:
                continue
            spans = item['locator'].get('parent_source_spans')
            if not spans:
                unmapped[path].append(parent)
                continue
            key = (path, item['version'], parent, json.dumps(spans, sort_keys=True, ensure_ascii=False))
            if key in parent_seen:
                continue
            parent_seen.add(key)
            parents[path].append({'id': item['locator'].get('parent_id') or item['id'], 'path': path,
                                  'text': parent, 'locator': {'source_spans': spans}})
    evidence_rows, group_rows, case_rows, skipped = [], [], [], []
    for case in cases:
        if not case.get('answerable') or case.get('scorable') is False:
            skipped.append({'id': case['id'], 'reason': 'unanswerable' if not case.get('answerable') else 'not_text_scorable'})
            continue
        groups = []
        for number, alternatives in enumerate(evidence_groups(case)):
            checked = []
            for alternative_index, expected in enumerate(alternatives):
                path = _path(expected['path'])
                entry = _inspect(expected, docs.get(path), children[path], parents[path], unmapped[path])
                entry.update(case_id=case['id'], group=number, alternative_index=alternative_index,
                             diagnostic_unit='alternative_representation', type=Path(path).suffix.lower() or '(none)')
                entry['either_span_covered'] = entry['child_span_covered'] or entry['parent_span_covered']
                entry['either_mapped_covered'] = entry['mapped_child_covered'] or entry['mapped_parent_covered']
                evidence_rows.append(entry)
                checked.append(entry)
            group = _group_result(checked)
            group_rows.append(group)
            groups.append(group)
        case_rows.append({'id': case['id'], 'category': case['category'], 'evidence_groups': groups,
                          **{field: bool(groups) and all(g[field] for g in groups) for field in
                             ('child_span_covered', 'either_span_covered', 'either_mapped_covered')}})
    by_source, by_type = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for row in evidence_rows:
        key = (row['case_id'], row['group'])
        by_source[row['path']][key].append(row)
        by_type[row['type']][key].append(row)
    def grouped_summary(buckets):
        return {name: _summary([_group_result(group) for group in groups.values()])
                for name, groups in sorted(buckets.items())}
    return {'schema_version': 'corpus-coverage-v2', 'metric': 'active_index_full_pool_evidence_coverage',
            'not_recall_at_k': True, 'catalog': str(catalog), 'catalog_snapshot_sha256': snapshot.hexdigest(),
            'dataset_sha256': dataset_hash, 'dataset_manifest_sha256': manifest.get('_manifest_sha256'),
            'active_chunks': active_count, 'active_mapped_parents_for_dev_sources': sum(map(len, parents.values())),
            'normalization': 'NFC, whitespace removed; case and scientific exponents preserved',
            'summary': _summary(group_rows), 'by_source': grouped_summary(by_source),
            'by_type': grouped_summary(by_type),
            'source_type_aggregation': 'Each required group counts once per represented source/type; only alternatives within that bucket can cover it. Cross-source alternative groups can occur in multiple buckets, so bucket totals are not necessarily additive.',
            'alternative_summary': _summary(evidence_rows, unit='alternative_representation'),
            'case_coverage': {field: {'covered': sum(x[field] for x in case_rows), 'total': len(case_rows)} for field in
                              ('child_span_covered', 'either_span_covered', 'either_mapped_covered')},
            'cases': case_rows, 'group_diagnostics': group_rows,
            'evidence_diagnostics': evidence_rows, 'evidence_diagnostics_unit': 'alternative_representation', 'skipped': skipped,
            'missing_case_ids': [x['id'] for x in case_rows if not x['either_span_covered']],
            'limitations': ['Full-pool coverage is an upper-bound diagnostic, not retrieval Recall@K or answer accuracy.',
                           'Parent expansion has no budget in this audit; it may be unavailable under a real context budget.',
                           'Strict span coverage uses the evaluator and accepts explicitly contiguous original-source spans within an item.',
                           'The mapped-text diagnostic can join display-adjacent fragments without established original-source adjacency; it is not verified coverage.',
                           'Cross-chunk reconstruction is diagnostic only, not proof a retrievable context contains the quote.',
                           'Missing exact text may reflect extraction changes, token decoding, or incorrect gold; this audit alone cannot identify the cause.',
                           'No blind examples, queries, model calls, index writes or gold edits are used.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=ROOT / '.rag/evaluation/quality-v2/dev.json')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    protected = {args.catalog.resolve(), args.cases.resolve(),
                 (args.cases.parent / 'manifest.json').resolve(), (args.cases.parent / 'source-checks.json').resolve()}
    if args.report.resolve() in protected or args.report.name.casefold() in {'dev.json', 'blind.json'}:
        parser.error('Report must not overwrite a catalog, dataset, manifest or source-check file')
    report = audit(args.catalog, args.cases)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.report), 'summary': report['summary'], 'case_coverage': report['case_coverage']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
