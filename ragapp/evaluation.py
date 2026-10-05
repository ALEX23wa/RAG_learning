"""Source-grounded scoring; automatic grades never substitute for manual audit."""
import hashlib
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import httpx


def normalized(text):
    # Preserve case, exponents and scientific symbols (mW is not MW).
    return re.sub(r'\s+', '', unicodedata.normalize('NFC', str(text)))


def _same_source(expected, evidence):
    return str(evidence.get('path', '')).replace('\\', '/') == str(expected.get('path', '')).replace('\\', '/')


def _location_matches(expected, actual):
    for key, value in expected.get('locator', {}).items():
        if key == 'line':
            if not actual.get('line_start', math.inf) <= value <= actual.get('line_end', -1):
                return False
        elif key in {'line_start', 'line_end'}:
            lo, hi = expected['locator'].get('line_start', value), expected['locator'].get('line_end', value)
            if actual.get('line_start', math.inf) > hi or actual.get('line_end', -1) < lo:
                return False
        elif actual.get(key) != value:
            return False
    return True


def located(expected, evidence):
    """Match authoritative physical spans, including merged/expanded contexts."""
    if not _same_source(expected, evidence):
        return False
    actual = evidence.get('locator') or {}
    spans = actual.get('source_spans') or [actual]
    return any(_location_matches(expected, span) for span in spans)


def _literal_char_map(text):
    """NFC/non-whitespace characters plus conservative original character ranges."""
    # Keep a combining sequence together, including Hangul starter composition.
    # Mapping a composed character to its entire original sequence prevents a
    # boundary through a combining mark from borrowing another physical span.
    units=[]; start=0; segment=''
    for position, character in enumerate(text):
        joined=segment+character
        if segment and not unicodedata.combining(character) and unicodedata.normalize('NFC',joined)==(
                unicodedata.normalize('NFC',segment)+unicodedata.normalize('NFC',character)):
            units.extend((c,start,position) for c in unicodedata.normalize('NFC',segment) if not c.isspace())
            start=position;segment=character
        else:segment=joined
    units.extend((c,start,len(text)) for c in unicodedata.normalize('NFC',segment) if not c.isspace())
    return ''.join(c for c,_,_ in units),[(a,b) for _,a,b in units]


def _physical_span(span):
    def positive(value):return type(value) is int and value>0
    if any(positive(span.get(k)) for k in ('page','paragraph','row')):return True
    if positive(span.get('line_start')) and positive(span.get('line_end')):
        return span['line_start']<=span['line_end']
    return bool(isinstance(span.get('image'),str) and span['image'])


def _literal_location_matches(expected, span):
    """Every contributing fragment must lie inside the declared gold range."""
    constraints=expected.get('locator') or {}
    for singular,first,last in [('line','line_start','line_end'),('paragraph','paragraph_start','paragraph_end'),
                                ('row','row_start','row_end')]:
        if any(key in constraints for key in (singular,first,last)):
            lo=constraints.get(first,constraints.get(singular));hi=constraints.get(last,constraints.get(singular,lo))
            a=span.get(first,span.get(singular));b=span.get(last,span.get(singular,a))
            if not all(type(v) is int for v in (lo,hi,a,b)) or not lo<=a<=b<=hi:return False
    ranges={'line','line_start','line_end','paragraph','paragraph_start','paragraph_end','row','row_start','row_end'}
    return all(span.get(key)==value for key,value in constraints.items() if key not in ranges)


def _contiguous_source_spans(left,right):
    """Require explicit ordered source adjacency, never page/function alone."""
    for key in ('page','sheet','table','section_path','symbol','function','image'):
        if key in left or key in right:
            if left.get(key)!=right.get(key):return False
    def integer_range(span,prefix):
        a,b=span.get(prefix+'start'),span.get(prefix+'end')
        return (a,b) if type(a) is int and type(b) is int and 0<=a<b else None
    a=integer_range(left,'document_char_');b=integer_range(right,'document_char_')
    if a is not None or b is not None:return a is not None and b is not None and a[1]==b[0]
    a=integer_range(left,'char_');b=integer_range(right,'char_')
    same_block=left.get('block_id') is not None and left.get('block_id')==right.get('block_id')
    same_page_text=(left.get('method')==right.get('method')=='pdf_text' and left.get('page')==right.get('page'))
    same_unit=any(left.get(key) is not None and left.get(key)==right.get(key) for key in ('paragraph','row','block'))
    if same_block or same_page_text or same_unit:
        return a is not None and b is not None and a[1]==b[0]
    # Numeric units establish adjacency at the parser's physical granularity.
    # Source offsets, when present, may not start halfway through the next unit.
    if b is not None and b[0]!=0:return False
    for start,end in [('line_start','line_end'),('paragraph','paragraph'),('row','row'),('block','block')]:
        lo,hi=right.get(start),left.get(end)
        if type(lo) is int and type(hi) is int and lo==hi+1:
            return start!='block' or left.get('page') is not None
    return False


def _mapped_literal_interval(expected,evidence,text,spans,start,end):
    contributing=[]
    for span in spans:
        if not isinstance(span,dict):return False
        a,b=span.get('chunk_start'),span.get('chunk_end')
        if type(a) is not int or type(b) is not int or not 0<=a<b<=len(text):return False
        if b<=start or a>=end:continue
        source=span.get('path',evidence['path'])
        if (not _physical_span(span) or not isinstance(source,str) or source.replace('\\','/')!=evidence['path'].replace('\\','/')
                or span.get('version',evidence.get('version'))!=evidence.get('version')):return False
        fragment=dict(span)
        if 'line_start' in fragment:
            if type(span['line_start']) is not int:return False
            fragment['line_start']=span['line_start']+text[a:max(a,start)].count('\n')
            fragment['line_end']=span['line_start']+text[a:max(a,min(b,end)-1)].count('\n')
        if not _literal_location_matches(expected,fragment):return False
        contributing.append(span)
    contributing.sort(key=lambda s:s['chunk_start'])
    cursor=start;previous=None
    for span in contributing:
        a,b=max(start,span['chunk_start']),min(end,span['chunk_end'])
        if a<cursor or text[cursor:a].strip():return False
        if previous is not None and not _contiguous_source_spans(previous,span):return False
        cursor=b;previous=span
    return bool(contributing) and not text[cursor:end].strip()


def span_matches(expected, evidence):
    """Match literal visible evidence and its complete, ordered physical mapping.

    Only NFC and whitespace normalization are allowed. A quote may cross mapped
    spans if each contributing location matches the gold and its source units are
    explicitly adjacent; non-whitespace gaps, overlaps and unknown joins fail.
    """
    quote = expected.get('span_text') or expected.get('quote')
    if not quote or not _same_source(expected, evidence):return False
    text=evidence.get('text','');location=evidence.get('locator') or {}
    wanted=normalized(quote)
    if not isinstance(text,str) or not wanted:return False
    if 'source_spans' not in location:
        if location.get('context_expanded') or location.get('parent_source_spans'):return False
        return _location_matches(expected,location) and wanted in normalized(text)
    spans=location['source_spans']
    if not isinstance(spans,list) or not spans:return False
    if wanted not in normalized(text):return False
    visible,positions=_literal_char_map(text)
    position=visible.find(wanted)
    while position>=0:
        start=min(a for a,b in positions[position:position+len(wanted)])
        end=max(b for a,b in positions[position:position+len(wanted)])
        if _mapped_literal_interval(expected,evidence,text,spans,start,end):return True
        position=visible.find(wanted,position+1)
    return False


def evidence_groups(case):
    """All groups are required; alternatives inside a group are interchangeable."""
    return case.get('evidence_groups', [[item] for item in case.get('evidence', [])])


def original_cell_matches(expected, evidence):
    """Match an original cell to a read-only calculator result, never an aggregate.

    This is a separate citation-support representation, not literal Recall@K.
    A broad gold row range may contain the exact requested cell. Both physical
    ranges and the complete cell value must agree; row/page overlap is not enough.
    """
    if expected.get('extraction') != 'original_xlsx_values' or not _same_source(expected,evidence):return False
    quote=expected.get('span_text')
    if not isinstance(quote,str):return False
    match=re.fullmatch(r'([A-Z]{1,3}[1-9][0-9]*)=([^\r\n]+)',quote)
    if not match:return False
    cell,value=match.groups();loc=evidence.get('locator') or {};gold=expected.get('locator') or {}
    if loc.get('kind')!='calculation' or loc.get('operation')!='header_lookup':return False
    if not gold.get('sheet') or loc.get('sheet')!=gold['sheet']:return False
    # Only the declared sheet/range format is supported, not ignored constraints.
    if set(gold)-{'sheet','range'}:return False
    def coordinate(text):
        m=re.fullmatch(r'([A-Z]{1,3})([1-9][0-9]*)',text or '')
        if not m:return None
        column=0
        for char in m[1]:column=column*26+ord(char)-64
        row=int(m[2])
        return (column,row) if column<=16384 and row<=1048576 else None
    point=coordinate(cell)
    def contains(area):
        if not isinstance(area,str):return False
        ends=area.split(':')
        if len(ends)==1:ends*=2
        if len(ends)!=2:return False
        lo,hi=map(coordinate,ends)
        return bool(point and lo and hi and lo[0]<=point[0]<=hi[0] and lo[1]<=point[1]<=hi[1])
    if not contains(gold.get('range')) or loc.get('range')!=f'{cell}:{cell}':return False
    text=evidence.get('text','');prefix='[Deterministic calculation from original table]\n'
    if not isinstance(text,str) or not text.startswith(prefix):return False
    try:data=json.loads(text[len(prefix):])
    except (ValueError,TypeError):return False
    if not isinstance(data,dict):return False
    # Do not accept detached metadata that disagrees with the visible result.
    for key in ('operation','sheet','range','cells','header_row','row_start','row_end'):
        if key not in data or data[key]!=loc.get(key):return False
    cells=data.get('cells')
    if not isinstance(cells,dict) or set(cells)!={cell} or not isinstance(cells[cell],dict):return False
    actual=cells[cell].get('value')
    if not isinstance(actual,str) or actual!=data.get('value') or actual!=data.get('column'):return False
    if not point or any(type(data.get(k)) is not int or data[k]!=point[1]
                        for k in ('header_row','row_start','row_end')):return False
    return normalized(actual)==normalized(value)


def recalled_groups(case, evidence, matcher=span_matches):
    return [any(matcher(expected, item) for expected in group for item in evidence)
            for group in evidence_groups(case)]


_NUMBER = re.compile(r'(?<![A-Za-z0-9_.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![0-9_.])')


def scoring_text(text):
    """Citation identifiers are metadata, never answer facts or measurements."""
    text = re.sub(r'[\[【]\s*S\d+(?:\s*[,，;；、]\s*S?\d+)*\s*[\]】]', '', text, flags=re.I)
    return re.sub(r'(?<![A-Za-z0-9_])S\d+(?![A-Za-z0-9_])', '', text, flags=re.I).replace('−', '-')


def fact_matches(pattern, text):
    """A regex match may not claim a numeric substring or source identifier."""
    numbers = list(_NUMBER.finditer(text))
    for match in re.finditer(pattern, text, re.I):
        if any(max(match.start(), number.start()) < min(match.end(), number.end())
               and not (match.start() <= number.start() and match.end() >= number.end()) for number in numbers):
            continue
        if (match.group() and match.group()[0].isdigit() and match.start() > 0
                and re.match(r'[A-Za-z0-9_.]', text[match.start()-1])):
            continue
        if (match.group() and match.group()[-1].isdigit() and match.end() < len(text)
                and re.match(r'[0-9_.]', text[match.end()])):
            continue
        return True
    return False


def numeric_grade(targets, text):
    """Match declared values and optional exact units/conditions, without inference.

    Legacy gold with value/tolerance only keeps a clearly labelled value-only
    check. Missing constraints are unknown, never a numeric success. Conditions
    must appear in the same punctuation-bounded clause as the matched value.
    """
    if not targets:
        return None, 'not_evaluated'
    numbers = list(_NUMBER.finditer(text))
    outcomes = []
    for target in targets:
        expected, tolerance = target.get('value'), target.get('tolerance', 0)
        if (type(expected) not in (int, float) or type(tolerance) not in (int, float)
                or not math.isfinite(expected) or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError('Numeric gold needs a finite value and nonnegative tolerance')
        unit = target.get('unit')
        conditions = target.get('conditions', [])
        if isinstance(conditions, str):
            conditions = [conditions]
        if (unit is not None and (not isinstance(unit, str) or not unit.strip())
                or not isinstance(conditions, list) or not all(isinstance(value, str) and value for value in conditions)):
            raise ValueError('Numeric units and conditions must be explicit nonempty text')
        matches = False
        for number in numbers:
            value = float(number.group().replace(',', ''))
            if not math.isfinite(value) or abs(value - expected) > tolerance:
                continue
            if unit and not re.match(r'\s*' + re.escape(unit) + r'(?![A-Za-z0-9_/%²³^])', text[number.end():]):
                continue
            left = max((match.end() for match in re.finditer(r'[。！？;；\n]|(?<!\d)\.(?!\d)', text[:number.start()])), default=0)
            end = re.search(r'[。！？;；\n]|(?<!\d)\.(?!\d)', text[number.end():])
            right = number.end() + end.start() if end else len(text)
            if all(normalized(condition) in normalized(text[left:right]) for condition in conditions):
                matches = True
                break
        outcomes.append(matches)
    return all(outcomes), 'value_unit_conditions' if all(target.get('unit') for target in targets) else 'value_only_or_partial_units'


def grade(case, answer):
    """Require correct cited sources; refusal words alone are not successful refusals."""
    text, citations, evidence = answer.get('answer', ''), answer.get('citations', []), answer.get('evidence', [])
    factual_text = scoring_text(text)
    found = [any(fact_matches(p, factual_text) for p in alternatives) for alternatives in case.get('facts', [])]
    contradictions = [p for p in case.get('forbidden', []) if fact_matches(p, factual_text)]
    legacy = recalled_groups(case, evidence[:10], located)
    groups = evidence_groups(case)
    has_spans = bool(groups) and all(any(e.get('span_text') or e.get('quote') for e in g) for g in groups)
    spans = recalled_groups(case, evidence[:10]) if has_spans else None
    literal_citation_support = recalled_groups(case, citations, span_matches if has_spans else located)
    cell_support = recalled_groups(case, citations, original_cell_matches)
    citation_support = [literal or cell for literal,cell in zip(literal_citation_support,cell_support)]
    numeric, numeric_status = numeric_grade(case.get('numeric'), factual_text)
    by_id = {e.get('id'): e for e in evidence if e.get('id')}
    citation_valid = bool(citations) and all(c.get('id') in by_id and c.get('locator') and c.get('version')
        and c.get('path') == by_id[c['id']].get('path') and c.get('version') == by_id[c['id']].get('version') for c in citations)
    refused = answer.get('status') == 'abstained' and not citations
    if case['answerable']:
        passed = (bool(found) and all(found) and numeric is not False and not contradictions and citation_valid
                  and bool(citation_support) and all(citation_support) and not refused)
    else:
        passed = refused and not contradictions
    return {'pass': bool(passed), 'facts_matched': found, 'contradictions': contradictions,
        'evidence_recalled': legacy, 'legacy_locator_recalled_at_10': legacy, 'span_recalled_at_10': spans,
        'citation_gold_support': citation_support, 'citation_literal_gold_support': literal_citation_support,
        'citation_original_cell_support': cell_support, 'scoring_schema': 'literal-plus-original-cell-v1',
        'numeric_pass': numeric, 'numeric_evaluation': numeric_status,
        'citation_structurally_valid': bool(citation_valid), 'explicit_abstention': refused,
        'manual_grounding_review': 'pending', 'gold_review_status': case.get('review_status', 'legacy_unrecorded')}


def runtime_fingerprint(manifest):
    explicit = manifest.get('runtime_fingerprint') or manifest.get('fingerprint')
    if explicit:
        return str(explicit)
    stable = {k: v for k, v in manifest.items() if k not in {'created', 'time', 'timestamp', 'uptime'}}
    return hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def verify_runtime(report, manifest, dataset_manifest=None):
    """Freeze the runtime per run; original-source hashes must match frozen gold."""
    fingerprint = runtime_fingerprint(manifest)
    if report.get('runtime_fingerprint') not in (None, fingerprint):
        raise ValueError('Runtime manifest changed; start a separate run')
    inventory = manifest.get('source_inventory')
    if inventory is not None and inventory.get('status') != 'available':
        raise ValueError('Runtime source inventory is unavailable or unreadable')
    documents = manifest.get('documents') or {}
    if isinstance(documents, list):
        documents = {item['path']: item for item in documents}
    observed = manifest.get('observed_sources')
    if observed is not None:
        for name, document in documents.items():
            if document.get('excluded') or document.get('status') not in {'ready', 'update_failed'}:
                continue
            actual = observed.get(name.replace('\\', '/')) or {}
            if actual.get('status') != 'available' or actual.get('hash') != document.get('hash'):
                raise ValueError('Original source changed or unavailable; sync/rebuild before evaluation: ' + name)
    for source in (manifest.get('source_fingerprints') or {}).values():
        if source.get('excluded') or source.get('status') not in {'ready', 'update_failed'}:
            continue
        if ('observed_status' in source and source['observed_status'] != 'available'
                or 'indexed_hash' in source and source.get('hash') != source['indexed_hash']):
            raise ValueError('Original source changed or unavailable; sync/rebuild before evaluation')
    if dataset_manifest:
        expected = dataset_manifest.get('sources') or dataset_manifest.get('documents') or {}
        actual = manifest.get('documents') or manifest.get('sources') or manifest.get('corpus', {}).get('documents') or {}
        if isinstance(actual, list):
            actual = {x['path']: x for x in actual}
        actual = {k.replace('\\', '/'): v for k, v in actual.items()}
        redacted = manifest.get('source_fingerprints') or {}
        for path, source in expected.items():
            normal_path = path.replace('\\', '/')
            row = actual.get(normal_path) or redacted.get(hashlib.sha256(normal_path.encode()).hexdigest())
            if row is None or (row.get('hash') or row.get('sha256')) != (source.get('hash') or source.get('sha256')):
                raise ValueError('Frozen source changed or missing: ' + path)
            if row.get('excluded') or ('status' in row and row['status'] not in {'ready', 'update_failed'}):
                raise ValueError('Frozen source is not available: ' + path)
    report['runtime_fingerprint'], report['runtime_manifest'] = fingerprint, manifest


def load_dataset(cases_path):
    path = Path(cases_path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    cases = payload['cases'] if isinstance(payload, dict) else payload
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    fingerprint = hashlib.sha256(raw).hexdigest()
    if manifest and path.name in manifest.get('datasets', {}):
        expected = manifest['datasets'][path.name]
        if isinstance(expected, dict):
            expected = expected['sha256']
        if expected != fingerprint:
            raise ValueError('Frozen dataset file hash mismatch')
    if manifest and manifest.get('source_checks_sha256'):
        checks = path.parent / 'source-checks.json'
        if not checks.exists() or hashlib.sha256(checks.read_bytes()).hexdigest() != manifest['source_checks_sha256']:
            raise ValueError('Frozen source-check report hash mismatch')
    if manifest:
        manifest['_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    return cases, fingerprint, manifest


def _snapshot_runtime_code(target, report):
    """Archive only a complete code manifest matching these actual source bytes."""
    from filelock import FileLock
    code = (report.get('runtime_manifest') or {}).get('code')
    if not code:
        return  # Test manifests and legacy reports may not contain source hashes.
    if not isinstance(code, dict) or not all(isinstance(name, str) for name in code):
        raise ValueError('Invalid runtime source manifest')
    if any('/' in name or '\\' in name for name in code):
        # The frozen-baseline runner verifies its own path-qualified source tree.
        # Do not pretend the currently imported source is that older snapshot.
        report['runtime_code_snapshot'] = {'status': 'skipped_external_code_map',
            'reason': 'Path-qualified code requires the independently verified frozen-baseline snapshot'}
        return
    source = {path.name: path.read_bytes() for path in Path(__file__).parent.glob('*.py')}
    actual = {name: hashlib.sha256(data).hexdigest() for name, data in source.items()}
    if code != actual:
        raise ValueError('Runtime source manifest differs from current complete ragapp source; cannot archive this run')
    digest = hashlib.sha256(json.dumps(code, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode()).hexdigest()
    relative = Path('runtime-snapshots') / digest
    snapshot = target.parent / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 'ragapp-source-snapshot-v1', 'code_digest': digest,
                'files': {'ragapp/' + name: value for name, value in sorted(code.items())}}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')
    # Sibling reports may share a snapshot. The lock prevents observing a partial
    # first copy; existing archived bytes are verified, never silently replaced.
    with FileLock(str(snapshot.parent / (digest + '.lock')), timeout=5):
        (snapshot / 'ragapp').mkdir(parents=True, exist_ok=True)
        for name, data in source.items():
            archived = snapshot / 'ragapp' / name
            if not archived.exists():
                archived.write_bytes(data)
            if hashlib.sha256(archived.read_bytes()).hexdigest() != code[name]:
                raise ValueError('Archived runtime source hash mismatch: ' + name)
        manifest_path = snapshot / 'manifest.json'
        if not manifest_path.exists():
            manifest_path.write_bytes(manifest_bytes)
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError('Archived runtime snapshot manifest changed')
    report['runtime_code_snapshot'] = {'status': 'verified', 'code_digest': digest,
        'path': relative.as_posix(), 'manifest_path': (relative / 'manifest.json').as_posix(),
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}


def save_report(target, report):
    """Save a recoverable checkpoint despite brief Windows file sharing locks.

    Each write has a unique fsynced temporary JSON. Replacement retries at most
    five times (0.75 seconds total backoff), only for access/sharing/busy errors.
    On failure the previous report and the complete checkpoint both remain; the
    exception gives its path for explicit recovery after the file lock clears.
    A full local runtime code manifest is archived and verified before saving.
    """
    import errno
    import os
    import tempfile
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _snapshot_runtime_code(target, report)
    payload = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    with tempfile.NamedTemporaryFile(mode='wb', dir=target.parent,
            prefix='.' + target.name + '.checkpoint-', suffix='.json', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    waits = (.05, .1, .2, .4)
    for attempt in range(len(waits) + 1):
        try:
            temporary.replace(target)
            return
        except OSError as exc:
            transient = exc.errno in {errno.EACCES, errno.EPERM, errno.EBUSY} or getattr(exc, 'winerror', None) in {5, 32, 33}
            if not transient or attempt == len(waits):
                raise RuntimeError('Report replacement failed; previous report retained. Recover checkpoint: '
                                   + str(temporary.resolve())) from exc
            time.sleep(waits[attempt])


def answer_metrics(cases, rows):
    """Provisional answer metrics shared by online and offline runners."""
    grades = [r['grade'] for r in rows]
    legacy = [x for g in grades for x in g['evidence_recalled']]
    spans = [x for g in grades for x in (g.get('span_recalled_at_10') or [])]
    complete = len(grades) == len(cases)
    by_case = {c['id']: c for c in cases}
    negatives = sum(not by_case[r['id']]['answerable'] for r in rows)
    positives = len(grades) - negatives
    refused = [r for r in rows if r['grade'].get('explicit_abstention')]
    correct_refusals = sum(not by_case[r['id']]['answerable'] for r in refused)
    quality = any(c.get('schema_version') == 'quality-v2' for c in cases)
    recall = spans if quality else legacy
    return {'completed': len(grades), 'total': len(cases), 'passed': sum(g['pass'] for g in grades),
        'accuracy': sum(g['pass'] for g in grades) / len(cases) if cases and complete else None,
        'legacy_locator_recall_at_10': sum(legacy) / len(legacy) if legacy and complete else None,
        'span_recall_at_10': sum(spans) / len(spans) if spans and complete else None,
        'evidence_recall_at_10': sum(recall) / len(recall) if recall and complete else None,
        'abstention_precision': correct_refusals / len(refused) if refused and complete else None,
        'abstention_recall': correct_refusals / negatives if negatives and complete else None,
        'answerable_false_abstention_rate': (len(refused) - correct_refusals) / positives if positives and complete else None,
        'faithfulness': None}


def run(cases_path, report_path, token, url='http://127.0.0.1:8000', *, trace_token=None):
    """Run/resume answers, freezing code/config/index/corpus for every dataset."""
    cases, fingerprint, dataset_manifest = load_dataset(cases_path)
    target = Path(report_path)
    report = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {
        'dataset_sha256': fingerprint, 'results': [], 'status': 'incomplete'}
    if report['dataset_sha256'] != fingerprint:
        raise ValueError('Frozen dataset changed; start a new run')
    if report['results'] and (not report.get('runtime_fingerprint') or not report.get('runtime_manifest')):
        raise ValueError('Existing results have no frozen runtime manifest; start a new run')
    report['dataset_path'] = str(Path(cases_path).resolve())
    if dataset_manifest:
        current = dataset_manifest['_manifest_sha256']
        if report.get('dataset_manifest_sha256') not in (None, current):
            raise ValueError('Frozen dataset manifest changed; start a new run')
        report['dataset_manifest_sha256'] = current
    strict = any(c.get('schema_version') == 'quality-v2' for c in cases)
    splits = {c.get('split') for c in cases}
    report['dataset_split'] = next(iter(splits)) if len(splits) == 1 else 'mixed'
    report['dataset_case_count'] = len(cases)
    completed = {r['id'] for r in report['results']}
    if len(completed) != len(report['results']) or not completed <= {case['id'] for case in cases}:
        raise ValueError('Checkpoint contains duplicate or foreign case IDs')
    with httpx.Client(timeout=300, headers={'Authorization': 'Bearer ' + token}) as client:
        def check_runtime():
            response = client.get(url + '/api/manifest')
            response.raise_for_status()
            verify_runtime(report, response.json(), dataset_manifest)
        check_runtime()
        save_report(target, report)
        for case in cases:
            if case['id'] in completed:
                continue
            started, response = time.perf_counter(), None
            try:
                check_runtime()
                for attempt in range(6):
                    response = client.post(url + '/api/chat', json={'question': case['question'], 'history': case.get('history', [])})
                    if response.status_code != 429 or attempt == 5:
                        break
                    time.sleep(min(60, max(13, float(response.headers.get('Retry-After', '60')))))
                response.raise_for_status()
                answer = response.json()
                if re.search(r'^\s*Sorry, to prevent abuse of free resources,.*(?:recharg|free quota)', answer.get('answer', ''), re.I | re.S):
                    raise RuntimeError('Provider quota message returned as HTTP 200; checkpoint retained')
                if answer.get('trace_id'):
                    trace_response = client.get(url + '/api/traces/' + answer['trace_id'],
                        headers={'Authorization': 'Bearer ' + (trace_token or token)})
                    if trace_response.status_code == 200:
                        answer['evaluation_trace'] = trace_response.json()
                    elif trace_response.status_code in {403, 404}:
                        answer['evaluation_trace_status'] = 'unavailable; candidate/context metrics are not inferred'
                    else:
                        trace_response.raise_for_status()
                check_runtime()
            except Exception as exc:
                report['blocked'] = {'id': case['id'], 'error': str(exc), 'time': time.time(),
                    'provider_detail': response.text[:1200] if response is not None else None}
                break
            from .experiments import score_retrieval
            trace = answer.get('evaluation_trace') or {}
            if trace.get('runtime_manifest'):
                verify_runtime(report, trace['runtime_manifest'], dataset_manifest)
            report['results'].append({'id': case['id'], 'question': case['question'], 'answer': answer,
                'case_metadata': {k: case[k] for k in ('answerable', 'category', 'scorable', 'review_status') if k in case},
                'retrieval_score': score_retrieval(case, {**answer, **trace}),
                'grade': grade(case, answer), 'elapsed': time.perf_counter() - started})
            report.pop('blocked', None)
            save_report(target, report)
    complete = len(report['results']) == len(cases)
    report['metrics'] = answer_metrics(cases, report['results'])
    report['evaluation_schema'] = 'quality-v2' if strict else 'legacy_locator'
    report['status'] = 'awaiting_manual_audit' if complete else 'incomplete'
    save_report(target, report)
    return report['metrics']
