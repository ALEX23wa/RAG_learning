"""Conservative evidence *representation* normalization, with original offsets.

This module is intentionally independent of evaluation/acceptance. Literal mode
is the default; display mode is for a separately reported sensitivity analysis.
Neither normalization nor an offset map proves that a source span was retrieved.
Call ``require_full_coverage`` on independently validated original ranges before
using normalized matches: ALL original characters, including removed whitespace
and table syntax, must be covered. Unknown gaps may never be bridged by the map.
"""
from dataclasses import dataclass
import re
import unicodedata


# Explicit typography only. No NFKC, case folding, numeric/unit conversion,
# minus/dash replacement, superscript flattening, or mathematical reordering.
PRINT_PUNCTUATION = str.maketrans({'：': ':', '，': ',', '；': ';', '（': '(', '）': ')',
                                 '“': '"', '”': '"', '‘': "'", '’': "'",
                                 '！': '!', '？': '?'})
ROW_SEPARATOR = '\u241e'  # Visible record separator, mapped to an original newline.
_TABLE_KINDS = {'table', 'table_row', 'table_schema', 'table_cell'}
_CODE_KINDS = {'code', 'code_block', 'source_code'}


@dataclass(frozen=True)
class CanonicalText:
    text: str
    # One half-open original Unicode-character range for EACH output character.
    # A composed/reordered NFC segment conservatively maps to its whole segment.
    offsets: tuple[tuple[int, int], ...]
    source_length: int
    deleted_ranges: tuple[tuple[int, int], ...]
    mode: str
    table_rows: tuple[int, ...] = ()  # Zero-based original line numbers.

    def original_range(self, start, end):
        """Enclosing original range for a nonempty canonical substring.

        This is a navigation aid, NOT proof of coverage. Use require_full_coverage
        as well; deleted characters outside this envelope still need coverage.
        """
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.text):
            raise ValueError('Expected a nonempty canonical character range')
        ranges = self.offsets[start:end]
        return min(value[0] for value in ranges), max(value[1] for value in ranges)

    def is_fully_covered(self, physical_ranges):
        """Require validated input-relative ranges to cover every source character.

        This checks only character coverage; caller must separately establish
        document identity/version and the validity of each physical locator.
        """
        ranges = []
        for value in physical_ranges:
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(type(number) is not int for number in value)
                    or not 0 <= value[0] <= value[1] <= self.source_length):
                return False
            ranges.append(tuple(value))
        cursor = 0
        for start, end in sorted(ranges):
            if start > cursor:
                return False
            cursor = max(cursor, end)
        return cursor == self.source_length

    def require_full_coverage(self, physical_ranges):
        if not self.is_fully_covered(physical_ranges):
            raise ValueError('Original characters, including deleted syntax/whitespace, have an uncovered gap')


def _escaped(text, index):
    backslashes = 0
    while index > 0 and text[index - 1] == '\\':
        index -= 1; backslashes += 1
    return backslashes % 2 == 1


def _protected_characters(text):
    r"""Protect balanced inline code and explicit $math$ / \(...\) from table parsing."""
    protected = set()
    expressions = (r'(`+)([^\n]*?)\1', r'(?<!\\)\$\$[^\n]*?(?<!\\)\$\$',
                   r'(?<![\\$])\$(?!\$)[^\n]*?(?<!\\)\$', r'\\\([^\n]*?\\\)', r'\\\[[^\n]*?\\\]')
    for pattern in expressions:
        for match in re.finditer(pattern, text):
            protected.update(range(match.start(), match.end()))
    return protected


def _row(line, *, strict_spacing=False):
    """Parse at least two cells; escaped/code/math pipes remain literal content."""
    body = line.rstrip('\r\n')
    protected = _protected_characters(body)
    pipes = [i for i, character in enumerate(body)
             if character == '|' and i not in protected and not _escaped(body, i)]
    if not pipes:
        return None
    left = pipes[0] if not body[:pipes[0]].strip() else None
    right = pipes[-1] if not body[pipes[-1]+1:].strip() else None
    borders = {index for index in (left, right) if index is not None}
    inner = [index for index in pipes if index not in borders]
    if not inner:
        return None  # In particular, |x| alone is NOT a table.
    if strict_spacing and any(not (i > 0 and i + 1 < len(body)
                                  and body[i-1].isspace() and body[i+1].isspace()) for i in inner):
        return None  # Untyped math-like |x|+|y| is too ambiguous even inside a table block.
    begin, finish = left + 1 if left is not None else 0, right if right is not None else len(body)
    boundaries = [begin] + [index + 1 for index in inner]
    ends = inner + [finish]
    cells = [body[a:b] for a, b in zip(boundaries, ends)]
    return {'borders': borders, 'cells': cells, 'inner': inner}


def _table_edits(text, locator, span_type):
    """Recognize complete Markdown tables or explicitly typed, unambiguous rows."""
    kinds = {str(locator.get(key, '')).lower() for key in ('type', 'kind')}
    if span_type:
        kinds.add(str(span_type).lower())
    if kinds & _CODE_KINDS:
        return set(), {}, ()
    explicit = bool(kinds & _TABLE_KINDS or locator.get('table_id') or 'table' in locator)
    lines = text.splitlines(keepends=True)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset); offset += len(line)
    # Fences are never treated as Markdown table syntax, even with table metadata.
    eligible, fence = [], None
    for line in lines:
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            eligible.append(False)
        else:
            eligible.append(fence is None and not line.startswith(('    ', '\t')))
    rows, dividers = {}, set()
    index = 0
    while index + 1 < len(lines):
        header = _row(lines[index]) if eligible[index] else None
        divider = _row(lines[index+1]) if eligible[index+1] else None
        if (header and divider and len(header['cells']) == len(divider['cells'])
                and all(re.fullmatch(r'\s*:?-{3,}:?\s*', cell) for cell in divider['cells'])):
            rows[index] = header; dividers.add(index+1)
            expected = len(header['cells']); cursor = index + 2
            while cursor < len(lines) and eligible[cursor]:
                row = _row(lines[cursor])
                if not row or len(row['cells']) != expected:
                    break
                rows[cursor] = row; cursor += 1
            index = cursor
        else:
            index += 1
    if explicit:
        for index, line in enumerate(lines):
            if eligible[index] and index not in rows and index not in dividers:
                row = _row(line, strict_spacing=True)
                if row:
                    rows[index] = row
    deleted, replacements = set(), {}
    for index in dividers:
        deleted.update(range(starts[index], starts[index] + len(lines[index])))
    for index, row in rows.items():
        deleted.update(starts[index] + value for value in row['borders'])
        # Internal pipes deliberately remain distinct cell boundaries.
        next_line = index + 1
        while next_line in dividers:
            next_line += 1
        if next_line in rows:
            newline = re.search(r'[\r\n]', lines[index])
            if newline:
                replacements[starts[index] + newline.start()] = ROW_SEPARATOR
    return deleted, replacements, tuple(sorted(rows))


def _nfc_segments(text):
    """Find canonical-combining segments, including Hangul starter composition."""
    if not text:
        return
    start = 0
    for index in range(1, len(text)):
        character = text[index]
        if unicodedata.combining(character):
            continue
        segment = text[start:index]
        if unicodedata.normalize('NFC', segment + character) != (unicodedata.normalize('NFC', segment)
                                                               + unicodedata.normalize('NFC', character)):
            continue
        yield start, index
        start = index
    yield start, len(text)


def canonicalize(text, *, mode='literal', locator=None, span_type=None):
    """Return mapped NFC text with whitespace removed; display mode is opt-in.

    Display mode maps only PRINT_PUNCTUATION and conservatively recognized table
    syntax. Internal pipes and ROW_SEPARATOR preserve cell/row boundaries. A
    plain extraction without known cell boundaries cannot be assumed equivalent
    to a Markdown table. No literal score or acceptance threshold is changed.
    """
    if not isinstance(text, str) or mode not in {'literal', 'display'}:
        raise ValueError('Expected text and literal/display mode')
    locator = {} if locator is None else locator
    if not isinstance(locator, dict):
        raise ValueError('Locator must be a mapping')
    deleted, replacements, rows = _table_edits(text, locator, span_type) if mode == 'display' else (set(), {}, ())
    output, offsets = [], []
    for start, end in _nfc_segments(text):
        original = text[start:end]
        segment = unicodedata.normalize('NFC', original)
        mappings = [(start+i, start+i+1) for i in range(len(segment))] if segment == original else [(start, end)] * len(segment)
        for character, (a, b) in zip(segment, mappings):
            if all(index in deleted for index in range(a, b)):
                continue
            if a in replacements and b == a + 1:
                character = replacements[a]
            elif character.isspace():
                deleted.update(range(a, b))
                continue
            elif mode == 'display':
                character = character.translate(PRINT_PUNCTUATION)
            output.append(character); offsets.append((a, b))
    # The mapping is conservative, so a composed sequence is represented, not
    # listed as deleted merely because NFC has fewer output characters.
    ranges = []
    for index in sorted(deleted):
        if ranges and index == ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], index+1)
        else:
            ranges.append((index, index+1))
    return CanonicalText(''.join(output), tuple(offsets), len(text), tuple(ranges), mode, rows)


def canonicalize_display(text, *, locator=None, span_type=None):
    return canonicalize(text, mode='display', locator=locator, span_type=span_type)
