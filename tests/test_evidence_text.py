"""Display sensitivity must preserve scientific meaning and source coverage."""
import unicodedata

import pytest

from ragapp.evidence_text import canonicalize, canonicalize_display, ROW_SEPARATOR


def test_literal_is_default_and_display_only_maps_explicit_print_punctuation():
    source = '定义： “功率”，（上界）；‘说明’！ 是否？'
    literal = canonicalize(source)
    display = canonicalize_display(source)
    assert literal.text == '定义：“功率”，（上界）；‘说明’！是否？'
    assert display.text == '定义:"功率",(上界);\'说明\'!是否?'
    assert len(display.text) == len(display.offsets)
    for index, (start, end) in enumerate(display.offsets):
        assert 0 <= start < end <= len(source)
        assert source[start:end].strip()


@pytest.mark.parametrize('text', ['mW MW', 'x² x2', 'm³/s', '−3 -3', 'A−B A-B', 'ＣＤ CD',
                                  'Ⅳ IV', '① 1', 'H₂O H2O', 'x_i xⁱ', 'a+b b+a', '。 、 …'])
def test_scientific_symbols_case_compatibility_forms_and_math_order_are_preserved(text):
    actual = canonicalize_display(text).text
    assert actual == ''.join(character for character in unicodedata.normalize('NFC', text) if not character.isspace())


def test_nfc_composition_reordering_and_hangul_preserve_original_mapping():
    source = 'e\u0301 x\u0301 a\u0315\u0300 \u1100\u1161\u11a8'
    result = canonicalize_display(source)
    assert result.text == ''.join(c for c in unicodedata.normalize('NFC', source) if not c.isspace())
    assert result.offsets[0] == (0, 2)
    assert result.original_range(0, 1) == (0, 2)
    # Every original non-whitespace character participates in a mapped segment.
    mapped = {i for a, b in result.offsets for i in range(a, b)}
    assert {i for i, c in enumerate(source) if not c.isspace()} <= mapped
    assert result.offsets[-1] == (len(source)-3, len(source))


def test_removing_whitespace_does_not_compose_across_it():
    source = 'a \u0301'
    result = canonicalize_display(source)
    assert result.text == 'a\u0301' and result.text != 'á'
    assert result.offsets == ((0, 1), (2, 3))
    assert result.deleted_ranges == ((1, 2),)


def test_legal_markdown_table_removes_borders_and_divider_but_keeps_cells_and_rows():
    source = '| A | B |\n| :--- | ---: |\n| 12 | 3 |\n| 1 | 23 |'
    result = canonicalize_display(source)
    assert result.text == 'A|B' + ROW_SEPARATOR + '12|3' + ROW_SEPARATOR + '1|23'
    assert result.table_rows == (0, 2, 3)
    assert '123' not in result.text
    assert canonicalize(source).text == '|A|B||:---|---:||12|3||1|23|'
    for i, character in enumerate(result.text):
        a, b = result.offsets[i]
        if character == ROW_SEPARATOR:
            assert source[a:b] == '\n'
        else:
            assert source[a:b] == character


def test_boundaryless_markdown_and_crlf_rows_are_valid_without_collapsing_cells():
    source = 'A | B\r\n--- | ---\r\n12 | 3\r\n'
    result = canonicalize_display(source)
    assert result.text == 'A|B' + ROW_SEPARATOR + '12|3'
    marker = result.text.index(ROW_SEPARATOR)
    a, b = result.offsets[marker]
    assert source[a:b] == '\r'
    assert (a+1, a+2) not in result.offsets


@pytest.mark.parametrize('source', ['| A | B |\n| -- | --- |\n| 1 | 2 |',
                                    '| A | B |\n| --- | --- | --- |\n| 1 | 2 |',
                                    '|x|', '|x| + |y|', 'a | b', 'x |= y', 'a || b'])
def test_untyped_non_table_or_invalid_markdown_never_drops_pipe(source):
    result = canonicalize_display(source)
    assert result.text.count('|') == source.count('|')
    assert result.table_rows == ()


@pytest.mark.parametrize('locator', [{'kind': 'table_row'}, {'type': 'table'}, {'table_id': 't'}, {'table': 1}])
def test_explicit_table_metadata_allows_only_unambiguous_rows(locator):
    assert canonicalize_display('| 12 | 3 |', locator=locator).text == '12|3'
    assert canonicalize_display('|x| + |y|', locator=locator).text == '|x|+|y|'
    assert canonicalize_display('|x|', locator=locator).text == '|x|'
    assert canonicalize_display('12|3', locator=locator).text == '12|3'
    assert canonicalize_display('|12|3|', locator=locator).text == '|12|3|'


def test_explicit_span_type_and_code_override_table_metadata():
    assert canonicalize_display('| A | B |', span_type='table').text == 'A|B'
    source = '| A | B |\n| --- | --- |\n| x | y |'
    actual = canonicalize_display(source, locator={'kind': 'code', 'table_id': 't'}, span_type='table')
    assert actual.text.count('|') == source.count('|') and actual.table_rows == ()


def test_fenced_and_indented_code_are_never_tables_even_inside_typed_block():
    for source in ('```python\n| A | B |\n| --- | --- |\n| x | y |\n```',
                   '~~~\n| A | B |\n| --- | --- |\n~~~',
                   '    | A | B |\n    | --- | --- |\n    | x | y |'):
        result = canonicalize_display(source, span_type='table')
        assert result.text.count('|') == source.count('|')
        assert result.table_rows == ()


def test_escaped_math_and_inline_code_pipes_remain_literal_inside_tables():
    source = '| A | B |\n| --- | --- |\n| $|x|$ | `a|b` |\n| a\\|b | \\(|y|\\) |'
    result = canonicalize_display(source)
    assert '$|x|$|`a|b`' in result.text
    assert 'a\\|b|\\(|y|\\)' in result.text
    assert result.table_rows == (0, 2, 3)
    escaped = source.index('\\|')
    assert (escaped, escaped+1) in result.offsets and (escaped+1, escaped+2) in result.offsets


def test_table_row_width_mismatch_is_not_silently_reinterpreted():
    source = '| A | B |\n| --- | --- |\n| 1 | 2 | 3 |'
    result = canonicalize_display(source)
    assert result.table_rows == (0,)
    assert result.text.endswith('|1|2|3|')


def test_deleted_syntax_and_whitespace_need_real_physical_coverage():
    source = '| A | B |\n| --- | --- |\n| 12 | 3 |'
    result = canonicalize_display(source)
    assert result.is_fully_covered([(0, len(source))])
    split = source.index('| ---')
    assert result.is_fully_covered([(0, split), (split, len(source))])
    # All output-character ranges still omit removed syntax: they are not proof.
    assert not result.is_fully_covered(result.offsets)
    with pytest.raises(ValueError, match='uncovered gap'):
        result.require_full_coverage(result.offsets)
    result.require_full_coverage([(0, len(source))])
    plain = canonicalize_display('a  b')
    assert plain.text == 'ab' and plain.original_range(0, 2) == (0, 4)
    assert not plain.is_fully_covered([(0, 1), (3, 4)])


@pytest.mark.parametrize('ranges', [[(0, 99)], [(-1, 2)], [(True, 2)], [[1]], [(2, 1)]])
def test_invalid_physical_ranges_do_not_become_coverage(ranges):
    assert canonicalize_display('abc').is_fully_covered(ranges) is False


def test_empty_and_all_removed_text_are_still_accountable():
    assert canonicalize('').is_fully_covered([])
    whitespace = canonicalize(' \n')
    assert whitespace.text == '' and whitespace.offsets == ()
    assert not whitespace.is_fully_covered([])
    assert whitespace.is_fully_covered([(0, 2)])
    with pytest.raises(ValueError):
        whitespace.original_range(0, 0)


def test_module_does_not_change_current_literal_evaluation():
    from ragapp.evaluation import normalized
    source = '定义：42（mW）'
    assert normalized(source) == canonicalize(source).text
    assert normalized(source) != canonicalize_display(source).text
