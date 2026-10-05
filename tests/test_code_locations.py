from ragapp.context import code_line_locations


def test_expanded_code_uses_original_lines_not_child_start_or_context_line_count():
    original = '# header\r\n\r\nRATE_2: float = 0.25\r\nSIZE_DAYS = 365\r\n'
    item = {'text': original, 'locator': {'line_start': 4, 'line_end': 4,
            'context_expanded': True, 'source_spans': [{'line_start': 1, 'line_end': 4}]}}
    assert code_line_locations(item, 'RATE_2 and SIZE_DAYS 的代码位置', original) == {'RATE_2': 3, 'SIZE_DAYS': 4}


def test_code_locations_reject_uncovered_absent_and_ambiguous_assignments():
    original = 'SIZE_DAYS = 365\n# SIZE_DAYS = 1\n\nSIZE_DAYS = 180\n'
    item = {'text': original, 'locator': {'line_start': 1, 'line_end': 4}}
    assert code_line_locations(item, 'SIZE_DAYS', original) == {}
    item['locator'] = {'line_start': 4, 'line_end': 4}
    assert code_line_locations(item, 'SIZE_DAYS', original) == {'SIZE_DAYS': 4}
    item['text'] = 'SIZE_DAYS = 365\n'
    assert code_line_locations(item, 'SIZE_DAYS', original) == {}
    item['locator'] = {'line_start': 2, 'line_end': 3}
    assert code_line_locations(item, 'SIZE_DAYS', original) == {}
