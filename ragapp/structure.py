"""Token-bounded structure chunks with offsets into the extracted source text.

Offsets are Unicode character offsets, never decoded tokenizer output. Physical
locations in source_spans remain authoritative when a chunk contains several rows
or paragraphs. Retrieval decoration is kept separate from citable text.
"""
import bisect
import hashlib
import json
import re


def token_offsets(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(pair) for pair in encoded['offset_mapping'] if pair[1] > pair[0]]


def _section(locator):
    section = locator.get('section_path')
    if section is not None:
        return list(section)
    return [locator['heading']] if locator.get('heading') else []


def _source(item, number):
    location = dict(item.get('locator', {}))
    text = item['text']
    source_start = 0
    if item.get('row_text') is not None:
        source_start = text.find(item['row_text'])
        if source_start < 0:
            raise ValueError('Table row must be a substring of its extracted block')
        text = item['row_text']
    location.setdefault('block_id', 'block-' + str(number))
    return {'text': text, 'locator': location, 'source_start': source_start,
            'image': item.get('image')}


def _key(item):
    loc = item['locator']
    section = tuple(_section(loc))
    if item.get('image') or loc.get('symbol') or loc.get('kind') == 'table_schema':
        return ('isolated', loc['block_id'])
    if loc.get('table_id'):
        return ('table', loc['table_id'], section)
    if 'page' in loc:
        kind = loc.get('type', 'text')
        if kind in {'table', 'equation', 'image', 'chart'}:
            return ('isolated', loc['block_id'])
        # MinerU labels headings separately, but a heading and its same-section
        # prose belong to one bounded parent. Physical spans retain both types.
        family = 'prose' if kind in {'text', 'paragraph', 'paragraph_title', 'title', 'heading'} else kind
        return ('page', loc['page'], section, family)
    return ('prose', section)


def _groups(blocks, tokenizer, maximum):
    """Make bounded parents; never merge separate sections, tables or images."""
    pending, count, key = [], 0, None
    emitted_tables = set()
    for number, raw in enumerate(blocks):
        if not raw.get('text') or not raw['text'].strip():
            continue
        item = _source(raw, number)
        loc = item['locator']
        table_id = loc.get('table_id')
        if table_id and table_id not in emitted_tables and loc.get('table_header'):
            if pending:
                yield pending
                pending, count, key = [], 0, None
            emitted_tables.add(table_id)
            header_loc = dict(loc.get('header_locator', loc))
            header_loc.update(kind='table_schema', table_id=table_id,
                              section_path=_section(loc), block_id=f'table-schema-{table_id}')
            yield [{'text': loc['table_header'], 'locator': header_loc,
                    'source_start': 0, 'image': None}]
        item_key = _key(item)
        size = len(token_offsets(tokenizer, item['text']))
        group_limit=maximum if item_key[0]=='table' else maximum*2
        if pending and (key != item_key or count + size + 2 > group_limit):
            yield pending
            pending, count = [], 0
        if size > maximum * 2:
            if pending:
                yield pending
                pending, count, key = [], 0, None
            for a,b,_,_ in _windows(item['text'],tokenizer,maximum*2,0,loc.get('statement_ends')):
                part={**item,'text':item['text'][a:b],
                      'source_start':item['source_start']+a,'locator':dict(loc)}
                if 'line_start' in loc:
                    part['locator']['line_start']=loc['line_start']+item['text'][:a].count('\n')
                yield [part]
            continue
        pending.append(item)
        count += size + (2 if len(pending)>1 else 0)
        key = item_key
        if item_key[0] == 'isolated':
            yield pending
            pending, count, key = [], 0, None
    if pending:
        yield pending


def _windows(text, tokenizer, maximum, overlap, statement_ends=None, input_fits=None):
    """Prefer sentence/line ends, retaining original whitespace and all content."""
    offsets = token_offsets(tokenizer, text)
    if not offsets:
        return
    starts = [a for a, _ in offsets]
    boundaries = (sorted(set(statement_ends)) if statement_ends else
                  [m.end() for m in re.finditer(r'[。！？!?；;](?:[”’\"\']|\s)*|(?<=[.])\s+|\n', text)])
    begin, char_begin = 0, 0
    while begin < len(offsets):
        end = min(begin + maximum, len(offsets))
        char_end = len(text) if end == len(offsets) else offsets[end - 1][1]
        if end < len(offsets):
            lower = char_begin if statement_ends else offsets[min(end - 1, begin + max(1, maximum // 2))][0]
            candidate = bisect.bisect_right(boundaries, char_end) - 1
            if candidate >= 0 and boundaries[candidate] > lower:
                char_end = boundaries[candidate]
                end = bisect.bisect_left(starts, char_end, lo=begin + 1, hi=end)
        # SentencePiece may add a token when a substring is encoded independently.
        def fits():
            piece=text[char_begin:char_end]
            return len(token_offsets(tokenizer,piece))<=maximum and (input_fits is None or input_fits(piece))
        while not fits() and end > begin + 1:
            end -= 1
            char_end = offsets[end - 1][1]
        if not fits():
            raise ValueError('One source token plus model template/special tokens exceeds the input budget')
        if char_end <= char_begin:
            raise ValueError('Tokenizer offsets do not advance')
        yield char_begin, char_end, begin, end
        if char_end >= len(text):
            break
        next_begin = max(begin + 1, end - overlap)
        if statement_ends and overlap:
            prior=bisect.bisect_right(boundaries,offsets[next_begin][0])-1
            if prior>=0 and boundaries[prior]>char_begin:
                next_begin=max(begin+1,bisect.bisect_left(starts,boundaries[prior]))
        # No overlap is required when the full row/paragraph ended at the boundary.
        begin = next_begin
        char_begin = offsets[begin][0]
        if overlap == 0:
            char_begin = min(char_begin, char_end)


def _span(source, start, end, chunk_start, chunk_end):
    location = dict(source['locator'])
    for key in ('table_header', 'header_locator', 'retrieval_text', 'source_spans', 'statement_ends'):
        location.pop(key, None)
    location.update(char_start=source['source_start'] + start,
                    char_end=source['source_start'] + end,
                    chunk_start=chunk_start, chunk_end=chunk_end)
    if 'document_char_start' in location:
        base=source['locator']['document_char_start']
        location['document_char_start']=base+source['source_start']+start
        location['document_char_end']=base+source['source_start']+end
    if 'line_start' in location:
        first = source['locator']['line_start']
        location['line_start'] = first + source['text'][:start].count('\n')
        location['line_end'] = first + source['text'][:max(start, end - 1)].count('\n')
    return location


def structural_chunks(blocks, tokenizer, maximum=384, overlap=48, *, input_fits=None):
    if maximum <= 0 or overlap < 0 or overlap >= maximum:
        raise ValueError('Require maximum > 0 and 0 <= overlap < maximum')
    order = 0
    for group in _groups(blocks, tokenizer, maximum):
        segments, position = [], 0
        for item in group:
            if segments:
                position += 2
            segments.append((position, position + len(item['text']), item))
            position += len(item['text'])
        parent = '\n\n'.join(item['text'] for item in group)
        first = group[0]['locator']
        parent_id = hashlib.sha256(json.dumps(
            [parent, [item['locator']['block_id'] for item in group], _section(first)],
            ensure_ascii=False).encode()).hexdigest()[:24]
        parent_spans = [_span(item, 0, len(item['text']), left, right)
                        for left, right, item in segments]
        statement_ends=[left+point-item['source_start'] for left,right,item in segments
                        for point in item['locator'].get('statement_ends',[])
                        if item['source_start']<point<=item['source_start']+len(item['text'])]
        for begin, end, token_begin, token_end in _windows(parent, tokenizer, maximum, overlap, statement_ends, input_fits):
            spans = []
            for left, right, item in segments:
                a, b = max(left, begin), min(right, end)
                if a < b:
                    spans.append(_span(item, a - left, b - left, a - begin, b - begin))
            if not spans:
                continue
            location = {k: v for k, v in spans[0].items()
                        if k not in {'chunk_start', 'chunk_end'}}
            if 'line_start' in location:
                location['line_end'] = spans[-1].get('line_end', location['line_end'])
            if all('paragraph' in s for s in spans):
                location.update(paragraph_start=spans[0]['paragraph'], paragraph_end=spans[-1]['paragraph'])
            if all('row' in s for s in spans):
                location.update(row_start=spans[0]['row'], row_end=spans[-1]['row'])
            location.update(section_path=_section(first), parent_id=parent_id, order=order,
                            source_spans=spans, parent_source_spans=parent_spans,
                            parent_char_start=begin, parent_char_end=end,
                            token_start=token_begin, token_end=token_end)
            text = parent[begin:end]
            decoration = ' > '.join(_section(first))
            if first.get('table_header'):
                decoration += '\n' + first['table_header']
            if first.get('units'):
                decoration += '\nUnits: ' + ' | '.join(first['units'])
            retrieval_text = decoration.strip() + '\n' + text if decoration.strip() else text
            location['retrieval_text'] = retrieval_text
            yield {'text': text, 'parent': parent, 'image': group[0].get('image'),
                   'locator': location, 'retrieval_text': retrieval_text}
            order += 1
