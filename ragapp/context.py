"""Traceable source expansion with a budget measured on serialized context."""
import copy
import json
import re
from .textbudget import token_count


def code_line_locations(item, question, original):
    """Locate explicit constant assignments in original, covered physical lines.

    Do not count expanded context from the child chunk's start or infer a line
    from model prose. Ambiguous multiple assignments intentionally have no single
    location. The returned map locates text, not execution or a call relationship.
    """
    names = set(re.findall(r'(?<![A-Za-z0-9_])[A-Z][A-Z_0-9]{3,}(?![A-Za-z0-9_])', question))
    loc = item['locator']
    spans = loc.get('source_spans') or [loc]
    matches = {}
    for symbol in names:
        found = []
        for number, line in enumerate(original.splitlines(), 1):
            covered = any(type(s.get('line_start')) is int and type(s.get('line_end')) is int
                          and s['line_start'] <= number <= s['line_end'] for s in spans)
            if (covered and line.strip() and line.strip() in item['text']
                    and re.match(r'^[ \t]*' + re.escape(symbol) + r'\s*(?::[^=\n]+)?=(?!=)', line)):
                found.append(number)
        if len(found) == 1:
            matches[symbol] = found[0]
    return matches


def public_evidence(item):
    result={k:v for k,v in item.items() if k not in {'parent','image','retrieval','retrieval_text',
                                                  'hit_locator','source_locator','query_ranks','query_rrf_score'}}
    if isinstance(result.get('locator'),dict):
        result['locator']={k:v for k,v in result['locator'].items()
            if k not in {'retrieval_text','parent_source_spans'}}
    return result


def prompt_evidence(item):
    """Keep physical provenance in prompts without repeating indexing/cache metadata."""
    physical={'page','paragraph','paragraph_start','paragraph_end','table','row','row_start','row_end',
              'sheet','range','line_start','line_end','matched_lines','symbol','image','embedded_image',
              'column','operation','kind','heading','section_path'}
    loc=item['locator']
    compact={k:v for k,v in loc.items() if k in physical}
    if loc.get('source_spans'):
        spans=[{k:v for k,v in s.items() if k in physical} for s in loc['source_spans']]
        compact['source_spans']=list({json.dumps(s,sort_keys=True,ensure_ascii=False):s for s in spans}.values())
    return {'citation':item['citation'],'source':item['path'],'version':item['version'],
            'locator':compact,'text':item['text']}


def build_context(evidence, tokenizer, budget=8000, expand=True):
    selected=[];seen=set();used=0
    # Cover different documents before spending the remaining budget on repetitions.
    ordered=[];documents=set()
    for row in evidence:
        if row['document_id'] not in documents:
            ordered.append(row);documents.add(row['document_id'])
    ordered.extend(row for row in evidence if row not in ordered)
    for original in ordered:
        row=copy.deepcopy(original);loc=row['locator']
        if isinstance(loc,str):loc=json.loads(loc);row['locator']=loc
        can_expand=expand and loc.get('parent_id') and loc.get('parent_source_spans') and row.get('parent')
        key=(row['document_id'],row['version'],loc.get('parent_id') if can_expand else row['text'])
        if key in seen:continue
        if can_expand:
            row['text']=row['parent']
            row['locator']['source_spans']=loc['parent_source_spans']
            row['locator']['context_expanded']=True
            # Expanded provenance consists of multiple exact spans, not the hit's original range.
            row['hit_locator']=copy.deepcopy(original['locator'])
        def cost(item):
            return token_count(tokenizer,json.dumps(prompt_evidence(item),ensure_ascii=False))
        size=cost(row)
        if used+size>budget and can_expand:
            row=copy.deepcopy(original);size=cost(row);key=(row['document_id'],row['version'],row['text'])
        if used+size>budget:continue
        selected.append(row);used+=size;seen.add(key)
    return selected,{'context_tokens':used,'budget':budget,'selected_ids':[r['id'] for r in selected],
                    'expanded':sum(bool(r['locator'].get('context_expanded')) for r in selected)}
