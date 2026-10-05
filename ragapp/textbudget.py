"""Use offsets to preserve original source text; never reconstruct evidence with decode."""
import re


def offsets(tokenizer, text):
    return tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']


def token_count(tokenizer, text):
    return len(offsets(tokenizer,text))


def windows(tokenizer, text, maximum, overlap=0):
    if maximum<1 or not 0<=overlap<maximum: raise ValueError('Invalid token window budget')
    spans=offsets(tokenizer,text)
    if not spans:return []
    result=[]; start=0
    while start<len(spans):
        end=min(len(spans),start+maximum)
        if end<len(spans):
            a,b=spans[start][0],spans[end-1][1]
            breaks=list(re.finditer(r'(?:[。！？.!?]\s+|\n)',text[a:b]))
            if breaks and breaks[-1].end()>=(b-a)*.6:
                boundary=a+breaks[-1].end()
                while end>start+1 and spans[end-1][1]>boundary:end-=1
        a,b=spans[start][0],spans[end-1][1]
        piece=text[a:b]
        # Retokenization can add a boundary token, so check the actual input.
        while token_count(tokenizer,piece)>maximum and end>start+1:
            end-=1;b=spans[end-1][1];piece=text[a:b]
        result.append((piece,a,b))
        if end==len(spans):break
        start=max(start+1,end-overlap)
    return result
