"""Pinned local model adapters. Qwen reranking scores yes/no tokens, not a random head."""
import numpy as np

TASK='Given a question, retrieve passages that contain evidence to answer the question'


def embedding_inputs(settings, texts, query):
    if 'multilingual-e5' in settings.embedding_model:
        return [('query: ' if query else 'passage: ')+t for t in texts]
    if 'Qwen3-Embedding' in settings.embedding_model and query:
        return [f'Instruct: {TASK}\nQuery: {t}' for t in texts]
    return list(texts)


class QwenReranker:
    def __init__(self, path, maximum):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).eval()
        self.maximum=maximum
        self.prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.yes=self.tokenizer.convert_tokens_to_ids('yes');self.no=self.tokenizer.convert_tokens_to_ids('no')

    def predict(self, pairs, batch_size=4, show_progress_bar=False):
        import torch
        result=[]
        for start in range(0,len(pairs),batch_size):
            texts=[self.prefix+f'<Instruct>: {TASK}\n<Query>: {q}\n<Document>: {d}'+self.suffix
                   for q,d in pairs[start:start+batch_size]]
            inputs=self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
            if inputs['input_ids'].shape[1]>self.maximum:
                raise ValueError('Qwen reranker input exceeds budget; window the evidence first')
            with torch.inference_mode():
                logits=self.model(**inputs,logits_to_keep=1).logits[:,-1,:]
                result.extend((logits[:,self.yes]-logits[:,self.no]).tolist())
        return np.asarray(result)
