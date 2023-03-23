import re
import base64
import string
import numpy as np
from contextlib import ExitStack
import itertools
from more_itertools import chunked
import torch
import os
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
from lsr.tokenizer import Tokenizer
from lsr.models import DualSparseEncoder

"""
This module provides a pt.Transformer-compatible interface to the lsr package.

Example indexing and retrieval for TREC DL 2019:

```python
import pyterrier as pt ; pt.init()
from lsr.transformer import LSR
from pyterrier_pisa import PisaIndex, PisaToksIndexer

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
lsr = LSR('macavaney/sparta-msmarco-distil') # load a trained LSR model

# index the corpus
index_pipeline = lsr >> PisaToksIndexer('my-index.pisa')
index = index_pipeline.index(dataset.get_corpus_iter())
```
"""

class LSR(pt.Transformer):
    def __init__(self, model_name, device=None, batch_size=32, text_field='text', fp16=False, topk=None):
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fp16 = fp16
        self.device = device
        self.model = DualSparseEncoder.from_pretrained(model_name).eval().to(device)
        self.tokenizer = Tokenizer.from_pretrained(os.path.join(model_name, 'tokenizer'))
        all_token_ids = list(range(self.tokenizer.get_vocab_size()))
        self.all_tokens = np.array(self.tokenizer.convert_ids_to_tokens(all_token_ids))
        self.batch_size = batch_size
        self.text_field = text_field
        self.topk = topk

    def encode_queries(self, texts, out_fmt='dict', topk=None):
        outputs = []
        if out_fmt != 'dict':
            assert topk is None, "topk only supported when out_fmt='dict'"
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fp16:
                stack.enter_context(torch.cuda.amp.autocast())
            for batch in chunked(texts, self.batch_size):
                enc = self.tokenizer(batch, padding=True, truncation=True, return_special_tokens_mask=True, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                res = self.model.encode_queries(**enc).cpu().float()
                if out_fmt == 'dict':
                    res = self.vec2dicts(res, topk=topk)
                    outputs.extend(res)
                else:
                    outputs.append(res.numpy())
        if out_fmt == 'np':
            outputs = np.concatenate(outputs, axis=0)
        elif out_fmt == 'np_list':
            outputs = list(itertools.chain.from_iterable(outputs))
        return outputs

    def encode_docs(self, texts, out_fmt='dict', topk=None):
        outputs = []
        if out_fmt != 'dict':
            assert topk is None, "topk only supported when out_fmt='dict'"
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fp16:
                stack.enter_context(torch.cuda.amp.autocast())
            for batch in chunked(texts, self.batch_size):
                enc = self.tokenizer(batch, padding=True, truncation=True, return_special_tokens_mask=True, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                res = self.model.encode_docs(**enc)
                if out_fmt == 'dict':
                    res = self.vec2dicts(res, topk=topk)
                    outputs.extend(res)
                else:
                    outputs.append(res.cpu().float().numpy())
        if out_fmt == 'np':
            outputs = np.concatenate(outputs, axis=0)
        elif out_fmt == 'np_list':
            outputs = list(itertools.chain.from_iterable(outputs))
        return outputs

    def vec2dicts(self, batch_output, topk=None):
        rtr = []
        idxs, cols = torch.nonzero(batch_output, as_tuple=True)
        weights = batch_output[idxs, cols]
        args = weights.argsort(descending=True)
        idxs = idxs[args]
        cols = cols[args]
        weights = weights[args]
        for i in range(batch_output.shape[0]):
            mask = (idxs==i)
            col = cols[mask]
            w = weights[mask]
            if topk is not None:
                col = col[:topk]
                w = w[:topk]
            d = {self.all_tokens[k]: v for k, v in zip(col.cpu().tolist(), w.cpu().tolist())}
            rtr.append(d)
        return rtr

    def query_encoder(self, matchop=False, sparse=True, topk=None):
        return LSRQueryEncoder(self, matchop, sparse=sparse, topk=topk or self.topk)

    def doc_encoder(self, text_field=None, sparse=True, topk=None):
        return LSRDocEncoder(self, text_field or self.text_field, sparse=sparse, topk=topk or self.topk)

    def scorer(self, text_field=None):
        return LSRScorer(self, text_field or self.text_field)

    def transform(self, inp):
        if all(c in inp.columns for c in ['qid', 'query', self.text_field]):
            return self.scorer()(inp)
        elif 'query' in inp.columns:
            return self.query_encoder()(inp)
        elif self.text_field in inp.columns:
            return self.doc_encoder()(inp)
        raise ValueError(f'unsupported columns: {inp.columns}; expecting "query", {repr(self.text_field)}, or both.')


class LSRQueryEncoder(pt.Transformer):
    def __init__(self, lsr: LSR, matchop=False, sparse=True, topk=None):
        self.lsr = lsr
        if not sparse:
            assert not matchop, "matchop only supported when sparse=True"
            assert topk is None, "topk only supported when sparse=True"
        self.matchop = matchop
        self.sparse = sparse
        self.topk = topk

    def encode(self, texts):
        return self.lsr.encode_queries(texts, out_fmt='dict' if self.sparse else 'np_list', topk=self.topk)

    def transform(self, inp):
        res = self.encode(inp['query'])
        if self.matchop:
            res = [_matchop(r) for r in res]
            inp = pt.model.push_queries(inp)
            return inp.assign(query=res)
        if self.sparse:
            return inp.assign(query_toks=res)
        return inp.assign(query_vec=res)


class LSRDocEncoder(pt.Transformer):
    def __init__(self, lsr: LSR, text_field, sparse=True, topk=None):
        self.lsr = lsr
        self.text_field = text_field
        self.sparse = sparse
        if not sparse:
            assert topk is None, "topk only supported when sparse=True"
        self.topk = topk

    def encode(self, texts):
        return self.lsr.encode_docs(texts, out_fmt='dict' if self.sparse else 'np_list', topk=self.topk)

    def transform(self, inp):
        res = self.encode(inp[self.text_field])
        if self.sparse:
            return inp.assign(toks=res)
        return inp.assign(doc_vec=res)


class LSRScorer(pt.Transformer):
    def __init__(self, lsr: LSR, text_field):
        self.lsr = lsr
        self.text_field = text_field

    def score(self, query_texts, doc_texts):
        q, inv_q = np.unique(query_texts.values if isinstance(query_texts, pd.Series) else np.array(query_texts), return_inverse=True)
        q = self.lsr.encode_queries(q, out_fmt='np')[inv_q]
        d, inv_d = np.unique(doc_texts.values if isinstance(doc_texts, pd.Series) else np.array(doc_texts), return_inverse=True)
        d = self.lsr.encode_docs(d, out_fmt='np')[inv_d]
        return np.einsum('bd,bd->b', q, d)

    def transform(self, inp):
        res = inp.assign(score=self.score(inp['query'], inp[self.text_field]))
        return add_ranks(res)


_alphnum_exp = re.compile('^[' + re.escape(string.ascii_letters + string.digits) + ']+$')

def _matchop(d):
    res = []
    for t, w in d.items():
        if not _alphnum_exp.match(t):
            encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8")
            t = f'#base64({encoded})'
        if w != 1:
            t = f'#combine:0={w}({t})'
        res.append(t)
    return ' '.join(res)
