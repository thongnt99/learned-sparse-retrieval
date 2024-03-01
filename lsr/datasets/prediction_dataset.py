from torch.utils.data import Dataset
from lsr.utils.dataset_utils import (
    read_collection,
    read_queries,
    read_qrels,
)
import random


class TextCollection(Dataset):
    def __init__(self, ids, texts,  type="query") -> None:
        super().__init__()
        self.ids = ids
        self.texts = texts
        self.id_key = f"{type}_id"
        self.text_key = f"{type}_text"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {self.id_key: self.ids[index], self.text_key: self.texts[index]}


class PredictionDataset:
    def __init__(self, qrels_path, queries_path, docs_path, query_field=["text"], doc_field=["text"], num_documents=-1):
        full_docs = read_collection(docs_path, text_fields=doc_field)
        full_queries = read_queries(queries_path, text_fields=query_field)
        self.qrels = read_qrels(qrels_path)
        doc_ids = list(full_docs.keys())
        doc_texts = list(full_docs.values())
        query_ids, query_texts = list(zip(*full_queries))
        if num_documents > 0:
            # include all relevant documents and randomly sample the remaining.
            rel_doc_ids = set()
            for qid in self.qrels:
                rel_doc_ids.update(list(self.qrels[qid].keys()))
            to_sample_doc_ids = sorted(doc_ids).difference(rel_doc_ids)
            random.seed(42)
            if num_documents > len(doc_ids):
                num_sample = num_documents - len(doc_ids)
                random.shuffle(to_sample_doc_ids)
                sample_ids = to_sample_doc_ids[:num_sample]
                doc_ids = list(rel_doc_ids) + sample_ids
            else:
                doc_ids = rel_doc_ids
            doc_texts = [full_docs[did] for did in doc_ids]
        self.docs = TextCollection(doc_ids, doc_texts, type="doc")
        self.queries = TextCollection(query_ids, query_texts, type="query")
