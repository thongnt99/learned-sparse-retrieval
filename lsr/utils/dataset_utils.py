from tqdm import tqdm
import ir_datasets
import json
from datasets import DownloadManager
import gzip
import pickle

IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"


def read_collection(collection_path: str, text_fields=["text"]):
    doc_dict = {}
    if collection_path.startswith(IRDS_PREFIX):
        irds_name = collection_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for doc in tqdm(
            dataset.docs_iter(),
            desc=f"Loading doc collection for ir_datasets: {irds_name}",
        ):
            doc = dict(doc)
            doc_id = doc.doc_id
            texts = [getattr(doc, field) for field in text_fields]
            text = " ".join(texts)
            doc_dict[doc_id] = text
    else:
        with open(collection_path, "r") as f:
            for line in tqdm(f, desc=f"Reading doc collection from {collection_path}"):
                doc_id, doc_text = line.strip().split("\t")
                doc_dict[doc_id] = doc_text
    return doc_dict


def read_queries(queries_path: str, text_fields=["text"]):
    queries = []
    if queries_path.startswith(IRDS_PREFIX):
        irds_name = queries_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for query in tqdm(
            dataset.queries_iter(), desc=f"Reading queries from {queries_path}"
        ):
            query_id = query.query_id
            texts = [getattr(query, field) for field in text_fields]
            text = " ".join(texts)
            queries.append((query_id, text))
    else:
        with open(queries_path, "r") as f:
            for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                query_id, query_text = line.strip().split("\t")
                queries.append((query_id, query_text))
    return queries


def read_qrels(qrels_path: str, rel_threshold=0):
    qid2pos = {}
    if qrels_path.startswith(IRDS_PREFIX):
        irds_name = qrels_path.repalce(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for qrel in dataset.qrels_iter():
            if qrel.relevance > rel_threshold:
                qid, did = qrel.query_id, qrel.doc_id
                if not qid in qid2pos:
                    qid2pos[qid] = []
                qid2pos[qid].append(did)
    else:
        qrels = json.load(open(qrels_path, "r"))
        for qid in qrels:
            qid2pos[str(qid)] = [str(did) for did in qrels[qid]]
    return qid2pos


file_map = {
    "sentence-transformers/msmarco-hard-negatives": "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
}


def read_ce_score(ce_path: str):
    if ce_path.startswith(HFG_PREFIX):
        hf_name = ce_path.replace(HFG_PREFIX, "")
        _url = file_map[hf_name]
        dl_manager = DownloadManager()
        ce_path = dl_manager.download(_url)
    res = {}
    with gzip.open(ce_path, "rb") as f:
        data = pickle.load(f)
        for qid in tqdm(data, desc=f"Preprocessing data from {ce_path}"):
            res[str(qid)] = {str(did): data[qid][did] for did in data[qid]}
    return res
