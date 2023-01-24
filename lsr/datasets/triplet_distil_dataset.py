import json
from torch.utils.data import Dataset
import ir_datasets
from tqdm import tqdm
import gzip
import pickle
import random

from lsr.utils.dataset_utils import (
    read_collection,
    read_queries,
    read_qrels,
    read_ce_score,
)


class TripletDistilDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.triplets = []
        with open(data_path, "r") as f:
            for line in tqdm(f, desc="Loading dataset"):
                cols = line.split("\t")
                assert len(cols) == 5, "wrong format"
                self.triplets.append(
                    (cols[0], cols[1], cols[2], float(cols[3]), float(cols[4]))
                )

    def __getitem__(self, idx):
        return self.triplets[idx]

    def __len__(self):
        return len(self.triplets)


class TripletIDDistilDataset(Dataset):
    """
    Dataset with teacher's scores for distillation
    """

    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        qrels_path: str,
        ce_score_dict: str,
        train_group_size=2,
    ):
        super().__init__()
        self.doc_dict = read_collection(collection_path)
        self.queries = read_queries(queries_path)
        self.qrels = read_qrels(qrels_path)
        self.ce_score = read_ce_score(ce_score_dict)
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q_id, q_text = self.queries[idx]
        if q_id in self.qrels:
            doc1_id = int(random.choice(self.qrels[q_id]))
        else:
            doc1_id = random.choice(list(self.ce_score[q_id].keys()))
        doc_list = [self.doc_dict[doc1_id]]
        score_list = [self.ce_score[q_id][doc1_id]]
        all_doc_ids = list(self.ce_score[q_id].keys())
        if len(all_doc_ids) < self.train_group_size - 1:
            neg_doc_ids = random.choices(all_doc_ids, k=self.train_group_size - 1)
        else:
            neg_doc_ids = random.sample(all_doc_ids, k=self.train_group_size - 1)
        doc_list.extend([self.doc_dict[doc_id] for doc_id in neg_doc_ids])
        score_list.extend([self.ce_score[q_id][doc_id] for doc_id in neg_doc_ids])
        return (q_text, doc_list, score_list)
