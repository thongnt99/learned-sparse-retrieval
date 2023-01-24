import random
from lsr.utils.dataset_utils import read_collection, read_queries, read_triplets
from torch.utils.data import Dataset
from collections import defaultdict


class MultipleNegatives(Dataset):
    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        triplet_ids_path: str,
        train_group_size: int,
    ):
        self.docs_dict = read_collection(collection_path)
        self.q_dict = dict(read_queries(queries_path))
        self.qids = list(self.q_dict.keys())
        self.triplets = read_triplets(triplet_ids_path)
        self.query2pos = defaultdict(list)
        self.query2neg = defaultdict(list)
        for qid, pos_id, neg_id in self.triplets:
            self.query2pos[qid].append(pos_id)
            self.query2neg[qid].append(neg_id)
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, item):
        qid = self.qids[item]
        query = self.q_dict[qid]
        pos_id = random.choice(self.query2pos[qid])
        pos_psg = self.docs_dict[pos_id]
        group_batch = []
        group_batch.append(pos_psg)
        if len(self.query2neg[qid]) < self.train_group_size - 1:
            negs = random.choices(self.query2neg[qid], k=self.train_group_size - 1)
        else:
            negs = random.sample(self.query2neg[qid], k=self.train_group_size - 1)
        group_batch.extend([self.docs_dict[neg] for neg in negs])
        return query, group_batch