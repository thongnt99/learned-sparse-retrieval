import json
from torch.utils.data import Dataset
from tqdm import tqdm
import gzip
import pickle
import random


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
    Dataset with hard negatives and teacher's scores for distillation
    Link to the dataset used in this framework: https://download.europe.naverlabs.com/splade/sigir22/data.tar.gz
    """

    def __init__(
        self,
        collection_path,
        queries_path,
        qrels_path,
        ce_score_dict,
        expansion_dict=None,
        train_group_size=2,
    ) -> None:
        super().__init__()
        self.doc_dict = {}
        self.q_dict = {}
        self.q_ids = []
        with open(collection_path, "r") as f:
            for line in tqdm(f, desc=f"Reading doc collection from {collection_path}"):
                doc_id, doc_text = line.strip().split("\t")
                self.doc_dict[int(doc_id)] = doc_text
        if expansion_dict is not None:
            with open(expansion_dict, "r") as f:
                for doc_id, exp_text in enumerate(f):
                    self.doc_dict[doc_id] = (
                        self.doc_dict[doc_id] + " " + exp_text.strip()
                    )
        with gzip.open(ce_score_dict, "rb") as f:
            self.ce_score = pickle.load(f)

        with open(qrels_path, "r") as f:
            self.qrels = json.load(f)

        with open(queries_path, "r") as f:
            for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                query_id, query_text = line.strip().split("\t")
                self.q_dict[int(query_id)] = query_text
                self.q_ids.append(int(query_id))
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, idx):
        q_id = self.q_ids[idx]
        q_text = self.q_dict[q_id]
        str_qid = str(q_id)
        if str_qid in self.qrels:
            doc1_id = int(random.choice(list(self.qrels[str_qid].keys())))
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
