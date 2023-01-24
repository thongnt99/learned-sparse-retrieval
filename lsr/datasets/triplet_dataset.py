from torch.utils.data import Dataset
from tqdm import tqdm
import torch

from lsr.utils.dataset_utils import read_collection, read_queries, read_triplets


class TripletIdsDataset(Dataset):
    """BM25 triplets of (query_id, pos_id, neg_id)"""

    def __init__(
        self, triplet_ids_path: str, queries_path: str, collection_path: str
    ) -> None:
        super().__init__()
        self.docs_dict = read_collection(collection_path)
        self.queries_dict = dict(read_queries(queries_path))
        self.triplets, _, _ = read_triplets(triplet_ids_path)

    def __getitem__(self, idx):
        qid, pos_id, neg_id = self.triplets[idx]
        return (self.queries_dict[qid], self.docs_dict[pos_id], self.docs_dict[neg_id])

    def __len__(self):
        return len(self.triplets)


class TripletTextDataset(Dataset):
    "BM25 triplets of (query_text, pos_text, neg_text)"

    def __init__(self, data_path) -> None:
        super().__init__()
        print(data_path)
        self.triple_lists = []
        with open(data_path, "r") as f:
            for line in tqdm(f, f"Loading triplets from {data_path}"):
                q, pos, neg = line.strip().split("\t")
                self.triple_lists.append((q, pos, neg))

    def __getitem__(self, idx):
        return self.triple_lists[idx]

    def __len__(self):
        return len(self.triple_lists)
