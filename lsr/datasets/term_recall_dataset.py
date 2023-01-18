import json
from torch.utils.data import Dataset


class TermRecallDataset(Dataset):
    """Term-recall dataset for training DeepCT"""

    def __init__(self, data_path: str):
        super().__init__()
        self.data = []
        with open(data_path, "r", encoding="UTF-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # format {"query": ... "term_recall": ..., "doc": ...}
        return self.data[idx]
