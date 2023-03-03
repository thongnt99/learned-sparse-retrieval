import json
from torch.utils.data import Dataset
from pathlib import Path

from lsr.utils.dataset_utils import download_file


class TermRecallDataset(Dataset):
    """Term-recall dataset for training DeepCT"""

    links = {
        "data/msmarco/deepct/myalltrain.relevant.docterm_recall": "http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/data/myalltrain.relevant.docterm_recall"
    }

    def __init__(self, data_path: str):
        super().__init__()
        self.data = []
        if not Path(data_path).exists():
            if data_path in self.links:
                download_file(self.links[data_path], data_path)
            else:
                raise Exception(f"File not found: {data_path}")
        with open(data_path, "r", encoding="UTF-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # format {"query": ... "term_recall": ..., "doc": ...}
        return self.data[idx]
