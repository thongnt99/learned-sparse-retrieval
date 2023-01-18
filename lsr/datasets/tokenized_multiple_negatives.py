import random
import datasets

print(datasets.__file__)
from datasets import load_dataset
from typing import Union, List, Callable, Dict

from torch.utils.data import Dataset


class GroupedMarcoTrainDataset(Dataset):
    """
        Dataset link: http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/
        Dataset which map each query to a group of positive and negative pre-tokenized passages. The format is as following:
        {
        "qry": {
            "qid": str,
            "query": List[int],
        },
        "pos": List[
            {
                "pid": str,
                "passage": List[int],
            }
        ],
        "neg": List[
            {
                "pid": str,
                "passage": List[int]
            }
        ]
    }
    """

    query_columns = ["qid", "query"]
    document_columns = ["pid", "passage"]

    def __init__(
        self,
        path_to_tsv: Union[List[str], str],
        train_group_size: int,
    ):
        self.nlp_dataset = load_dataset(
            "json",
            data_files=path_to_tsv,
            ignore_verifications=False,
            features=datasets.Features(
                {
                    "qry": {
                        "qid": datasets.Value("string"),
                        "query": [datasets.Value("int32")],
                    },
                    "pos": [
                        {
                            "pid": datasets.Value("string"),
                            "passage": [datasets.Value("int32")],
                        }
                    ],
                    "neg": [
                        {
                            "pid": datasets.Value("string"),
                            "passage": [datasets.Value("int32")],
                        }
                    ],
                }
            ),
        )["train"]
        self.total_len = len(self.nlp_dataset)
        self.train_group_size = train_group_size

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        group = self.nlp_dataset[item]
        group_batch = []
        query = group["qry"]["query"]
        pos_psg = random.choice(group["pos"])["passage"]
        group_batch.append(pos_psg)
        if len(group["neg"]) < self.train_group_size - 1:
            negs = random.choices(group["neg"], k=self.train_group_size - 1)
        else:
            negs = random.sample(group["neg"], k=self.train_group_size - 1)
        group_batch.extend([neg["passage"] for neg in negs])
        return query, group_batch
