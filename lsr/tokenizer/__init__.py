from lsr.utils import get_absolute_class_name, get_class_from_str
from transformers import AutoTokenizer
from abc import ABC
import json
import os
import torch

CLASS_CONFIG = "class_config.json"


class Tokenizer(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwds):
        raise NotImplementedError("Not yet implemented")

    def save_pretrained(self, output_dir):
        raise NotImplementedError("Not yet implemented")

    @staticmethod
    def from_pretrained(tokenizer_name_or_dir):
        class_config_path = os.path.join(tokenizer_name_or_dir, CLASS_CONFIG)
        tokenizer_class_str = json.load(open(class_config_path, "r"))["tokenizer_class"]
        tokenizer_class = get_class_from_str(tokenizer_class_str)
        return tokenizer_class.from_pretrained(tokenizer_name_or_dir)


class HFTokenizer:
    def __init__(self, tokenizer_name, is_fast=None, nearest_neighbours=None) -> None:
        if is_fast != None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_fast=is_fast
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if nearest_neighbours is None:
            self.nearest_neighbours = None
        else:
            self.nearest_neighbours = torch.load(nearest_neighbours)

    def __call__(self, *args, **kwds):
        outputs = self.tokenizer(*args, **kwds)
        if not self.nearest_neighbours is None:
            _input_ids = outputs["input_ids"]
            topk_nnbs = self.nearest_neighbours.index_select(
                0, _input_ids.flatten()
            ).reshape(_input_ids.size(0), _input_ids.size(1), -1)
            outputs["topk_nnbs"] = topk_nnbs
        return outputs

    def encode_plus(self, *args, **kwds):
        return self.tokenizer.encode_plus(*args, **kwds)

    def pad(self, *args, **kwds):
        return self.tokenizer.pad(*args, **kwds)

    def save_pretrained(self, output_dir):
        self.tokenizer.save_pretrained(output_dir)
        class_config = {"tokenizer_class": get_absolute_class_name(self)}
        config_path = os.path.join(output_dir, CLASS_CONFIG)
        json.dump(class_config, open(config_path, "w"))

    def get_vocab_size(self):
        return len(self.tokenizer)

    def convert_ids_to_tokens(self, list_ids):
        return self.tokenizer.convert_ids_to_tokens(list_ids)

    @classmethod
    def from_pretrained(cls, tokenizer_name_or_dir):
        return cls(tokenizer_name_or_dir)
