from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils.sparse_rep import SparseRep
from torch import nn
import torch
from transformers import PretrainedConfig


class BinaryEncoderConfig(PretrainedConfig):
    """
    Configuration class for BinaryEncoder.
    """

    model_type = "BINARY"

    def __init__(
        self, vocab_size: int = 30522, scale=0.3, **kwargs,
    ):
        """
        Construct a configuration for Binary Encoder
        Parameters
        ----------
        vocab_size: int
            number of items in the vocabulary. Used for determining the shape of output vector
        scale: float
            scaling factor for the binary output
        """
        self.vocab_size = vocab_size
        self.scale = scale
        super().__init__(**kwargs)


class BinaryEncoder(SparseEncoder):
    config_class = BinaryEncoderConfig
    """
    BinaryEncoder assigns a weight of 1 to tokens present in the input and 0 to the remaining tokens.
    """

    def __init__(self, config: BinaryEncoderConfig = BinaryEncoderConfig()):
        """
        Construct a BinaryEncoder instance
        Parameters
        ----------
        config: BinaryEncoderConfig
            configuration for the BinaryEncoder
        """
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.scale = nn.Parameter(torch.tensor(config.scale))

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        bin_weights = (
            torch.ones_like(input_ids, dtype=torch.float)
            * kwargs["attention_mask"]
            * (1 - kwargs["special_tokens_mask"])
        ) * self.scale
        size = torch.tensor(
            (input_ids.size(0), self.vocab_size), device=input_ids.device
        )
        return SparseRep(indices=input_ids, values=bin_weights, size=size,)
