from lsr.models import SparseEncoder
from lsr.utils.functional import FunctionalFactory
from lsr.utils.sparse_rep import SparseRep
from transformers import AutoModel
import torch
from torch import nn
from transformers import PretrainedConfig


class TransformerMLPConfig(PretrainedConfig):
    """
    Configuration for TransformerMLPSparseEncoder
    """

    model_type = "MLP"

    def __init__(
        self,
        tf_base_model_name_or_dir="distilbert-base-uncased",
        activation="relu",
        norm="log1p",
        scale=1.0,
        **kwargs,
    ):
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.activation = activation
        self.norm = norm
        self.scale = scale
        super().__init__(**kwargs)


class TransformerMLPSparseEncoder(SparseEncoder):
    """
    MLP on top of Transformer layers
    """

    config_class = TransformerMLPConfig

    def __init__(self, config: TransformerMLPConfig):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.tf_base_model_name_or_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.activation = FunctionalFactory.get(config.activation)
        self.norm = FunctionalFactory.get(config.norm)
        self.linear.apply(self._init_weights)
        self.scale = nn.Parameter(torch.tensor(config.scale))

    def _init_weights(self, module):
        """Initialize the weights (needed this for the inherited from_pretrained method to work)"""
        torch.nn.init.kaiming_normal(module.weight.data, nonlinearity="relu")

    def forward(self, to_scale=False, **kwargs):
        special_tokens_mask = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs)
        tok_weights = self.linear(output.last_hidden_state).squeeze(-1)  # bs x len x 1
        tok_weights = (
            self.norm(self.activation(tok_weights))
            * kwargs["attention_mask"]
            * (1 - special_tokens_mask)
        )
        if to_scale:
            tok_weights = tok_weights * self.scale
        size = torch.tensor(
            (tok_weights.size(0), self.model.config.vocab_size),
            device=tok_weights.device,
        )
        return SparseRep(indices=kwargs["input_ids"], values=tok_weights, size=size)
