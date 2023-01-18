from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils.functional import FunctionalFactory
from lsr.utils.sparse_rep import SparseRep
from transformers import AutoModelForMaskedLM
from torch import nn
from transformers import PretrainedConfig


class TransformerCLSMLMConfig(PretrainedConfig):
    model_type = "CLS_MLM"

    def __init__(
        self,
        tf_base_model_name_or_dir="distilbert-base-uncased",
        activation="relu",
        norm="log1p",
        **kwargs,
    ):
        self.tf_base_model_name_or_dir = tf_base_model_name_or_dir
        self.activation = activation
        self.norm = norm
        super().__init__(**kwargs)


class TransformerCLSMLPSparseEncoder(SparseEncoder):
    """
    Masked Language Model's head on top of CLS's token only
    """

    config_class = TransformerCLSMLMConfig

    def __init__(self, config: TransformerCLSMLMConfig):
        super(TransformerCLSMLPSparseEncoder, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            config.tf_base_model_name_or_dir
        )
        self.activation = FunctionalFactory.get(config.activation)
        self.norm = FunctionalFactory.get(config.norm)

    def forward(self, **kwargs):
        _ = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs)
        lex_weights = output.logits[:, 0, :]
        lex_weights = self.norm(self.activation(lex_weights))
        return SparseRep(dense=lex_weights)
