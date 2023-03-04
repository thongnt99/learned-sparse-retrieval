from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, AutoModel

from lsr.models.sparse_encoder import SparseEncoder


class DualSparseConfig(PretrainedConfig):
    model_type = "DualEncoder"

    def __init__(self, shared=False, **kwargs):
        self.shared = shared
        super().__init__(**kwargs)


class DualSparseEncoder(PreTrainedModel):
    """
    DualSparseEncoder class that encapsulates encoder(s) for query and document.

    Attributes
    ----------
    shared: bool
        to use a shared encoder for both query/document
    encoder: lsr.models.SparseEncoder
        a shared encoder for encoding both queries and documnets. This encoder is used only if 'shared' is True, otherwise 'None'
    query_encoder: lsr.models.SparseEncoder
        a separate encoder for encoding queries, only if 'shared' is False
    doc_encoder: lsr.models.SparseEncoder
        a separate encoder for encoding documents, only if 'shared' is False
    Methods
    -------
    from_pretrained(model_dir_or_name: str)
    """

    config_class = DualSparseConfig

    def __init__(
        self,
        query_encoder: SparseEncoder,
        doc_encoder: SparseEncoder = None,
        config: DualSparseConfig = DualSparseConfig(),
    ):
        super().__init__(config)
        self.config = config
        if self.config.shared:
            self.encoder = query_encoder
        else:
            self.query_encoder = query_encoder
            self.doc_encoder = doc_encoder

    def encode_queries(self, to_dense=True, **queries):
        """
        Encode a batch of queries with the query encoder
        Arguments
        ---------
        to_dense: bool
            If True, return the output vectors in dense format; otherwise, return in the sparse format (indices, values)
        queries:
            Input dict with {"input_ids": torch.Tensor, "attention_mask": torch.Tensor , "special_tokens_mask": torch.Tensor }
        """
        if to_dense:
            if self.config.shared:
                return self.encoder(**queries).to_dense(reduce="sum")
            else:
                return self.query_encoder(**queries).to_dense(reduce="sum")
        else:
            if self.config.shared:
                return self.encoder(**queries)
            else:
                return self.query_encoder(**queries)

    def encode_docs(self, to_dense=True, **docs):
        """
        Encode a batch of documents with the document encoder
        """
        if to_dense:
            if self.config.shared:
                return self.encoder(**docs).to_dense(reduce="amax")
            else:
                return self.doc_encoder(**docs).to_dense(reduce="amax")
        else:
            if self.config.shared:
                return self.encoder(**docs)
            else:
                return self.doc_encoder(**docs)

    def forward(self, loss, queries, docs_batch, labels=None):
        """Compute the loss given (queries, docs, labels)"""
        q_reps = self.encode_queries(**queries)
        docs_batch_rep = self.encode_docs(**docs_batch)
        if labels is None:
            output = loss(q_reps, docs_batch_rep)
        else:
            output = loss(q_reps, docs_batch_rep, labels)
        return output

    def save_pretrained(self, model_dir):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir)
        if self.config.shared:
            self.encoder.save_pretrained(model_dir, "/shared_encoder")
        else:
            self.query_encoder.save_pretrained(model_dir + "/query_encoder")
            self.doc_encoder.save_pretrained(model_dir, "/doc_encoder")

    @classmethod
    def from_pretrained(cls, model_dir_or_name):
        """Load query and doc encoder from a directory"""
        config = DualSparseConfig.from_pretrained(model_dir_or_name)
        if config.shared:
            shared_encoder = AutoModel.from_pretrained(
                model_dir_or_name + "/shared_encoder"
            )
            return cls(shared_encoder, config=config)
        else:
            query_encoder = AutoModel.from_pretrained(
                model_dir_or_name + "/query_encoder"
            )
            doc_encoder = AutoModel.from_pretrained(model_dir_or_name + "/doc_encoder")
            return cls(query_encoder, doc_encoder, config)


from .binary import BinaryEncoder, BinaryEncoderConfig
from .mlp import TransformerMLPSparseEncoder, TransformerMLPConfig
from .mlm import (
    TransformerMLMSparseEncoder,
    TransformerMLMConfig,
)
from .cls_mlm import TransformerCLSMLPSparseEncoder, TransformerCLSMLMConfig

AutoConfig.register("BINARY", BinaryEncoderConfig)
AutoModel.register(BinaryEncoderConfig, BinaryEncoder)
AutoConfig.register("MLP", TransformerMLPConfig)
AutoModel.register(TransformerMLPConfig, TransformerMLPSparseEncoder)
AutoConfig.register("MLM", TransformerMLMConfig)
AutoModel.register(TransformerMLMConfig, TransformerMLMSparseEncoder)
AutoConfig.register("CLS_MLM", TransformerCLSMLMConfig)
AutoModel.register(TransformerCLSMLMConfig, TransformerCLSMLPSparseEncoder)
