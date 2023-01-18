from transformers import PreTrainedModel


class SparseEncoder(PreTrainedModel):
    """
    Abstract class for sparse encoder
    Methods:
    --------
    init(*args, **kwargs):
        initialize the encoder with necessary arguments
    forward(**kwargs):
        implementation of the forward pass
    save_pretrained(model_directory: str):
        save the encoder's checkpoint to a directory
    from_pretrained(model_name_or_dir):
        to the weights of the model by name or path
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        """Abstract forward method: should be overidden by child class"""
        raise NotImplementedError("Not yet implemented")
