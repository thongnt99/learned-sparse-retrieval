from torch import nn
import torch
from lsr.losses import Loss, cross_dot_product, num_non_zero


class MultipleNegativeLoss(Loss):
    """
    The MultipleNegativeLoss implements the CrossEntropyLoss underneath. There are one positive document and multiple negative documents per query.
    For each query, this loss considers two type of negatives:
        1. The query's own negatives sampled from traning data.
        2. Documents (both positive and negative) from other queries. (in-batch negatives)
    """

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        """
        Constructing MultipleNegativeLoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(MultipleNegativeLoss, self).__init__(q_regularizer, d_regularizer)
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, q_reps, d_reps, labels=None):
        """
        Calculating the MultipleNegativeLoss over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size*group_size x vocab_size).
            group_size is the numer of documents (positive & negative) per query. The first document of the group is positive, the rest are negatives.
        Returns
        -------
        tuple (loss, q_reg, d_reg, log)
            a tuple of averaged loss, query regularization, doc regularization and log (for experiment tracking)
        """
        sim_matrix = cross_dot_product(q_reps, d_reps)
        # cross_dot_product(q_reps, d_reps)
        reg_q_output = (
            torch.tensor(0.0, device=q_reps.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_reps)
        )
        reg_d_output = (
            torch.tensor(0.0, device=d_reps.device)
            if (self.d_regularizer is None)
            else self.d_regularizer(d_reps)
        )

        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        q_num = q_reps.size(0)
        d_num = d_reps.size(0)
        assert d_num % q_num == 0
        doc_group_size = d_num // q_num
        labels = torch.arange(0, d_num, doc_group_size, device=sim_matrix.device)
        ce_loss = self.ce(sim_matrix, labels)
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(d_reps),
        }
        return (ce_loss, reg_q_output, reg_d_output, to_log)
