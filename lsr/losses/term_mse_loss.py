from cProfile import label
from torch import nn
import torch
from lsr.losses import Loss, dot_product, num_non_zero


class TermMSELoss(Loss):
    """
    TermMSELoss calculates the mean squared distance between the predicted term salience and ground-truth term salience.
    The loss is applied for a pair of query and relevant document with a shared ground-truth term salience.
    """

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        """
        Constructing TermMSELoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(TermMSELoss, self).__init__(q_regularizer, d_regularizer)
        self.mse = nn.MSELoss()

    def forward(self, q_reps, d_reps, labels):
        """
        Calculating the TermMSELoss over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size x vocab_size)
        labels: torch.Tensor
            batch of term scores (size: batch_size x vocab_size)
        Returns
        -------
        tuple (loss, q_reg, d_reg, log)
            a tuple of averaged loss, query regularization, doc regularization and log (for experiment tracking)
        """
        query_mse = self.mse(q_reps, labels)
        doc_mse = self.mse(d_reps, labels)
        sum_mse = query_mse + doc_mse
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
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(d_reps),
            "loss_no_reg": sum_mse.detach(),
        }
        return (
            sum_mse,
            reg_q_output,
            reg_d_output,
            to_log,
        )
