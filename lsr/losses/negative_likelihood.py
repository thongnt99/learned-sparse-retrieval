from torch import nn
import torch
from lsr.losses import Loss, dot_product, num_non_zero


class NegativeLikelihoodLoss(Loss):
    """
    NegativeLikelihoodLoss returns the negative likelihood of positive documents. Minimizing this loss would increase the likelihood of the positive documents.
    """

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        """
        Constructing NegativeLikelihoodLoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(NegativeLikelihoodLoss, self).__init__(q_regularizer, d_regularizer)
        self.sm = nn.Softmax(dim=1)

    def forward(self, q_reps, d_reps, labels=None):
        """
        Calculating the NegativeLikelihoodLoss over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size*group_size x vocab_size).
            group_size is the numer of documents (positive & negative) per query. The first document of the group is positive, the rest are negatives.
        labels: None
            not used in this loss.
        Returns
        -------
        tuple (loss, q_reg, d_reg, log)
            a tuple of averaged loss, query regularization, doc regularization and log (for experiment tracking)
        """
        q_num = q_reps.size(0)
        d_num = d_reps.size(0)
        assert d_num % q_num == 0
        doc_group_size = d_num // q_num

        sim_matrix = dot_product(
            q_reps.unsqueeze(1),
            d_reps.view(q_num, doc_group_size, -1),
        )
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
        prob = self.sm(sim_matrix)
        negative_likelihood = (1.0 - prob[:, 0]).mean()
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(d_reps),
            "loss_no_reg": negative_likelihood.detach(),
        }
        return (
            negative_likelihood,
            reg_q_output,
            reg_d_output,
            to_log,
        )
