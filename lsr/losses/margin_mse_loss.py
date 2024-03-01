from torch import nn
import torch
from lsr.losses import Loss, dot_product, num_non_zero


class MarginMSELoss(Loss):
    """
    The class for the MarginMSE loss.
    The MarginMSE is used distillation from a teacher ranking model (T) to a student model (S).
    MarginMSE(q,d1,d2) = MSE(S(q,d1)-S(q,d2), T(q,d1) - T(q,d2))
    """

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        """
        Constructing MarginMSELoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(MarginMSELoss, self).__init__(q_regularizer, d_regularizer)
        self.mse = nn.MSELoss()

    def forward(self, q_reps, p_reps, n_reps,  labels):
        """
        Calculating the MarginMSE over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size*2 x vocab_size).
            The number of documents needed is twice the number of query as we need a pair of documents for each query to calculate the margin.
            Documents in even positions (0, 2, 4...) are positive (relevant) documents, documents in odd positions (1, 3, 5...) are negative (non-relvant) documents.
        labels: torch.Tensor
            Teacher's margin between positive and negative documents. labels[i] = teacher(q_reps[i], d_reps[i*2]) - teacher(q_reps[i], d_reps[i*2+1])
        Returns
        -------
        tuple (loss, q_reg, d_reg, log)
            a tuple of averaged loss, query regularization, doc regularization and log (for experiment tracking)
        """
        batch_size = q_reps.size(0)
        # p_score, n_score = labels.transpose(0, 1)
        # similarity with negative documents
        p_rel = dot_product(q_reps, p_reps)
        # similarity with positive documents
        n_rel = dot_product(q_reps, n_reps)
        distance = p_rel - n_rel
        ce_distance = labels[:, 0] - labels[:, 1]
        reg_q_output = (
            torch.tensor(0.0, device=q_reps.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_reps)
        )
        reg_d_output = (
            torch.tensor(0.0, device=p_reps.device)
            if (self.d_regularizer is None)
            else (self.d_regularizer(p_reps)*0.5 + self.d_regularizer(n_reps)*0.5 + self.d_regularizer(n_reps)) / 2
        )
        mse_loss = self.mse(distance, ce_distance)
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(p_reps),
            "loss_no_reg": mse_loss.detach(),
        }
        return (
            mse_loss,
            reg_q_output,
            reg_d_output,
            to_log,
        )
