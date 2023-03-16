from torch import nn
import torch
from lsr.losses import Loss, dot_product, num_non_zero


class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss with sparse regularizations on queries and documents.
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
        super(CrossEntropyLoss, self).__init__(q_regularizer, d_regularizer)
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, q_reps, d_reps, labels=None):
        """
        Calculating the CrossEntropyLoss over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size*group_size x vocab_size).
            group_size is the numer of documents (positive & negative) per query.
        labels: torch.Tensor
            a tensor of size (batch_size, 1). labels[i]=j means that d_reps[i*group_size + j] is relevant document.
            If labels is None, the first document is the relevant one by default. labels[i] = i*group_size for every i.
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
            q_reps.unsqueeze(1), d_reps.view(q_num, doc_group_size, -1),
        )

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
        if labels is None:
            labels = torch.zeros(0, q_num, dtype=torch.int, device=sim_matrix.device)

        ce_loss = self.ce(sim_matrix, labels)
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(d_reps),
            "loss_no_reg": ce_loss.detach(),
        }
        return (
            ce_loss,
            reg_q_output,
            reg_d_output,
            to_log,
        )
