from torch import nn
import torch
from lsr.losses import Loss, dot_product, num_non_zero


class JointKLMSE(Loss):
    def __init__(self, q_regularizer=None, d_regularizer=None, kl_weight=1.0, mse_weight=0.05) -> None:
        """
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(JointKLMSE, self).__init__(q_regularizer, d_regularizer)
        self.loss = torch.nn.KLDivLoss(reduction="none")
        self.mse = nn.MSELoss()
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight

    def forward(self, q_reps, p_reps, n_reps, labels):
        """
        Calculating the KL over a batch of query and document
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
        # p_reps, n_reps = d_reps.view(batch_size, 2, -1).transpose(0, 1)
        labels = labels.view(batch_size, 2)
        teacher_scores = torch.softmax(labels, dim=1)
        # similarity with negative documents
        p_rel = dot_product(q_reps, p_reps)
        # similarity with positive documents
        n_rel = dot_product(q_reps, n_reps)
        distance = p_rel - n_rel
        ce_distance = labels[:, 0] - labels[:, 1]
        mse_loss = self.mse(distance, ce_distance) * self.mse_weight
        student_scores = torch.stack([p_rel, n_rel], dim=1)
        student_scores = torch.log_softmax(student_scores, dim=1)
        kl_loss = self.loss(student_scores, teacher_scores).sum(
            dim=1).mean(dim=0) * self.kl_weight
        reg_q_output = (
            torch.tensor(0.0, device=q_reps.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_reps)
        )
        reg_d_output = (
            torch.tensor(0.0, device=p_reps.device)
            if (self.d_regularizer is None)
            else (self.d_regularizer(p_reps) + self.d_regularizer(n_reps)) / 2
        )
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(p_reps),
            "kl_loss": kl_loss.detach(),
            "mse_loss": mse_loss.detach(),
        }
        return (kl_loss+mse_loss, reg_q_output, reg_d_output, to_log)
