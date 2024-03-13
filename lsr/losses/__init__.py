from .joint_kl_mse_loss import JointKLMSE
from .distil_kl_loss import DistilKLLoss
from .negative_likelihood import NegativeLikelihoodLoss
from .cross_entropy_loss import CrossEntropyLoss
from .term_mse_loss import TermMSELoss
from .multiple_negative_loss import MultipleNegativeLoss
from .margin_mse_loss import MarginMSELoss
from abc import ABC
from curses import A_DIM
from torch import nn, Tensor
import torch


class Loss(nn.Module, ABC):
    """
    The loss abstract class
    """

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        super(Loss, self).__init__()
        self.q_regularizer = q_regularizer
        self.d_regularizer = d_regularizer

    def forward(self, *args, **kwargs):
        raise NotImplementedError("the loss function is not yet implemented")


def dot_product(a: Tensor, b: Tensor):
    """
    Calculating row-wise dot product between two tensors a and b.
    a and b must have the same dimensionality.
    Parameters
    ----------
    a: torch.Tensor
        size: batch_size x vector_dim
    b: torch.Tensor
        size: batch_size x vector_dim
    Returns
    -------
    torch.Tensor: size of (batch_size x 1)
        dot product for each pair of vectors
    """
    return (a * b).sum(dim=-1)


def cross_dot_product(a: Tensor, b: Tensor):
    """
    Calculating the cross doc product between each row in a with every row in b. a and b must have the same number of columns, but can have varied nuber of rows.
    Parameters
    ----------
    a: torch.Tensor
        size: (batch_size_1,  vector_dim)
    b: torch.Tensor
        size: (batch_size_2, vector_dim)
    Returns
    -------
    torch.Tensor: of size (batch_size_1, batch_size_2) where the value at (i,j) is dot product of a[i] and b[j].
    """
    return torch.mm(a, b.transpose(0, 1))


def num_non_zero(a: Tensor):
    """
    Calculating the average number of non-zero columns in each row.
    Parameters
    ----------
    a: torch.Tensor
        the input tensor
    """
    return (a > 0).float().sum(dim=1).mean()
