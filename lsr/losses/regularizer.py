from torch import nn
import torch


class Regularizer(nn.Module):
    """Base class for sparse regularizer.
    Attributes
    ----------
    weight : float
        the weight of regularizer in the loss
    T : int
        the weight get exponentially increased over T steps
    """

    def __init__(self, weight=0.0, T=1000) -> None:
        """
        Intializing the regularizer with weight and decaying steps
        Parameters
        ----------
            weight: float
                the regularizer's weight in the loss
            T: int
                warming up steps
        """
        super().__init__()
        self.weight_T = weight
        self.weight_t = 0
        self.T = T
        self.t = 0

    def step(self):
        """
        Perform a warming up step.
        The weight starts with zero and get expoentially increased step by step until the T-th step.
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.weight_t = self.weight_T * (self.t / self.T) ** 2

    def forward(self, reps):
        """
        reps: batch representation
        """
        raise NotImplementedError("This is an abstract regularizer only.")


class FLOPs(Regularizer):
    """
    Implementation of the FLOPs regularizer which is a mooth approximation for number of term overlap between a query and a document.
    Paper: https://arxiv.org/abs/2004.05665
    """

    def forward(self, reps):
        return (torch.abs(reps).mean(dim=0) ** 2).sum() * self.weight_t


class L1(Regularizer):
    """
    Implementation of the L1 regularizer
    """

    def forward(self, reps):
        return torch.abs(reps).sum(dim=1).mean() * self.weight_t
