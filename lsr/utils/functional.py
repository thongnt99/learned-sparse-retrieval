"""Contains common normalization or activation function wrapped in class"""
import torch
from torch import nn


class Log1P(nn.Module):
    """This is a class warper of torch.log1p function"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        return torch.log1p(inputs)


class NoNorm(nn.Module):
    """This module return the inputs itself"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        return inputs


class AllOne(nn.Module):
    """This module return 1.0 for any inputs"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        return 1.0


class FunctionalFactory:
    name2class = {
        "log1p": Log1P,
        "identity": NoNorm,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
    }

    @classmethod
    def get(cls, name):
        return cls.name2class[name]()

    @classmethod
    def register(cls, name, class_path):
        if name in cls.name2class:
            raise Exception(f"Name {name} already registered")
        cls.name2class[name] = class_path
