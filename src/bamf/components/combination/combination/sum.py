import torch
import torch.nn as nn


class ElementwiseSum(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args):
        return torch.add(*args)