import torch
import torch.nn as nn


class ElementwiseProduct(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args):
        return torch.mul(*args)