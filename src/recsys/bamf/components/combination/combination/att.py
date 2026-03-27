import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self, 
        dim: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, *args):
        score0 = self.mlp(args[0])
        score1 = self.mlp(args[1])
        weight = (
            torch.exp(score0) 
            / (
                torch.exp(score0) + torch.exp(score1)
            )
        )
        return weight * args[0] + (1-weight) * args[1]