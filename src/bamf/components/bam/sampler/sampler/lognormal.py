import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions import LogNormal
from ..score import Dot


class LogNormalSampler(nn.Module):
    def __init__(
        self,
        dim: int,
        hyper_approx: float,
        hyper_prior: float,
    ):
        super().__init__()

        self.dim = dim
        self.std_approx = hyper_approx
        self.std_prior = hyper_prior

        self._set_up_components()

    def forward(self, Q, K):
        # approx
        approx_exp = self.score_fn_approx(Q, K)
        approx = self._build_approx_dist(approx_exp)

        # prior
        prior_exp = self.score_fn_prior(K).squeeze(-1)
        prior = self._build_prior_dist(prior_exp)
        
        # sampling
        samples = (
            approx.rsample()
            if self.training
            else torch.exp(approx_exp)
        )

        # kl(q||p)
        kl = kl_divergence(p=approx, q=prior)
        
        return samples, kl

    def _build_approx_dist(self, exp_val):
        scale = torch.full_like(exp_val, self.std_approx)
        loc = exp_val - 0.5 * (scale ** 2)
        dist = LogNormal(loc, scale)
        return dist

    def _build_prior_dist(self, exp_val):
        scale = torch.full_like(exp_val, self.std_prior)
        loc = exp_val - 0.5 * (scale ** 2)
        dist = LogNormal(loc, scale)
        return dist

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.score_fn_approx = Dot()

        self.score_fn_prior = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),

            nn.Linear(self.dim, 1),
            nn.Softmax(dim=1),
        )