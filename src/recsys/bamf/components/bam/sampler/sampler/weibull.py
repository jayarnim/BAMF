import math
import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from .. import kl
from torch.distributions import Weibull, Gamma
from ..score import Dot


class WeibullSampler(nn.Module):
    def __init__(
        self,
        dim: int,
        hyper_approx: float,
        hyper_prior: float,
    ):
        super().__init__()

        self.dim = dim
        self.k = hyper_approx
        self.beta = hyper_prior

        self._set_up_components()

    def forward(self, Q, K):
        # approx
        approx_exp = self.score_fn_approx(Q, K)
        approx = self.build_approx_dist(approx_exp)
        # prior
        prior_exp = self.score_fn_prior(K).squeeze(-1)
        prior = self.build_prior_dist(prior_exp)
        return torch.exp(approx_exp), approx, prior

    def estimate(self, Q, K):
        exp_val, approx, prior = self.forward(Q, K)
        samples = approx.rsample()
        kl = kl_divergence(approx, prior)
        return samples, kl

    @torch.no_grad()
    def predict(self, Q, K):
        exp_val, approx, prior = self.forward(Q, K)
        kl = kl_divergence(approx, prior)
        return exp_val, kl

    def build_approx_dist(self, exp_val):
        k = torch.full_like(exp_val, self.k)
        lambda_ = torch.exp(exp_val) / torch.exp(torch.lgamma(1 + 1.0 / k))
        dist = Weibull(lambda_, k)
        return dist

    def build_prior_dist(self, exp_val):
        beta = torch.full_like(exp_val, self.beta)
        alpha = torch.exp(beta) * beta
        dist = Gamma(alpha, beta)
        return dist

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            dim=self.dim,
        )
        self.score_fn_approx = Dot(**kwargs)

        self.score_fn_prior = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),

            nn.Linear(self.dim, 1),
            nn.Softmax(dim=1),
        )