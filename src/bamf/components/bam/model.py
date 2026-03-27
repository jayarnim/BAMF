import torch
import torch.nn as nn
from .sampler.builder import sampler_builder
from .simplex import LinearProjection


class BayesianAttentionModules(nn.Module):
    def __init__(
        self,
        dim: int,
        dist: str,
        hyper_approx: float,
        hyper_prior: float,
        beta: float,
    ):
        """
        Bayesian Attention Modules (Fan et al., 2020)
        -----

        Args:
            dim (int):
                dimensionality of query, key, and value.
                (assuming that the query, key, and value dimensions are the same.)
            dist (str):
                approximate distribution for attention score.
                (e.g. `lognormal`, `weibull`)
            hyper_approx (float):
                hyper-parameter of approximate distribution.
                if approximate distribution is `lognormal`, hyper-parameter is `std`.
                if approximate distribution is `weibull`, hyper-parameter is `k`.
            hyper_prior (float):
                hyper-parameter of prior distribution.
                if approximate distribution is `lognormal`, prior distribution is `lognormal` and hyper-parameter is `std`.
                if approximate distribution is `weibull`, prior distribution is `gamma` and hyper-parameter is `beta`.            
            beta (float):
                smoothing factor for normalization @ simplex.
                (range: (0,1])
        """
        super().__init__()

        self.dim = dim
        self.dist = dist
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.beta = beta

        self._set_up_components()

    def forward(
        self,
        Q: torch.Tensor,                    # (B,D)
        K: torch.Tensor,                    # (B,H,D)
        V: torch.Tensor,                    # (B,H,D)
        mask: torch.Tensor,                 # (B,H)
    ):
        # Q: (B,D) -> (B,1,D) -> (B,H,D)
        Q_exp = Q.unsqueeze(1).expand_as(K)
        # attention scores: (B,H)
        scores, kl_entry = self.sampler(Q_exp, K)
        # masking: (B,H) -> (B,H)
        scores_masked = scores.masked_fill(~mask, float('-inf'))
        # simplex projection: (B,H) -> (B,H)
        weights = self.simplex_fn(scores_masked)
        # context vector: (B,H,1) x (B,H,D) -> (B,H,D) -> (B,D)
        context = torch.sum(weights.unsqueeze(-1) * V, dim=1)
        # kl mean: (B,H) -> scalar
        kl_mean = (
            (kl_entry * mask).sum(dim=1) 
            / mask.sum(dim=1).clamp_min(1)
        ).mean()
        return context, kl_mean

    def _set_up_components(self):
        self._create_layers()
    
    def _create_layers(self):
        kwargs = dict(
            name=self.dist,
            dim=self.dim,
            hyper_approx=self.hyper_approx,
            hyper_prior=self.hyper_prior,
        )
        self.sampler = sampler_builder(**kwargs)

        kwargs = dict(
            beta=self.beta,
        )
        self.simplex_fn = LinearProjection(**kwargs)