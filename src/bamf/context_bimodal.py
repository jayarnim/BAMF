from typing import Literal
import torch
import torch.nn as nn
from .components.embedding.builder import embedding_builder
from .components.matching import MatrixFactorizationLayer
from .components.prediction import ProjectionLayer
from .components.bam.model import BayesianAttentionModules


class Module(nn.Module):
    def __init__(
        self,
        histories: dict[str, torch.Tensor],
        num_users: int,
        num_items: int,
        embedding_dim: int,
        dist: Literal["lognormal", "weibull"],
        hyper_approx: float,
        hyper_prior: float,
        beta: float,
    ):
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.dist = dist
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.beta = beta
        self.histories = histories

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        kwargs = dict(
            user_idx=user_idx,
            item_idx=item_idx,
        )
        user_emb, item_emb, mask = self.embedding(**kwargs)

        kwargs = dict(
            Q=user_emb["anchor"],
            K=user_emb["history"],
            V=user_emb["history"],
            mask=mask["user"],
        )
        user_pooled, user_kl = self.pooling["user"](**kwargs)

        kwargs = dict(
            Q=item_emb["anchor"],
            K=item_emb["history"],
            V=item_emb["history"],
            mask=mask["item"],
        )
        item_pooled, item_kl = self.pooling["item"](**kwargs)

        X_pred = self.matching(user_pooled, item_pooled)

        return X_pred, (user_kl+item_kl)/2

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        X_pred, kl = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred).squeeze(-1)
        return logit, kl

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            name="bimodal",
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            histories=self.histories,
        )
        self.embedding = embedding_builder(**kwargs)

        kwargs = dict(
            dim=self.embedding_dim,
            dist=self.dist,
            hyper_approx=self.hyper_approx,
            hyper_prior=self.hyper_prior,
            beta=self.beta,
        )
        components = dict(
            user=BayesianAttentionModules(**kwargs),
            item=BayesianAttentionModules(**kwargs),
        )
        self.pooling = nn.ModuleDict(components)

        self.matching = MatrixFactorizationLayer(**kwargs)

        kwargs = dict(
            dim=self.embedding_dim,
        )
        self.prediction = ProjectionLayer(**kwargs)