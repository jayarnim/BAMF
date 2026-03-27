import torch
import torch.nn as nn
from .components.embedding.builder import embedding_builder
from .components.combination.builder import combination_builder
from .components.matching import MatrixFactorizationLayer
from .components.prediction import ProjectionLayer
from .components.bam.model import BayesianAttentionModules


class Module(nn.Module):
    def __init__(
        self,
        histories: dict[str, torch.Tensor],
        cfg,
    ):
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = cfg.num_users
        self.num_items = cfg.num_items
        self.embedding_dim = cfg.embedding_dim
        self.dist = cfg.dist
        self.hyper_approx = cfg.hyper_approx
        self.hyper_prior = cfg.hyper_prior
        self.beta = cfg.beta
        self.comb = cfg.comb
        self.histories = histories

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
        method: str,
    ):
        pooling = dict(
            user=getattr(self.pooling["user"], method),
            item=getattr(self.pooling["item"], method),
        )

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
        user_pooled, user_kl = pooling["user"](**kwargs)

        kwargs = dict(
            Q=item_emb["anchor"],
            K=item_emb["history"],
            V=item_emb["history"],
            mask=mask["item"],
        )
        item_pooled, item_kl = pooling["item"](**kwargs)

        user_combined = self.combination["user"](user_emb["anchor"], user_pooled)
        item_combined = self.combination["item"](item_emb["anchor"], item_pooled)

        X_pred = self.matching(user_combined, item_combined)

        return X_pred, (user_kl+item_kl)/2

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        X_pred, kl = self.forward(user_idx, item_idx, "estimate")
        logit = self.prediction(X_pred).squeeze(-1)
        return logit, kl

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        X_pred, kl = self.forward(user_idx, item_idx, "predict")
        logit = self.prediction(X_pred).squeeze(-1)
        return logit, kl

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
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

        kwargs = dict(
            name=self.comb,
            dim=self.embedding_dim,
        )
        components = dict(
            user=combination_builder(**kwargs),
            item=combination_builder(**kwargs),
        )
        self.combination = nn.ModuleDict(components)

        self.matching = MatrixFactorizationLayer(**kwargs)

        kwargs = dict(
            dim=(
                self.embedding_dim*2
                if self.comb=="cat"
                else self.embedding_dim
            ),
        )
        self.prediction = ProjectionLayer(**kwargs)