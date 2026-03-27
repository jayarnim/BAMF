import torch
import torch.nn as nn
from ..viewer import HistoryIDXViewer

class IDXEmbeddingWithHistory(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        histories: dict[str, torch.Tensor],
    ):
        super().__init__()

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.histories = histories

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ):
        kwargs = dict(
            anchor_idx=user_idx,
            target_idx=item_idx,
        )
        user_hist_idx, user_mask = self.viewer["user"](**kwargs)

        kwargs = dict(
            anchor_idx=item_idx,
            target_idx=user_idx,
        )
        item_hist_idx, item_mask = self.viewer["item"](**kwargs)

        user_emb_slice = dict(
            anchor=self.user["anchor"](user_idx),
            history=self.item["history"](user_hist_idx),
        )

        item_emb_slice = dict(
            anchor=self.item["anchor"](item_idx),
            history=self.user["history"](item_hist_idx),
        )
        
        mask = dict(
            user=user_mask,
            item=item_mask,
        )

        return user_emb_slice, item_emb_slice, mask

    def _set_up_components(self):
        self._create_layers()
        self._create_embeddings()
        self._init_embeddings()

    def _create_layers(self):
        kwargs = dict(
            histories=self.histories["user"],
            padding_idx=self.num_items,
        )
        viewer_user = HistoryIDXViewer(**kwargs)

        kwargs = dict(
            histories=self.histories["item"],
            padding_idx=self.num_users,
        )
        viewer_item = HistoryIDXViewer(**kwargs)

        components = dict(
            user=viewer_user,
            item=viewer_item,
        )
        self.viewer = nn.ModuleDict(components)

    def _create_embeddings(self):
        kwargs = dict(
            num_embeddings=self.num_users+1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_users,
        )
        components = dict(
            anchor=nn.Embedding(**kwargs),
            history=nn.Embedding(**kwargs),
        )
        self.user = nn.ModuleDict(components)

        kwargs = dict(
            num_embeddings=self.num_items+1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_items,
        )
        components = dict(
            anchor=nn.Embedding(**kwargs),
            history=nn.Embedding(**kwargs),
        )
        self.item = nn.ModuleDict(components)

    def _init_embeddings(self):
        embeddings = [
            self.user,
            self.item,
        ]

        for emb in embeddings:
            for name, val in emb.items():
                kwargs = dict(
                    tensor=val.weight, 
                    mean=0.0, 
                    std=0.01,
                )
                nn.init.normal_(**kwargs)