from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairwiseEngine:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion,
        beta: float,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.beta = beta
        self.scaler = GradScaler(device=DEVICE)

    def __call__(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        epoch: int,
    ):
        kwargs = dict(
            dataloader=trn_loader,
            epoch=epoch,
            obj="TRN",
        )
        self.model.train()
        trn_nll, trn_kl = self._epoch_step(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            epoch=epoch,
            obj="VAL",
        )
        with torch.no_grad():
            self.model.eval()
            val_nll, val_kl = self._epoch_step(**kwargs)

        return (trn_nll, trn_kl), (val_nll, val_kl)

    def _epoch_step(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        obj: str,
    ):
        epoch_nll = 0.0
        epoch_kl = 0.0

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"EPOCH {epoch+1} {obj}"
        )

        for user_idx, pos_idx, neg_idx in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                pos_idx=pos_idx.to(DEVICE), 
                neg_idx=neg_idx.to(DEVICE),
                obj=obj,
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_nll, batch_kl = self._batch_step(**kwargs)
                batch_loss = batch_nll + self.beta * batch_kl

            # backward pass
            if obj=="TRN":
                self._run_backprop(batch_loss)

            # accumulate loss
            epoch_nll += batch_nll.item()
            epoch_kl += batch_kl.item()

        return epoch_nll / len(dataloader), epoch_kl / len(dataloader)

    def _batch_step(self, user_idx, pos_idx, neg_idx, obj):
        fn = (
            self.model.estimate
            if obj=="TRN"
            else self.model.predict
        )
        pos_logit, pos_kl = fn(user_idx, pos_idx)
        neg_logit, neg_kl = fn(user_idx, neg_idx)
        nll = self.criterion(pos_logit, neg_logit)
        return nll, (pos_kl + neg_kl)/2

    def _run_backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()