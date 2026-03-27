from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointwiseEngine(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion,
        annealer,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.annealer = annealer
        self.scaler = GradScaler(device=DEVICE)
        self.current_epoch = 0

    def __call__(
        self, 
        dataloader: torch.utils.data.dataloader.DataLoader, 
    ):
        self.current_epoch += 1

        # train
        self.model.train()

        # reset epoch loss
        epoch_nll = 0.0
        epoch_kl = 0.0

        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {self.current_epoch} TRN"
        )

        for user_idx, item_idx, label in tqdm(**kwargs):
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(DEVICE),
                item_idx=item_idx.to(DEVICE), 
                label=label.to(DEVICE),
            )

            # forward pass
            with autocast(DEVICE.type):
                batch_nll, batch_kl = self.batch_step(**kwargs)
                batch_loss = batch_nll + self.annealer(self.current_epoch) * batch_kl
            
            # backward pass
            self.backprop(batch_loss)

            # accumulate loss
            epoch_nll += batch_nll.item()
            epoch_kl += batch_kl.item()

        return epoch_nll / len(dataloader), epoch_kl / len(dataloader)

    def batch_step(self, user_idx, item_idx, label):
        logit, kl = self.model.predict(user_idx, item_idx)
        nll = self.criterion(logit, label)
        return nll, kl

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()