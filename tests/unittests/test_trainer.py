# coding=utf-8
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mlcb.model.abc_model import ABCModel
from mlcb.trainer import Trainer


class MockModel(ABCModel):
    def __init__(self):
        super().__init__()

        self.fc = Linear(1, 1)
        self.cnt_training_step = 0
        self.cnt_validation_step = 0

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        self.cnt_training_step += 1

        x, y = batch
        if x.dim() == 1:
            x = torch.unsqueeze(x, -1)
        if y.dim() == 1:
            y = torch.unsqueeze(y, -1)
        out = self.fc(x)

        # loss = torch.mean(torch.abs(out - y))
        loss = mse_loss(out, y)
        epoch = self.trainer.current_epoch
        step = self.trainer.current_step

        logger.info(
            f"[EPOCH {epoch:03d}][STEPS {step:03d}] training loss: {loss}, x: {x}, out: {out}, y: {y}"
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        self.cnt_validation_step += 1

        x, y = batch
        out = self.fc(x)

        loss = mse_loss(out, y)
        epoch = self.trainer.current_epoch
        step = self.trainer.current_step

        # logger.info(
        #     f"[EPOCH {epoch:03d}][STEPS {step:03d}] validation loss: {loss}, x: {x}, out: {out}, y: {y}"
        # )


class MockDataset(Dataset):
    def __init__(self, start: int, end: int):
        self.x = [float(i) for i in range(start, end)]
        self.y = [float(i * 2) for i in range(start, end)]

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return (
            torch.tensor(self.x[index], dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.x)


def test_trainer_toy_dataset():
    train_dataset = MockDataset(0, 100)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataset = MockDataset(20000, 20200)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    model = MockModel()
    optimizer = SGD(model.parameters(), 0.00001)

    num_epochs = 2

    trainer = Trainer(max_epochs=num_epochs)
    trainer.fit(model, optimizer, train_dataloader, val_dataloader)
    assert model.cnt_training_step == len(train_dataloader) * num_epochs
    assert model.cnt_validation_step == len(val_dataloader) * num_epochs
