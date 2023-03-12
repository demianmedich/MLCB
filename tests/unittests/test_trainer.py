# coding=utf-8
import random
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mlcb.model.abc_model import ABCHookBasedModel, ABCModel
from mlcb.trainer import Trainer


class MockModel(ABCModel):
    def __init__(self) -> None:
        super().__init__()
        self.fc = Linear(1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class MockHookBasedModelModel(ABCHookBasedModel):
    def __init__(self):
        super().__init__()

        self.model = MockModel()
        self.cnt_training_step = 0
        self.cnt_validation_step = 0
        self.cnt_test_step = 0

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        self.cnt_training_step += 1

        x, y = batch
        if x.dim() == 1:
            x = torch.unsqueeze(x, -1)
        if y.dim() == 1:
            y = torch.unsqueeze(y, -1)
        out = self.model(x)

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
        out = self.model(x)
        return out, y

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        self.cnt_test_step += 1

        x, y = batch
        out = self.model(x)
        return out, y

    def on_validation_end(self, output_list: list[Any]) -> None:
        prediction_list, y_list = [], []
        for prediction, y in output_list:
            prediction_list.append(prediction)
            y_list.append(y)

        val_loss = mse_loss(torch.vstack(prediction_list), torch.vstack(y_list))
        epoch = self.trainer.current_epoch

        logger.info(f"[EPOCH {epoch:03d}] validation loss: {val_loss}")

    def on_test_end(self, output_list: list[Any]) -> None:
        prediction_list, y_list = [], []
        for prediction, y in output_list:
            prediction_list.append(prediction)
            y_list.append(y)

        test_loss = mse_loss(torch.vstack(prediction_list), torch.vstack(y_list))
        epoch = self.trainer.current_epoch

        logger.info(f"[EPOCH {epoch:03d}] test loss: {test_loss}")


class MockDataset(Dataset):
    def __init__(self, start: int, end: int, sample_size: int):
        self.start = start
        self.end = end

        x_list = [float(i) for i in range(start, end)]
        y_list = [2 * x + 1 for x in x_list]

        idx_list = random.sample([i for i in range(start, end)], sample_size)

        self.x_list = []
        self.y_list = []
        for idx in idx_list:
            self.x_list.append(x_list[idx])
            self.y_list.append(y_list[idx])

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return (
            torch.tensor(self.x_list[index], dtype=torch.float32),
            torch.tensor(self.y_list[index], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.x_list)


def test_trainer_toy_dataset():
    train_dataset = MockDataset(0, 10000, 500)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = MockDataset(0, 10000, 25)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataset = MockDataset(0, 10000, 50)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    model = MockHookBasedModelModel()
    optimizer = SGD(model.parameters(), 0.000000001)

    num_epochs = 2

    trainer = Trainer(max_epochs=num_epochs)
    trainer.fit(model, optimizer, train_dataloader, val_dataloader)
    assert model.cnt_training_step == len(train_dataloader) * num_epochs
    assert model.cnt_validation_step == len(val_dataloader) * num_epochs

    trainer.test(model, test_dataloader)
    assert model.cnt_test_step == len(test_dataloader)
