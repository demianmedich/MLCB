# coding=utf-8
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mlcb.model.abc_model import ABCHookBasedModel


class Trainer:
    def __init__(
        self,
        num_gpus: int = 1,
        num_nodes: int = 1,
        max_epochs: int = 1,
        max_steps: int = -1,
    ) -> None:
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.max_epochs = max_epochs
        self.max_steps = max_steps

        self._current_epoch = 0
        self._current_step = 0

        self._model: ABCHookBasedModel | None = None
        self._optimizers: list[Optimizer] = []
        # TODO: Supports learning-rate scheduler

    @property
    def optimizers(self) -> list[Optimizer]:
        return self._optimizers

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_step(self) -> int:
        return self._current_step

    def fit(
        self,
        model: ABCHookBasedModel,
        optimizers: Optimizer | Iterable[Optimizer],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        # TODO: Supports distributed training
        self._set_model(model)
        self._set_optimizers(optimizers)
        if ckpt_path:
            self._load_checkpoint(Path(ckpt_path))

        self._train_on_device(train_dataloader, val_dataloader)

    def _set_model(self, model: ABCHookBasedModel):
        self._model = model
        model.trainer = self

    def _set_optimizers(self, optimizers: Optimizer | Iterable[Optimizer]):
        if isinstance(optimizers, Optimizer):
            self._optimizers.append(optimizers)
        elif isinstance(optimizers, Iterable):
            self._optimizers.extend(optimizers)
        else:
            raise TypeError(f"Unsupported type {type(optimizers)}")

    def _load_checkpoint(self, ckpt_path: Path):
        # TODO: Supports checkpoint loading
        ...

    def _train_on_device(
        self, train_dataloader: DataLoader, val_dataloader: DataLoader | None
    ) -> None:
        model = self._model
        model.on_fit_start()

        # TODO: Supports sanity check loop
        model.on_train_start()
        for epoch in range(1, self.max_epochs + 1):
            self._current_epoch = epoch
            self._fit_loop(train_dataloader)

            if val_dataloader is not None:
                self._validation_loop(val_dataloader)
        model.on_train_end()

        model.on_fit_end()

    def _fit_loop(self, dataloader: DataLoader) -> None:
        model = self._model
        model.on_train_epoch_start()

        train_output_list = []

        # TODO: Supports TQDM progress bar
        for i, batch in enumerate(dataloader):
            self._current_step += 1

            batch = self._transfer_batch_to_device(batch)
            train_output = model.training_step(batch, i)
            if train_output is None:
                continue
            train_output_list.append(train_output)

            if not model.automatic_optimization:
                continue

            # TODO: Supports gradient accumulation
            self._optimizer_zero_grad()
            self._backward(train_output)
            self._optimizer_step()

        model.on_train_epoch_end(train_output_list)

    def _validation_loop(self, dataloader: DataLoader) -> None:
        model = self._model

        model.eval()
        torch.set_grad_enabled(False)
        model.on_validation_start()

        val_output_list = []
        for i, batch in enumerate(dataloader):
            batch = self._transfer_batch_to_device(batch)
            val_output = model.validation_step(batch, i)
            val_output_list.append(val_output)

        model.on_validation_end(val_output_list)
        model.train()
        torch.set_grad_enabled(True)

    def _optimizer_zero_grad(self) -> None:
        for optim in self._optimizers:
            optim.zero_grad()

    def _backward(self, train_output: Tensor | dict[str, Any]) -> None:
        if isinstance(train_output, dict):
            if "loss" not in train_output:
                raise KeyError("training_step() should contains key `loss`.")
            train_output = train_output["loss"]
        train_output.backward()

    def _optimizer_step(self) -> None:
        for optim in self._optimizers:
            optim.step()

    def _transfer_batch_to_device(self, batch: Any) -> Any:
        # TODO: Supports tensor-to-device
        return batch
