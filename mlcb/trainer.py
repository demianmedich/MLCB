# coding=utf-8
from pathlib import Path
from typing import Iterable, Any

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mlcb.model.abc_model import ABCModel


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

        self._model: ABCModel | None = None
        self._optimizers: list[Optimizer] = []
        # TODO: Supports learning-rate scheduler

    @property
    def optimizers(self) -> list[Optimizer]:
        return self._optimizers

    def fit(
            self,
            model: ABCModel,
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

    def _set_model(self, model: ABCModel):
        self._model = model

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
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader | None
    ) -> None:

        model = self._model
        model.on_fit_start()

        # TODO: Supports sanity check loop
        model.on_train_start()
        for epoch in range(1, self.max_epochs + 1):
            self._fit_loop(train_dataloader)
        model.on_train_end()

        if val_dataloader is not None:

        model.on_fit_end()

    def _fit_loop(self, dataloader: DataLoader) -> None:
        model = self._model
        model.on_train_epoch_start()

        # TODO: Supports TQDM progress bar
        for i, batch in enumerate(dataloader):
            model.training_step(batch, i)

        model.on_train_epoch_end()

    def _to_device(self, batch: Any) -> Any:
        # TODO: Supports tensor-to-device
        return batch
