# coding=utf-8
import weakref
from abc import ABCMeta, abstractmethod

from torch.nn import Module
from typing import TYPE_CHECKING, Any, Optional

from torch.optim import Optimizer
from mlcb.hooks import ModelHooks

if TYPE_CHECKING:
    from mlcb.trainer import Trainer


class ABCModel(ModelHooks, Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        self._trainer: Optional["Trainer"] = None
        self._automatic_optimization: bool = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Inference method.

        This is for inference method not same with torch.Module.forward()
        """
        raise NotImplementedError()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """Compute training loss and returns it and some additional metrics.

        Args:
            batch (Tensor | (Tensor, ...) | [Tensor, ...]):
                The output of your DataLoader. A tensor, tuple or list.
            batch_idx (int): The index of this batch

        Returns:
            Tensor: The loss tensor
            dict: A dictionary include `loss`
            None: Training will skip to the next batch

        Example:
            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss_fn(out, y)
                return loss

        To use multiple optimizers, you can switch to `manual_optimization` and control their stepping:

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False

            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with generator
                ...
                opt1.step()

                # do training_step with discriminator
                ...
                opt2.step()
        """
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        pass

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        pass

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        pass

    @property
    def trainer(self) -> Optional["Trainer"]:
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: Optional["Trainer"]) -> None:
        if trainer is not None and not isinstance(trainer, weakref.ProxyTypes):
            trainer = weakref.proxy(trainer)
        self._trainer = trainer

    @property
    def optimizers(self) -> Optimizer | list[Optimizer]:
        trainer = self.trainer
        if trainer is None:
            raise Exception("self.trainer is `None`.")

        opts = trainer._optimizers

        if len(opts) == 1 and isinstance(opts[0], Optimizer):
            return opts[0]
        return opts

    @property
    def automatic_optimization(self) -> bool:
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, value: bool) -> None:
        self._automatic_optimization = value

    def freeze(self) -> None:
        """Freeze all model parameters.

        Set `requires_grad` of all parameters to `False` and call model's eval().

        Example:
            model = BaseModel()
            model.freeze()
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all model parameters.

        Set `requires_grad` of all parameters to `True` and call model's train().

        Example:
            model = BaseModel()
            model.unfreeze()
        """
        for param in self.parameters():
            param.requires_grad = True
        self.train()
