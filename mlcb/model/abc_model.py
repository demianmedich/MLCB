# coding=utf-8
import weakref
from abc import ABCMeta
from typing import TYPE_CHECKING, Any, Optional

from torch.nn import Module
from torch.optim import Optimizer

from mlcb.hooks import ModelHooks

if TYPE_CHECKING:
    from mlcb.trainer import Trainer


class ABCModel(Module, metaclass=ABCMeta):
    """Class for Basic Model Unit"""

    def forward(self, *args, **kwargs) -> Any:
        """Feed-forward method

        Return last layer output of this model.
        """
        raise NotImplementedError()


class ABCHookBasedModel(ModelHooks, Module, metaclass=ABCMeta):
    """Class for Hook based Model using ABCModel.

    This model will be used in `Trainer`
    """

    def __init__(self):
        super().__init__()

        self._trainer: Optional["Trainer"] = None
        self._automatic_optimization: bool = True

    def forward(self, *args, **kwargs) -> Any:
        """Inference method.

        This is for inference method not same with torch.Module.forward().

        Returns:
            inference result
        """
        pass

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
        """Execute validation step on a mini-batch.

        In this step, you'd might generate examples or calculate something related with evaluation metric.

        Args:
            batch: The output of your DataLoader
            batch_idx: The index of a mini-batch

        Returns:
            Any: Any object or values
            None: this step will skip to the next batch
        """
        pass

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        """Execute test step on a mini-batch.

        In this step, you'd might generate examples or calculate something related with evaluation metric.

        Args:
            batch: The output of your DataLoader
            batch_idx: The index of a mini-batch

        Returns:
            Any: Any object or values
            None: this step will skip to the next batch
        """
        pass

    def prediction_step(self, batch: Any, batch_idx: int) -> Any:
        """Execute prediction step to calculate

        Args:
            batch: The output of your DataLoader
            batch_idx: The index of a mini-batch

        Returns:
            Any: Any object or values
            None: this step will skip to the next batch
        """
        pass

    def evaluate_epoch_metrics(
        self, stage: str, predictions: list[Any], labels: list[Any]
    ) -> None:
        """Evaluate metrics using given predictions and labels.

        Args:
            stage: string one of [`val`, `test`]
            predictions: a list of prediction steps
            labels: a list of label
        """
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

        opts = trainer.optimizers

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
