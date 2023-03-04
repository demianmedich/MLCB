# coding=utf-8
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Iterable, Literal

from loguru import logger
from pydantic import BaseSettings
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mlcb.util import rank_zero_warn


class BaseConfig(BaseSettings):
    @abstractmethod
    def instantiate(self, *args, **kwargs) -> Any:
        """abstract class method to instantiate config to object"""
        raise NotImplementedError()


class ModelConfig(ABC, BaseConfig):
    ...


class DataLoaderConfig(ABC, BaseConfig):
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = False
    pin_memory: bool = False


class OptimizerConfig(ABC, BaseConfig):
    lr: float = 0.1


class LrSchedulerConfig(ABC, BaseConfig):
    step_interval: Literal["step", "epoch"] = "step"


TaskMode = Literal["train", "eval", "predict"]


class BaseTaskConfig(ABC, BaseConfig):
    mode: TaskMode
    train_dataloader_cfg: DataLoaderConfig | None = None
    val_dataloader_cfg: DataLoaderConfig | None = None
    test_dataloader_cfg: DataLoaderConfig | None = None
    predict_dataloader_cfg: DataLoaderConfig | None = None
    model_cfg: ModelConfig | None = None
    optimizer_cfg: OptimizerConfig | None = None
    lr_scheduler_cfg: LrSchedulerConfig | None = None


DataLoaderType = DataLoader | Iterable[DataLoader] | None


class BaseTask(metaclass=ABCMeta):
    def __init__(self, cfg: BaseTaskConfig):
        # TODO: cfg를 이용해서 각 cfg가 존재하면, 학습에 필요한 것들을 초기화할 것.
        self._mode = cfg.mode
        self._train_dataloader: DataLoader | None = None
        self._val_dataloader: DataLoaderType = None
        self._test_dataloader: DataLoaderType = None
        self._predict_dataloader: DataLoaderType = None

    @property
    def train_dataloader(self) -> DataLoader | None:
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoaderType:
        return self._val_dataloader

    @property
    def test_dataloader(self) -> DataLoaderType:
        return self._test_dataloader

    @property
    def predict_dataloader(self) -> DataLoaderType:
        return self._predict_dataloader

    def setup_dataloader(self, cfg: BaseTaskConfig):
        if cfg.train_dataloader_cfg is not None:
            self._train_dataloader = cfg.train_dataloader_cfg.instantiate()
        if cfg.val_dataloader_cfg is not None:
            self._val_dataloader = cfg.val_dataloader_cfg.instantiate()
        if cfg.test_dataloader_cfg is not None:
            self._test_dataloader = cfg.test_dataloader_cfg.instantiate()
        if cfg.predict_dataloader_cfg is not None:
            self._predict_dataloader = cfg.predict_dataloader_cfg.instantiate()

    def setup_model(self, cfg: BaseTaskConfig, from_checkpoint: bool):
        """"""
        rank_zero_warn("You must implement setup_model() to use optimizer")

    def setup_optimizer(self) -> Optimizer | list[Optimizer]:
        rank_zero_warn("You must implement setup_optimizer() to use optimizer")

    def setup_lr_scheduler(self):
        rank_zero_warn("")

    def start_task(self):
        logger.info(f"{self.__class__.__name__} start_task()")

        self.setup_model()

        if self._mode == "train":
            self.train()
        elif self._mode == "eval":
            if self.val_dataloader:
                self.validate()

        elif self._mode == "predict":
            self.predict()

        logger.info(f"{self.__class__.__name__} start_task() done")
