# coding=utf-8
from typing import Iterable

from torch.utils.data import DataLoader

DATALOADERS = DataLoader | Iterable[DataLoader]


class ModelHooks:
    """Hooks to be used in BaseModel"""

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.
        If on DDP it is called on every process.
        """
        pass

    def on_fit_end(self) -> None:
        """Called at the very end of fit.
        If on DDP it is called on every process.
        """
        pass

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""
        pass

    def on_train_end(self) -> None:
        """Called at the end of training before logger experiment is closed."""
        pass

    def on_train_epoch_start(self) -> None:
        """Called at the beginning of training epoch"""
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch"""
        pass

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self) -> None:
        """Called at the end of validation."""
        pass

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        pass

    def on_test_end(self) -> None:
        """Called at the end of testing."""
        pass

    def on_predict_start(self) -> None:
        """Called at the beginning of predicting."""
        pass

    def on_predict_end(self) -> None:
        """Called at the end of predicting."""
        pass


class DataHooks:
    """Hooks to be used for data related method"""

    def __init__(self) -> None:
        """
        Attributes:
            prepare_data_per_node:
                If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data.
            allow_zero_length_dataloader_with_multiple_devices:
                If True, dataloader with zero length within local rank is allowed.
                Default value is False.
        """
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def prepare_data(self) -> None:
        """Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single
        process, so you can safely add your downloading logic within.

        .. warning:: DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In a distributed environment, ``prepare_data`` can be called in two ways
        (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = True


            # call on GLOBAL_RANK=0 (great for shared file systems)
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = False

        This is called before requesting the dataloaders:

        .. code-block:: python

            model.prepare_data()
            initialize_distributed()
            model.setup(stage)
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()
            model.predict_dataloader()
        """
        pass

    def setup(self, stage: str):
        """Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process
        when using DDP.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        Example::

            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)
        """

    def teardown(self, stage: str) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """

    def train_dataloader(self) -> DATALOADERS:
        """Implement one or more PyTorch DataLoaders for training.

        Return:
            A collection of :class:`torch.utils.data.DataLoader` specifying training samples.
            In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Example::

            # single dataloader
            def train_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                return loader

            # multiple dataloaders, return as list
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a list of tensors: [batch_mnist, batch_cifar]
                return [mnist_loader, cifar_loader]

            # multiple dataloader, return as dict
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a dict of tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
                return {'mnist': mnist_loader, 'cifar': cifar_loader}
        """
        pass

    def val_dataloader(self) -> DATALOADERS:
        pass

    def test_dataloader(self) -> DATALOADERS:
        pass

    def predict_dataloader(self) -> DATALOADERS:
        pass
