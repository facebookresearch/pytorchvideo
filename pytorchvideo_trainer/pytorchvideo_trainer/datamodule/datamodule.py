# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import hydra
import pytorch_lightning as pl
import pytorchvideo.data
import torch
from hydra.core.config_store import ConfigStore

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from pytorchvideo_trainer.datamodule.transforms import build_transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchrecipes.core.conf import DataModuleConf
from torchrecipes.utils.config_utils import get_class_name_str


class PyTorchVideoDataModule(pl.LightningDataModule):
    """
    A PyTorch-Lightning DataModule module supporting all the dataloaders
    in PyTorchVideo for different phases (train, validation and testing) of
    Lightning tranining.

    Supports loading any aribtrary iterable and map-style PyTorchVideo dataset
    upon following the config schema detailed below.

    Args:
        dataloader (DataLoaderConf):
            An OmegaConf / Hydra Config object consisting of dataloder
            config for each phase i.e, train, val and test.

            The Hydra schema for this config is as defined in
            `pytorchvideo_trainer.datamodule.datamodule.DataLoaderConf`

            One such example config can be found at
            `pytorchvideo_trainer/conf/datamodule/dataloader/kinetics_classification.yaml`

        transforms (TransformsConf):
            An OmegaConf / Hydra Config object consisting of transforms
            config for each phase i.e, train, val and test.

            The Hydra schema for this config is as defined in
            `pytorchvideo_trainer.datamodule.datamodule.TransformsConf`

            One such example config used for Resnet50 video model traning can be found at
            `pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_slow.yaml`
    """

    def __init__(
        self,
        dataloader: DataLoaderConf,
        transforms: TransformsConf,
    ) -> None:
        super().__init__()
        self.config: Dict[str, Any] = {
            "train": dataloader.train,
            "val": dataloader.val,
            "test": dataloader.test,
        }
        self.transforms: Dict[str, Any] = {
            "train": build_transforms(transforms.train),
            "val": build_transforms(transforms.val),
            "test": build_transforms(transforms.test),
        }
        self.datasets: dict[str, Any] = {"train": None, "val": None, "test": None}

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:
            self.datasets["train"] = self._get_dataset(
                phase="train", transforms=self.transforms["train"]
            )
            self.datasets["val"] = self._get_dataset(
                phase="val", transforms=self.transforms["val"]
            )
        if stage == "test" or stage is None:
            self.datasets["test"] = self._get_dataset(
                phase="test", transforms=self.transforms["test"]
            )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer uses.
        """
        if (
            self.trainer
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            self.datasets["train"].video_sampler.set_epoch(self.trainer.current_epoch)

        return self._get_dataloader("train")

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Defines the val DataLoader that the PyTorch Lightning Trainer uses.
        """
        return self._get_dataloader("val")

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Defines the test DataLoader that the PyTorch Lightning Trainer uses.
        """
        return self._get_dataloader("test")

    def _get_dataloader(self, phase: str) -> DataLoader:
        assert self.datasets[phase] is not None, "Failed to get the {} dataset!".format(
            phase
        )

        if isinstance(self.datasets[phase], torch.utils.data.IterableDataset):
            return torch.utils.data.DataLoader(
                self.datasets[phase],
                batch_size=self.config[phase].batch_size,
                num_workers=self.config[phase].num_workers,
                pin_memory=self.config[phase].pin_memory,
                drop_last=self.config[phase].drop_last,
                collate_fn=hydra.utils.instantiate(self.config[phase].collate_fn),
                worker_init_fn=hydra.utils.instantiate(
                    self.config[phase].worker_init_fn
                ),
            )
        else:
            sampler = None
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                logging.info(
                    "Distributed Environmnet detected, using DistributedSampler for dataloader."
                )
                sampler = DistributedSampler(self.datasets[phase])

            return torch.utils.data.DataLoader(
                self.datasets[phase],
                batch_size=self.config[phase].batch_size,
                num_workers=self.config[phase].num_workers,
                pin_memory=self.config[phase].pin_memory,
                drop_last=self.config[phase].drop_last,
                sampler=sampler,
                shuffle=(False if sampler else self.config[phase].shuffle),
                collate_fn=hydra.utils.instantiate(self.config[phase].collate_fn),
                worker_init_fn=hydra.utils.instantiate(
                    self.config[phase].worker_init_fn
                ),
            )

    def _get_dataset(
        self,
        phase: str,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> pytorchvideo.data.LabeledVideoDataset:

        video_sampler = RandomSampler
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logging.info(
                "Distributed Environmnet detected, using DistributedSampler for dataset."
            )
            video_sampler = DistributedSampler

        dataset = hydra.utils.instantiate(
            self.config[phase].dataset,
            transform=transforms,
            video_sampler=video_sampler,
        )
        return dataset


@dataclass
class PhaseDataLoaderConf:

    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    batch_size: int = MISSING
    shuffle: bool = True

    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    collate_fn: Optional[Any] = None
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    worker_init_fn: Optional[Any] = None

    ## Dataset Related
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    dataset: Any = MISSING


@dataclass
class DataLoaderConf:
    train: PhaseDataLoaderConf = MISSING
    val: PhaseDataLoaderConf = MISSING
    test: PhaseDataLoaderConf = MISSING


@dataclass
class TransformsConf:

    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    train: List[Any] = MISSING

    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    val: List[Any] = MISSING

    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    test: List[Any] = MISSING


@dataclass
class VideoClassificationDataModuleConf(DataModuleConf):
    _target_: str = get_class_name_str(PyTorchVideoDataModule)

    dataloader: DataLoaderConf = MISSING
    transforms: TransformsConf = MISSING


cs = ConfigStore()

cs.store(
    group="schema/datamodule",
    name="ptv_video_classification_data_module_conf",
    node=VideoClassificationDataModuleConf,
    package="datamodule",
)
