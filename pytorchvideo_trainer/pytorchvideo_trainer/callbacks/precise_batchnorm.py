# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Generator

import torch
from fvcore.nn.precise_bn import update_bn_stats
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader


class PreciseBn(Callback):
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this callaback, the BN stats are recomputed with fixed weights, to make
    the running average more precise during Training Phase. Specifically, it
    computes the true average of per-batch mean/variance instead of the
    running average. See Sec. 3 of the paper "Rethinking Batch in BatchNorm"
    for details.
    """

    def __init__(self, num_batches: int) -> None:
        """
        Args:
            num_batches (int): Number of steps / mini-batches to
            perform to sample for updating the precise batchnorm
            stats.
        """
        self.num_batches = num_batches

    def _get_precise_bn_loader(
        self, data_loader: DataLoader, pl_module: LightningModule
    ) -> Generator[torch.Tensor, None, None]:
        for batch in data_loader:
            inputs = batch[pl_module.modality_key]
            if isinstance(inputs, list):
                inputs = [x.to(pl_module.device) for x in inputs]
            else:
                inputs = inputs.to(pl_module.device)
            yield inputs

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """
        Called at the end of every epoch only during the training
        phase.

        Args:
            trainer (Trainer): A PyTorch-Lightning trainer object.
            pl_module (LightningModule): A PyTorch-Lightning module.
            Typically supported modules include -
            pytorchvideo_trainer.module.VideoClassificationModule, etc.
        """
        # pyre-ignore[16]
        dataloader = trainer.datamodule.train_dataloader()
        precise_bn_loader = self._get_precise_bn_loader(
            data_loader=dataloader, pl_module=pl_module
        )
        update_bn_stats(
            model=pl_module.model,  # pyre-ignore[6]
            data_loader=precise_bn_loader,
            num_iters=self.num_batches,
        )
