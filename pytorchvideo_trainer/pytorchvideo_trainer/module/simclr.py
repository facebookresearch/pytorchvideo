# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pytorchvideo.models.resnet import create_resnet
from pytorchvideo.models.weight_init import init_net_weights
from pytorchvideo_trainer.module.byol import create_mlp_util
from pytorchvideo_trainer.module.ssl_helper import SSLBaseModule
from pytorchvideo_trainer.module.video_classification import (
    Batch,
    BatchKey,
    EnsembleMethod,
)
from torchrecipes.core.conf import ModuleConf
from torchrecipes.utils.config_utils import get_class_name_str


class SimCLR(nn.Module):
    """
    Skeletal NN.Module for the SimCLR model that supports
    arbitrary bacbone and projector models.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            backbone (nn.Module): backbone for simclr, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector (nn.Module): An mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
        """
        super().__init__()

        if projector is not None:
            backbone = nn.Sequential(
                backbone,
                projector,
            )
        init_net_weights(backbone)
        self.backbone = backbone

    def forward(
        self, x_list: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x_list (list(tensor) or tensor): Expects a list of 2 tensors
                for trainin phase and single tensor for the train and val
                phases. Here all tensors are expected to be of the shape,
                N x C x T x H x W.
        """
        if not self.training:
            assert isinstance(
                x_list, torch.Tensor
            ), "Expected tensor for test/val phase in SimCLR"
            if self.backbone is not None:
                x_list = self.backbone(x_list)
            x_list = F.normalize(x_list, p=2, dim=1)
            return x_list

        assert (
            isinstance(x_list, list) and len(x_list) == 2
        ), f"Invalid list input to SimCLR. Expected len 2 but received {len(x_list)}"

        for i, x in enumerate(x_list):
            if self.backbone is not None:
                x = self.backbone(x)
            x = F.normalize(x, p=2, dim=1)
            x_list[i] = x

        return x_list


def create_simclr_resnet_50(
    # Backbone
    backbone_creator: Callable = create_resnet,  # pyre-ignore[24]
    backbone_embed_dim: int = 128,
    dim_in: int = 2048,
    # Projector
    # TODO: Standardize projector conf across all SSL tasks
    mlp_activation: Callable = nn.ReLU,  # pyre-ignore[24]
    mlp_inner_dim: int = 2048,
    mlp_depth: int = 1,
    mlp_norm: Optional[Callable] = None,  # pyre-ignore[24]
) -> SimCLR:
    """
    Builds a Resnet video model with a projector for SimCLR
    SSL traning task.
    """
    backbone = backbone_creator(
        model_num_class=backbone_embed_dim,
        dropout_rate=0.0,
    )
    backbone.blocks[-1].proj = None
    projector = create_mlp_util(
        dim_in,
        backbone_embed_dim,
        mlp_inner_dim,
        mlp_depth,
        norm=mlp_norm,  # pyre-ignore[6]
    )
    simclr = SimCLR(
        backbone=backbone,
        projector=projector,
    )
    return simclr


class SimCLRModule(SSLBaseModule):
    """
    The Lightning Base module for SimCLR SSL video task.

    For more details refer to,
    1. A Simple Framework for Contrastive Learning of Visual Representations :
        https://arxiv.org/abs/2002.05709
    2. A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning

    Args:
        model (OmegaConf): An omega conf object intializing the neural-network modle.
            Example configs can be found in `pytorchvideo_trainer/conf/module/model`
        loss(OmegaConf): An omega conf object intializing the loss function.
            Example configs can be found in `pytorchvideo_trainer/conf/module/loss`
        optim (OmegaConf): An omega conf object for constructing the optimizer object.
            The associated config schema can be found at
            `pytorchvideo_trainer.module.optimizer.OptimizerConf`.
            Example configs can be found in `pytorchvideo_trainer/conf/module/optim`
        metrics (OmegaConf): The metrics to track, which will be used for both train,
            validation and test. Example configs can be found in
            `pytorchvideo_trainer/conf/module/metricx`
        lr_scheduler (OmegaConf): An omega conf object associated with learning rate
            scheduler used during trainer.
            The associated config schema can be found at
            `pytorchvideo_trainer.module.lr_policy.LRSchedulerConf`.
            Example configs can be found in `pytorchvideo_trainer/conf/module/lr_scheduler`
        modality_key (str): The modality key used in data processing, default: "video".
        ensemble_method (str): The data ensembling method to control how we accumulate
            the testing results at video level, which is optional. Users may choose from
            ["sum", "max", None], If it is set to None, no data ensembling will be applied.
        knn_memory (OmegaConf): An optional hydra / omeaga conf, if set, initializes KNN
            Memory module to use. Example config can be found at,
            `pytorchvideo_trainer/conf/module/knn_memory`.
        num_sync_devices (int): Number of gpus to sync bathcnorm over. Only works if
            pytorch lightning trainer's sync_batchnorm parameter is to false.
    """

    def __init__(
        self,
        model: Any,  # pyre-ignore[2]
        loss: Any,  # pyre-ignore[2]
        optim: Any,  # pyre-ignore[2]
        metrics: List[Any],  # pyre-ignore[2]
        lr_scheduler: Optional[Any] = None,  # pyre-ignore[2]
        modality_key: BatchKey = "video",
        ensemble_method: Optional[EnsembleMethod] = None,
        knn_memory: Optional[Any] = None,  # pyre-ignore[2]
        num_sync_devices: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            loss=loss,
            optim=optim,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            modality_key=modality_key,
            ensemble_method=ensemble_method,
            knn_memory=knn_memory,
            momentum_anneal_cosine=False,
            num_sync_devices=num_sync_devices,
        )

    def training_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> None:

        self.cur_epoch_step += 1  # pyre-ignore[16]

        self.manual_zero_opt_grad()
        self.manual_update_lr()

        inputs = batch[self.modality_key]  # pyre-ignore[6]
        partial_loss = 0.0
        for i in range(len(inputs) - 1):
            y_hat = self(inputs[i : i + 2])
            loss = self.loss(y_hat)
            self.manual_backward(loss)
            partial_loss += loss.detach()

        partial_loss /= len(inputs) - 1
        self.log("Losses/train_loss", partial_loss, on_step=True, on_epoch=True)

        if self.knn_memory is not None:
            # pyre-ignore[29]
            self.knn_memory.update(y_hat[0], batch["video_index"])

        self.manual_opt_step()


@dataclass
class SimCLRModuleConf(ModuleConf):
    _target_: str = get_class_name_str(SimCLRModule)
    model: Any = MISSING  # pyre-ignore[4]
    loss: Any = MISSING  # pyre-ignore[4]
    optim: Any = MISSING  # pyre-ignore[4]
    metrics: List[Any] = MISSING  # pyre-ignore[4]
    lr_scheduler: Optional[Any] = None  # pyre-ignore[4]
    modality_key: str = "video"
    ensemble_method: Optional[str] = None
    num_sync_devices: Optional[int] = 1
    knn_memory: Optional[Any] = None  # pyre-ignore[4]


cs = ConfigStore()
cs.store(
    group="schema/module",
    name="simclr_module_conf",
    node=SimCLRModuleConf,
    package="module",
)
