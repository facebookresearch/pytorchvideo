# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pytorchvideo.models.resnet import create_resnet
from pytorchvideo.models.weight_init import init_net_weights
from pytorchvideo_trainer.module.ssl_helper import create_mlp_util, SSLBaseModule
from pytorchvideo_trainer.module.video_classification import (
    Batch,
    BatchKey,
    EnsembleMethod,
)
from torchrecipes.core.conf import ModuleConf
from torchrecipes.utils.config_utils import get_class_name_str


class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent A New Approach to Self-Supervised Learning
    Details can be found in:
    https://arxiv.org/pdf/2006.07733.pdf
    """

    def __init__(
        self,
        mmt: float,
        backbone: nn.Module,
        predictor: nn.Module,
        backbone_mmt: nn.Module,
        projector: Optional[nn.Module] = None,
        projector_mmt: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            backbone (nn.Module): backbone for byol, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector (nn.Module): An mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
            backbone_mmt (nn.Module): backbone for byol, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector_mmt (nn.Module): Am mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
            predictor (nn.Module): predictor MLP of BYOL of similar structure as the
                projector MLP.
            mmt (float): momentum update ratio for the momentum backbone.
        """
        super().__init__()

        self.mmt: float = mmt
        if projector is not None:
            backbone = nn.Sequential(
                backbone,
                projector,
            )
        init_net_weights(backbone)
        self.backbone = backbone

        if projector_mmt is not None:
            backbone_mmt = nn.Sequential(
                backbone_mmt,
                projector_mmt,
            )
        init_net_weights(backbone_mmt)
        self.backbone_mmt = backbone_mmt

        for p in self.backbone_mmt.parameters():
            p.requires_grad = False

        init_net_weights(predictor)
        self.predictor = predictor

        self._copy_weights_to_backbone_mmt()

    def _copy_weights_to_backbone_mmt(self) -> None:
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p
        for name, p in self.backbone_mmt.named_parameters():
            p.data.copy_(dist[name].data)

    @torch.no_grad()
    def momentum_update_backbone(self) -> None:
        """
        Momentum update on the backbone.
        """
        m = self.mmt
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p
        for name, p in self.backbone_mmt.named_parameters():
            # pyre-ignore[41]
            p.data = dist[name].data * (1.0 - m) + p.data * m

    @torch.no_grad()
    def forward_backbone_mmt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward momentum backbone.
        Args:
            x (tensor): input to be forwarded of shape N x C x T x H x W
        """
        with torch.no_grad():
            proj = self.backbone_mmt(x)
        return F.normalize(proj, dim=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Args:
            x (tensor): input to be forwarded of shape N x C x T x H x W
        """
        if not self.training:
            x = self.backbone(x)
            x = F.normalize(x, dim=1)
            return x

        proj = self.backbone(x)
        pred = self.predictor(proj)
        pred = F.normalize(pred, dim=1)

        out_proj = F.normalize(proj, dim=1)

        return out_proj, pred  # pyre-ignore[7]


def create_byol_resnet_50(
    # Backbone
    backbone_creator: Callable = create_resnet,  # pyre-ignore[24]
    backbone_embed_dim: int = 128,
    head_pool: Callable = nn.AdaptiveAvgPool3d,  # pyre-ignore[24]
    head_output_size: Tuple[int, int, int] = (1, 1, 1),
    head_activation: Callable = None,  # pyre-ignore[9,24]
    dropout_rate: float = 0.0,
    # Projector
    projector_dim_in: int = 2048,
    projector_inner_dim: int = 4096,
    projector_depth: int = 2,
    # Predictor
    predictor_inner_dim: int = 4096,
    predictor_depth: int = 2,
    predictor_norm: Callable = nn.BatchNorm1d,  # pyre-ignore[24]
    projector_norm: Callable = nn.BatchNorm1d,  # pyre-ignore[24]
    mmt: float = 0.99,
) -> BYOL:
    """
    Builds a Resnet video backbone, projector and predictors models for
    BYOL SSL task.
    """

    def _make_bacbone_and_projector():  # pyre-ignore[3]
        backbone = backbone_creator(
            dropout_rate=dropout_rate,
            head_activation=head_activation,
            head_output_with_global_average=True,
            head_pool=head_pool,
            head_output_size=head_output_size,
        )

        backbone.blocks[-1].proj = None  # Overwite head projection
        projector = create_mlp_util(
            projector_dim_in,
            backbone_embed_dim,
            projector_inner_dim,
            projector_depth,
            norm=projector_norm,
        )
        return backbone, projector

    backbone, projector = _make_bacbone_and_projector()
    backbone_mmt, projector_mmt = _make_bacbone_and_projector()

    predictor = create_mlp_util(
        backbone_embed_dim,
        backbone_embed_dim,
        predictor_inner_dim,
        predictor_depth,
        norm=predictor_norm,
    )
    byol_model = BYOL(
        mmt=mmt,
        backbone=backbone,
        projector=projector,
        predictor=predictor,
        backbone_mmt=backbone_mmt,
        projector_mmt=projector_mmt,
    )
    return byol_model


class BYOLModule(SSLBaseModule):
    """
    The Lightning Base module for BYOL SSL video task.

    For more details refer to,
    1. Bootstrap your own latent: A new approach to self-supervised Learning:
        https://arxiv.org/abs/2006.07733
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
        momentum_anneal_cosine (bool): For MoCo and BYOL tasks, if set to true, cosine
            anneals the momentum term used from updating the backbone-history model.
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
        momentum_anneal_cosine: bool = False,
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
            momentum_anneal_cosine=momentum_anneal_cosine,
            num_sync_devices=num_sync_devices,
        )

    def training_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> None:
        self.cur_epoch_step += 1  # pyre-ignore[16]

        if self.momentum_anneal_cosine:
            self._cosine_anneal_momentum()

        self.manual_zero_opt_grad()
        self.manual_update_lr()

        inputs = batch[self.modality_key]  # pyre-ignore[6]

        self.model.momentum_update_backbone()  # pyre-ignore[29]
        keys = self._compute_keys(inputs)

        partial_loss = 0.0
        for k, vids in enumerate(inputs):
            other_keys = keys[:k] + keys[k + 1 :]
            assert len(other_keys) > 0, "Length of keys cannot be zero"

            proj, pred = self.model(vids)
            loss_k = self.loss(pred, other_keys[0])
            for i in range(1, len(other_keys)):
                loss_k += self.loss(pred, other_keys[i])
            loss_k /= len(other_keys)

            self.manual_backward(loss_k)
            partial_loss += loss_k.detach()

        if self.knn_memory is not None:
            self.knn_memory.update(proj, batch["video_index"])  # pyre-ignore[29,61]

        partial_loss /= len(inputs) * 2.0  # to have same loss as symmetric loss
        self.log("Losses/train_loss", partial_loss, on_step=True, on_epoch=True)

        self.manual_opt_step()

    @torch.no_grad()
    def _compute_keys(self, x: torch.Tensor) -> List[torch.Tensor]:
        keys = []
        for sub_x in x:
            # pyre-ignore[29]
            keys.append(self.model.forward_backbone_mmt(sub_x).detach())
        return keys


@dataclass
class BYOLModuleConf(ModuleConf):
    _target_: str = get_class_name_str(BYOLModule)
    model: Any = MISSING  # pyre-ignore[4]
    loss: Any = MISSING  # pyre-ignore[4]
    optim: Any = MISSING  # pyre-ignore[4]
    metrics: List[Any] = MISSING  # pyre-ignore[4]
    lr_scheduler: Optional[Any] = None  # pyre-ignore[4]
    modality_key: str = "video"
    ensemble_method: Optional[str] = None
    num_sync_devices: Optional[int] = 1
    knn_memory: Optional[Any] = None  # pyre-ignore[4]
    momentum_anneal_cosine: bool = False


cs = ConfigStore()
cs.store(
    group="schema/module",
    name="byol_module_conf",
    node=BYOLModuleConf,
    package="module",
)
