# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import pytorchvideo_trainer.module.distributed_utils as du
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


def create_moco_resnet_50(
    # Backbone
    backbone_creator: Callable = create_resnet,  # pyre-ignore[24]
    backbone_embed_dim: int = 128,
    head_pool: Callable = nn.AdaptiveAvgPool3d,  # pyre-ignore[24]
    head_output_size: Tuple[int, int, int] = (1, 1, 1),
    head_activation: Callable = None,  # pyre-ignore[9,24]
    dropout_rate: float = 0.0,
    # Projector
    projector_dim_in: int = 2048,
    projector_inner_dim: int = 2048,
    projector_depth: int = 3,
    projector_norm: Optional[Callable] = None,  # pyre-ignore[24]
    mmt: float = 0.994,
) -> nn.Module:
    def _make_bacbone_and_projector():  # pyre-ignore[3]
        backbone = backbone_creator(
            dropout_rate=dropout_rate,
            head_activation=head_activation,
            head_output_with_global_average=True,
            head_pool=head_pool,
            head_output_size=head_output_size,
            stem_conv_kernel_size=(1, 7, 7),
            head_pool_kernel_size=(8, 7, 7),
        )

        backbone.blocks[-1].proj = None  # Overwite head projection
        projector = create_mlp_util(
            projector_dim_in,
            backbone_embed_dim,
            projector_inner_dim,
            projector_depth,
            norm=projector_norm,  # pyre-ignore[6]
        )
        return backbone, projector

    backbone, projector = _make_bacbone_and_projector()
    backbone_mmt, projector_mmt = _make_bacbone_and_projector()

    moco_model = MOCO(
        mmt=mmt,
        backbone=backbone,
        projector=projector,
        backbone_mmt=backbone_mmt,
        projector_mmt=projector_mmt,
    )
    return moco_model


class MOCO(nn.Module):
    """
    Momentum Contrast for unsupervised Visual Representation Learning
    Details can be found in:
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        mmt: float,
        backbone: nn.Module,
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
            out_proj = F.normalize(proj, dim=1)
        return out_proj

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Args:
            x (tensor): input to be forwarded of shape N x C x T x H x W
        """
        proj = self.backbone(x)
        out_proj = F.normalize(proj, dim=1)
        return out_proj


class MOCOV2Module(SSLBaseModule):
    """
    The Lightning Base module for MoCo SSL video task.

    For more details refer to,
    1. Momentum Contrast for unsupervised Visual Representation Learning:
        https://arxiv.org/abs/1911.05722
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
        dim (int): Dimentionality of features in the stored queue. Set to be same as
            embedding dimentions for the SSL model.
        k (int): Queue size for stored features.
        batch_suffle (bool): If true, performs shuffling of the computed keys.
        local_shuffle_bn (bool): If true, only performs shuffling of keys with in the
            current node.
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
        dim: int,
        k: int,
        batch_shuffle: bool,
        local_shuffle_bn: bool,
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

        self.dim: int = dim
        self.k: int = k
        self.batch_shuffle_on = batch_shuffle
        self.local_shuffle_bn = local_shuffle_bn
        self.register_buffer("ptr", torch.tensor([0]))
        self.ptr.requires_grad = False
        stdv = 1.0 / math.sqrt(self.dim / 3)
        self.register_buffer(
            "queue_x",
            torch.rand(self.k, self.dim).mul_(2 * stdv).add_(-stdv),
        )
        self.queue_x.requires_grad = False
        self.local_process_group = None  # pyre-ignore[4]

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.
        If on DDP it is called on every process
        """
        dataloader = self.trainer.datamodule.train_dataloader()
        if self.knn_memory is not None:
            self.knn_memory.init_knn_labels(dataloader)  # pyre-ignore[29]

        world_size = self.trainer.world_size
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.local_shuffle_bn
            and self.batch_shuffle_on
        ):
            self._create_local_process_group()

        # TODO: For ad's dataloder this might be different
        # pyre-ignore[16]
        self.no_update_iters = self.k // world_size // dataloader.batch_size

    def _create_local_process_group(self) -> None:
        assert self.trainer.num_gpus > 1, "Error creating local process group in MoCo"

        for i in range(self.trainer.num_nodes):
            ranks_on_i = list(
                range(i * self.trainer.num_gpus, (i + 1) * self.trainer.num_gpus)
            )
            pg = torch.distributed.new_group(ranks=ranks_on_i)
            if i == torch.distributed.get_rank() // self.trainer.num_gpus:
                self.local_process_group = pg

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

            proj = self.model(vids)
            q_knn = proj
            queue_neg = torch.einsum("nc,kc->nk", [proj, self.queue_x.clone().detach()])

            for k, key in enumerate(other_keys):
                out_pos = torch.einsum("nc,nc->n", [proj, key]).unsqueeze(-1)
                lgt_k = torch.cat([out_pos, queue_neg], dim=1)
                if k == 0:
                    logits = lgt_k
                else:
                    logits = torch.cat([logits, lgt_k], dim=0)
            loss_k = self.loss(logits)  # pyre-ignore[61]

            self.manual_backward(loss_k)
            partial_loss += loss_k.detach()

        if self.knn_memory is not None:
            self.knn_memory.update(q_knn, batch["video_index"])  # pyre-ignore[29,61]

        partial_loss /= len(inputs) * 2.0  # to have same loss as symmetric loss
        self.log("Losses/train_loss", partial_loss, on_step=True, on_epoch=True)
        self._dequeue_and_enqueue(keys)

        if (
            self.trainer.current_epoch == 0
            and self.cur_epoch_step < self.no_update_iters
        ):
            print(
                f"No update: Epoch {self.trainer.current_epoch}"
                + f" Step {self.cur_epoch_step}/{self.no_update_iters}"
            )
            return

        self.manual_opt_step()

    @torch.no_grad()
    def _compute_keys(self, x: torch.Tensor) -> List[torch.Tensor]:
        keys = []
        for sub_x in x:
            if self.batch_shuffle_on:
                with torch.no_grad():
                    sub_x, idx_restore = self._batch_shuffle(sub_x)
            with torch.no_grad():
                # pyre-ignore[29]
                key = self.model.forward_backbone_mmt(sub_x).detach()

            if self.batch_shuffle_on:
                key = self._batch_unshuffle(key, idx_restore).detach()
            keys.append(key)
        return keys

    @torch.no_grad()
    def _batch_shuffle(self, x: torch.Tensor):  # pyre-ignore[3]
        world_size = self.trainer.world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if self.local_shuffle_bn:
                assert self.local_process_group is not None
                x = du.cat_all_gather(x, self.local_process_group)
                gpu_idx = du.get_local_rank(self.local_process_group)
                world_size = self.trainer.num_gpus
            else:
                x = du.cat_all_gather(x)
                gpu_idx = torch.distributed.get_rank()

        idx_randperm = torch.randperm(x.shape[0]).to(self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(idx_randperm, src=0)
        else:
            gpu_idx = 0
        idx_randperm = idx_randperm.view(world_size, -1)
        x = x[idx_randperm[gpu_idx, :]]  # pyre-ignore[61]
        idx_restore = torch.argsort(idx_randperm.view(-1))
        idx_restore = idx_restore.view(world_size, -1)

        return x, idx_restore

    @torch.no_grad()
    def _batch_unshuffle(
        self, x: torch.Tensor, idx_restore: torch.Tensor
    ) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if self.local_shuffle_bn:
                assert self.local_process_group is not None
                x = du.cat_all_gather(x, self.local_process_group)
                gpu_idx = du.get_local_rank(self.local_process_group)
            else:
                x = du.cat_all_gather(x)
                gpu_idx = torch.distributed.get_rank()
        else:
            gpu_idx = 0

        idx = idx_restore[gpu_idx, :]
        x = x[idx]
        return x

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        keys: List[torch.Tensor],
    ) -> None:
        assert len(keys) > 0, "need to have multiple views for adding them to queue"
        ptr = int(self.ptr.item())
        for key in keys:
            # write the current feat into queue, at pointer
            num_items = int(key.size(0))
            assert (
                self.k % num_items == 0
            ), "Queue size should be a multiple of batchsize"
            assert ptr + num_items <= self.k
            self.queue_x[ptr : ptr + num_items, :] = key
            # move pointer
            ptr += num_items
            # reset pointer
            if ptr == self.k:
                ptr = 0
            self.ptr[0] = ptr


@dataclass
class MOCOV2ModuleConf(ModuleConf):
    _target_: str = get_class_name_str(MOCOV2Module)
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
    dim: int = MISSING
    k: int = MISSING
    batch_shuffle: bool = MISSING
    local_shuffle_bn: bool = MISSING


cs = ConfigStore()
cs.store(
    group="schema/module",
    name="moco_v2_module_conf",
    node=MOCOV2ModuleConf,
    package="module",
)
