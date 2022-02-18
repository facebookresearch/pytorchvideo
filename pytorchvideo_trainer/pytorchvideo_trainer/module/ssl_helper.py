# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytorchvideo_trainer.module.distributed_utils as du
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from pytorch_lightning.trainer import Trainer
from pytorchvideo_trainer.module.video_classification import (
    Batch,
    BatchKey,
    EnsembleMethod,
    VideoClassificationModule,
)


def create_mlp_util(
    dim_in: int,
    dim_out: int,
    mlp_dim: int,
    num_layers: int,
    norm: Callable,  # pyre-ignore[24]
    bias: bool = True,
    xavier_init: bool = True,
) -> nn.Module:
    """
    A utility method for creating the MLP that gets attached to the SSL
    bacbone network either in the form of the projector or predictor.

    Consists of multiple squences of "Linear -> Norm -> Relu" layers.

    Args:
        dim_in (int): Input dimension size to the MLP.
        dim_out (int): Output dimension size of MLP.
        mlp_dim (int):  Dimentions size for the inner layers of MLP.
        num_layers (int): Number of layer in the MLP.
        norm (callabe): Type of normalization to apply between layers.
            Examples include BatchNorm, SyncBatchNorm, etc
        bias (bool): If set true, enables bias for the final layer.
        xavier_init (bool): If set to true, performs Xavier weight
            initialization for all linear layers.
    """
    if num_layers == 1:
        return nn.Linear(dim_in, dim_out)

    b = False if norm is not None else bias
    mlp_layers = [nn.Linear(dim_in, mlp_dim, bias=b)]
    mlp_layers[-1].xavier_init = xavier_init
    for i in range(1, num_layers):
        if norm:
            mlp_layers.append(norm(mlp_dim))
        mlp_layers.append(nn.ReLU(inplace=True))
        if i == num_layers - 1:
            d = dim_out
            b = bias
        else:
            d = mlp_dim
        mlp_layers.append(nn.Linear(mlp_dim, d, bias=b))
        mlp_layers[-1].xavier_init = xavier_init
    return nn.Sequential(*mlp_layers)


def create_classification_model_from_ssl_checkpoint(
    ssl_checkpoint_path: str,
    checkpoint_type: str,
    mlp: Optional[nn.Module] = None,
    detach_backbone: bool = False,
) -> nn.Module:

    """
    A utlity function for extracting the bacbone from the PyTorch Lightning's
    SSL checkpoints. Used for supervided finetuning the SSL pre-trained models
    in video classification task.

    Extracts bacbone from the checkpoints of the SimCLR, BYOL and MoCoV2 SSL
    tasks and attaches the given MLP to the backbone.

    Args:
        ssl_checkpoint_path (str): Path to the lightning checkpoint for the
            said SSL task.
        checkpoint_type (str): Type of the SSL task the checkpoint belongs to.
            Should be one of ["simclr, "byol", "mocov_v2"]
        mlp (nn.Module): If specified, the MLP module to attach to the bacbone
            for the supervised finetuning phase.
        detach_bacbone: If true, detaches bacbone and no gradient are tracked and
            updated for the bacbone. Only updates the MLP weights during finetuning.

    Returns:
        model (SSLFineTuningModel): Returns an instance of `SSLFineTuningModel`,
            consisting of bacbone and mlp.
    """

    if checkpoint_type == "simclr":
        from pytorchvideo_trainer.module.simclr import SimCLRModule as M

        lightning_module = M.load_from_checkpoint(ssl_checkpoint_path)
        backbone = lightning_module.model.backbone[0]
    elif checkpoint_type == "byol":
        from pytorchvideo_trainer.module.byol import BYOLModule as M

        lightning_module = M.load_from_checkpoint(ssl_checkpoint_path)
        backbone = lightning_module.model.backbone[0]
    elif checkpoint_type == "moco_v2":
        from pytorchvideo_trainer.module.moco_v2 import MOCOV2Module as M

        lightning_module = M.load_from_checkpoint(ssl_checkpoint_path)
        backbone = lightning_module.model.backbone[0]
    else:
        raise ValueError("Incorrect SSL checkpoint type.")

    # pyre-ignore[6]
    return SSLFineTuningModel(backbone, mlp, detach_backbone)


class SSLFineTuningModel(nn.Module):
    """
    Model consisting of a backbone sequentially followed by an an MLP.
    Used for supervised finetuning of the SSL pre-trained models.

    Args:
        backbone (nn.Module): A model whole weights are conditionally
            updated based on the betach_backbone parameter.
        mlp (nn.Module): If specified, the MLP module to attach to the bacbone
            for the supervised finetuning phase.
        detach_bacbone: If true, detaches bacbone and no gradient are tracked and
            updated for the bacbone. Only updates the MLP weights during finetuning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mlp: nn.Module,
        detach_backbone: bool,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.mlp = mlp
        self.detach_backbone = detach_backbone

        for p in self.backbone.parameters():
            p.requires_grad = False if detach_backbone else True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if self.detach_backbone:
            x = x.detach()
        if self.mlp is not None:
            x = self.mlp(x)
        return x


class KnnMemory(nn.Module):
    """
    KNN Memory object that keeps track of the features generated by the SSL model
    during the traing phase and performs nearest neighbours inference during the
    test and validation phases for video classfication.

    KNN memory requires that you provide the labels and video indices for the
    dataset used for the SSL training phase.

    Args:
        length (int): Size of the KNN memory. Set to be equal to the training dataset size.
        dim (int): Feture dimention generated by the SSL model.
        momentum (float): The rate at which to update the features in memory during the SSL-
            training phase.
        downstream_classes (int): Number of classes in the dataset.
        temperature (float): Temperature scaling to use during the inference phase. Typically,
            set to the same value as the loss temperature used in SSL.
        knn_k (int): Number of nearest neighbours to aggregate metrics over for inference.
        deive (str): Device to store the memory module on.
    """

    def __init__(
        self,
        length: int,
        dim: int,
        momentum: float = 1.0,
        downstream_classes: int = 400,
        temperature: float = 1.0,
        knn_k: int = 200,
        device: str = "cpu",
    ) -> None:
        super(KnnMemory, self).__init__()
        self.length = length
        self.dim = dim
        self.momentum = momentum
        self.temperature = temperature
        self.downstream_classes = downstream_classes
        self.knn_k = knn_k
        stdv = 1.0 / math.sqrt(dim / 3)
        self.device = device
        self.register_buffer(
            "memory",
            torch.rand(length, dim, device=self.device).mul_(2 * stdv).add_(-stdv),
        )

    def resize(self, length: int, dim: int) -> None:
        """
        Resizes the memory and intialized it fresh.

        Args:
            length (int): Size of the KNN memory. Set to be equal to the training
                dataset size.
            dim (int): Feture dimention generated by the SSL model.
        """
        self.length = length
        self.dim = dim
        stdv = 1.0 / math.sqrt(dim / 3)
        del self.memory
        self.memory = (
            torch.rand(length, dim, device=self.device).mul_(2 * stdv).add_(-stdv)
        )

    @torch.no_grad()
    def get(self, ind: torch.Tensor) -> torch.Tensor:
        """
        Fetches features from the memory based on the video index.

        Args:
            ind (int): Index of the video / clip for which to fetch the features.
        """
        batch_size = ind.size(0)
        selected_mem = self.memory[ind.view(-1), :]
        out = selected_mem.view(batch_size, -1, self.dim)
        return out

    @torch.no_grad()
    def update(self, mem: torch.Tensor, ind: torch.Tensor) -> None:
        """
        Peforms feature update in the memory based on the new features realized by the
        SSL model. Called during the SSL training phase.

        Args:
            mem (tensor): Features of the same N x C genereated by the SSL model.
                N is the batch size and C is the feature dimention generated by the
                SSL Model.
            ind (tensor): A 1-D tensor of video indices associated the given features.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            mem, ind = du.all_gather([mem, ind])
        mem = mem.view(mem.size(0), 1, -1)
        mem_old = self.get(ind).to(mem.device)

        mem_update = mem * self.momentum + mem_old * (1 - self.momentum)
        mem_update = F.normalize(mem_update, p=2, dim=1)
        self.memory[ind.view(-1), :] = mem_update.squeeze().to(self.memory.device)

    @torch.no_grad()
    def init_knn_labels(self, train_loader: Trainer) -> None:
        """
        Called before traning, intializes the KNN Memory and resizes it based on the
        labels and number of samples in the train dataloader.

        Args:
            train_loader (dataloader): Trainining dataloader containing an attribute
                `dataset._labeled_videos` which holds mapping from video indices to
                labels.
        """
        # TODO: Make sure all dataloader's have this property `dataset._labeled_videos`
        self.num_imgs = len(train_loader.dataset._labeled_videos)  # pyre-ignore[16]
        # pyre-ignore[16]
        self.train_labels = np.zeros((self.num_imgs,), dtype=np.int32)
        for i in range(self.num_imgs):  # pyre-ignore[6]
            # pyre-ignore[29]
            self.train_labels[i] = train_loader.dataset._labeled_videos[i][1]["label"]
        self.train_labels = torch.LongTensor(self.train_labels).to(self.device)
        if self.length != self.num_imgs:
            self.resize(self.num_imgs, self.dim)  # pyre-ignore[6]

    def forward(self, inputs: torch.Tensor) -> None:
        pass

    @torch.no_grad()
    def eval_knn(self, q_knn: torch.Tensor) -> torch.Tensor:
        """
        Peforms KNN nearest neighbour aggregations and returns predictions
        for the qurried features.

        Args:
            q_nn (tensor): Features generated by the SSL model during the inference
                phase. Expected to be of shape N x C where, N is the batch size and
                C is the feature dimention generated by the SSL Model.
        """
        device = q_knn.device
        batch_size = q_knn.size(0)
        dist = torch.einsum(
            "nc,mc->nm",
            q_knn.view(batch_size, -1),
            self.memory.view(self.memory.size(0), -1).to(device),
        )
        yd, yi = dist.topk(self.knn_k, dim=1, largest=True, sorted=True)

        K = yi.shape[1]
        C = self.downstream_classes
        candidates = self.train_labels.view(1, -1).expand(batch_size, -1)
        candidates = candidates.to(device)
        yi = yi.to(device)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot = torch.zeros((batch_size * K, C)).to(device)
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = (yd.clone().div_(self.temperature).exp_()).to(device)
        probs = torch.mul(
            retrieval_one_hot.view(batch_size, -1, C),
            yd_transform.view(batch_size, -1, 1),
        )
        preds = torch.sum(probs, 1)
        return preds


class SSLBaseModule(VideoClassificationModule):
    """
    The Lightning Base module supporting SimCLR, MoCo and BYOL SSL tasks.

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
        momentum_anneal_cosine: bool = False,  # TODO: Refactor out mmt from base class.
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
            num_sync_devices=num_sync_devices,
        )

        self.knn_memory: nn.Module = instantiate(knn_memory)
        self.automatic_optimization = False
        self.momentum_anneal_cosine = momentum_anneal_cosine
        if self.momentum_anneal_cosine:
            self.initial_mmt: float = self.model.mmt  # pyre-ignore[8]

        if ensemble_method is not None:
            assert (
                self.knn_memory is not None
            ), "Test-Ensembling is only supported with KNN module"

    def on_fit_start(self) -> None:
        """
        Called at the very beginning of fit.
        If on DDP it is called on every process.

        Peforms conversion of model batchnorm layers into syncbatchnom
        and intialized the KNN module using the dataloader labels.
        """

        self._convert_to_sync_bn()
        if self.knn_memory is not None:
            dataloader = self.trainer.datamodule.train_dataloader()
            self.knn_memory.init_knn_labels(dataloader)  # pyre-ignore[29]

    def _test_step_with_data_ensembling(self, batch: Batch, batch_idx: int) -> None:
        """
        Operates on a single batch of data from the test set.
        """
        assert (
            isinstance(batch, dict)
            and self.modality_key in batch
            and "label" in batch
            and "video_index" in batch
            and self.knn_memory is not None
        ), (
            f"Returned batch [{batch}] is not a map with '{self.modality_key}' and"
            + "'label' and 'video_index' keys"
        )

        y_hat = self(batch[self.modality_key])
        y_hat = (
            self.knn_memory.eval_knn(y_hat) if self.knn_memory is not None else y_hat
        )
        preds = torch.nn.functional.softmax(y_hat, dim=-1)
        labels = batch["label"]
        video_ids = torch.tensor(batch["video_index"], device=self.device)

        self._ensemble_at_video_level(preds, labels, video_ids)

    def _step(self, batch: Batch, batch_idx: int, phase_type: str) -> Dict[str, Any]:
        """
        If KNN Memory is enabled, evaluates metrics using the labels of neighbours
        during the validation and test phases.
        """
        assert (
            isinstance(batch, dict)
            and self.modality_key in batch
            and ("label" in batch or self.knn_memory is None)
            and phase_type in ["val", "test"]
        ), (
            f"Returned batch [{batch}] is not a map with '{self.modality_key}' and"
            + "'label' keys"
        )

        if self.knn_memory is not None:
            y_hat = self(batch[self.modality_key])
            y_hat = self.knn_memory.eval_knn(y_hat)
            pred = torch.nn.functional.softmax(y_hat, dim=-1)
            metrics_result = self._compute_metrics(pred, batch["label"], phase_type)
            self.log_dict(metrics_result, on_epoch=True)

    def training_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> None:
        """Missing method implemented in subsequent derived SSL task modules."""
        pass

    @torch.no_grad()
    def _cosine_anneal_momentum(self) -> None:
        """
        For MoCo and BYOL tasks, if self.momentum_anneal_cosine set to true,
        cosine anneals the momentum term used from updating the backbone-history
        model.
        """
        # pyre-ignore[6]
        exact_epoch = float(self.cur_epoch_step) / float(
            self._num_training_steps_per_epoch()
        )
        exact_epoch += self.trainer.current_epoch
        new_mmt = (
            1.0
            - (1.0 - self.initial_mmt)
            * (
                math.cos(math.pi * float(exact_epoch) / float(self.trainer.max_epochs))
                + 1.0
            )
            * 0.5
        )
        self.model.mmt = new_mmt  # pyre-ignore[16]
        self.log("MMT", new_mmt, on_step=True, prog_bar=True)
