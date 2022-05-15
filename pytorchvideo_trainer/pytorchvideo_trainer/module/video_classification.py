# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING, OmegaConf
from pytorch_lightning.utilities import rank_zero_info
from pytorchvideo_trainer.datamodule.transforms import MixVideoBatchWrapper
from pytorchvideo_trainer.module.lr_policy import get_epoch_lr, LRSchedulerConf, set_lr
from pytorchvideo_trainer.module.optimizer import construct_optimizer
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchrecipes.core.conf import ModuleConf
from torchrecipes.utils.config_utils import get_class_name_str


class Batch(TypedDict):
    """
    PyTorchVideo batches are dictionaries containing each modality or metadata of
    the batch collated video clips. For Kinetics it has the below keys and types.
    """

    video: torch.Tensor  # (B, C, T, H, W)
    audio: torch.Tensor  # (B, S)
    label: torch.Tensor  # (B, 1)
    video_index: List[int]  # len(video_index) == B


BatchKey = Literal["video", "audio", "label", "video_index"]
EnsembleMethod = Literal["sum", "max"]


class VideoClassificationModule(pl.LightningModule):
    """
    The Lightning module supporting the video classification task.

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
        num_classes (int): The number of classes in the dataset.
        num_sync_devices (int): Number of gpus to sync bathcnorm over. Only works if
            pytorch lightning trainer's sync_batchnorm parameter is to false.
        batch_transform (OmegaConf): An optional omega conf object, for constructing the
            data transform method that act upon the entire mini batch. Examples include,
            MixVideo transform, etc.
        clip_gradient_norm (float): Performs gradient clipping if set to a positive value.
            Since, we use Pytorch-lightning's manual optimization approach gradient clipping
            has to be be set in the lightning module instead of the Trainer object.
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
        num_classes: int = 400,
        num_sync_devices: int = 1,
        batch_transform: Optional[Any] = None,  # pyre-ignore[2]
        clip_gradient_norm: float = 0.0,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.model: nn.Module = instantiate(model, _convert_="all")
        self.loss: nn.Module = instantiate(loss)
        self.batch_transform = instantiate(batch_transform)  # pyre-ignore[4]
        rank_zero_info(OmegaConf.to_yaml(optim))
        self.optim: torch.optim.Optimizer = construct_optimizer(self.model, optim)
        self.lr_scheduler_conf: LRSchedulerConf = lr_scheduler
        self.modality_key: BatchKey = modality_key
        self.ensemble_method: Optional[EnsembleMethod] = ensemble_method
        self.num_classes: int = num_classes
        self.clip_gradient_norm = clip_gradient_norm

        self.metrics: Mapping[str, nn.Module] = {
            metric_conf.name: instantiate(metric_conf.config) for metric_conf in metrics
        }

        self.train_metrics: nn.ModuleDict = nn.ModuleDict()
        self.val_metrics: nn.ModuleDict = nn.ModuleDict()
        self.test_metrics: nn.ModuleDict = nn.ModuleDict()

        self.save_hyperparameters()

        # These are used for data ensembling in the test stage.
        self.video_preds: Dict[int, torch.Tensor] = {}
        self.video_labels: Dict[int, torch.Tensor] = {}
        self.video_clips_cnts: Dict[int, int] = {}

        # Sync BatchNorm
        self.num_sync_devices = num_sync_devices

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_metrics.update(self.metrics)
            self.val_metrics.update(self.metrics)
        else:
            self.test_metrics.update(self.metrics)

    # pyre-ignore[14]: *args, **kwargs are not torchscriptable.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def _num_training_steps_per_epoch(self) -> int:
        """training steps per epoch inferred from datamodule and devices."""
        dataloader = self.trainer.datamodule.train_dataloader()
        world_size = self.trainer.world_size

        # TODO: Make sure other dataloaders has this property
        dataset_size = self.trainer.limit_train_batches
        dataset_size *= len(dataloader.dataset._labeled_videos)

        # TODO: Make sure other dataloaders has this property
        return dataset_size // world_size // dataloader.batch_size

    def manual_update_lr(self) -> None:
        """Utility function for manually updating the optimizer learning rate"""

        opt = self.optimizers()

        if self.lr_scheduler_conf is not None:
            # pyre-ignore[6]
            exact_epoch = float(self.cur_epoch_step) / float(
                self._num_training_steps_per_epoch()
            )
            exact_epoch += self.trainer.current_epoch
            lr = get_epoch_lr(exact_epoch, self.lr_scheduler_conf)
            self.log("LR", lr, on_step=True, prog_bar=True)
            self.log("ExactE", exact_epoch, on_step=True, prog_bar=True)

            if isinstance(opt, list):
                for op in opt:
                    set_lr(op, lr)  # pyre-ignore[6]
            else:
                set_lr(opt, lr)  # pyre-ignore[6]

    def manual_zero_opt_grad(self) -> None:
        """Utility function for zeroing optimzer gradients"""
        opt = self.optimizers()
        if isinstance(opt, list):
            for op in opt:
                op.zero_grad()  # pyre-ignore[16]
        else:
            opt.zero_grad()

    def manual_opt_step(self) -> None:
        """Utility function for manually stepping the optimzer"""
        opt = self.optimizers()
        if isinstance(opt, list):
            for op in opt:
                op.step()
        else:
            opt.step()

    def training_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> None:
        """
        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping the
        dict and feeding it through the model/loss.
        """
        self.cur_epoch_step += 1  # pyre-ignore[16]

        if self.batch_transform is not None:
            batch = self.batch_transform(batch)

        self.manual_zero_opt_grad()
        self.manual_update_lr()

        # Forward/backward
        loss = self._step(batch, batch_idx, "train")
        self.manual_backward(loss)  # pyre-ignore[6]
        if self.clip_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_gradient_norm
            )
        self.manual_opt_step()

    def validation_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Operates on a single batch of data from the validation set.
        """
        return self._step(batch, batch_idx, "val")

    def test_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Operates on a single batch of data from the test set.
        """
        if self.ensemble_method:
            self._test_step_with_data_ensembling(batch, batch_idx)
        else:
            return self._step(batch, batch_idx, "test")

    def _test_step_with_data_ensembling(self, batch: Batch, batch_idx: int) -> None:
        """
        Operates on a single batch of data from the test set.
        """
        assert (
            isinstance(batch, dict)
            and self.modality_key in batch
            and "label" in batch
            and "video_index" in batch
        ), (
            f"Returned batch [{batch}] is not a map with '{self.modality_key}' and"
            + "'label' and 'video_index' keys"
        )

        y_hat = self(batch[self.modality_key])
        preds = torch.nn.functional.softmax(y_hat, dim=-1)
        labels = batch["label"]
        video_ids = torch.tensor(batch["video_index"], device=self.device)

        self._ensemble_at_video_level(preds, labels, video_ids)

    def on_train_epoch_start(self) -> None:
        self._reset_metrics("train")
        self.cur_epoch_step = 0.0  # pyre-ignore[16]

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._reset_metrics("test")

    def on_test_epoch_end(self) -> None:
        """Pytorch-Lightning's method for aggregating test metrics at the end of epoch"""
        if self.ensemble_method:
            for video_id in self.video_preds:
                self.video_preds[video_id] = (
                    self.video_preds[video_id] / self.video_clips_cnts[video_id]
                )
            video_preds = torch.stack(list(self.video_preds.values()), dim=0)
            video_labels = torch.tensor(
                list(self.video_labels.values()),
                device=self.device,
            )
            metrics_result = self._compute_metrics(video_preds, video_labels, "test")
            self.log_dict(metrics_result)

    def _ensemble_at_video_level(
        self, preds: torch.Tensor, labels: torch.Tensor, video_ids: torch.Tensor
    ) -> None:
        """
        Ensemble multiple predictions of the same view together. This relies on the
        fact that the dataloader reads multiple clips of the same video at different
        spatial crops.
        """
        for i in range(preds.shape[0]):
            vid_id = int(video_ids[i])
            self.video_labels[vid_id] = labels[i]
            if vid_id not in self.video_preds:
                self.video_preds[vid_id] = torch.zeros(
                    (self.num_classes), device=self.device, dtype=preds.dtype
                )
                self.video_clips_cnts[vid_id] = 0

            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[i]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(self.video_preds[vid_id], preds[i])
            self.video_clips_cnts[vid_id] += 1

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Tuple[Iterable[torch.optim.Optimizer], Iterable[_LRScheduler]],
    ]:
        """Pytorch-Lightning's method for configuring optimizer"""
        return self.optim

    def _step(self, batch: Batch, batch_idx: int, phase_type: str) -> Dict[str, Any]:
        assert (
            isinstance(batch, dict) and self.modality_key in batch and "label" in batch
        ), (
            f"Returned batch [{batch}] is not a map with '{self.modality_key}' and"
            + "'label' keys"
        )

        y_hat = self(batch[self.modality_key])
        if phase_type == "train":
            loss = self.loss(y_hat, batch["label"])
            self.log(
                f"Losses/{phase_type}_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            loss = None

        ## TODO: Move MixUP transform metrics to sperate method.
        if (
            phase_type == "train"
            and self.batch_transform is not None
            and isinstance(self.batch_transform, MixVideoBatchWrapper)
        ):
            _top_max_k_vals, top_max_k_inds = torch.topk(
                batch["label"], 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(batch["label"].shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(batch["label"].shape[0]), top_max_k_inds[:, 1]
            y_hat = y_hat.detach()
            y_hat[idx_top1] += y_hat[idx_top2]
            y_hat[idx_top2] = 0.0
            batch["label"] = top_max_k_inds[:, 0]

        pred = torch.nn.functional.softmax(y_hat, dim=-1)
        metrics_result = self._compute_metrics(pred, batch["label"], phase_type)
        self.log_dict(metrics_result, on_epoch=True)

        return loss

    def _compute_metrics(
        self, pred: torch.Tensor, label: torch.Tensor, phase_type: str
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = getattr(self, f"{phase_type}_metrics")
        metrics_result = {}
        for name, metric in metrics_dict.items():
            metrics_result[f"Metrics/{phase_type}/{name}"] = metric(pred, label)
        return metrics_result

    def _reset_metrics(self, phase_type: str) -> None:
        metrics_dict = getattr(self, f"{phase_type}_metrics")
        for _, metric in metrics_dict.items():
            metric.reset()

    def _convert_to_sync_bn(self) -> None:
        """
        Converts BatchNorm into sync-batchnorm.
        If pytorch lightning trainer's sync_batchnorm parameter is to true,
        performs global sync-batchnorm across all nodes and gpus. Else,
        if perform local sync-batchnorm acroos specified number of gpus.
        """
        if (
            hasattr(self.trainer.training_type_plugin, "sync_batchnorm")
            and self.trainer.training_type_plugin.sync_batchnorm
        ):
            print("Using Global Synch BatchNorm.")
            return None

        if self.num_sync_devices > 1:
            print(f"Using local Synch BatchNorm over {self.num_sync_devices} devices.")
            pg = create_syncbn_process_group(self.num_sync_devices)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model, process_group=pg
            )

    def on_fit_start(self) -> None:
        """
        Called at the very beginning of fit.
        If on DDP it is called on every process.
        """
        self._convert_to_sync_bn()


def create_syncbn_process_group(group_size: int) -> List[int]:
    """
    Creates process groups to be used for syncbn of a give ``group_size`` and returns
    process group that current GPU participates in.

    Args:
        group_size (int): number of GPU's to collaborate for sync bn. group_size should
            be >=2 else, no action is taken.
    """
    assert (
        group_size > 1
    ), f"Invalid group size {group_size} to convert to sync batchnorm."

    world_size = torch.distributed.get_world_size()
    assert world_size >= group_size
    assert world_size % group_size == 0

    group = None
    for group_num in range(world_size // group_size):
        group_ids = range(group_num * group_size, (group_num + 1) * group_size)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank() // group_size == group_num:
            group = cur_group
            # can not drop out and return here,
            # every process must go through creation of all subgroups

    assert group is not None
    return group


@dataclass
class VideoClassificationModuleConf(ModuleConf):
    _target_: str = get_class_name_str(VideoClassificationModule)
    model: Any = MISSING  # pyre-ignore[4]
    loss: Any = MISSING  # pyre-ignore[4]
    optim: Any = MISSING  # pyre-ignore[4]
    metrics: List[Any] = MISSING  # pyre-ignore[4]
    lr_scheduler: Optional[Any] = None  # pyre-ignore[4]
    modality_key: str = "video"
    ensemble_method: Optional[str] = None
    num_classes: int = 400
    num_sync_devices: Optional[int] = 1


@dataclass
class VideoClassificationModuleConfVisionTransformer(VideoClassificationModuleConf):

    batch_transform: Optional[Any] = None  # pyre-ignore[4]
    clip_gradient_norm: float = 0.0


cs = ConfigStore()
cs.store(
    group="schema/module",
    name="video_classification_module_conf",
    node=VideoClassificationModuleConf,
    package="module",
)

cs.store(
    group="schema/module",
    name="video_classification_module_conf_vision_transformer",
    node=VideoClassificationModuleConfVisionTransformer,
    package="module",
)


def create_classification_model_from_modelzoo(
    checkpoint_path: str,
    model: nn.Module,
) -> nn.Module:
    """
    Builds a model from PyTorchVideo's model zoo checkpoint.

    Example config for building this method can be found at -
    `pytorchvideo_trainer/conf/module/model/from_model_zoo_checkpoint.yaml`

    Args:
        checkpoint_path (str): Path the pretrained model weights.
        model (nn.Module): Module to load the checkpoints into.
    Returns:
        model (nn.Module): Returns the model with pretrained weights loaded.
    """

    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    state_dict = checkpoint["model_state"]
    model.load_state_dict(state_dict)
    return model


def create_classification_model_from_lightning(
    checkpoint_path: str,
) -> nn.Module:
    """
    Builds a model from pytorchvideo_trainer's PytorchLightning checkpoint.

    Example config for building this method can be found at -
    `pytorchvideo_trainer/conf/module/model/from_lightning_checkpoint.yaml`

    Args:
        checkpoint_path (str): Path the pretrained model weights.
    Returns:
        model (nn.Module): Returns the model with pretrained weights loaded.
    """
    lightning_model = VideoClassificationModule.load_from_checkpoint(checkpoint_path)
    return lightning_model.model
