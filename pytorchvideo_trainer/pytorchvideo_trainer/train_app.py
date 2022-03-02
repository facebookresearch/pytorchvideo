# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import hydra
import numpy as np
import submitit
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorchvideo_trainer.datamodule.datamodule import VideoClassificationDataModuleConf
from pytorchvideo_trainer.module.video_classification import (
    VideoClassificationModuleConf,
)
from torchrecipes.core.base_train_app import BaseTrainApp, TrainOutput
from torchrecipes.core.conf import TrainAppConf, TrainerConf
from torchrecipes.utils.config_utils import get_class_name_str


class VideoClassificationTrainApp(BaseTrainApp):
    """
    This app is used to launch the video tasks (both Classfication and SSL).
    Main point of entry for all training, validation and test phases.

    The hydra/Omega conf schema used by the train app is as defined in
    `VideoClassificationTrainAppConf`

    Args:
        module (OmegaConf): Hydra/Omega conf object associated with the initialization of the
            pytorch-lightning module. Supported config schema's include,
            1. `pytorchvide_trainer.module.video_classification.VideoClassificationModuleConf`
            2. `pytorchvide_trainer.module.simclr.SimCLRModuleConf`
            3. `pytorchvide_trainer.module.byol.BYOLModuleConf`
            4. `pytorchvide_trainer.module.moco_v2.MOCOV2ModuleConf`
            and more. Example definitions of the config can be found in
            `pytorchvide_trainer/conf.module`
        trainer (OmegaConf):  Hydra/Omega conf object associated with the initialization of the
            pytorch-lightning Trainer object. Supported config schema can be found in
            `github.com/facebookresearch/recipes/blob/main/torchrecipes/core/conf/__init__.py`
        datamodule (OmegaConf): Hydra/Omega conf object associated with the initialization of
            the pytorch-lightning DataModule object. Supported config schema can be found at,
            `pytorchvideo_trainer.datamodule.datamodule.VideoClassificationDataModuleConf`
        logger (OmegaConf): Hydra/Omega conf object associated with the initialization of the
            pytorch-lightning's tensboard logger object. Example config can be found at,
            `pytorchvideo_trainer/conf/logger`
        callbacks (List[OmegaConf]): Hydra/Omega conf object associated with the intialization
            of a series of pytorch-ligtning Callbacks that act upon the lightning module. Expect
            a list or iterable config object wherein, each element represent the hydra conf of
            a single callback. Thus, supports loading multiple callabacks at a time. Example
            configs can be found at `pytorchvideo_trainer/conf/callbacks`
        submitit_conf (OmegaConf): Hydra/Omega conf to be used by the `submitit_launcher` for
            launching the train app. Example config file can be found at,
            `pytorchvideo_trainer/conf/submitit_conf`
    """

    def __init__(
        self,
        module: VideoClassificationModuleConf,
        trainer: TrainerConf,
        datamodule: VideoClassificationDataModuleConf,
        logger: Any,  # pyre-ignore[2]
        callbacks: Optional[Any] = None,  # pyre-ignore[2]
        submitit_conf: Optional[Any] = None,  # pyre-ignore[2]
    ) -> None:

        self.logger_conf: DictConfig = logger
        self.callbacks_conf: DictConfig = callbacks
        self.submitit_conf: DictConfig = submitit_conf
        # This has to happen at last because it depends on the value above.
        super().__init__(module, trainer, datamodule)

    def get_data_module(self) -> Optional[LightningDataModule]:
        """
        Instantiate a LightningDataModule.
        """
        return hydra.utils.instantiate(
            self.datamodule_conf,
            _recursive_=False,
        )

    def get_lightning_module(self) -> LightningModule:
        """
        Instantiate a LightningModule.
        """
        return hydra.utils.instantiate(
            self.module_conf,
            _recursive_=False,
        )

    def get_callbacks(self) -> List[Callback]:
        """
        Creates a list of callbacks that feeds into trainer.
        You can add additional ModelCheckpoint here too.
        """
        callbacks = []
        if self.trainer_conf.logger:
            callbacks.extend(
                [
                    LearningRateMonitor(),
                ]
            )
        if self.callbacks_conf is None:
            return callbacks

        for cb_conf in self.callbacks_conf.values():
            callbacks.append(
                hydra.utils.instantiate(
                    cb_conf,
                    _recursive_=False,
                ),
            )

        return callbacks

    def _make_reproducible_conf(self) -> DictConfig:
        conf = OmegaConf.create()
        conf._target_ = "pytorchvideo_trainer.train_app.VideoClassificationTrainApp"
        conf.module = self.module_conf
        conf.trainer = self.trainer_conf
        conf.datamodule = self.datamodule_conf
        conf.logger = self.logger_conf
        conf.callbacks = self.callbacks_conf
        conf.submitit_conf = self.submitit_conf
        return conf

    def get_logger(self) -> TensorBoardLogger:
        """
        Creates a logger that feeds into trainer.
        Override this method to return a logger for trainer.
        """
        logger = hydra.utils.instantiate(
            self.logger_conf,
            _recursive_=False,
        )

        @rank_zero_only
        def log_params() -> None:  # pyre-ignore[53]
            if os.environ["PTV_TRAINER_ENV"] == "oss":
                from iopath.common.file_io import g_pathmgr

                conf_to_log = self._make_reproducible_conf()
                conf_save_path = os.path.join(logger.log_dir, "train_app_conf.yaml")
                g_pathmgr.mkdirs(logger.log_dir)
                if not g_pathmgr.exists(conf_save_path):
                    with g_pathmgr.open(conf_save_path, mode="w") as f:
                        f.write(OmegaConf.to_yaml(conf_to_log))
            else:
                from stl.lightning.io import filesystem

                fs = filesystem.get_filesystem(logger.log_dir)
                conf_to_log = self._make_reproducible_conf()
                fs.makedirs(logger.log_dir, exist_ok=True)
                conf_save_path = os.path.join(logger.log_dir, "train_app_conf.yaml")
                if not fs.exists(conf_save_path):
                    with fs.open(conf_save_path, mode="w") as f:
                        f.write(OmegaConf.to_yaml(conf_to_log))

        log_params()
        return logger

    def test(self) -> TrainOutput:  # pyre-ignore[15]
        """
        Triggers PyTorch-lightning's testing phase.
        """
        trainer, _ = self._get_trainer()
        trainer.test(self.module, datamodule=self.datamodule)
        return TrainOutput(tensorboard_log_dir=self.root_dir)

    def predict(self) -> TrainOutput:  # pyre-ignore[15]
        """
        Triggers PyTorch-lightning's prediction phase.
        """
        trainer, _ = self._get_trainer()
        trainer.predict(self.module, datamodule=self.datamodule)
        return TrainOutput(tensorboard_log_dir=self.root_dir)


def run_app_in_certain_mode(
    cfg: TrainAppConf, mode: str, env: str = "oss"
) -> TrainOutput:

    os.environ["PTV_TRAINER_ENV"] = env

    rank_zero_info(OmegaConf.to_yaml(cfg))

    # TODO: Move this to config and replace with `seed_everything`
    np.random.seed(0)
    torch.manual_seed(0)
    app = hydra.utils.instantiate(cfg, _recursive_=False)

    if mode == "train":
        rank_zero_info("MODE set to train, run train only.")
        return app.train()
    elif mode == "test":
        rank_zero_info("MODE set to test, run test only.")
        return app.test()
    elif mode == "predict":
        rank_zero_info("MODE set to predict, run train and predict.")
        app.train()
        return app.predict()
    else:
        # By default, run train and test
        app.train()
        return app.test()


project_defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"schema/module": "video_classification_module_conf"},
    {"schema/module/optim": "optim_conf"},
    {"schema/datamodule": "ptv_video_classification_data_module_conf"},
    {"datamodule/dataloader": "kinetics_classification"},
    {"logger": "ptl"},
    {"datamodule/transforms": "kinetics_classification_slow"},
    {"module/model": "slow_r50"},
    {"module/loss": "cross_entropy"},
    {"module/optim": "sgd"},
    {"module/metrics": "accuracy"},
    {"schema/trainer": "trainer"},
    {"trainer": "cpu"},
]


@dataclass
class VideoClassificationTrainAppConf(TrainAppConf):
    _target_: str = get_class_name_str(VideoClassificationTrainApp)
    datamodule: VideoClassificationDataModuleConf = MISSING
    module: VideoClassificationModuleConf = MISSING
    trainer: TrainerConf = MISSING

    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    logger: Any = MISSING

    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    callbacks: Optional[Any] = None

    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    defaults: List[Any] = field(default_factory=lambda: project_defaults)

    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    submitit_conf: Optional[Any] = None


cs = ConfigStore()
cs.store(
    name="video_classification_train_app_conf",
    node=VideoClassificationTrainAppConf,
)


@hydra.main(config_path="conf", config_name=None)
# pyre-ignore[2]
def submitit_launcher(cfg) -> None:

    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    submitit_conf = cfg.get("submitit_conf", None)
    logger_conf = cfg.get("logger", None)
    assert submitit_conf is not None, "Missing submitit config"

    if logger_conf is not None:
        assert (
            logger_conf.save_dir is not None
        ), "set save_dir in logger conf to a valid path"
        submitit_dir = os.path.join(logger_conf.save_dir, logger_conf.name)
    else:
        assert submitit_conf.log_save_dir is not None
        submitit_dir = submitit_conf.log_save_dir

    submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    executor = submitit.AutoExecutor(folder=submitit_dir)
    job_kwargs = {
        "slurm_time": submitit_conf.time,
        "name": cfg.logger.name if logger_conf is not None else submitit_conf.name,
        "slurm_partition": submitit_conf.partition,
        "gpus_per_node": cfg.trainer.gpus,
        "tasks_per_node": cfg.trainer.gpus,  # one task per GPU
        "cpus_per_task": submitit_conf.cpus_per_task,
        "nodes": cfg.trainer.num_nodes,
    }
    if submitit_conf.get("mem", None) is not None:
        job_kwargs["slurm_mem"] = submitit_conf.mem
    if submitit_conf.get("constraints", None) is not None:
        job_kwargs["constraints"] = submitit_conf.constraints

    executor.update_parameters(**job_kwargs)
    job = executor.submit(run_app_in_certain_mode, cfg, submitit_conf.mode)
    print("Submitit Job ID:", job.job_id)


if __name__ == "__main__":
    submitit_launcher()
