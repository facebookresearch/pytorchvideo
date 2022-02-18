# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
from typing import Any

import hydra
from hydra import compose, initialize_config_module
from hydra.utils import instantiate  # @manual
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorchvideo_trainer.datamodule.datamodule import VideoClassificationDataModuleConf
from pytorchvideo_trainer.train_app import VideoClassificationTrainAppConf
from util import create_small_kinetics_dataset, run_locally, tempdir


class TestMain(unittest.TestCase):
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def get_datamodule(self, cfg: VideoClassificationDataModuleConf) -> Any:
        test_data_module = instantiate(
            cfg,
            _recursive_=False,
        )
        return test_data_module

    def train(self, cfg: VideoClassificationTrainAppConf) -> None:
        print(OmegaConf.to_yaml(cfg))
        test_module = hydra.utils.instantiate(cfg.module, _recursive_=False)
        test_data_module = self.get_datamodule(cfg.datamodule)
        # pyre-fixme[6]: Expected `SupportsKeysAndGetItem[Variable[_KT],
        #  Variable[_VT]]` for 1st param but got `TrainerConf`.
        trainer_params = dict(cfg.trainer)
        trainer_params["logger"] = True
        trainer_params["checkpoint_callback"] = False
        trainer_params["fast_dev_run"] = True
        pl_trainer = Trainer(**trainer_params)
        pl_trainer.fit(model=test_module, datamodule=test_data_module)

    @run_locally
    @tempdir
    def test_train_video_model(self, root_dir: str) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            create_small_kinetics_dataset(root_dir)
            # Config is relative to a module
            cfg = compose(
                config_name="video_classification_train_app_conf",
                overrides=[
                    f"datamodule.dataloader.train.dataset.data_path={root_dir}/train.csv",
                    f"datamodule.dataloader.val.dataset.data_path={root_dir}/val.csv",
                    f"datamodule.dataloader.test.dataset.data_path={root_dir}/val.csv",
                    f"datamodule.dataloader.train.dataset.video_path_prefix={root_dir}",
                    f"datamodule.dataloader.val.dataset.video_path_prefix={root_dir}",
                    f"datamodule.dataloader.test.dataset.video_path_prefix={root_dir}",
                    "datamodule.dataloader.train.num_workers=0",
                    "datamodule.dataloader.val.num_workers=0",
                    "datamodule.dataloader.test.num_workers=0",
                    "datamodule.dataloader.train.batch_size=2",
                    "datamodule.dataloader.val.batch_size=2",
                    "datamodule.dataloader.test.batch_size=2",
                    "+module/lr_scheduler=cosine_with_warmup",
                    "trainer.logger=true",
                ],
            )
            self.assertEqual(cfg.trainer.max_epochs, 1)

            self.train(cfg)

    @run_locally
    @tempdir
    def test_train_video_model_simclr(self, root_dir: str) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            create_small_kinetics_dataset(root_dir)
            # Config is relative to a module
            cfg = compose(
                config_name="simclr_train_app_conf",
                overrides=[
                    f"datamodule.dataloader.train.dataset.data_path={root_dir}/train.csv",
                    f"datamodule.dataloader.val.dataset.data_path={root_dir}/val.csv",
                    f"datamodule.dataloader.test.dataset.data_path={root_dir}/val.csv",
                    f"datamodule.dataloader.train.dataset.video_path_prefix={root_dir}",
                    f"datamodule.dataloader.val.dataset.video_path_prefix={root_dir}",
                    f"datamodule.dataloader.test.dataset.video_path_prefix={root_dir}",
                    "datamodule.dataloader.train.num_workers=0",
                    "datamodule.dataloader.val.num_workers=0",
                    "datamodule.dataloader.test.num_workers=0",
                    "module.knn_memory.length=50",
                    "module.knn_memory.knn_k=2",
                    "datamodule.dataloader.train.batch_size=2",
                    "datamodule.dataloader.val.batch_size=2",
                    "datamodule.dataloader.test.batch_size=2",
                    "trainer.logger=true",
                ],
            )
            self.assertEqual(cfg.trainer.max_epochs, 1)

            self.train(cfg)

    @run_locally
    @tempdir
    def test_train_video_model_byol(self, root_dir: str) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            create_small_kinetics_dataset(root_dir)
            # Config is relative to a module
            cfg = compose(
                config_name="byol_train_app_conf",
                overrides=[
                    f"datamodule.dataloader.train.dataset.data_path={root_dir}/train.csv",
                    f"datamodule.dataloader.val.dataset.data_path={root_dir}/val.csv",
                    f"datamodule.dataloader.test.dataset.data_path={root_dir}/val.csv",
                    f"datamodule.dataloader.train.dataset.video_path_prefix={root_dir}",
                    f"datamodule.dataloader.val.dataset.video_path_prefix={root_dir}",
                    f"datamodule.dataloader.test.dataset.video_path_prefix={root_dir}",
                    "datamodule.dataloader.train.num_workers=0",
                    "datamodule.dataloader.val.num_workers=0",
                    "datamodule.dataloader.test.num_workers=0",
                    "module.knn_memory.length=50",
                    "module.knn_memory.knn_k=2",
                    "datamodule.dataloader.train.batch_size=2",
                    "datamodule.dataloader.val.batch_size=2",
                    "datamodule.dataloader.test.batch_size=2",
                    "trainer.logger=true",
                ],
            )
            self.assertEqual(cfg.trainer.max_epochs, 1)

            self.train(cfg)
