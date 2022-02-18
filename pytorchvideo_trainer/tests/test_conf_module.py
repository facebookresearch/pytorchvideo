# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import hydra
from hydra.experimental import compose, initialize_config_module
from pytorchvideo_trainer.module.byol import BYOLModule
from pytorchvideo_trainer.module.moco_v2 import MOCOV2Module
from pytorchvideo_trainer.module.simclr import SimCLRModule
from pytorchvideo_trainer.module.video_classification import VideoClassificationModule


class TestVideoClassificationModuleConf(unittest.TestCase):
    def test_init_with_hydra(self) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            test_conf = compose(
                config_name="video_classification_train_app_conf",
                overrides=["module/model=slow_r50"],
            )
            test_module = hydra.utils.instantiate(test_conf.module, _recursive_=False)
            self.assertIsInstance(test_module, VideoClassificationModule)
            self.assertIsNotNone(test_module.model)


class TestVideoSimCLRModuleConf(unittest.TestCase):
    def test_init_with_hydra(self) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            test_conf = compose(
                config_name="simclr_train_app_conf",
            )
            test_module = hydra.utils.instantiate(test_conf.module, _recursive_=False)
            self.assertIsInstance(test_module, SimCLRModule)
            self.assertIsNotNone(test_module.model)


class TestVideoBYOLModuleConf(unittest.TestCase):
    def test_init_with_hydra(self) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            test_conf = compose(
                config_name="byol_train_app_conf",
            )
            test_module = hydra.utils.instantiate(test_conf.module, _recursive_=False)
            self.assertIsInstance(test_module, BYOLModule)
            self.assertIsNotNone(test_module.model)


class TestVideoMOCOV2ModuleConf(unittest.TestCase):
    def test_init_with_hydra(self) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            test_conf = compose(
                config_name="moco_v2_train_app_conf",
                # overrides=["module/model=resnet"],
            )
            test_module = hydra.utils.instantiate(test_conf.module, _recursive_=False)
            self.assertIsInstance(test_module, MOCOV2Module)
            self.assertIsNotNone(test_module.model)
