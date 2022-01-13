# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

from hydra.experimental import compose, initialize_config_module
from hydra.utils import instantiate  # @manual
from pytorchvideo_trainer.datamodule.datamodule import PyTorchVideoDataModule


class TestKineticsDataModuleConf(unittest.TestCase):
    def test_init_with_hydra(self) -> None:
        with initialize_config_module(config_module="pytorchvideo_trainer.conf"):
            test_conf = compose(
                config_name="video_classification_train_app_conf",
                overrides=[
                    "datamodule/dataloader=kinetics_classification",
                    "datamodule/transforms=kinetics_classification_slow",
                ],
            )
            print(test_conf)
            kinetics_data_module = instantiate(
                test_conf.datamodule,
                _recursive_=False,
            )
            self.assertIsInstance(kinetics_data_module, PyTorchVideoDataModule)
            self.assertIsNotNone(kinetics_data_module.transforms["train"])
            self.assertIsNotNone(kinetics_data_module.transforms["val"])
            self.assertIsNotNone(kinetics_data_module.transforms["test"])
