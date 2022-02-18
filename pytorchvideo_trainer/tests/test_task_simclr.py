# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict
from torchrecipes.core.base_train_app import BaseTrainApp
from util import (
    BaseTrainAppTestCase,
    create_small_kinetics_dataset,
    run_locally,
    tempdir,
)


class TestSimCLRTrainApp(BaseTrainAppTestCase):
    def get_train_app(
        self,
        root_dir: str,
        fast_dev_run: bool = True,
        logger: bool = False,
    ) -> BaseTrainApp:
        create_small_kinetics_dataset(root_dir)
        overrides = [
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
            "trainer.logger=false",
        ]
        app = self.create_app_from_hydra(
            config_module="pytorchvideo_trainer.conf",
            config_name="simclr_train_app_conf",
            overrides=overrides,
        )
        trainer_overrides = {"fast_dev_run": fast_dev_run, "logger": logger}
        self.mock_trainer_params(app, trainer_overrides)
        return app

    @run_locally
    @tempdir
    def test_simclr_app_train_test_30_views(self, root_dir: str) -> None:
        train_app = self.get_train_app(
            root_dir=root_dir, fast_dev_run=False, logger=False
        )
        output = train_app.train()
        self.assertIsNotNone(output)
        output = train_app.test()
        self.assertIsNotNone(output)

        video_clips_cnts = getattr(train_app.module, "video_clips_cnts", None)
        num_ensemble_views = getattr(train_app.datamodule, "num_ensemble_views", 10)
        num_spatial_crops = getattr(train_app.datamodule, "num_spatial_crops", 3)
        self.assertIsNotNone(video_clips_cnts)
        for _, sample_cnts in video_clips_cnts.items():
            self.assertEqual(num_ensemble_views * num_spatial_crops, sample_cnts)
