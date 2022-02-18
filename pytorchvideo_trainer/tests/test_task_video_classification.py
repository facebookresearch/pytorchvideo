# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict
from torchrecipes.core.base_train_app import BaseTrainApp
from util import (
    BaseTrainAppTestCase,
    create_small_kinetics_dataset,
    run_locally,
    tempdir,
)


class TestVideoClassificationTrainApp(BaseTrainAppTestCase):
    def get_train_app(
        self,
        root_dir: str,
        precise_bn_num_batches: int = 0,
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
            "datamodule.dataloader.train.batch_size=2",
            "datamodule.dataloader.val.batch_size=2",
            "datamodule.dataloader.test.batch_size=2",
            "+module/lr_scheduler=cosine_with_warmup",
            "trainer.logger=false",
        ]
        if precise_bn_num_batches > 0:
            overrides.extend(
                [
                    "+callbacks=precise_bn",
                    f"callbacks.precise_bn.num_batches={precise_bn_num_batches}",
                    "datamodule.dataloader.train.batch_size=2",
                    "datamodule.dataloader.val.batch_size=2",
                    "datamodule.dataloader.test.batch_size=2",
                ]
            )
        app = self.create_app_from_hydra(
            config_module="pytorchvideo_trainer.conf",
            config_name="video_classification_train_app_conf",
            overrides=overrides,
        )
        trainer_overrides = {"fast_dev_run": fast_dev_run, "logger": logger}
        self.mock_trainer_params(app, trainer_overrides)
        return app

    @run_locally
    @tempdir
    def test_video_classification_app_train(self, root_dir: str) -> None:
        train_app = self.get_train_app(root_dir=root_dir, logger=False)
        output = train_app.train()
        self.assertIsNotNone(output)

    @run_locally
    @tempdir
    def test_video_classification_app_train_with_precise_bn(
        self, root_dir: str
    ) -> None:
        train_app = self.get_train_app(
            root_dir=root_dir, precise_bn_num_batches=2, logger=False
        )
        output = train_app.train()
        self.assertIsNotNone(output)

    @run_locally
    @tempdir
    def test_video_classification_app_test(self, root_dir: str) -> None:
        train_app = self.get_train_app(root_dir=root_dir)
        output = train_app.test()
        self.assertIsNotNone(output)

    @run_locally
    @tempdir
    def test_video_classification_app_test_30_views(self, root_dir: str) -> None:
        train_app = self.get_train_app(root_dir=root_dir, fast_dev_run=False)
        train_app.test()
        video_clips_cnts = getattr(train_app.module, "video_clips_cnts", None)
        num_ensemble_views = getattr(train_app.datamodule, "num_ensemble_views", 10)
        num_spatial_crops = getattr(train_app.datamodule, "num_spatial_crops", 3)
        self.assertIsNotNone(video_clips_cnts)
        for _, sample_cnts in video_clips_cnts.items():
            self.assertEqual(num_ensemble_views * num_spatial_crops, sample_cnts)
