# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


def register_components() -> None:
    """
    Calls register_components() for all subfolders so we can register
    subcomponents to Hydra's ConfigStore.
    """
    import pytorchvideo_trainer.datamodule.datamodule  # noqa
    import pytorchvideo_trainer.module.byol  # noqa
    import pytorchvideo_trainer.module.lr_policy  # noqa
    import pytorchvideo_trainer.module.moco_v2  # noqa
    import pytorchvideo_trainer.module.optimizer  # noqa
    import pytorchvideo_trainer.module.simclr  # noqa
    import pytorchvideo_trainer.module.video_classification  # noqa
    import pytorchvideo_trainer.train_app  # noqa
