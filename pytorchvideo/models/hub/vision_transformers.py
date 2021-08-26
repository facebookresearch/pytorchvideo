# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any

import torch.nn as nn
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)

from .utils import MODEL_ZOO_ROOT_DIR, hub_model_builder


checkpoint_paths = {
    "mvit_base_16x4": "{}/kinetics/MVIT_B_16x4.pyth".format(MODEL_ZOO_ROOT_DIR),
}


mvit_video_base_config = {
    "spatial_size": 224,
    "temporal_size": 16,
    "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
    "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
    "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    "pool_kv_stride_adaptive": [1, 8, 8],
    "pool_kvq_kernel": [3, 3, 3],
}


def mvit_base_16x4(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Multiscale Vision Transformers model architecture [1] trained with an 16x4
    setting on the Kinetics400 dataset. Model with pretrained weights has top1
    accuracy of 79.0.

    [1] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra
    Malik, Christoph Feichtenhofer, "Multiscale Vision Transformers"
    https://arxiv.org/pdf/2104.11227.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics400 dataset.
        progress (bool): If True, displays a progress bar of the download to stderr.
        kwargs: Use these to modify any of the other model settings. All the
            options are defined in create_multiscale_vision_transformers.

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """

    return hub_model_builder(
        model_builder_func=create_multiscale_vision_transformers,
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["mvit_base_16x4"],
        default_config=mvit_video_base_config,
        **kwargs,
    )
