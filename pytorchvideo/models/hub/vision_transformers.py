# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any

import torch.nn as nn
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)

from .utils import hub_model_builder, MODEL_ZOO_ROOT_DIR


checkpoint_paths = {
    "mvit_base_16x4": "{}/kinetics/MVIT_B_16x4.pyth".format(MODEL_ZOO_ROOT_DIR),
    "mvit_base_32x3": "{}/kinetics/MVIT_B_32x3_f294077834.pyth".format(
        MODEL_ZOO_ROOT_DIR
    ),
    "mvit_base_16": "{}/imagenet/MVIT_B_16_f292487636.pyth".format(MODEL_ZOO_ROOT_DIR),
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

mvit_video_base_32x3_config = {
    "spatial_size": 224,
    "temporal_size": 32,
    "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
    "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
    "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    "pool_kv_stride_adaptive": [1, 8, 8],
    "pool_kvq_kernel": [3, 3, 3],
}

mvit_image_base_16_config = {
    "spatial_size": 224,
    "temporal_size": 1,
    "depth": 16,
    "conv_patch_embed_kernel": [7, 7],
    "conv_patch_embed_stride": [4, 4],
    "conv_patch_embed_padding": [3, 3],
    "use_2d_patch": True,
    "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
    "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
    "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    "pool_kv_stride_adaptive": [1, 4, 4],
    "pool_kvq_kernel": [1, 3, 3],
}


def mvit_base_16x4(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Multiscale Vision Transformers model architecture [1] trained with an 16x4
    setting on the Kinetics400 dataset. Model with pretrained weights has top1
    accuracy of 78.9%.

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


def mvit_base_32x3(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Multiscale Vision Transformers model architecture [1] trained with an 32x3
    setting on the Kinetics400 dataset. Model with pretrained weights has top1
    accuracy of 80.3%.

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
        checkpoint_path=checkpoint_paths["mvit_base_32x3"],
        default_config=mvit_video_base_32x3_config,
        **kwargs,
    )


def mvit_base_16(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Multiscale Vision Transformers model architecture [1] with a depth 16 trained on
    ImageNet-1k dataset. Model with pretrained weights has top1 accuracy of 83%.

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
        checkpoint_path=checkpoint_paths["mvit_base_16"],
        default_config=mvit_image_base_16_config,
        **kwargs,
    )
