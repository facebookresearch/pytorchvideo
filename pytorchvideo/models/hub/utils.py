# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Callable, Dict, Optional

import torch.nn as nn
from torch.hub import load_state_dict_from_url


MODEL_ZOO_ROOT_DIR = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo"


def hub_model_builder(
    model_builder_func: Callable,
    pretrained: bool = False,
    progress: bool = True,
    checkpoint_path: str = "",
    default_config: Optional[Dict[Any, Any]] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    model_builder_func (Callable): Model builder function.
    pretrained (bool): Whether to load a pretrained model or not. Default: False.
    progress (bool): Whether or not to display a progress bar to stderr. Default: True.
    checkpoint_path (str): URL of the model weight to download.
    default_config (Dict): Default model configs that is passed to the model builder.
    **kwargs: (Any): Additional model configs. Do not modify the model configuration
    via the kwargs for pretrained model.
    """
    if pretrained:
        assert len(kwargs) == 0, "Do not change kwargs for pretrained model."

    if default_config is not None:
        for argument, value in default_config.items():
            if kwargs.get(argument) is None:
                kwargs[argument] = value

    model = model_builder_func(**kwargs)
    if pretrained:
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            checkpoint_path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)
    return model
