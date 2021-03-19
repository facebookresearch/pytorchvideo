# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
from pytorchvideo.accelerator.efficient_blocks.efficient_block_base import (
    EfficientBlockBase,
)


def _add_input_tensor_size_lut_hook(
    module: nn.Module,
    input_tensor_size_lut: Dict,
    hook_handle_list: List,
    base_name: str = "",
) -> None:
    """
    This helper function recursively goes through all modules in a network, registers
    forward hook function to each module. The hook function records the input tensor
    size in forward in input_tensor_size_lut[base_name].
    Args:
        module (nn.Module): input module to add hook recursively.
        input_tensor_size_lut (dict): lut to record input tensor size for hook function.
        hook_handle_list (list): a list to contain hook handles.
        base_name (str): name for module input.
    """

    def hook_fn(_, _in, _out):
        if isinstance(_in[0], torch.Tensor):
            input_tensor_size_lut[base_name] = tuple(_in[0].size())
        return

    handle = module.register_forward_hook(hook_fn)
    hook_handle_list.append(handle)
    for name, child in module.named_children():
        _add_input_tensor_size_lut_hook(
            child,
            input_tensor_size_lut,
            hook_handle_list,
            base_name=f"{base_name}.{name}",
        )


def _convert_module(
    module: nn.Module,
    input_tensor_size_lut: Dict,
    base_name: str = "",
) -> None:
    """
    This helper function recursively goes through sub-modules in a network. If current
    module is a efficient block (instance of EfficientBlockBase) with convert() method,
    its convert() method will be called, and the input tensor size (needed by efficient
    blocks for mobile cpu) will be provided by matching module name in
    input_tensor_size_lut.
    Otherwise if the input module is a non efficient block, this function will try to go
    through child modules of input module to look for any efficient block in lower
    hierarchy.
    Args:
        module (nn.Module): input module for convert.
        input_tensor_size_lut (dict): input tensor size look-up table.
        base_name (str): module name for input module.
    """
    if isinstance(module, EfficientBlockBase):
        module.convert(input_tensor_size_lut[base_name])
    else:
        for name, child in module.named_children():
            _convert_module(
                child, input_tensor_size_lut, base_name=f"{base_name}.{name}"
            )


def convert_to_deployable_form(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> nn.Module:
    """
    This function takes an input model, and returns a deployable model copy.
    Args:
        model (nn.Module): input model for conversion. The model can include a mix of
            efficient blocks (instances of EfficientBlockBase) and non efficient blocks.
            The efficient blocks will be converted by calling its convert() method, while
            other blocks will stay unchanged.
        input_tensor (torch.Tensor): input tensor for model. Note current conversion for
            deployable form in mobile cpu only works for single input tensor size (i.e.,
            the future input tensor to converted model should have the same size as
            input_tensor specified here).
    """
    input_tensor_size_lut = {}
    hook_handle_list = []
    _add_input_tensor_size_lut_hook(model, input_tensor_size_lut, hook_handle_list)
    # Run forward to fill in input tensor lut.
    model.eval()
    model(input_tensor)
    # Remove forward hooks.
    for handle in hook_handle_list:
        handle.remove()
    model_converted = deepcopy(model)
    model_converted.eval()
    _convert_module(model_converted, input_tensor_size_lut)
    return model_converted
