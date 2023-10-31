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
    Recursively adds a forward hook to each module in a network, recording input tensor sizes
    in the provided input_tensor_size_lut dictionary.

    Args:
        module (nn.Module): The input module to add hooks to, recursively.
        input_tensor_size_lut (dict): A dictionary to record input tensor sizes for the hook function.
        hook_handle_list (list): A list to contain hook handles.
        base_name (str): The base name for the input module.

    This helper function iterates through the input `module` and its children, registering a forward
    hook for each module. The hook function records the input tensor size for the module in the
    `input_tensor_size_lut` dictionary using the `base_name` as the key.

    Note: Forward hooks are useful for monitoring and analyzing the input tensor sizes as they pass
    through each module in a neural network.
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
    convert_for_quantize: bool = False,
    native_conv3d_op_qnnpack: bool = False,
) -> None:
    """
    Recursively traverses sub-modules in a neural network and performs module conversion
    if applicable. For efficient blocks (instances of EfficientBlockBase) with a 'convert'
    method, it calls the 'convert' method with the input tensor size obtained from the
    'input_tensor_size_lut'. If the module is not an efficient block, it explores child
    modules to find efficient blocks in lower hierarchy.

    Args:
        module (nn.Module): The input module for conversion.
        input_tensor_size_lut (dict): A dictionary containing input tensor sizes for reference.
        base_name (str): The name of the module.
        convert_for_quantize (bool): Whether this module is intended for quantization.
        native_conv3d_op_qnnpack (bool): Whether the QNNPACK version has native int8 Conv3d.

    This helper function is designed for recursively exploring a neural network and converting
    specific modules, such as efficient blocks. If a module is an instance of EfficientBlockBase
    and has a 'convert' method, it calls the 'convert' method with the input tensor size from
    the 'input_tensor_size_lut'. If the module is not an efficient block, it continues to explore
    its child modules in search of efficient blocks in lower hierarchies.

    Note: Module conversion is a common step in optimizing and adapting neural networks for
    specific hardware or use cases, such as mobile CPUs.
    """
    if isinstance(module, EfficientBlockBase):
        module.convert(
            input_tensor_size_lut[base_name],
            convert_for_quantize=convert_for_quantize,
            native_conv3d_op_qnnpack=native_conv3d_op_qnnpack,
        )
    else:
        for name, child in module.named_children():
            _convert_module(
                child,
                input_tensor_size_lut,
                base_name=f"{base_name}.{name}",
                convert_for_quantize=convert_for_quantize,
                native_conv3d_op_qnnpack=native_conv3d_op_qnnpack,
            )


def convert_to_deployable_form(
    model: nn.Module,
    input_tensor: torch.Tensor,
    convert_for_quantize: bool = False,
    native_conv3d_op_qnnpack: bool = False,
) -> nn.Module:
    """
    Converts an input model into a deployable form and returns a copy of the modified model.

    Args:
        model (nn.Module): The input model for conversion. The model can consist of a mix
            of efficient blocks (instances of EfficientBlockBase) and non-efficient blocks.
            Efficient blocks are converted using their `convert()` method, while other
            blocks remain unchanged.
        input_tensor (torch.Tensor): The input tensor used for the model. The conversion for
            deployable form on mobile CPU is designed for a single input tensor size. The
            future input tensor to the converted model should match the size of the
            `input_tensor` specified here.
        convert_for_quantize (bool): Indicates whether this module is intended for quantization.
        native_conv3d_op_qnnpack (bool): Specifies whether the QNNPACK version has native
            int8 Conv3d support.

    Returns:
        nn.Module: A copy of the input model converted into a deployable form.

    This function prepares the input model for deployment by performing the following steps:
    1. Captures input tensor sizes during forward pass.
    2. Executes a forward pass to record input tensor sizes.
    3. Removes forward hooks used for input tensor size capture.
    4. Creates a deep copy of the input model for conversion.
    5. Converts the copied model by applying the `_convert_module` function.
    6. Returns the converted model suitable for deployment.
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
    _convert_module(
        model_converted,
        input_tensor_size_lut,
        convert_for_quantize=convert_for_quantize,
        native_conv3d_op_qnnpack=native_conv3d_op_qnnpack,
    )
    return model_converted
