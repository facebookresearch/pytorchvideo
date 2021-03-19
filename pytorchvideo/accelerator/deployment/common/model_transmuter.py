# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from typing import List

import torch.nn as nn


"""
This file contains top-level transmuter to convert user input model (nn.Module) into
an equivalent model composed of efficientBlocks for target device.
Specifically, each target device has a transmuter list, which contains transmuter
functions to convert module into equivalent efficientBlock. Each transmuter list is
registered in EFFICIENT_BLOCK_TRANSMUTER_REGISTRY to be accessed by top-level transmuter.
"""
EFFICIENT_BLOCK_TRANSMUTER_REGISTRY = {}


def _find_equivalent_efficient_module(
    module_input: nn.Module,
    efficient_block_transmuter_list: List,
    module_name: str = "",
):
    """
    Given module_input, search through efficient_block_registry to see whether the
    module_input can be replaced with equivalent efficientBlock. Returns None if no
    equivalent efficientBlock is found, else returns an instance of equivalent
    efficientBlock.
    Args:
        module_input (nn.Module): module to be replaced by equivalent efficientBlock
        efficient_block_transmuter_list (list): a transmuter list that contains transmuter
            functions for available efficientBlocks
        module_name (str): name of module_input in original model
    """
    eq_module_hit_list = []
    for iter_func in efficient_block_transmuter_list:
        eq_module = iter_func(module_input)
        if eq_module is not None:
            eq_module_hit_list.append(eq_module)
    if len(eq_module_hit_list) > 0:
        # Check for multiple matches.
        if len(eq_module_hit_list) > 1:
            logging.warning(f"{module_name} has multiple matches:")
            for iter_match in eq_module_hit_list:
                logging.warning(f"{iter_match.__class__.__name__} is a match.")
            logging.warning(
                f"Will use {eq_module_hit_list[0]} as it has highest priority."
            )
        return eq_module_hit_list[0]
    return None


def transmute_model(
    model: nn.Module,
    target_device: str = "mobile_cpu",
    prefix: str = "",
):
    """
    Recursively goes through user input model and replace module in place with available
    equivalent efficientBlock for target device.
    Args:
        model (nn.Module): user input model to be transmuted
        target_device (str): name of target device, used to access transmuter list in
            EFFICIENT_BLOCK_TRANSMUTER_REGISTRY
        prefix (str): name of current hierarchy in user model
    """
    assert (
        target_device in EFFICIENT_BLOCK_TRANSMUTER_REGISTRY
    ), f"{target_device} not registered in EFFICIENT_BLOCK_TRANSMUTER_REGISTRY!"
    transmuter_list = EFFICIENT_BLOCK_TRANSMUTER_REGISTRY[target_device]
    for name, child in model.named_children():
        equivalent_module = _find_equivalent_efficient_module(
            child, transmuter_list, module_name=f"{prefix}.{name}"
        )
        if equivalent_module is not None:
            model._modules[name] = equivalent_module
            logging.info(
                f"Replacing {prefix}.{name} ({child.__class__.__name__}) with "
                f"{equivalent_module.__class__.__name__}"
            )
        else:
            transmute_model(
                child,
                target_device=target_device,
                prefix=f"{prefix}.{name}",
            )
