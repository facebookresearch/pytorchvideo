# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Learning rate policy."""
import math
from dataclasses import dataclass
from typing import Callable, List

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class LRSchedulerConf:
    # common
    lr_policy: str = MISSING
    lr: float = MISSING
    max_iters: int = MISSING
    warmup_iters: int = MISSING
    warmup_start_lr: float = MISSING

    # cosine
    cosine_end_lr: float = MISSING
    cosine_after_warmup: bool = MISSING

    # LRS
    steps: List[int] = MISSING
    lrs: List[float] = MISSING


cs = ConfigStore()
cs.store(
    group="schema/module/lr_scheduler",
    name="lr_scheduler_conf",
    node=LRSchedulerConf,
    package="module.lr_scheduler",
)


def get_lr_at_epoch(cfg: LRSchedulerConf, cur_epoch: float) -> float:
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.

    Args:
        cfg (LRSchedulerConf): Hydra / omega conf object associated with
            Learningrate scheduler. The schema can be found in
            `LRSchedulerConf` and the example configs can be found in
            `pytorchvideo_trainer/conf/module/lr_scheduler`.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.lr_policy)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.warmup_iters:
        lr_start = cfg.warmup_start_lr
        lr_end = get_lr_func(cfg.lr_policy)(cfg, cfg.warmup_iters)
        alpha = (lr_end - lr_start) / cfg.warmup_iters
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg: LRSchedulerConf, cur_epoch: float) -> float:
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter ,SGDR: Stochastic Gradient
    Descent With Warm Restarts.

    Args:
        cfg (CfgNode): Hydra / omega conf object associated with
            Learningrate scheduler. The schema can be found in
            `LRSchedulerConf` and the example configs can be found in
            `pytorchvideo_trainer/conf/module/lr_scheduler`.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = cfg.warmup_iters if cfg.cosine_after_warmup else 0.0
    assert cfg.cosine_end_lr < cfg.lr
    return (
        cfg.cosine_end_lr
        + (cfg.lr - cfg.cosine_end_lr)
        * (math.cos(math.pi * (cur_epoch - offset) / (cfg.max_iters - offset)) + 1.0)
        * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg: LRSchedulerConf, cur_epoch: float) -> float:
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.

    Args:
        cfg (CfgNode): configs. Hydra / omega conf object associated with
            Learningrate scheduler. The schema can be found in
            `LRSchedulerConf` and the example configs can be found in
            `pytorchvideo_trainer/conf/module/lr_scheduler`.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.lrs[ind] * cfg.lr


def get_step_index(cfg: LRSchedulerConf, cur_epoch: float) -> int:
    """
    Retrieves the lr step index for the given epoch.

    Args:
        cfg (CfgNode): Hydra / omega conf object associated with
            Learningrate scheduler. The schema can be found in
            `LRSchedulerConf` and the example configs can be found in
            `pytorchvideo_trainer/conf/module/lr_scheduler`.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.steps + [cfg.max_iters]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy: str) -> Callable:  # pyre-ignore[24]
    """
    Given the configs, retrieve the specified lr policy function.

    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]


def get_epoch_lr(cur_epoch: float, cfg: LRSchedulerConf) -> float:
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).

    Args:
        cfg (config): Hydra / omega conf object associated with
            Learningrate scheduler. The schema can be found in
            `LRSchedulerConf` and the example configs can be found in
            `pytorchvideo_trainer/conf/module/lr_scheduler`.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
