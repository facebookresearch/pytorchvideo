# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-ignore-all-errors

from dataclasses import dataclass

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class OptimizerConf:
    method: str = MISSING
    lr: float = MISSING
    weight_decay: float = 1e-4
    bn_weight_decay: float = 0.0
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = True
    zero_weight_decay_1d_param: bool = False
    lars_on: bool = False


# TODO: Refactor contruct_optimer to torch.optim conf + construct_param_group
def construct_optimizer(
    model: torch.nn.Module, cfg: OptimizerConf  # noqa
) -> torch.optim.Optimizer:
    """
    Constructs a stochastic gradient descent or ADAM (or ADAMw) optimizer
    with momentum. i.e, constructs a torch.optim.Optimizer with zero-weight decay
    Batchnorm and/or no-update 1-D parameters support, based on the config.

    Supports wrapping the optimizer with Layer-wise Adaptive Rate Scaling
    (LARS): https://arxiv.org/abs/1708.03888

    Args:
        model (nn.Module): model to perform stochastic gradient descent
            optimization or ADAM optimization.
        cfg (OptimizerConf): Hydra/Omega conf object consisting hyper-parameters
            of SGD or ADAM, includes base learning rate,  momentum, weight_decay,
            dampening and etc. The supported config schema is `OptimizerConf`.
            Example config files can be found at,
            `pytorchvideo_trainer/conf/module/optim`
    """
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []
    skip = {}

    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()  # pyre-ignore[29]

    for name, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                no_grad_parameters.append(p)
            elif is_bn:
                bn_parameters.append(p)
            elif name in skip:
                zero_parameters.append(p)
            elif cfg.zero_weight_decay_1d_param and (
                len(p.shape) == 1 or name.endswith(".bias")
            ):
                zero_parameters.append(p)
            else:
                non_bn_parameters.append(p)

    optim_params = [
        {
            "params": bn_parameters,
            "weight_decay": cfg.bn_weight_decay,
            "apply_LARS": False,
        },
        {
            "params": non_bn_parameters,
            "weight_decay": cfg.weight_decay,
            "apply_LARS": cfg.lars_on,
        },
        {
            "params": zero_parameters,
            "weight_decay": 0.0,
            "apply_LARS": cfg.lars_on,
        },
    ]
    optim_params = [x for x in optim_params if len(x["params"])]  # pyre-ignore[6]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_parameters
    ) + len(zero_parameters) + len(
        no_grad_parameters
    ), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(list(model.parameters())),
    )
    print(
        "bn {}, non bn {}, zero {} no grad {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
        )
    )

    if cfg.method == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            dampening=cfg.dampening,
            nesterov=cfg.nesterov,
        )
    elif cfg.method == "adam":
        optimizer = torch.optim.Adam(
            optim_params,
            lr=cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.weight_decay,
        )
    elif cfg.method == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=cfg.lr,
            eps=1e-08,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise NotImplementedError("Does not support {} optimizer".format(cfg.method))

    if cfg.lars_on:
        optimizer = LARS(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    return optimizer


cs = ConfigStore()
cs.store(
    group="schema/module/optim",
    name="optim_conf",
    node=OptimizerConf,
    package="module.optim",
)


class LARS(torch.optim.Optimizer):
    """
    This class is adapted from
    https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py to
    include ignoring LARS application specific parameters (e.g. 1D params)

    Args:
        optimizer (torch.optim): Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr.
            See https://arxiv.org/abs/1708.03888
        clip (bool): Decides between clipping or scaling mode of LARS. If `clip=True` the
            learning rate is set to `min(optimizer_lr, local_lr)` for each parameter.
            If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps (float): epsilon kludge to help with numerical stability while calculating
        adaptive_lr.
        ignore_1d_param (float): If true, does not update 1 dimentional parameters.
    """

    def __init__(
        self,
        optimizer,
        trust_coefficient=0.02,
        clip=True,
        eps=1e-8,
        ignore_1d_param=True,
    ) -> None:
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip
        self.ignore_1d_param = ignore_1d_param

        self.defaults = self.optim.defaults

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self, closure=None):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group["weight_decay"] if "weight_decay" in group else 0
                weight_decays.append(weight_decay)
                apply_LARS = group["apply_LARS"] if "apply_LARS" in group else True
                if not apply_LARS:
                    continue
                group["weight_decay"] = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.ignore_1d_param and p.ndim == 1:  # ignore bias
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = (
                            self.trust_coefficient
                            * (param_norm)
                            / (grad_norm + param_norm * weight_decay + self.eps)
                        )

                        # clip learning rate for LARS
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied
                            # by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group["lr"], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]
