# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

dependencies = ["torch"]
from pytorchvideo.models.hub import (  # noqa: F401, E402
    slow_r50,
    slowfast_r50,
    slowfast_r101,
    x3d_m,
    x3d_s,
    x3d_xs,
    efficient_x3d_xs,
    efficient_x3d_s,
)
