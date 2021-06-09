# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .csn import csn_r101
from .efficient_x3d_mobile_cpu import efficient_x3d_s, efficient_x3d_xs
from .r2plus1d import r2plus1d_r50
from .resnet import c2d_r50, i3d_r50, slow_r50, slow_r50_detection
from .slowfast import (
    slowfast_16x8_r101_50_50,
    slowfast_r50,
    slowfast_r50_detection,
    slowfast_r101,
)
from .x3d import x3d_l, x3d_m, x3d_s, x3d_xs
