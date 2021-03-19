# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import List


def set_attributes(self, params: List[object] = None) -> None:
    """
    An utility function used in classes to set attributes from the input list of parameters.
    Args:
        params (list): list of parameters.
    """
    if params:
        for k, v in params.items():
            if k != "self":
                setattr(self, k, v)


def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
    """
    Round width of filters based on width multiplier
    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int): the minimum width after multiplication.
        divisor (int): the new width should be dividable by divisor.
        ceil (bool): If True, use ceiling as the rounding method.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_width = min_width or divisor
    if ceil:
        width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
    else:
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def round_repeats(repeats, multiplier):
    """
    Round number of layers based on depth multiplier.
    """
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
