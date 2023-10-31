# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import List


def set_attributes(self, params: List[object] = None) -> None:
    """
    Utility function used in classes to set attributes from the input dictionary of parameters.
    
    Args:
        params: A List of attribute names and their corresponding values.
    """
    for key, value in params.items():
        setattr(self, key, value)

def round_width(width: int, multiplier: float, min_width: int = 8, divisor: int = 8, ceil: bool = False) -> int:
    """
    Round the width of filters based on a width multiplier.
    
    Args:
        width (int): The channel dimensions of the input.
        multiplier (float): The multiplication factor.
        min_width (int, optional): The minimum width after multiplication.
        divisor (int, optional): The new width should be divisible by divisor.
        ceil (bool, optional): If True, use ceiling as the rounding method.

    Returns:
        int: The rounded width value.
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

def round_repeats(repeats: int, multiplier: float) -> int:
    """
    Round the number of layers based on a depth multiplier.

    Args:
        repeats (int): The original number of layers.
        multiplier (float): The depth multiplier.

    Returns:
        int: The rounded number of layers.
    """
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
