from typing import List

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {26: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}


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
