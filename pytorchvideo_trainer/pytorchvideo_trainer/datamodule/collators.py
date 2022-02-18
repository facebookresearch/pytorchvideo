# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Callable, Dict, List

from torch.utils.data._utils.collate import default_collate


# pyre-ignore[2]
def multiple_samples_collate(batch: List[Dict[str, List[Any]]]) -> Dict[str, Any]:
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.

    To be used when working with,
    `pytorchvideo_trainer.datamodule.transforms.RepeatandConverttoList`
    """
    batch_dict = {}
    for k in batch[0].keys():
        v_iter = []
        for sample_dict in batch:
            v_iter += sample_dict[k]
        batch_dict[k] = default_collate(v_iter)

    return batch_dict


# pyre-ignore[24]
_COLLATORS: Dict[str, Callable] = {
    "multiple_samples_collate": multiple_samples_collate,
}


def build_collator_from_name(name: str) -> Callable:  # pyre-ignore[24]
    """
    A utility function that returns the function handles to specific collators
    in `_COLLATORS` dictionary object based on the queried key. Used in
    `pytorchvideo_trainer.datamodule.PyTorchVideoDataModule`, etc.

    Arg:
        name (str): name of the qurried collators. The key should be present in
        `_COLLATORS` dictionary object
    """
    assert (
        name in _COLLATORS
    ), f"Inavalid Collator method. Available methods are {_COLLATORS.keys()}"
    return _COLLATORS[name]
