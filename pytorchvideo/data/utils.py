# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import itertools
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


def thwc_to_cthw(data: torch.Tensor) -> torch.Tensor:
    """
    Permute tensor from (time, height, weight, channel) to
    (channel, height, width, time).
    """
    return data.permute(3, 0, 1, 2)


class MultiProcessSampler(torch.utils.data.Sampler):
    """
    MultiProcessSampler splits sample indices from a PyTorch Sampler evenly across
    workers spawned by a PyTorch DataLoader.
    """

    def __init__(self, sampler: torch.utils.data.Sampler) -> None:
        self._sampler = sampler

    def __iter__(self):
        """
        Returns:
            Iterator for underlying PyTorch Sampler indices split by worker id.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers != 0:

            # Split sampler indexes by worker.
            video_indexes = range(len(self._sampler))
            worker_splits = np.array_split(video_indexes, worker_info.num_workers)
            worker_id = worker_info.id
            worker_split = worker_splits[worker_id]
            if len(worker_split) == 0:
                logger.warning(
                    f"More data workers({worker_info.num_workers}) than videos"
                    f"({len(self._sampler)}). For optimal use of processes "
                    "reduce num_workers."
                )
                return iter(())

            iter_start = worker_split[0]
            iter_end = worker_split[-1] + 1
            worker_sampler = itertools.islice(iter(self._sampler), iter_start, iter_end)
        else:

            # If no worker processes found, we return the full sampler.
            worker_sampler = iter(self._sampler)

        return worker_sampler
