# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch


class Video(ABC):
    """
    Video provides an interface to access clips from a video container.
    """

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Returns:
            duration of the video in seconds
        """
        pass

    @abstractmethod
    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the internal video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            video_data_dictonary: A dictionary mapping strings to tensor of the clip's
                underlying data.

        """
        pass
