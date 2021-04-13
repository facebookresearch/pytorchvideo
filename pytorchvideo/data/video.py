# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import io
import pathlib
from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional

import torch
from iopath.common.file_io import g_pathmgr


class Video(ABC):
    """
    Video provides an interface to access clips from a video container.
    """

    @classmethod
    def from_path(cls, file_path: str, decode_audio: bool = True):
        """
        Fetches the given video path using PathManager (allowing remote uris to be
        fetched) and constructs the EncodedVideo object.

        Args:
            file_path (str): a PathManager file-path.
        """
        # We read the file with PathManager rather than pyav so that we can read from
        # remote uris.
        with g_pathmgr.open(file_path, "rb") as fh:
            video_file = io.BytesIO(fh.read())

        return cls(video_file, pathlib.Path(file_path).name, decode_audio)

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

    @abstractmethod
    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_audio: bool = True,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
        """
        pass
