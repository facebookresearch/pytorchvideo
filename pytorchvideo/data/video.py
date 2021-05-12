# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional

import torch
from iopath.common.file_io import g_pathmgr


class VideoPathHandler(object):
    """
    Utility class that handles all deciphering and caching of video paths for
    encoded and frame videos.
    """

    def __init__(self) -> None:
        # Pathmanager isn't guaranteed to be in correct order,
        # sorting is expensive, so we cache paths in case of frame video and reuse.
        self.path_order_cache = {}

    def video_from_path(self, filepath, decode_audio=False, decoder="pyav", fps=30):
        try:
            is_file = g_pathmgr.isfile(filepath)
            is_dir = g_pathmgr.isdir(filepath)
        except NotImplementedError:

            # Not all PathManager handlers support is{file,dir} functions, when this is the
            # case, we default to assuming the path is a file.
            is_file = True
            is_dir = False

        if is_file:
            from pytorchvideo.data.encoded_video import EncodedVideo

            return EncodedVideo.from_path(filepath, decode_audio, decoder)
        elif is_dir:
            from pytorchvideo.data.frame_video import FrameVideo

            assert not decode_audio, "decode_audio must be False when using FrameVideo"
            return FrameVideo.from_directory(
                filepath, fps, path_order_cache=self.path_order_cache
            )
        else:
            raise FileNotFoundError(f"{filepath} not found.")


class Video(ABC):
    """
    Video provides an interface to access clips from a video container.
    """

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

    def close(self):
        pass
