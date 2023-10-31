# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional

import torch
from iopath.common.file_io import g_pathmgr


class VideoPathHandler:
    """
    Utility class that handles all deciphering and caching of video paths for
    encoded and frame videos.
    """

    def __init__(self) -> None:
        # Pathmanager isn't guaranteed to be in correct order,
        # sorting is expensive, so we cache paths in case of frame video and reuse.
        self.path_order_cache = {}

    def video_from_path(
        self, filepath, decode_video=True, decode_audio=False, decoder="pyav", fps=30
    ):
        """
        Returns a video object (either EncodedVideo or FrameVideo) based on the provided file path.

        Args:
            filepath (str): Path to the video file or directory containing frame images.
            decode_video (bool): Whether to decode the video (only for EncodedVideo).
            decode_audio (bool): Whether to decode the audio (only for EncodedVideo).
            decoder (str): The video decoder to use (only for EncodedVideo).
            fps (int): Frames per second (only for FrameVideo).

        Returns:
            Union[EncodedVideo, FrameVideo]: A video object based on the provided file path.
        
        Raises:
            FileNotFoundError: If the file or directory specified by `filepath` does not exist.
        """
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

            return EncodedVideo.from_path(
                filepath,
                decode_video=decode_video,
                decode_audio=decode_audio,
                decoder=decoder,
            )
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
        Initializes the Video object with a file-like object containing the encoded video.

        Args:
            file (BinaryIO): A file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
            video_name (Optional[str]): An optional name for the video.
            decode_audio (bool): Whether to decode audio from the video.
        """
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            float: The duration of the video in seconds.
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
            start_sec (float): The clip start time in seconds.
            end_sec (float): The clip end time in seconds.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary mapping strings to tensors
            of the clip's underlying data. It may include video frames and audio.
        """
        pass

    def close(self):
        """
        Closes any resources associated with the Video object.
        """
        pass
    