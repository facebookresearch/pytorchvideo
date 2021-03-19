# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import io
import logging
import pathlib
from typing import BinaryIO, Dict, Optional

import torch
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.decoder import DecoderType

from .encoded_video_pyav import EncodedVideoPyAV
from .encoded_video_torchvision import EncodedVideoTorchVision
from .video import Video


logger = logging.getLogger(__name__)


def select_video_class(decoder: str) -> Video:
    """
    Select the class for accessing clips based on provided decoder string

    Args:
        decoder (str): Defines what type of decoder used to decode a video.
    """
    if DecoderType(decoder) == DecoderType.PYAV:
        video_cls = EncodedVideoPyAV
    elif DecoderType(decoder) == DecoderType.TORCHVISION:
        video_cls = EncodedVideoTorchVision
    else:
        raise NotImplementedError(f"Unknown decoder type {decoder}")
    return video_cls


class EncodedVideo(Video):
    """
    EncodedVideo is an abstraction for accessing clips from an encoded video.
    It supports selective decoding when header information is available.
    """

    @classmethod
    def from_path(
        cls, file_path: str, decode_audio: bool = True, decoder: str = "pyav"
    ):
        """
        Fetches the given video path using PathManager (allowing remote uris to be
        fetched) and constructs the EncodedVideo object.

        Args:
            file_path (str): a PathManager file-path.
        """
        # We read the file with PathManager so that we can read from remote uris.
        with g_pathmgr.open(file_path, "rb") as fh:
            video_file = io.BytesIO(fh.read())

        return cls(video_file, pathlib.Path(file_path).name, decode_audio, decoder)

    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.

            decoder (str): Defines what type of decoder used to decode a video.
        """
        video_cls = select_video_class(decoder)
        self.encoded_video = video_cls(file, video_name, decode_audio)

    @property
    def name(self) -> Optional[str]:
        """
        Returns:
            name: the name of the stored video if set.
        """
        return self.encoded_video.name

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self.encoded_video.duration

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_data:
                A dictionary mapping the entries at "video" and "audio" to a tensors.

                "video": A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.float32 and
                in the range [0 - 255].

                "audio": A tensor of the clip's audio samples with shape:
                (samples). The samples are of type torch.float32 and
                in the range [0 - 255].

            Returns None if no video or audio found within time range.

        """
        return self.encoded_video.get_clip(start_sec, end_sec)

    def close(self):
        """
        Closes the internal video container.
        """
        self.encoded_video.close()
