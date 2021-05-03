# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import io
import logging
import pathlib

from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.decoder import DecoderType

from .video import Video


logger = logging.getLogger(__name__)


def select_video_class(decoder: str) -> Video:
    """
    Select the class for accessing clips based on provided decoder string

    Args:
        decoder (str): Defines what type of decoder used to decode a video.
    """
    if DecoderType(decoder) == DecoderType.PYAV:
        from .encoded_video_pyav import EncodedVideoPyAV

        video_cls = EncodedVideoPyAV
    elif DecoderType(decoder) == DecoderType.TORCHVISION:
        from .encoded_video_torchvision import EncodedVideoTorchVision

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

        video_cls = select_video_class(decoder)
        return video_cls(video_file, pathlib.Path(file_path).name, decode_audio)
