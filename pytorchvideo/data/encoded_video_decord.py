# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
from typing import BinaryIO, Dict, Optional, TypeVar

import torch

from .utils import thwc_to_cthw
from .video import Video


logger = logging.getLogger(__name__)

try:
    import decord
except ImportError:
    _HAS_DECORD = False
else:
    _HAS_DECORD = True

if _HAS_DECORD:
    decord.bridge.set_bridge("torch")

DecordDevice = TypeVar("DecordDevice")


class EncodedVideoDecord(Video):
    """

    Accessing clips from an encoded video using Decord video reading API
    as the decoding backend. For more details, please refer to -
    `Decord <https://github.com/dmlc/decord>`
    """

    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_video: bool = True,
        decode_audio: bool = True,
        sample_rate: int = 44100,
        mono: bool = True,
        width: int = -1,
        height: int = -1,
        num_threads: int = 0,
        fault_tol: int = -1,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
            video_name (str): An optional name assigned to the video.
            decode_video (bool): If disabled, video is not decoded.
            decode_audio (bool): If disabled, audio is not decoded.
            sample_rate: int, default is -1
                Desired output sample rate of the audio, unchanged if `-1` is specified.
            mono: bool, default is True
                Desired output channel layout of the audio. `True` is mono layout. `False`
                is unchanged.
            width : int, default is -1
                Desired output width of the video, unchanged if `-1` is specified.
            height : int, default is -1
                Desired output height of the video, unchanged if `-1` is specified.
            num_threads : int, default is 0
                Number of decoding thread, auto if `0` is specified.
            fault_tol : int, default is -1
                The threshold of corupted and recovered frames. This is to prevent silent fault
                tolerance when for example 50% frames of a video cannot be decoded and duplicate
                frames are returned. You may find the fault tolerant feature sweet in many
                cases, but not for training models. Say `N = # recovered frames`
                If `fault_tol` < 0, nothing will happen.
                If 0 < `fault_tol` < 1.0, if N > `fault_tol * len(video)`,
                raise `DECORDLimitReachedError`.
                If 1 < `fault_tol`, if N > `fault_tol`, raise `DECORDLimitReachedError`.
        """
        if not decode_video:
            raise NotImplementedError()

        self._decode_audio = decode_audio
        self._video_name = video_name
        if not _HAS_DECORD:
            raise ImportError(
                "decord is required to use EncodedVideoDecord decoder. Please "
                "install with 'pip install decord' for CPU-only version and refer to"
                "'https://github.com/dmlc/decord' for GPU-supported version"
            )
        try:
            if self._decode_audio:
                self._av_reader = decord.AVReader(
                    uri=file,
                    ctx=decord.cpu(0),
                    sample_rate=sample_rate,
                    mono=mono,
                    width=width,
                    height=height,
                    num_threads=num_threads,
                    fault_tol=fault_tol,
                )
            else:
                self._av_reader = decord.VideoReader(
                    uri=file,
                    ctx=decord.cpu(0),
                    width=width,
                    height=height,
                    num_threads=num_threads,
                    fault_tol=fault_tol,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to open video {video_name} with Decord. {e}")

        if self._decode_audio:
            self._fps = self._av_reader._AVReader__video_reader.get_avg_fps()
        else:
            self._fps = self._av_reader.get_avg_fps()

        self._duration = float(len(self._av_reader)) / float(self._fps)

    @property
    def name(self) -> Optional[str]:
        """
        Returns:
            name: the name of the stored video if set.
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def close(self):
        if self._av_reader is not None:
            del self._av_reader
            self._av_reader = None

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
        if start_sec > end_sec or start_sec > self._duration:
            raise RuntimeError(
                f"Incorrect time window for Decord decoding for video: {self._video_name}."
            )

        start_idx = math.ceil(self._fps * start_sec)
        end_idx = math.ceil(self._fps * end_sec)
        end_idx = min(end_idx, len(self._av_reader))
        frame_idxs = list(range(start_idx, end_idx))
        audio = None

        try:
            outputs = self._av_reader.get_batch(frame_idxs)
        except Exception as e:
            logger.debug(f"Failed to decode video with Decord: {self._video_name}. {e}")
            raise e

        if self._decode_audio:
            audio, video = outputs
            if audio is not None:
                audio = list(audio)
                audio = torch.cat(audio, dim=1)
                audio = torch.flatten(audio)
                audio = audio.to(torch.float32)
        else:
            video = outputs

        if video is not None:
            video = video.to(torch.float32)
            video = thwc_to_cthw(video)

        return {
            "video": video,
            "audio": audio,
        }
