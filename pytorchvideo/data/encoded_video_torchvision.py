# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from fractions import Fraction
from typing import BinaryIO, Dict, Optional

import numpy as np
import torch

from .utils import pts_to_secs, secs_to_pts, thwc_to_cthw
from .video import Video


logger = logging.getLogger(__name__)


class EncodedVideoTorchVision(Video):
    """

    Accessing clips from an encoded video using Torchvision video reading API
    (torch.ops.video_reader.read_video_from_memory) as the decoding backend.
    """

    """
    av_seek_frame is imprecise so seek to a timestamp earlier by a margin
    The unit of margin is second
    """
    SEEK_FRAME_MARGIN = 0.25

    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_video: bool = True,
        decode_audio: bool = True,
    ) -> None:
        """
        Initialize an EncodedVideoTorchVision object.

        Args:
            file (BinaryIO): A file-like object containing the encoded video.
            video_name (str, optional): An optional name assigned to the video.
            decode_video (bool): Whether to decode the video.
            decode_audio (bool): Whether to decode the audio.
        """
        if not decode_video:
            raise NotImplementedError()

        self._video_tensor = torch.tensor(
            np.frombuffer(file.getvalue(), dtype=np.uint8)
        )
        self._video_name = video_name
        self._decode_audio = decode_audio

        (
            self._video,
            self._video_time_base,
            self._video_start_pts,
            video_duration,
            self._audio,
            self._audio_time_base,
            self._audio_start_pts,
            audio_duration,
        ) = self._torch_vision_decode_video()

        # Take the largest duration of either video or duration stream.
        if audio_duration is not None and video_duration is not None:
            self._duration = max(
                pts_to_secs(
                    video_duration, self._video_time_base, self._video_start_pts
                ),
                pts_to_secs(
                    audio_duration, self._audio_time_base, self._audio_start_pts
                ),
            )
        elif video_duration is not None:
            self._duration = pts_to_secs(
                video_duration, self._video_time_base, self._video_start_pts
            )

        elif audio_duration is not None:
            self._duration = pts_to_secs(
                audio_duration, self._audio_time_base, self._audio_start_pts
            )

    @property
    def name(self) -> Optional[str]:
        """
        Get the name of the stored video.

        Returns:
            str or None: The video's name if set, otherwise None.
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        Get the video's duration in seconds.

        Returns:
            float: The video's duration in seconds.
        """
        return self._duration

    def close(self):
        """
        Close the video (not implemented).
        """
        pass

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds). Returned frames will be in
        [start_sec, end_sec). Note that 1) if you want to avoid float precision issue
        and need accurate frames, please use Fraction for start_sec and end_sec.
        2) As end_sec is exclusive, so you may need to use
        `get_clip(start_sec, duration + EPS)` to get the last frame.

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
        video_frames = None
        if self._video is not None:
            video_start_pts = secs_to_pts(
                start_sec,
                self._video_time_base,
                self._video_start_pts,
                round_mode="ceil",
            )
            video_end_pts = secs_to_pts(
                end_sec,
                self._video_time_base,
                self._video_start_pts,
                round_mode="ceil",
            )
            video_frames = [
                f
                for f, pts in self._video
                if pts >= video_start_pts and pts < video_end_pts
            ]

        audio_samples = None
        if self._decode_audio and self._audio:
            audio_start_pts = secs_to_pts(
                start_sec,
                self._audio_time_base,
                self._audio_start_pts,
                round_mode="ceil",
            )
            audio_end_pts = secs_to_pts(
                end_sec,
                self._audio_time_base,
                self._audio_start_pts,
                round_mode="ceil",
            )
            audio_samples = [
                f
                for f, pts in self._audio
                if pts >= audio_start_pts and pts < audio_end_pts
            ]
            audio_samples = torch.cat(audio_samples, axis=0)
            audio_samples = audio_samples.to(torch.float32)

        if video_frames is None or len(video_frames) == 0:
            logger.warning(
                f"No video found within {start_sec} and {end_sec} seconds. "
                f"Video starts at time 0 and ends at {self.duration}."
            )

            video_frames = None

        if video_frames is not None:
            video_frames = thwc_to_cthw(torch.stack(video_frames)).to(torch.float32)

        return {
            "video": video_frames,
            "audio": audio_samples,
        }

    def _torch_vision_decode_video(
        self, start_pts: int = 0, end_pts: int = -1
    ) -> float:
        """
        Decode the video in the specified PTS range.

        Args:
            start_pts (int): The starting Presentation TimeStamp (PTS) to decode.
            end_pts (int): The ending Presentation TimeStamp (PTS) to decode.

        Returns:
            tuple: A tuple containing video and audio data as well as other information.
        """
        video_and_pts = None
        audio_and_pts = None

        width, height, min_dimension, max_dimension = 0, 0, 0, 0
        video_start_pts, video_end_pts = start_pts, end_pts
        video_timebase_num, video_timebase_den = 0, 1

        samples, channels = 0, 0
        audio_start_pts, audio_end_pts = start_pts, end_pts
        audio_timebase_num, audio_timebase_den = 0, 1

        try:
            tv_result = torch.ops.video_reader.read_video_from_memory(
                self._video_tensor,
                self.SEEK_FRAME_MARGIN,
                # Set getPtsOnly=0, i.e., read full video rather than just header
                0,
                # Read video stream
                1,
                width,
                height,
                min_dimension,
                max_dimension,
                video_start_pts,
                video_end_pts,
                video_timebase_num,
                video_timebase_den,
                # Read audio stream
                self._decode_audio,
                samples,
                channels,
                audio_start_pts,
                audio_end_pts,
                audio_timebase_num,
                audio_timebase_den,
            )
        except Exception as e:
            logger.warning(f"Failed to decode video of name {self._video_name}. {e}")
            raise e

        (
            vframes,
            vframes_pts,
            vtimebase,
            _,
            vduration,
            aframes,
            aframe_pts,
            atimebase,
            _,
            aduration,
        ) = tv_result

        if vduration < 0:
            # No header information to infer video duration
            video_duration = int(vframes_pts[-1])
        else:
            video_duration = int(vduration)

        video_and_pts = list(zip(vframes, vframes_pts))
        video_start_pts = int(vframes_pts[0])
        video_time_base = Fraction(int(vtimebase[0]), int(vtimebase[1]))

        audio_and_pts = None
        audio_time_base = None
        audio_start_pts = None
        audio_duration = None
        if self._decode_audio:
            if aduration < 0:
                # No header information to infer audio duration
                audio_duration = int(aframe_pts[-1])
            else:
                audio_duration = int(aduration)

            audio_and_pts = list(zip(aframes, aframe_pts))
            audio_start_pts = int(aframe_pts[0])
            audio_time_base = Fraction(int(atimebase[0]), int(atimebase[1]))

        return (
            video_and_pts,
            video_time_base,
            video_start_pts,
            video_duration,
            audio_and_pts,
            audio_time_base,
            audio_start_pts,
            audio_duration,
        )
