# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
from fractions import Fraction
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import av
import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from .utils import pts_to_secs, secs_to_pts, thwc_to_cthw


logger = logging.getLogger(__name__)


class EncodedVideoPyAV(EncodedVideo):
    """
    EncodedVideoPyAV is an abstraction for accessing clips from an encoded video using
    PyAV as the decoding backend. It supports selective decoding when header information
    is available.
    """

    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_video: bool = True,
        decode_audio: bool = True,
        perform_seek: bool = True,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
            perform_seek:
                Whether or not to seek time to the underlying video container.

                NOTE: seeks may be slow on larger files, e.g. on a networked filesystem
        """
        self.perform_seek = perform_seek
        self._video_name = video_name
        self._decode_video = decode_video
        self._decode_audio = decode_audio

        try:
            self._container = av.open(file)
        except Exception as e:
            raise RuntimeError(f"Failed to open video {video_name}. {e}")

        if self._container is None or len(self._container.streams.video) == 0:
            raise RuntimeError(f"Video stream not found {video_name}")

        # Retrieve video header information if available.
        video_stream = self._container.streams.video[0]
        self._video_time_base = video_stream.time_base
        self._video_start_pts = video_stream.start_time
        if self._video_start_pts is None:
            self._video_start_pts = 0.0

        video_duration = video_stream.duration

        # Retrieve audio header information if available.
        audio_duration = None
        self._has_audio = None
        if self._decode_audio:
            self._has_audio = self._container.streams.audio
            if self._has_audio:
                self._audio_time_base = self._container.streams.audio[0].time_base
                self._audio_start_pts = self._container.streams.audio[0].start_time
                if self._audio_start_pts is None:
                    self._audio_start_pts = 0.0

                audio_duration = self._container.streams.audio[0].duration

        # If duration isn't found in header the whole video is decoded to
        # determine the duration.
        self._video, self._audio, self._selective_decoding = (None, None, True)
        if audio_duration is None and video_duration is None:
            self._selective_decoding = False
            self._video, self._audio = self._pyav_decode_video()
            if self._video is None:
                raise RuntimeError("Unable to decode video stream")

            video_duration = self._video[-1][1]
            if self._audio is not None:
                audio_duration = self._audio[-1][1]

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
    def rate(self) -> Union[str, Fraction]:
        """
        Returns:
            rate: the frame rate of the video
        """
        return self._container.streams.video[0].rate

    @property
    def bit_rate(self) -> int:
        """
        Returns:
            bit_rate: the bit rate of the underlying video
        """
        return self._container.streams.video[0].bit_rate

    @property
    def pix_fmt(self) -> int:
        """
        Returns:
            pix_fmt: the pixel format of the underlying video
        """
        return self._container.streams.video[0].pix_fmt

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
        if self._selective_decoding:
            self._video, self._audio = self._pyav_decode_video(start_sec, end_sec)

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
        if self._has_audio and self._audio is not None:
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
            logger.debug(
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

    def close(self):
        """
        Closes the internal video container.
        """
        if self._container is not None:
            self._container.close()

    def _pyav_decode_video(
        self, start_secs: float = 0.0, end_secs: float = math.inf
    ) -> float:
        """
        Selectively decodes a video between start_pts and end_pts in time units of the
        self._video's timebase.
        """
        video_and_pts = None
        audio_and_pts = None
        try:
            if self._decode_video:
                pyav_video_frames, _ = _pyav_decode_stream(
                    self._container,
                    secs_to_pts(
                        start_secs,
                        self._video_time_base,
                        self._video_start_pts,
                        round_mode="ceil",
                    ),
                    secs_to_pts(
                        end_secs,
                        self._video_time_base,
                        self._video_start_pts,
                        round_mode="ceil",
                    ),
                    self._container.streams.video[0],
                    {"video": 0},
                    perform_seek=self.perform_seek,
                )
                if len(pyav_video_frames) > 0:
                    video_and_pts = [
                        (torch.from_numpy(frame.to_rgb().to_ndarray()), frame.pts)
                        for frame in pyav_video_frames
                    ]

            if self._has_audio:
                pyav_audio_frames, _ = _pyav_decode_stream(
                    self._container,
                    secs_to_pts(
                        start_secs,
                        self._audio_time_base,
                        self._audio_start_pts,
                        round_mode="ceil",
                    ),
                    secs_to_pts(
                        end_secs,
                        self._audio_time_base,
                        self._audio_start_pts,
                        round_mode="ceil",
                    ),
                    self._container.streams.audio[0],
                    {"audio": 0},
                    perform_seek=self.perform_seek,
                )

                if len(pyav_audio_frames) > 0:
                    audio_and_pts = [
                        (
                            torch.from_numpy(np.mean(frame.to_ndarray(), axis=0)),
                            frame.pts,
                        )
                        for frame in pyav_audio_frames
                    ]

        except Exception as e:
            logger.debug(f"Failed to decode video: {self._video_name}. {e}")

        return video_and_pts, audio_and_pts


def _pyav_decode_stream(
    container: av.container.input.InputContainer,
    start_pts: int,
    end_pts: int,
    stream: av.video.stream.VideoStream,
    stream_name: dict,
    buffer_size: int = 0,
    perform_seek: bool = True,
) -> Tuple[List, float]:
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
    Returns:
        result (list): list of decoded frames.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """

    # Seeking in the stream is imprecise. Thus, seek to an earlier pts by a
    # margin pts.
    margin = 1024

    # NOTE:
    # Don't want to seek if iterating through a video due to slow-downs. I
    # believe this is some PyAV bug where seeking after a certain point causes
    # major slow-downs
    if perform_seek:
        seek_offset = max(start_pts - margin, 0)
        container.seek(int(seek_offset), any_frame=False, backward=True, stream=stream)
    frames = {}
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts >= start_pts and frame.pts < end_pts:
            frames[frame.pts] = frame
        elif frame.pts >= end_pts:
            break

    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts
