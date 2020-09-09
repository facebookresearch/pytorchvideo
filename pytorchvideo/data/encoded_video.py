# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import pathlib
from typing import List, Optional, Tuple

import av
import numpy as np
import torch

from .utils import thwc_to_cthw


logger = logging.getLogger(__name__)


class EncodedVideo:
    """
    EncodedVideo is an abstraction for accessing clips from an encoded video using
    selective decoding. It supports selective decoding when header information is
    available PyAV is used as the decoding backend.
    """

    def __init__(self, file_path: str) -> None:
        """
        Args:
            file_path (str): a file a or file-like object (e.g. io.BytesIO or
                io.StringIO) that contains the encoded video.
        """
        self._file_path = file_path

        try:
            self._container = av.open(self._file_path)
        except Exception as e:
            logger.warning(f"Failed to open path {self._file_path}. {e}")
            raise e

        # Retrieve video header information if available.
        self._video_time_base = self._container.streams.video[0].time_base
        self._video_start_pts = self._container.streams.video[0].start_time
        if self._video_start_pts is None:
            self._video_start_pts = 0.0

        video_duration = self._container.streams.video[0].duration

        # Retrieve audio header information if available.
        self._has_audio = self._container.streams.audio
        audio_duration = None
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
            if self._video is not None:
                video_duration = self._video[-1][1]

            if self._audio is not None:
                audio_duration = self._audio[-1][1]

        # Take the largest duration of either video or duration stream.
        if audio_duration is not None and video_duration is not None:
            self._duration = max(
                self._pts_to_secs(
                    video_duration, self._video_time_base, self._video_start_pts
                ),
                self._pts_to_secs(
                    audio_duration, self._audio_time_base, self._audio_start_pts
                ),
            )
        elif video_duration is not None:
            self._duration = self._pts_to_secs(
                video_duration, self._video_time_base, self._video_start_pts
            )

        elif audio_duration is not None:
            self._duration = self._pts_to_secs(
                audio_duration, self._audio_time_base, self._audio_start_pts
            )

    @property
    def name(self) -> str:
        """
        Returns:
            name: the name of the stored video extracted from the video path.
        """
        return pathlib.Path(self._file_path).name

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_video: A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.uint8 and
                in the range [0 - 255].
            clip_audio: A tensor of the clip's audio samples with shape:
                (samples). The samples are of type torch.uint8 and
                in the range [0 - 255].
            Returns None if no video or audio found within time range.
        """
        if self._selective_decoding:
            self._video, self._audio = self._pyav_decode_video(start_sec, end_sec)

        if self._video is None and self._audio is None:
            logger.warning(
                f"No video or audio found within {start_sec} and {end_sec} seconds. "
                "Video starts at time 0 and ends at {self.duration}."
            )
            return None

        video_start_pts = self._secs_to_pts(
            start_sec, self._video_time_base, self._video_start_pts
        )
        video_end_pts = self._secs_to_pts(
            end_sec, self._video_time_base, self._video_start_pts
        )
        video_frames = [
            f
            for f, pts in self._video
            if pts >= video_start_pts and pts <= video_end_pts
        ]

        audio_samples = None
        if self._has_audio:
            audio_start_pts = self._secs_to_pts(
                start_sec, self._audio_time_base, self._audio_start_pts
            )
            audio_end_pts = self._secs_to_pts(
                end_sec, self._audio_time_base, self._audio_start_pts
            )
            audio_samples = [
                f
                for f, pts in self._audio
                if pts >= audio_start_pts and pts <= audio_end_pts
            ]
            audio_samples = torch.cat(audio_samples, axis=0)

        if len(video_frames) == 0 and len(audio_samples) == 0:
            return None

        return thwc_to_cthw(torch.stack(video_frames)), audio_samples

    def close(self):
        """
        Closes the internal video container.
        """
        if self._container is not None:
            self._container.close()

    def _secs_to_pts(
        self, time_in_seconds: float, time_base: float, start_pts: float
    ) -> float:
        """
        Converts a time (in seconds) to the given time base and start_pts offset
        presentation time.

        Returns:
            pts (float): The time in the given time base.
        """
        if time_in_seconds == math.inf:
            return math.inf

        time_base = float(time_base)
        return int(time_in_seconds / time_base) + start_pts

    def _pts_to_secs(
        self, time_in_seconds: float, time_base: float, start_pts: float
    ) -> float:
        """
        Converts a present time with the given time base and start_pts offset to seconds.

        Returns:
            time_in_seconds (float): The corresponding time in seconds.
        """
        if time_in_seconds == math.inf:
            return math.inf

        return (time_in_seconds - start_pts) * float(time_base)

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
            pyav_video_frames, _ = _pyav_decode_stream(
                self._container,
                self._secs_to_pts(
                    start_secs, self._video_time_base, self._video_start_pts
                ),
                self._secs_to_pts(
                    end_secs, self._video_time_base, self._video_start_pts
                ),
                self._container.streams.video[0],
                {"video": 0},
            )
            if len(pyav_video_frames) > 0:
                video_and_pts = [
                    (torch.from_numpy(frame.to_rgb().to_ndarray()), frame.pts)
                    for frame in pyav_video_frames
                ]

            if self._has_audio:
                pyav_audio_frames, _ = _pyav_decode_stream(
                    self._container,
                    self._secs_to_pts(
                        start_secs, self._audio_time_base, self._audio_start_pts
                    ),
                    self._secs_to_pts(
                        end_secs, self._audio_time_base, self._audio_start_pts
                    ),
                    self._container.streams.audio[0],
                    {"audio": 0},
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
            logger.warning(f"Failed to decode video at path {self._file_path}. {e}")
            raise e

        return video_and_pts, audio_and_pts


def _pyav_decode_stream(
    container: av.container.input.InputContainer,
    start_pts: float,
    end_pts: float,
    stream: av.video.stream.VideoStream,
    stream_name: dict,
    buffer_size: int = 0,
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
    seek_offset = max(start_pts - margin, 0)
    container.seek(int(seek_offset), any_frame=False, backward=True, stream=stream)
    frames = {}
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts >= start_pts and frame.pts <= end_pts:
            frames[frame.pts] = frame
        elif frame.pts > end_pts:
            break

    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts
