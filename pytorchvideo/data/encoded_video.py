# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import pathlib
from typing import List, Tuple

import av
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
            self._video = av.open(self._file_path)
        except Exception as e:
            logger.warning(f"Failed to open path {self._file_path}. {e}")
            raise e

        self._time_base = self._video.streams.video[0].time_base
        self._start_pts = self._video.streams.video[0].start_time
        self._duration_pts = self._video.streams.video[0].duration
        if self._start_pts is None:
            self._start_pts = 0.0

        # If duration isn't found in video header the whole video is decoded to
        # determine the duration.
        self._frames = None
        self._selective_decoding = True
        if self._duration_pts is None:
            self._frames, self._duration_pts = self._pyav_decode_video()
            self._selective_decoding = False

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
        return (self._duration_pts - self._start_pts) * float(self._time_base)

    def get_clip(self, start_sec: float, end_sec: float) -> torch.Tensor:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_frames: A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.uint8 and
                in the range [0 - 255]. Returns None if no frames are found.
        """
        start_pts = self._seconds_to_video_pts(start_sec)
        end_pts = self._seconds_to_video_pts(end_sec)
        if self._selective_decoding:
            self._frames, _ = self._pyav_decode_video(start_pts, end_pts)

        if self._frames is None:
            logger.warning(
                f"No frames found within {start_sec} and {end_sec}. Video starts "
                f"at time 0 and ends at {self.duration}."
            )
            return None

        clip_frames = [
            f for f, pts in self._frames if pts >= start_pts and pts <= end_pts
        ]
        if len(clip_frames) == 0:
            return None

        return thwc_to_cthw(torch.stack(clip_frames))

    def close(self):
        """
        Closes the internal video container.
        """
        if self._video is not None:
            self._video.close()

    def _seconds_to_video_pts(self, time_in_seconds: float) -> float:
        """
        Converts a time in seconds to the video's time base and offset relative to
        the video's start_pts.

        Returns:
            video_pts (float): The time in the video's time base.
        """
        time_base = float(self._time_base)
        return int(time_in_seconds / time_base) + self._start_pts

    def _pyav_decode_video(
        self, start_pts: float = 0.0, end_pts: float = math.inf
    ) -> float:
        """
        Selectively decodes a video between start_pts and end_pts in time units of the
        self._video's timebase.
        """
        frames_and_pts = None
        duration = None
        try:
            pyav_frames, duration = _pyav_decode_stream(
                self._video,
                start_pts,
                end_pts,
                self._video.streams.video[0],
                {"video": 0},
            )
            if len(pyav_frames) > 0:
                frames_and_pts = [
                    (torch.from_numpy(frame.to_rgb().to_ndarray()), frame.pts)
                    for frame in pyav_frames
                ]

        except Exception as e:
            logger.warning(f"Failed to decode video at path {self._file_path}. {e}")
            raise e

        return frames_and_pts, duration


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
