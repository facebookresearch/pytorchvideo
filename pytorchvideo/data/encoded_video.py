#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
from typing import List, Tuple

import av
import numpy as np
import torch


logger = logging.getLogger(__name__)


class EncodedVideo:
    """
    EncodedVideo is an abstraction for accessing clips from an encoded video.
    It supports selective decoding when header information is available. PyAV is
    used as the decoding backend.
    """

    def __init__(self, file_path: str) -> None:
        """
        If header information isn't available the whole video will be decoded to
        retrieve metadata information.
        Args:
            file_path (str): a file a or file-like object (e.g. io.BytesIO or
                io.StringIO) that contains the encoded video.
        """
        self._file_path = file_path
        self._video_decoded = False

        try:
            self._video = av.open(self._file_path)
        except Exception as e:
            logger.warning(f"Failed to open path {self._file_path}\n{e}")
            raise e

        # Try to fetch the decoding information from the video's header. Some
        # videos do not include the decoding information, for that case we decode the
        # whole video.
        self._time_base = self._video.streams.video[0].time_base
        self._start_pts = self._video.streams.video[0].start_time
        duration = self._video.streams.video[0].duration
        no_header_info = (
            duration is None or self._time_base is None or self._start_pts is None
        )
        if no_header_info:
            duration = self._pyav_decode_video()
            self._time_base = 1
            self._start_pts = 0
            self._video_decoded = True

        self._end_pts = self._start_pts + duration

    def _pyav_decode_video(
        self, start_pts: float = 0.0, end_pts: float = math.inf
    ) -> float:
        try:
            pyav_frames, duration = _pyav_decode_stream(
                self._video,
                start_pts,
                end_pts,
                self._video.streams.video[0],
                {"video": 0},
            )
            self._decoded_pts = [frame.pts for frame in pyav_frames]
            frames = [frame.to_rgb().to_ndarray() for frame in pyav_frames]
            if len(frames) > 0:
                self._decoded_frames = torch.as_tensor(np.stack(frames))

        except Exception as e:
            logger.warning(f"Failed to decode video at path {self._file_path}\n{e}")
            raise e

        return duration

    @property
    def start_pts(self) -> float:
        """
        Returns:
            start_pts: the video's beginning presentation timestamp in the video's
                timebase.
        """
        return self._start_pts

    @property
    def end_pts(self) -> float:
        """
        Returns:
            end_pts: the video's end presentation timestamp in the video's timebase.
        """
        return self._end_pts

    def seconds_to_video_pts(self, time_in_seconds: float) -> float:
        """
        Converts a time in seconds to the video's time base and offset relative to
        the video's start_pts.

        Returns:
            video_pts (float): The time in the video's time base.
        """
        time_base = float(self._time_base)
        return int(time_in_seconds / time_base) + self._start_pts

    def get_clip(self, start_pts: float, end_pts: float) -> torch.Tensor:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in the videos presentation time base. Uses selective decoding between the time
        points if header information was found in the video.

        Args:
            start_pts (float): the start time in the video's time base
            end_pts (float): the end time in the video's time base
        Returns:
            clip_frames: A tensor of the clip's RGB frames with shape:
                (time, height, width, channels). The frames are of type torch.uint8 and
                in the range [0 - 255]. Returns an empty tensor if no frames found.
        """
        if not self._video_decoded:
            self._pyav_decode_video(start_pts, end_pts)

        clip_frames = []
        for frame, pts in zip(self._decoded_frames, self._decoded_pts):
            if pts >= start_pts and pts <= end_pts:
                clip_frames.append(frame)
            else:
                break

        if len(clip_frames) == 0:
            logger.warning(
                f"No frames found within {start_pts} and {end_pts}. Video starts"
                "at time {self._start_pts} and ends at {self._end_pts}."
            )
            return torch.tensor(0)

        return torch.stack(clip_frames)

    def close(self):
        """
        Closes the internal video container.
        """
        if self._video is not None:
            self._video.close()


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
