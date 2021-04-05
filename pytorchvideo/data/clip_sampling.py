# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random
from abc import ABC, abstractmethod
from typing import NamedTuple


class ClipInfo(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_start_sec  (float): clip start time.
        clip_end_sec (float): clip end time.
        clip_index (int): clip index in the video.
        aug_index (int): augmentation index for the clip. Different augmentation methods
            might generate multiple views for the same clip.
        is_last_clip (bool): a bool specifying whether there are more clips to be
            sampled from the video.
    """

    clip_start_sec: float
    clip_end_sec: float
    clip_index: int
    aug_index: int
    is_last_clip: bool


class ClipSampler(ABC):
    """
    Interface for clip sampler's which take a video time, previous sampled clip time,
    and returns a named-tuple `ClipInfo`.
    """

    def __init__(self, clip_duration: float) -> None:
        self._clip_duration = clip_duration
        self._current_clip_index = 0
        self._current_aug_index = 0

    @abstractmethod
    def __call__(self, last_clip_time: float, video_duration: float) -> ClipInfo:
        pass


def make_clip_sampler(sampling_type: str, *args) -> ClipSampler:
    """
    Constructs the clip samplers found in this module from the given arguments.
    Args:
        sampling_type (str): choose clip sampler to return. It has two options:
            - uniform: constructs and return UniformClipSampler
            - random: construct and return RandomClipSampler
        *args: the args to pass to the chosen clip sampler constructor
    """
    if sampling_type == "uniform":
        return UniformClipSampler(*args)
    elif sampling_type == "random":
        return RandomClipSampler(*args)
    elif sampling_type == "constant_clips_per_video":
        return ConstantClipsPerVideoSampler(*args)
    else:
        raise NotImplementedError(f"{sampling_type} not supported")


class UniformClipSampler(ClipSampler):
    """
    Evenly splits the video into clips of size clip_duration.
    """

    def __init__(self, clip_duration: float) -> None:
        super().__init__(clip_duration)

    def __call__(self, last_clip_time: float, video_duration: float) -> ClipInfo:
        """
        Args:
            last_clip_time (float): the last clip end time sampled from this video. This
                should be 0.0 if the video hasn't had clips sampled yet.
                segments, clip_index is the segment index to sample.
            video_duration: (float): the duration of the video that's being sampled in seconds
        Returns:
            a named-tuple `ClipInfo`: includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip), where the times are in
                seconds and is_last_clip is False when there is still more of time in the video
                to be sampled.

        """
        clip_start_sec = last_clip_time
        clip_end_sec = clip_start_sec + self._clip_duration
        clip_index = self._current_clip_index
        self._current_clip_index += 1
        is_last_clip = (clip_end_sec + self._clip_duration) > video_duration
        return ClipInfo(clip_start_sec, clip_end_sec, clip_index, 0, is_last_clip)


class RandomClipSampler(ClipSampler):
    """
    Randomly samples clip of size clip_duration from the videos.
    """

    def __init__(self, clip_duration: float) -> None:
        super().__init__(clip_duration)

    def __call__(self, last_clip_time: float, video_duration: float) -> ClipInfo:
        """
        Args:
            last_clip_time (float): Not used for RandomClipSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled
        Returns:
            a named-tuple `ClipInfo`: includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.

        """
        max_possible_clip_start = max(video_duration - self._clip_duration, 0)
        clip_start_sec = random.uniform(0, max_possible_clip_start)
        return ClipInfo(
            clip_start_sec, clip_start_sec + self._clip_duration, 0, 0, True
        )


class ConstantClipsPerVideoSampler(ClipSampler):
    """
    Evenly splits the video into clips_per_video increments and samples clips of size
    clip_duration at these increments.
    """

    def __init__(
        self, clip_duration: float, clips_per_video: int, augs_per_clip: int = 1
    ) -> None:
        super().__init__(clip_duration)
        self._clips_per_video = clips_per_video
        self._augs_per_clip = augs_per_clip

    def __call__(self, last_clip_time: float, video_duration: float) -> ClipInfo:
        """
        Args:
            last_clip_time (float): Not used for ConstantClipsPerVideoSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled.
        Returns:
            a named-tuple `ClipInfo`: includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                is_last_clip is True after clips_per_video clips have been sampled or the end
                of the video is reached.

        """
        max_possible_clip_start = max(video_duration - self._clip_duration, 0)
        uniform_clip = max_possible_clip_start / self._clips_per_video
        clip_start_sec = uniform_clip * self._current_clip_index
        clip_index = self._current_clip_index
        aug_index = self._current_aug_index

        self._current_aug_index += 1
        if self._current_aug_index >= self._augs_per_clip:
            self._current_clip_index += 1
            self._current_aug_index = 0

        # Last clip is True if sampled self._clips_per_video or if end of video is reached.
        is_last_clip = False
        if (
            self._current_clip_index >= self._clips_per_video
            or uniform_clip * self._current_clip_index > max_possible_clip_start
        ):
            self._current_clip_index = 0
            is_last_clip = True

        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self._clip_duration,
            clip_index,
            aug_index,
            is_last_clip,
        )
