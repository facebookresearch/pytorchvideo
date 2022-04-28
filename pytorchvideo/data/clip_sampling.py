# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union


class ClipInfo(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_start_sec  (Union[float, Fraction]): clip start time.
        clip_end_sec (Union[float, Fraction]): clip end time.
        clip_index (int): clip index in the video.
        aug_index (int): augmentation index for the clip. Different augmentation methods
            might generate multiple views for the same clip.
        is_last_clip (bool): a bool specifying whether there are more clips to be
            sampled from the video.
    """

    clip_start_sec: Union[float, Fraction]
    clip_end_sec: Union[float, Fraction]
    clip_index: int
    aug_index: int
    is_last_clip: bool


class ClipInfoList(NamedTuple):
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

    clip_start_sec: List[float]
    clip_end_sec: List[float]
    clip_index: List[float]
    aug_index: List[float]
    is_last_clip: List[float]


class ClipSampler(ABC):
    """
    Interface for clip samplers that take a video time, previous sampled clip time,
    and returns a named-tuple ``ClipInfo``.
    """

    def __init__(self, clip_duration: Union[float, Fraction]) -> None:
        self._clip_duration = Fraction(clip_duration)
        self._current_clip_index = 0
        self._current_aug_index = 0

    @abstractmethod
    def __call__(
        self,
        last_clip_end_time: Union[float, Fraction],
        video_duration: Union[float, Fraction],
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        pass

    def reset(self) -> None:
        """Resets any video-specific attributes in preperation for next video"""
        pass


def make_clip_sampler(sampling_type: str, *args) -> ClipSampler:
    """
    Constructs the clip samplers found in ``pytorchvideo.data.clip_sampling`` from the
    given arguments.

    Args:
        sampling_type (str): choose clip sampler to return. It has three options:

            * uniform: constructs and return ``UniformClipSampler``
            * random: construct and return ``RandomClipSampler``
            * constant_clips_per_video: construct and return ``ConstantClipsPerVideoSampler``

        *args: the args to pass to the chosen clip sampler constructor.
    """
    if sampling_type == "uniform":
        return UniformClipSampler(*args)
    elif sampling_type == "random":
        return RandomClipSampler(*args)
    elif sampling_type == "constant_clips_per_video":
        return ConstantClipsPerVideoSampler(*args)
    elif sampling_type == "random_multi":
        return RandomMultiClipSampler(*args)
    else:
        raise NotImplementedError(f"{sampling_type} not supported")


class UniformClipSampler(ClipSampler):
    """
    Evenly splits the video into clips of size clip_duration.
    """

    def __init__(
        self,
        clip_duration: Union[float, Fraction],
        stride: Optional[Union[float, Fraction]] = None,
        backpad_last: bool = False,
        eps: float = 1e-6,
    ):
        """
        Args:
            clip_duration (Union[float, Fraction]):
                The length of the clip to sample (in seconds).
            stride (Union[float, Fraction], optional):
                The amount of seconds to offset the next clip by
                default value of None is equivalent to no stride => stride == clip_duration.
            eps (float):
                Epsilon for floating point comparisons. Used to check the last clip.
            backpad_last (bool):
                Whether to include the last frame(s) by "back padding".

                For instance, if we have a video of 39 frames (30 fps = 1.3s)
                with a stride of 16 (0.533s) with a clip duration of 32 frames
                (1.0667s). The clips will be (in frame numbers):

                with backpad_last = False
                - [0, 31]

                with backpad_last = True
                - [0, 31]
                - [8, 39], this is "back-padded" from [16, 48] to fit the last window
        Note that you can use Fraction for clip_duration and stride if you want to
        avoid float precision issue and need accurate frames in each clip.
        """
        super().__init__(clip_duration)
        self._stride = stride if stride is not None else self._clip_duration
        self._eps = eps
        self._backpad_last = backpad_last

        assert self._stride > 0, "stride must be positive"

    def _clip_start_end(
        self,
        last_clip_end_time: Union[float, Fraction],
        video_duration: Union[float, Fraction],
        backpad_last: bool,
    ) -> Tuple[Fraction, Fraction]:
        """
        Helper to calculate the start/end clip with backpad logic
        """
        delta = self._stride - self._clip_duration
        last_end_time = -delta if last_clip_end_time is None else last_clip_end_time
        clip_start = Fraction(last_end_time + delta)
        clip_end = Fraction(clip_start + self._clip_duration)
        if backpad_last:
            buffer_amount = max(0, clip_end - video_duration)
            clip_start -= buffer_amount
            clip_start = Fraction(max(0, clip_start))  # handle rounding
            clip_end = Fraction(clip_start + self._clip_duration)

        return clip_start, clip_end

    def __call__(
        self,
        last_clip_end_time: Optional[float],
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        """
        Args:
            last_clip_end_time (float): the last clip end time sampled from this video. This
                should be 0.0 if the video hasn't had clips sampled yet.
            video_duration: (float): the duration of the video that's being sampled in seconds
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info: (ClipInfo): includes the clip information (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip), where the times are in
            seconds and is_last_clip is False when there is still more of time in the video
            to be sampled.
        """
        clip_start, clip_end = self._clip_start_end(
            last_clip_end_time, video_duration, backpad_last=self._backpad_last
        )

        # if they both end at the same time - it's the last clip
        _, next_clip_end = self._clip_start_end(
            clip_end, video_duration, backpad_last=self._backpad_last
        )
        if self._backpad_last:
            is_last_clip = abs(next_clip_end - clip_end) < self._eps
        else:
            is_last_clip = (next_clip_end - video_duration) > self._eps

        clip_index = self._current_clip_index
        self._current_clip_index += 1

        if is_last_clip:
            self.reset()

        return ClipInfo(clip_start, clip_end, clip_index, 0, is_last_clip)

    def reset(self):
        self._current_clip_index = 0


class UniformClipSamplerTruncateFromStart(UniformClipSampler):
    """
    Evenly splits the video into clips of size clip_duration.
    If truncation_duration is set, clips sampled from [0, truncation_duration].
    If truncation_duration is not set, defaults to UniformClipSampler.
    """

    def __init__(
        self,
        clip_duration: Union[float, Fraction],
        stride: Optional[Union[float, Fraction]] = None,
        backpad_last: bool = False,
        eps: float = 1e-6,
        truncation_duration: float = None,
    ) -> None:
        super().__init__(clip_duration, stride, backpad_last, eps)
        self.truncation_duration = truncation_duration

    def __call__(
        self,
        last_clip_end_time: float,
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfo:

        truncated_video_duration = video_duration
        if self.truncation_duration is not None:
            truncated_video_duration = min(self.truncation_duration, video_duration)

        return super().__call__(
            last_clip_end_time, truncated_video_duration, annotation
        )


class RandomClipSampler(ClipSampler):
    """
    Randomly samples clip of size clip_duration from the videos.
    """

    def __call__(
        self,
        last_clip_end_time: float,
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        """
        Args:
            last_clip_end_time (float): Not used for RandomClipSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info (ClipInfo): includes the clip information of (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
            clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.

        """
        max_possible_clip_start = max(video_duration - self._clip_duration, 0)
        clip_start_sec = Fraction(random.uniform(0, max_possible_clip_start))
        return ClipInfo(
            clip_start_sec, clip_start_sec + self._clip_duration, 0, 0, True
        )


class RandomMultiClipSampler(RandomClipSampler):
    """
    Randomly samples multiple clips of size clip_duration from the videos.
    """

    def __init__(self, clip_duration: float, num_clips: int) -> None:
        super().__init__(clip_duration)
        self._num_clips = num_clips

    def __call__(
        self,
        last_clip_end_time: Optional[float],
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfoList:

        (
            clip_start_list,
            clip_end_list,
            clip_index_list,
            aug_index_list,
            is_last_clip_list,
        ) = (
            self._num_clips * [None],
            self._num_clips * [None],
            self._num_clips * [None],
            self._num_clips * [None],
            self._num_clips * [None],
        )
        for i in range(self._num_clips):
            (
                clip_start_list[i],
                clip_end_list[i],
                clip_index_list[i],
                aug_index_list[i],
                is_last_clip_list[i],
            ) = super().__call__(last_clip_end_time, video_duration, annotation)

        return ClipInfoList(
            clip_start_list,
            clip_end_list,
            clip_index_list,
            aug_index_list,
            is_last_clip_list,
        )


class RandomMultiClipSamplerTruncateFromStart(RandomMultiClipSampler):
    """
    Randomly samples multiple clips of size clip_duration from the videos.
    If truncation_duration is set, clips sampled from [0, truncation_duration].
    If truncation_duration is not set, defaults to RandomMultiClipSampler.
    """

    def __init__(
        self, clip_duration: float, num_clips: int, truncation_duration: float = None
    ) -> None:
        super().__init__(clip_duration, num_clips)
        self.truncation_duration = truncation_duration

    def __call__(
        self,
        last_clip_end_time: Optional[float],
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfoList:

        truncated_video_duration = video_duration
        if self.truncation_duration is not None:
            truncated_video_duration = min(self.truncation_duration, video_duration)

        return super().__call__(
            last_clip_end_time, truncated_video_duration, annotation
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

    def __call__(
        self,
        last_clip_end_time: Optional[float],
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        """
        Args:
            last_clip_end_time (float): Not used for ConstantClipsPerVideoSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled.
            annotation (Dict): Not used by this sampler.
        Returns:
            a named-tuple `ClipInfo`: includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                is_last_clip is True after clips_per_video clips have been sampled or the end
                of the video is reached.

        """
        max_possible_clip_start = Fraction(max(video_duration - self._clip_duration, 0))
        uniform_clip = Fraction(
            max_possible_clip_start, max(self._clips_per_video - 1, 1)
        )
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

        if is_last_clip:
            self.reset()

        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self._clip_duration,
            clip_index,
            aug_index,
            is_last_clip,
        )

    def reset(self):
        self._current_clip_index = 0
        self._current_aug_index = 0
