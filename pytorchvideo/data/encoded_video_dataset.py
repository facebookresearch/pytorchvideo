# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, List, Optional, Tuple, Type

import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo

from .labeled_video_paths import LabeledVideoPaths
from .utils import MultiProcessSampler


logger = logging.getLogger(__name__)


class EncodedVideoDataset(torch.utils.data.IterableDataset):
    """
    EncodedVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as an encoded video (e.g. mp4, avi).
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        """
        Args:
            labeled_video_paths List[Tuple[str, Optional[dict]]]]) : List containing
                    video file paths and associated labels

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. The clip output is a dictionary with the
                following format:
                    {
                        'video': <video_tensor>,
                        'label': <index_label>,
                        'index': <clip_index>
                    }
                If transform is None, the raw clip output in the above format is
                returned unmodified.
        """
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._video_sampler = video_sampler(self._labeled_videos)
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()
        self._num_consecutive_failures = 0

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._next_clip_start_time = 0.0

    @property
    def video_sampler(self):
        return self._video_sampler

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A video clip with the following format if transform is None:
                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'index': <clip_index>
                }
            Otherwise, the transform defines the clip output.
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        # Called when failed to decode video or retrieve clip.
        def retry_next():
            self._num_consecutive_failures += 1
            if self._num_consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
                )

            return self.__next__()

        # Reuse previously stored video if there are still clips to be sampled from
        # the last loaded video.
        if self._loaded_video_label:
            video, info_dict = self._loaded_video_label
        else:
            video_index = next(self._video_sampler_iter)
            try:
                video_path, info_dict = self._labeled_videos[video_index]
                video = EncodedVideo(video_path)
                self._loaded_video_label = (video, info_dict)
            except OSError as e:
                logger.warning(e)
                retry_next()

        clip_start, clip_end, is_last_clip = self._clip_sampler(
            self._next_clip_start_time, video.duration
        )
        clip_data = video.get_clip(clip_start, clip_end)
        frames = clip_data["video"]
        audio_samples = clip_data["audio"]
        self._next_clip_start_time = clip_end

        if is_last_clip or frames is None:
            # Close the loaded encoded video and reset the last sampled clip time ready
            # to sample a new video on the next iteration.
            self._loaded_video_label[0].close()
            self._loaded_video_label = None
            self._next_clip_start_time = 0.0

            if frames is None:
                retry_next()

        sample_dict = {
            "video": frames,
            "audio": audio_samples,
            "video_name": video.name,
            **info_dict,
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        self._num_consecutive_failures = 0
        return sample_dict

    def __iter__(self):
        return self


def labeled_encoded_video_dataset(
    data_path: pathlib.path,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[dict], Any]] = None,
    video_path_prefix: str = "",
) -> EncodedVideoDataset:
    """
    A helper function to create EncodedVideoDataset object for Ucf101 and Kinectis datasets.

    Args:
        data_path (pathlib.Path): Path to the data. The path type defines how the
        data should be read:
            - For a file path, the file is read and each line is parsed into a
                video path and label.
            - For a directory, the directory structure defines the classes
                (i.e. each subdirectory is a class).
        See the LabeledVideoPaths class documentation for specific formatting
        details and examples.

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. The clip output is a dictionary with the
                following format:
                    {
                        'video': <video_tensor>,
                        'label': <index_label>,
                        'index': <clip_index>
                    }
                If transform is None, the raw clip output in the above format is
                returned unmodified.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in EncodedVideoDataset. All the video paths before loading
                are prefixed with this path.

    """
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = EncodedVideoDataset(
        labeled_video_paths, clip_sampler, video_sampler, transform
    )
    return dataset
