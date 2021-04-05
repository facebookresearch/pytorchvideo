# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import multiprocessing
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
        decode_audio: bool = True,
        decoder: str = "pyav",
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
                        'video': <video_tensor>
                        'label': <index_label>
                        'video_index': <video_index>
                        'clip_index': <clip_index>
                        'aug_index': <aug_index>, augmentation index as augmentations
                            might generate multiple views for one clip.
                    }
                If transform is None, the raw clip output in the above format is
                returned unmodified.

            decoder (str): Defines what type of decoder used to decode a video.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
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
                    'video_index': <video_index>
                    'clip_index': <clip_index>
                    'aug_index': <aug_index>, augmentation index as augmentations
                        might generate multiple views for one clip.
                }
            Otherwise, the transform defines the clip output.
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = EncodedVideo.from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except (RuntimeError, OSError) as e:
                    logger.warning(
                        "Failed to load video from {} with error {}; trial {}".format(
                            video_path,
                            e,
                            i_try,
                        )
                    )
                    continue

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(self._next_clip_start_time, video.duration)
            # Only load the clip once and reuse previously stored clip if there are multiple
            # views for augmentations to perform on the same clip.
            if aug_index == 0:
                self._loaded_clip = video.get_clip(clip_start, clip_end)
            self._next_clip_start_time = clip_end

            clip_is_null = (
                self._loaded_clip is None
                or self._loaded_clip["video"] is None
                or (self._loaded_clip["audio"] is None and self._decode_audio)
            )
            if is_last_clip or clip_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._next_clip_start_time = 0.0

                if clip_is_null:
                    logger.warning(
                        "Failed to meta load video {}; trial {}".format(
                            video.name, i_try
                        )
                    )
                    continue

            frames = self._loaded_clip["video"]
            audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


def labeled_encoded_video_dataset(
    data_path: pathlib.path,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[dict], Any]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
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
                        'video_index': <video_index>
                        'clip_index': <clip_index>
                        'aug_index': <aug_index>, augmentation index as augmentations
                            might generate multiple views for one clip.
                    }
                If transform is None, the raw clip output in the above format is
                returned unmodified.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in EncodedVideoDataset. All the video paths before loading
                are prefixed with this path.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    # PathManager may configure the multiprocessing context in a way that conflicts
    # with PyTorch DataLoader workers. To avoid this, we make sure the PathManager
    # calls (made by LabeledVideoPaths) are wrapped in their own sandboxed process.
    try:
        with multiprocessing.Pool(processes=1) as pool:
            res = pool.apply_async(LabeledVideoPaths.from_path, (data_path,))
            labeled_video_paths = res.get(timeout=100)
    except multiprocessing.TimeoutError:
        labeled_video_paths = LabeledVideoPaths.from_path(data_path)

    labeled_video_paths.path_prefix = video_path_prefix
    dataset = EncodedVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset
