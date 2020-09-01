# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, List, Tuple, Type

import av
import torch.utils.data
from fvcore.common.file_io import PathManager
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.datasets.folder import make_dataset

from .utils import MultiProcessSampler


logger = logging.getLogger(__name__)


class Kinetics(torch.utils.data.IterableDataset):
    """
        Action recognition video dataset for Kinetics-{400,600,700} stored as encoded videos.
        <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>

        This dataset handles the loading, decoding and clip sampling for the videos.
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        data_path: pathlib.Path,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Callable[[dict], Any] = None,
    ) -> None:
        """
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
        """
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = LabeledVideoPaths.from_path(data_path)
        self._video_sampler = video_sampler(self._labeled_videos)
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()
        self._num_consecutive_failures = 0

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._last_clip_end_time = 0.0

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
            video, label = self._loaded_video_label
        else:
            video_index = next(self._video_sampler_iter)
            try:
                video_path, label = self._labeled_videos[video_index]
                video = EncodedVideo(video_path)
                self._loaded_video_label = (video, label)
            except (av.error.ValueError, OSError) as e:
                logger.warning(e)
                retry_next()

        clip_start, clip_end, is_last_clip = self._clip_sampler(
            self._last_clip_end_time, video.duration
        )
        frames = video.get_clip(clip_start, clip_end)
        self._last_clip_end_time = clip_end

        if is_last_clip or frames is None:
            # Close the loaded encoded video and reset the last sampled clip time ready
            # to sample a new video on the next iteration.
            self._loaded_video_label[0].close()
            self._loaded_video_label = None
            self._last_clip_end_time = 0.0

            if frames is None:
                retry_next()

        sample_dict = {"video": frames, "label": label, "video_name": video.name}
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        self._num_consecutive_failures = 0
        return sample_dict

    def __iter__(self):
        return self


class LabeledVideoPaths(torch.utils.data.Dataset):
    """
    LabeledVideoPaths contains pairs of video path and integer index label.
    """

    @classmethod
    def from_path(cls, data_path: pathlib.Path) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object depending on the path
        type.
        - If it is a directory path it uses the LabeledVideoPaths.from_directory function.
        - If it's a file it uses the LabeledVideoPaths.from_csv file.
        Args:
            file_path (pathlib.Path): The path to the file to be read.
        """
        data_path = pathlib.Path(data_path)
        if data_path.is_file():
            return LabeledVideoPaths.from_csv(data_path)
        elif data_path.is_dir():
            return LabeledVideoPaths.from_directory(data_path)
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_csv(cls, file_path: pathlib.Path) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object by reading a file with the
        following format:
            <path> <integer_label>
            ...
            <path> <integer_label>

        Args:
            file_path (pathlib.Path): The path to the file to be read.
        """
        assert PathManager.exists(file_path), f"{file_path} not found."
        video_paths_and_label = []
        with PathManager.open(file_path, "r") as f:
            for path_label in f.read().splitlines():
                file_path, label = path_label.rsplit(None, 1)
                video_paths_and_label.append((file_path, int(label)))

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    @classmethod
    def from_directory(cls, dir_path: pathlib.Path) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object by parsing the structure
        of the given directory's subdirectories into the classification labels. It
        expects the directory format to be the following:
             dir_path/<class_name>/<video_name>.mp4

        Classes are indexed from 0 to the number of classes, alphabetically.

        E.g.
            dir_path/class_x/xxx.ext
            dir_path/class_x/xxy.ext
            dir_path/class_x/xxz.ext
            dir_path/class_y/123.ext
            dir_path/class_y/nsdf3.ext
            dir_path/class_y/asd932_.ext

        Would produce two classes labeled 0 and 1 with 3 videos paths associated with each.

        Args:
            dir_path (pathlib.Path): Root directory to the video class directories .
        """
        assert PathManager.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted((f for f in dir_path.iterdir() if f.is_dir()))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        video_paths_and_label = make_dataset(
            dir_path, class_to_idx, extensions=("mp4",)
        )
        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {dir_path}."
        return cls(video_paths_and_label)

    def __init__(self, sample_paths_and_labels: List[Tuple[str, int]]) -> None:
        """
        Args:
            sample_paths_and_labels [(str, int)]: a list of tuples containing the video
                path and integer label.
        """
        self._sample_paths_and_labels = sample_paths_and_labels

    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        Args:
            index (int): the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        return self._sample_paths_and_labels[index]

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._sample_paths_and_labels)
