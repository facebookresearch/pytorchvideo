#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import pathlib
import random
from typing import List, Tuple

import torch.utils.data
from fvcore.common.file_io import PathManager
from torchvision.datasets.folder import make_dataset

from .encoded_video import EncodedVideo


class LabeledVideoPaths:
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


class Kinetics(torch.utils.data.Dataset):
    """
        Action recognition video dataset for Kinetics-{400,600,700}
        <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>

        This dataset handles the loading, decoding and clip sampling for the videos.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        clip_sampling_type: str,
        clip_duration: float,
        clips_per_video: int,
        transform=None,
        num_retries=10,
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

            clip_sampling_type (str): Defines how clips should be sampled from each
                video. It has two options:
                    - uniform: evenly splits the video into clips_per_video increments and
                      samples clips of size clip_duration at these increments.
                    - random: randomly samples clip of size clip_duration from the videos.

            clip_duration (float): Duration of clips in seconds.
            clips_per_video (int): Number of clips per video.
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

            num_retries (int): Number of times to retry reading a video on failure. Each
                time a new a video is tried.
        """
        assert clip_sampling_type in [
            "uniform",
            "random",
        ], f"{clip_sampling_type} not supported"

        self._clip_duration = clip_duration
        self._clips_per_video = clips_per_video
        self._sampling_type = clip_sampling_type
        self._num_retries = num_retries
        self._transform = transform

        # Determine how to parse the data based on whether a file or directory is given.
        data_path = pathlib.Path(data_path)
        if data_path.is_file():
            self._labeled_videos = LabeledVideoPaths.from_csv(data_path)
        elif data_path.is_dir():
            self._labeled_videos = LabeledVideoPaths.from_directory(data_path)
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    def __getitem__(self, index: int) -> dict:
        """
        Samples a video clip associated to the given index. If video reading or decoding
        fails, the sampling to retried on a random video within the dataset.

        Args:
            index (int): index for the video clip.

        Returns:
            A video clip with the following format if transform is None:
                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'index': <clip_index>
                }
            Otherwise, the transform defines the clip output.
        """
        # On failure we retry the video loading and decoding self._num_retries times.
        # Each time we randomly select a new video from the dataset.
        for _ in range(self._num_retries):
            video_index = index // self._clips_per_video
            video_path, label = self._labeled_videos[video_index]

            video = EncodedVideo(video_path)
            clip_duration_pts = video.seconds_to_video_pts(self._clip_duration)
            if self._sampling_type == "uniform":
                clip_index = index % self._clips_per_video

                # Evenly split the video into self._clips_per_video clips and then
                # sample one based on the clip_index.
                uniform_clip_pts = (
                    video.end_pts - video.start_pts
                ) / self._clips_per_video
                clip_start_pts = uniform_clip_pts * clip_index + video.start_pts

            elif self._sampling_type == "random":

                # Randomly sample a clip of size self._clip_duration.
                delta = max(video.end_pts - clip_duration_pts, video.start_pts)
                clip_start_pts = random.uniform(video.start_pts, delta)

            clip_end_pts = clip_start_pts + clip_duration_pts
            frames = video.get_clip(clip_start_pts, clip_end_pts)
            video.close()

            # If decoding failed (wrong format, video is too short, etc...),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self._labeled_videos) - 1)
                continue

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            sample_dict = {"video": frames, "label": label, "index": index}
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

            return sample_dict

        raise RuntimeError(f"Failed to fetch video after {self._num_retries} retries.")

    def __len__(self) -> int:
        """
        Returns:
            The number of video clips in the dataset.
        """
        return len(self._labeled_videos) * self._clips_per_video
