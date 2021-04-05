# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch.utils.data
from iopath.common.file_io import g_pathmgr

from .clip_sampling import ClipSampler
from .encoded_video_dataset import EncodedVideoDataset


logger = logging.getLogger(__name__)


class Hmdb51LabeledVideoPaths:
    """
    Pre-processor for Hmbd51 dataset mentioned here -
        https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

    This dataset consists of classwise folds with each class consisting of 3
        folds (splits).

    The videos directory is of the format,
        video_dir_path/class_x/<somevideo_name>.avi
        ...
        video_dir_path/class_y/<somevideo_name>.avi

    The splits/fold directory is of the format,
        folds_dir_path/class_x_test_split_1.txt
        folds_dir_path/class_x_test_split_2.txt
        folds_dir_path/class_x_test_split_3.txt
        ...
        folds_dir_path/class_y_test_split_1.txt
        folds_dir_path/class_y_test_split_2.txt
        folds_dir_path/class_y_test_split_3.txt

    And each text file in the splits directory class_x_test_split_<1 or 2 or 3>.txt
        <a video as in video_dir_path/class_x> <0 or 1 or 2>
        where 0,1,2 corresponds to unused, train split respectively.

    Each video has name of format
        <some_name>_<tag1>_<tag2>_<tag_3>_<tag4>_<tag5>_<some_id>.avi
    For more details on tags -
        https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
    """

    _allowed_splits = [1, 2, 3]
    _split_type_dict = {"train": 1, "test": 2, "unused": 0}

    @classmethod
    def from_dir(
        cls, data_path: str, split_id: int = 1, split_type: str = "train"
    ) -> Hmdb51LabeledVideoPaths:
        """
        Factory function that creates Hmdb51LabeledVideoPaths object form a splits/folds
        directory.

        Args:
            data_path (str): The path to the splits/folds directory of HMDB51.
            split_id (int): Fold id to be loaded. Belongs to [1,2,3]
            split_type (str): Split/Fold type to be loaded. It belongs to one of the
                following,
                - "train"
                - "test"
                - "unused" (This is a small set of videos that are neither
                of part of test or train fold.)
        """
        data_path = pathlib.Path(data_path)
        if not data_path.is_dir():
            return RuntimeError(f"{data_path} not found or is not a directory.")
        if not int(split_id) in cls._allowed_splits:
            return RuntimeError(
                f"{split_id} not found in allowed split id's {cls._allowed_splits}."
            )
        file_name_format = "_test_split" + str(int(split_id))
        file_paths = sorted(
            (
                f
                for f in data_path.iterdir()
                if f.is_file() and f.suffix == ".txt" and file_name_format in f.stem
            )
        )
        return cls.from_csvs(file_paths, split_type)

    @classmethod
    def from_csvs(
        cls, file_paths: List[Union[pathlib.Path, str]], split_type: str = "train"
    ) -> Hmdb51LabeledVideoPaths:
        """
        Factory function that creates Hmdb51LabeledVideoPaths object form a list of
        split files of .txt type

        Args:
            file_paths (List[Union[pathlib.Path, str]]) : The path to the splits/folds
                    directory of HMDB51.
            split_type (str): Split/Fold type to be loaded.
                - "train"
                - "test"
                - "unused"
        """
        video_paths_and_label = []
        for file_path in file_paths:
            file_path = pathlib.Path(file_path)
            assert g_pathmgr.exists(file_path), f"{file_path} not found."
            if not (file_path.suffix == ".txt" and "_test_split" in file_path.stem):
                return RuntimeError(f"Ivalid file: {file_path}")

            action_name = "_"
            action_name = action_name.join((file_path.stem).split("_")[:-2])
            with g_pathmgr.open(file_path, "r") as f:
                for path_label in f.read().splitlines():
                    line_split = path_label.rsplit(None, 1)

                    if not int(line_split[1]) == cls._split_type_dict[split_type]:
                        continue

                    file_path = os.path.join(action_name, line_split[0])
                    meta_tags = line_split[0].split("_")[-6:-1]
                    video_paths_and_label.append(
                        (file_path, {"label": action_name, "meta_tags": meta_tags})
                    )

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: List[Tuple[str, Optional[dict]]], path_prefix=""
    ) -> None:
        """
        Args:
            paths_and_labels [(str, int)]: a list of tuples containing the video
                path and integer label.
        """
        self._paths_and_labels = paths_and_labels
        self._path_prefix = path_prefix

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, dict]:
        """
        Args:
            index (int): the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        path, label = self._paths_and_labels[index]
        return (os.path.join(self._path_prefix, path), label)

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._paths_and_labels)


def Hmdb51(
    data_path: pathlib.Path,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[dict], Any]] = None,
    video_path_prefix: str = "",
    split_id: int = 1,
    split_type: str = "train",
    decode_audio=True,
    decoder: str = "pyav",
) -> EncodedVideoDataset:
    """
    A helper function to create EncodedVideoDataset object for HMDB51 dataset

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

        decoder (str): Defines which backend should be used to decode videos.
    """
    labeled_video_paths = Hmdb51LabeledVideoPaths.from_dir(
        data_path, split_id=split_id, split_type=split_type
    )
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
