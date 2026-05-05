# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import os
import pathlib
from typing import List, Optional, Tuple, Dict, Union, Callable, cast

from iopath.common.file_io import g_pathmgr
from torchvision.datasets.folder import (
    make_dataset,
    has_file_allowed_extension,
    find_classes,
)


def make_dataset_from_video_folders(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples."
        )

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_folder(x: str) -> bool:
            if g_pathmgr.ls(x):
                return has_file_allowed_extension(g_pathmgr.ls(x)[0], extensions)
            else:
                return False

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, fnames, _ in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_folder(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = (
            f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class LabeledVideoPaths:
    """
    LabeledVideoPaths contains pairs of video path and integer index label.
    """

    @classmethod
    def from_path(cls, data_path: str) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object depending on the path
        type.
        - If it is a directory path it uses the LabeledVideoPaths.from_directory function.
        - If it's a file it uses the LabeledVideoPaths.from_csv file.
        Args:
            file_path (str): The path to the file to be read.
        """

        if g_pathmgr.isfile(data_path):
            return LabeledVideoPaths.from_csv(data_path)
        elif g_pathmgr.isdir(data_path):
            class_0 = g_pathmgr.ls(data_path)[0]
            video_0 = g_pathmgr.ls(pathlib.Path(data_path) / class_0)[0]
            video_0_path = pathlib.Path(data_path) / class_0 / video_0
            if g_pathmgr.isfile(video_0_path):
                return LabeledVideoPaths.from_directory(data_path)
            else:
                return LabeledVideoPaths.from_directory_of_video_folders(data_path)
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_csv(cls, file_path: str) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object by reading a file with the
        following format:
            <path> <integer_label>
            ...
            <path> <integer_label>

        Args:
            file_path (str): The path to the file to be read.
        """
        assert g_pathmgr.exists(file_path), f"{file_path} not found."
        video_paths_and_label = []
        with g_pathmgr.open(file_path, "r") as f:
            for path_label in f.read().splitlines():
                line_split = path_label.rsplit(None, 1)

                # The video path file may not contain labels (e.g. for a test split). We
                # assume this is the case if only 1 path is found and set the label to
                # -1 if so.
                if len(line_split) == 1:
                    file_path = line_split[0]
                    label = -1
                else:
                    file_path, label = line_split

                video_paths_and_label.append((file_path, int(label)))

        assert len(video_paths_and_label) > 0, (
            f"Failed to load dataset from {file_path}."
        )
        return cls(video_paths_and_label)

    @classmethod
    def from_directory(cls, dir_path: str) -> LabeledVideoPaths:
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
            dir_path (str): Root directory to the video class directories .
        """
        assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted(
            (f.name for f in pathlib.Path(dir_path).iterdir() if f.is_dir())
        )
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        video_paths_and_label = make_dataset(
            dir_path, class_to_idx, extensions=("mp4", "avi")
        )
        assert len(video_paths_and_label) > 0, (
            f"Failed to load dataset from {dir_path}."
        )
        return cls(video_paths_and_label)

    @classmethod
    def from_directory_of_video_folders(cls, dir_path: str) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object by parsing the structure
        of the given directory's subdirectories into the classification labels. It
        expects the directory format to be the following:
             dir_path/<class_name>/<video_name>/<frame_name>.jpg

        Classes are indexed from 0 to the number of classes, alphabetically.

        E.g.
            dir_path/class_x/vid1/xxx.ext
            dir_path/class_x/vid1/xxy.ext
            dir_path/class_x/vid2/xxz.ext
            dir_path/class_y/vid3/123.ext
            dir_path/class_y/vid4/nsdf3.ext
            dir_path/class_y/vid4/asd932_.ext

        Would produce two classes labeled 0 and 1 with 2 videos paths associated with each.

        Args:
            dir_path (str): Root directory to the video class directories .
        """
        assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted(
            (f.name for f in pathlib.Path(dir_path).iterdir() if f.is_dir())
        )
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        video_paths_and_label = make_dataset_from_video_folders(
            dir_path, class_to_idx, extensions=("jpg", "png")
        )
        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {dir_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: List[Tuple[str, Optional[int]]], path_prefix=""
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

    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        Args:
            index (int): the path and label index.

        Returns:
            The path and label tuple for the given index.
        """
        path, label = self._paths_and_labels[index]
        return (os.path.join(self._path_prefix, path), {"label": label})

    def __len__(self) -> int:
        """
        Returns:
            The number of video paths and label pairs.
        """
        return len(self._paths_and_labels)
