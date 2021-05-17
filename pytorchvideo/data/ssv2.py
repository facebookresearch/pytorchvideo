# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import csv
import functools
import json
import os
import random
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo

from .utils import MultiProcessSampler


class SSv2(torch.utils.data.IterableDataset):
    """
    Action recognition video dataset for
    `Something-something v2 (SSv2) <https://20bn.com/datasets/something-something>`_ stored
    as image frames.

    This dataset handles the parsing of frames, loading and clip sampling for the
    videos. All io is done through :code:`iopath.common.file_io.PathManager`, enabling
    non-local storage uri's to be used.
    """

    def __init__(
        self,
        label_name_file: str,
        video_label_file: str,
        video_path_label_file: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,
        rand_sample_frames: bool = False,
    ) -> None:
        """
        Args:
            label_name_file (str): SSV2 label file that contains the label names and
                indexes.

            video_label_file (str): a file that contains video ids and the corresponding
                video label.

            video_path_label_file (str): a file that contains frame paths for each
                video and the corresponding frame label. The file must be a space separated
                csv of the format: (original_vido_id video_id frame_id path labels).

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Optional[Callable]): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            video_path_prefix (str): prefix path to add to all paths from data_path.

            frames_per_clip (Optional[int]): The number of frames per clip to sample.

            rand_sample_frames (bool): If True, randomly sampling frames for each clip.
        """

        torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.SSv2.__init__")

        self._transform = transform
        self._clip_sampler = clip_sampler
        self._path_to_videos, self._labels = _read_video_paths_and_labels(
            label_name_file,
            video_label_file,
            video_path_label_file,
            prefix=video_path_prefix,
        )
        self._video_sampler = video_sampler(self._path_to_videos)
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()
        self._frame_filter = (
            functools.partial(
                SSv2._sample_clip_frames,
                frames_per_clip=frames_per_clip,
                rand_sample=rand_sample_frames,
            )
            if frames_per_clip is not None
            else None
        )

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video = None
        self._next_clip_start_time = 0.0

    @staticmethod
    def _sample_clip_frames(
        frame_indices: List[int], frames_per_clip: int, rand_sample: bool
    ) -> List[int]:
        """
        Use segment-based input frame sampling that splits eachvideo into segments,
        and from each of them, we sample one frame to form a clip.

        Args:
            frame_indices (list): list of frame indices.
            frames_per_clip (int): The number of frames per clip to sample.
            rand_sample (bool): if True, randomly sampling frames.

        Returns:
            (list): Outputs a subsampled list with num_samples frames.
        """
        num_frames = len(frame_indices)

        seg_size = float(num_frames - 1) / frames_per_clip
        seq = []
        for i in range(frames_per_clip):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if rand_sample:
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return [frame_indices[idx] for idx in seq]

    @property
    def video_sampler(self):
        return self._video_sampler

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        if self._loaded_video:
            video, video_index = self._loaded_video
        else:
            video_index = next(self._video_sampler_iter)
            path_to_video_frames = self._path_to_videos[video_index]
            video = FrameVideo.from_frame_paths(path_to_video_frames)
            self._loaded_video = (video, video_index)

        clip_start, clip_end, clip_index, aug_index, is_last_clip = self._clip_sampler(
            self._next_clip_start_time, video.duration, {}
        )
        # Only load the clip once and reuse previously stored clip if there are multiple
        # views for augmentations to perform on the same clip.
        if aug_index == 0:
            self._loaded_clip = video.get_clip(0, video.duration, self._frame_filter)

        self._next_clip_start_time = clip_end

        if is_last_clip:
            self._loaded_video = None
            self._next_clip_start_time = 0.0

        sample_dict = {
            "video": self._loaded_clip["video"],
            "label": self._labels[video_index],
            "video_name": str(video_index),
            "video_index": video_index,
            "clip_index": clip_index,
            "aug_index": aug_index,
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict

    def __iter__(self):
        return self


def _read_video_paths_and_labels(
    label_name_file: str,
    video_label_file: str,
    video_path_label_file: str,
    prefix: str = "",
) -> Tuple[List[str], List[int]]:
    """
    Args:
        label_name_file (str): ssv2 label file that contians the label names and
            indexes. ('/path/to/folder/something-something-v2-labels.json')
        video_label_file (str): a file that contains video ids and the corresponding
            video label. (e.g., '/path/to/folder/something-something-v2-train.json')
        video_path_label_file (str): a file that contains frame paths for each
            video and the corresponding frame label. The file must be a space separated
            csv of the format:
                `original_vido_id video_id frame_id path labels`
        prefix (str): prefix path to add to all paths from video_path_label_file.

    Returns:
        image_paths (list): list of list containing path to each frame.
        labels (list): list containing label of each video.
    """
    # Loading image paths.
    paths = defaultdict(list)
    with g_pathmgr.open(video_path_label_file, "r") as f:
        # Space separated CSV with format: original_vido_id video_id frame_id path labels
        csv_reader = csv.DictReader(f, delimiter=" ")
        for row in csv_reader:
            assert len(row) == 5
            video_name = row["original_vido_id"]
            path = os.path.join(prefix, row["path"])
            paths[video_name].append(path)

    # Loading label names.
    with g_pathmgr.open(label_name_file, "r") as f:
        label_name_dict = json.load(f)

    with g_pathmgr.open(video_label_file, "r") as f:
        video_label_json = json.load(f)

    labels = []
    image_paths = []
    for video in video_label_json:
        video_name = video["id"]
        if video_name in paths:
            template = video["template"]
            template = template.replace("[", "")
            template = template.replace("]", "")
            label = int(label_name_dict[template])
            image_paths.append(paths[video_name])
            labels.append(label)

    return image_paths, labels
