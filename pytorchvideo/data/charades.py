# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import csv
import functools
import itertools
import os
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo

from .utils import MultiProcessSampler


class Charades(torch.utils.data.IterableDataset):
    """
    Action recognition video dataset for Charades stored as image frames.
    <https://prior.allenai.org/projects/charades>

    This dataset handles the parsing of frames, loading and clip sampling for the
    videos. All io reading is done with PathManager, enabling non-local storage
    uri's to be used.
    """

    # Number of classes represented by this dataset's annotated labels.
    NUM_CLASSES = 157

    def __init__(
        self,
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,
    ) -> None:
        """
        Args:
            data_path (str): Path to the data file. This file must be a space
                separated csv with the format:
                    `original_vido_id video_id frame_id path labels`

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Optional[Callable]): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. The clip output is a dictionary with the
                following format:
                    {
                        'video': <video_tensor>,
                        'label': <index_label> for clip-level label,
                        'video_label': <index_label> for video-level label,
                        'video_index': <video_index>,
                        'clip_index': <clip_index>,
                        'aug_index': <aug_index>, augmentation index as augmentations
                            might generate multiple views for one clip.
                    }
                If transform is None, the raw clip output in the above format is
                returned unmodified.
            video_path_prefix (str): prefix path to add to all paths from data_path.
            frames_per_clip (Optional[int]): The number of frames per clip to sample.
        """
        self._transform = transform
        self._clip_sampler = clip_sampler
        (
            self._path_to_videos,
            self._labels,
            self._video_labels,
        ) = _read_video_paths_and_labels(data_path, prefix=video_path_prefix)
        self._video_sampler = video_sampler(self._path_to_videos)
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()
        self._frame_filter = (
            functools.partial(
                Charades._sample_clip_frames,
                frames_per_clip=frames_per_clip,
            )
            if frames_per_clip is not None
            else None
        )

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0

    @staticmethod
    def _sample_clip_frames(
        frame_indices: List[int], frames_per_clip: int
    ) -> List[int]:
        """
        Args:
            frame_indices (list): list of frame indices.
            frames_per+clip (int): The number of frames per clip to sample.

        Returns:
            (list): Outputs a subsampled list with num_samples frames.
        """
        num_frames = len(frame_indices)
        indices = torch.linspace(0, num_frames - 1, frames_per_clip)
        indices = torch.clamp(indices, 0, num_frames - 1).long()

        return [frame_indices[idx] for idx in indices]

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
                    'label': <index_label> for clip-level label,
                    'video_label': <index_label> for video-level label,
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>, augmentation index as augmentations
                        might generate multiple views for one clip.
                }
            Otherwise, the transform defines the clip output.
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
            self._next_clip_start_time, video.duration
        )
        # Only load the clip once and reuse previously stored clip if there are multiple
        # views for augmentations to perform on the same clip.
        if aug_index == 0:
            self._loaded_clip = video.get_clip(clip_start, clip_end, self._frame_filter)
        frames, frame_indices = (
            self._loaded_clip["video"],
            self._loaded_clip["frame_indices"],
        )
        self._next_clip_start_time = clip_end

        if is_last_clip:
            self._loaded_video = None
            self._next_clip_start_time = 0.0

        # Merge unique labels from each frame into clip label.
        labels_by_frame = [
            self._labels[video_index][i]
            for i in range(min(frame_indices), max(frame_indices) + 1)
        ]
        sample_dict = {
            "video": frames,
            "label": labels_by_frame,
            "video_label": self._video_labels[video_index],
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
    video_path_label_file: List[str], prefix: str = ""
) -> Tuple[List[str], List[int]]:
    """
    Args:
        video_path_label_file (List[str]): a file that contains frame paths for each
            video and the corresponding frame label. The file must be a space separated
            csv of the format:
                `original_vido_id video_id frame_id path labels`

        prefix (str): prefix path to add to all paths from video_path_label_file.

    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with g_pathmgr.open(video_path_label_file, "r") as f:

        # Space separated CSV with format: original_vido_id video_id frame_id path labels
        csv_reader = csv.DictReader(f, delimiter=" ")
        for row in csv_reader:
            assert len(row) == 5
            video_name = row["original_vido_id"]
            path = os.path.join(prefix, row["path"])
            image_paths[video_name].append(path)
            frame_labels = row["labels"].replace('"', "")
            label_list = []
            if frame_labels:
                label_list = [int(x) for x in frame_labels.split(",")]

            labels[video_name].append(label_list)

    # Extract image paths from dictionary and return paths and labels as list.
    video_names = image_paths.keys()
    image_paths = [image_paths[key] for key in video_names]
    labels = [labels[key] for key in video_names]
    # Aggregate labels from all frames to form video-level labels.
    video_labels = [list(set(itertools.chain(*label_list))) for label_list in labels]
    return image_paths, labels, video_labels
