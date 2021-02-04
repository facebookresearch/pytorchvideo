# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch
from pytorchvideo.data.dataset_manifest_utils import (
    EncodedVideoInfo,
    VideoClipInfo,
    VideoDatasetType,
    VideoDataset,
    VideoInfo,
)
from pytorchvideo.data.utils import DataclassFieldCaster, load_dataclass_dict_from_csv
from pytorchvideo.data.video import Video

USER_SCENE_MAP = {
    0: "none",
    1: "indoor",
    2: "nature",
    3: "crowded_environment",
    4: "urban",
}

USER_ACTIVITY_MAP = {
    0: "none",
    1: "walking",
    2: "running",
    3: "standing",
    4: "biking",
    5: "driving",
    6: "playing",
    7: "cooking",
    8: "eating",
    9: "observing",
    10: "in_conversation",
    11: "browsing",
    12: "shopping",
}

USER_ATTENTION_MAP = {
    0: "none",
    1: "paying_attention",
    2: "interacting",
}


@dataclass
class ActivityData(DataclassFieldCaster):
    """
    Class representing a contiguous activity video segment from the DoMSEV dataset.
    """

    video_id: str
    start_time: float  # Start time of the activity, in seconds
    stop_time: float  # Stop time of the activity, in seconds
    start_frame: int  # 0-indexed ID of the start frame (inclusive)
    stop_frame: int  # 0-index ID of the stop frame (inclusive)
    activity_id: int
    activity_name: str


class ClipSampling(Enum):
    RandomOffsetUniform = 1
    # TODO(T84168155): Switch to the action-based clip sampling in D25699256


class DomsevDataset(torch.utils.data.Dataset):
    """
    Egocentric activity classification video dataset for DoMSEV stored as
    an encoded video (with frame-level labels).
    <https://www.verlab.dcc.ufmg.br/semantic-hyperlapse/cvpr2018-dataset/>

    This dataset handles the loading, decoding, and configurable clip
    sampling for the videos.
    """

    def __init__(
        self,
        video_data_manifest_file_path: str,
        video_info_file_path: str,
        activities_file_path: str,
        clip_sampling: ClipSampling = ClipSampling.RandomOffsetUniform,
        dataset_type: VideoDatasetType = VideoDatasetType.Frame,
        seconds_per_clip: float = 10.0,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        frame_filter: Optional[Callable[[List[int]], List[int]]] = None,
        multithreaded_io: bool = False,
    ) -> None:
        f"""
        Args:
            video_data_manifest_file_path (str):
                The path to a json file outlining the available video data for the
                associated videos.  File must be a csv (w/header) with columns:
                {[f.name for f in dataclass_fields(EncodedVideoInfo)]}

                To generate this file from a directory of video frames, see helper
                functions in Module: pytorchvideo.data.domsev.utils

            video_info_file_path (str):
                Path or URI to manifest with basic metadata of each video.
                File must be a csv (w/header) with columns:
                {[f.name for f in dataclass_fields(VideoInfo)]}

            activities_file_path (str):
                Path or URI to manifest with activity annotations for each video.
                File must be a csv (w/header) with columns:
                {[f.name for f in dataclass_fields(ActivityData)]}

            clip_sampling (ClipSampling):
                The type of sampling to perform to perform on the videos of the dataset.

            dataset_type (VideoDatasetType): The dataformat in which dataset
                video data is store (e.g. video frames, encoded video etc).

            transform (Optional[Callable[[Dict[str, Any]], Any]]):
                This callable is evaluated on the clip output before the clip is returned.
                It can be used for user-defined preprocessing and augmentations to the clips.

                    The clip input is a dictionary with the following format:
                        {{
                            'video': <video_tensor>,
                            'audio': <audio_tensor>,
                            'activities': <activities_tensor>,
                            'start_time': <float>,
                            'stop_time': <float>
                        }}

                If transform is None, the raw clip output in the above format is
                returned unmodified.

            frame_filter (Optional[Callable[[List[int]], List[int]]]):
                This callable is evaluated on the set of available frame inidices to be
                included in a sampled clip. This can be used to subselect frames within
                a clip to be loaded.

            multithreaded_io (bool):
                Boolean to control whether parllelizable io operations are performed across
                multiple threads.
        """
        assert video_info_file_path
        assert activities_file_path
        assert video_data_manifest_file_path

        # Populate video and metadata data providers
        self._videos: Dict[str, Video] = VideoDataset._load_videos(
            video_data_manifest_file_path,
            video_info_file_path,
            multithreaded_io,
            dataset_type,
        )

        self._activities: Dict[str, List[ActivityData]] = load_dataclass_dict_from_csv(
            activities_file_path, ActivityData, "video_id", list_per_key=True
        )

        clip_sampler = DomsevDataset._define_clip_structure_generator(
            seconds_per_clip, clip_sampling
        )
        transform = DomsevDataset._transform_generator(transform)

        # Sample datapoints
        self._clips: List[VideoClipInfo] = clip_sampler(self._videos, self._activities)

        self._transform = transform
        self._frame_filter = frame_filter

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Samples a video clip associated to the given index.

        Args:
            index (int): index for the video clip.

        Returns:
            A video clip with the following format if transform is None:
                {{
                    'video_id': <str>,
                    'video': <video_tensor>,
                    'audio': <audio_tensor>,
                    'activities': <activities_tensor>,
                    'start_time': <float>,
                    'stop_time': <float>
                }}
            Otherwise, the transform defines the clip output.
        """
        clip = self._clips[index]

        # Filter activities by only the ones that appear within the clip boundaries
        activities_in_video = self._activities[clip.video_id]
        activities_in_clip = [
            a
            for a in activities_in_video
            if (a.start_time <= clip.stop_time and a.stop_time >= clip.start_time)
        ]

        # Convert the list of ActivityData objects to a tensor of just the activity class IDs
        activity_class_ids = [
            activities_in_clip[i].activity_id for i in range(len(activities_in_clip))
        ]
        activity_class_ids_tensor = torch.tensor(activity_class_ids)

        clip_data = {
            "video_id": clip.video_id,
            **self._videos[clip.video_id].get_clip(clip.start_time, clip.stop_time),
            "activities": activity_class_ids_tensor,
            "start_time": clip.start_time,
            "stop_time": clip.stop_time,
        }

        if self._transform:
            clip_data = self._transform(clip_data)

        return clip_data

    def __len__(self) -> int:
        """
        Returns:
            The number of video clips in the dataset.
        """
        return len(self._clips)

    @staticmethod
    def _define_clip_structure_generator(
        seconds_per_clip: float, clip_sampling: ClipSampling
    ) -> Callable[
        [Dict[str, Video], Dict[str, List[ActivityData]]], List[VideoClipInfo]
    ]:
        """
        Args:
            seconds_per_clip (float): The length of each sampled clip in seconds.
            clip_sampling (ClipSampling):
                The type of sampling to perform to perform on the videos of the dataset.

        Returns:
            A function that takes a dictionary of videos and a dictionary of the activities
            for each video and outputs a list of sampled clips.
        """
        if not clip_sampling == ClipSampling.RandomOffsetUniform:
            raise NotImplementedError(
                f"Only {ClipSampling.RandomOffsetUniform} is implemented. "
                f"{clip_sampling} not implemented."
            )

        def define_clip_structure(
            videos: Dict[str, Video], activities: Dict[str, List[ActivityData]]
        ) -> List[VideoClipInfo]:
            clips = []
            for video_id, video in videos.items():
                offset = random.random() * seconds_per_clip
                num_clips = int((video.duration - offset) // seconds_per_clip)

                for i in range(num_clips):
                    start_time = i * seconds_per_clip + offset
                    stop_time = start_time + seconds_per_clip
                    clip = VideoClipInfo(video_id, start_time, stop_time)
                    clips.append(clip)
            return clips

        return define_clip_structure

    @staticmethod
    def _transform_generator(
        transform: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Args:
            transform (Callable[[Dict[str, Any]], Dict[str, Any]]): A function that performs
            any operation on a clip before it is returned in the default transform function.

        Returns:
            A function that performs any operation on a clip and returns the transformed clip.
        """

        def transform_clip(clip: Dict[str, Any]) -> Dict[str, Any]:
            for key in clip:
                if clip[key] is None:
                    clip[key] = torch.tensor([])

            if transform:
                clip = transform(clip)

            return clip

        return transform_clip
