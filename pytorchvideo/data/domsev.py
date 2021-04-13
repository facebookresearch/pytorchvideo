# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from pytorchvideo.data.dataset_manifest_utils import (
    EncodedVideoInfo,
    VideoClipInfo,
    VideoDataset,
    VideoDatasetType,
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


# Utility functions
def seconds_to_frame_index(
    time_in_seconds: float, fps: int, zero_indexed: Optional[bool] = True
) -> int:
    """Converts a point in time (in seconds) within a video clip to its closest
    frame indexed (rounding down), based on a specified frame rate.

    Args:
        time_in_seconds (float): The point in time within the video.
        fps (int): The frame rate (frames per second) of the video.
        zero_indexed (Optional[bool]): Whether the returned frame should be
            zero-indexed (if True) or one-indexed (if False).

    Returns:
        (int) The index of the nearest frame (rounding down to the nearest integer).
    """
    frame_idx = math.floor(time_in_seconds * fps)
    if not zero_indexed:
        frame_idx += 1
    return frame_idx


def frame_index_to_seconds(
    frame_index: int, fps: int, zero_indexed: Optional[bool] = True
) -> float:
    """Converts a frame index within a video clip to the corresponding
    point in time (in seconds) within the video, based on a specified frame rate.

    Args:
        frame_index (int): The index of the frame within the video.
        fps (int): The frame rate (frames per second) of the video.
        zero_indexed (Optional[bool]): Whether the specified frame is zero-indexed
            (if True) or one-indexed (if False).

    Returns:
        (float) The point in time within the video.
    """
    if not zero_indexed:
        frame_index -= 1
    time_in_seconds = frame_index / fps
    return time_in_seconds


def get_overlap_for_time_range_pair(
    t1_start: float, t1_stop: float, t2_start: float, t2_stop: float
) -> Optional[Tuple[float, float]]:
    """Calculates the overlap between two time ranges, if one exists.

    Returns:
        (Optional[Tuple]) A tuple of <overlap_start_time, overlap_stop_time> if
        an overlap is found, or None otherwise.
    """
    # Check if there is an overlap
    if (t1_start <= t2_stop) and (t2_start <= t1_stop):
        # Calculate the overlap period
        overlap_start_time = max(t1_start, t2_start)
        overlap_stop_time = min(t1_stop, t2_stop)
        return (overlap_start_time, overlap_stop_time)
    else:
        return None


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
        clip_sampler: Callable[
            [Dict[str, Video], Dict[str, List[ActivityData]]], List[VideoClipInfo]
        ],
        dataset_type: VideoDatasetType = VideoDatasetType.Frame,
        frames_per_second: int = 1,
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

            clip_sampler: Callable[
                [Dict[str, Video], Dict[str, List[ActivityData]]], List[VideoClipInfo]
            ],

            dataset_type (VideoDatasetType): The dataformat in which dataset
                video data is store (e.g. video frames, encoded video etc).

            frames_per_second (int): The FPS of the stored videos. (NOTE:
                this is variable and may be different than the original FPS
                reported on the DoMSEV dataset website -- it depends on the
                subsampling and frame extraction done internally at Facebook).

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

        # Sample datapoints
        self._clips: List[VideoClipInfo] = clip_sampler(self._videos, self._activities)

        self._frames_per_second = frames_per_second
        self._user_transform = transform
        self._transform = self._transform_clip
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

        # Filter activities by only the ones that appear within the clip boundaries,
        # and unpack the activities so there is one per frame in the clip
        activities_in_video = self._activities[clip.video_id]
        activities_in_clip = []
        for activity in activities_in_video:
            overlap_period = get_overlap_for_time_range_pair(
                clip.start_time, clip.stop_time, activity.start_time, activity.stop_time
            )
            if overlap_period is not None:
                overlap_start_time, overlap_stop_time = overlap_period

                # Convert the overlapping period between clip and activity to
                # 0-indexed start and stop frame indexes, so we can unpack 1
                # activity label per frame.
                overlap_start_frame = seconds_to_frame_index(
                    overlap_start_time, self._frames_per_second
                )
                overlap_stop_frame = seconds_to_frame_index(
                    overlap_stop_time, self._frames_per_second
                )

                # Append 1 activity label per frame
                for _ in range(overlap_start_frame, overlap_stop_frame):
                    activities_in_clip.append(activity)

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

    def _transform_clip(self, clip: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms a given video clip, according to some pre-defined transforms
        and an optional user transform function (self._user_transform).

        Args:
            clip (Dict[str, Any]): The clip that will be transformed.

        Returns:
            (Dict[str, Any]) The transformed clip.
        """
        for key in clip:
            if clip[key] is None:
                clip[key] = torch.tensor([])

        if self._user_transform:
            clip = self._user_transform(clip)

        return clip
