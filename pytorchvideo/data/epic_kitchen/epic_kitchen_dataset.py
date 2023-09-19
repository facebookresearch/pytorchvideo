# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import ast
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Callable, Dict, List, Optional

import torch
from pytorchvideo.data.dataset_manifest_utils import (
    EncodedVideoInfo,
    get_seconds_from_hms_time,
    VideoClipInfo,
    VideoDataset,
    VideoDatasetType,
    VideoFrameInfo,
    VideoInfo,
)
from pytorchvideo.data.frame_video import FrameVideo
from pytorchvideo.data.utils import DataclassFieldCaster, load_dataclass_dict_from_csv
from pytorchvideo.data.video import Video


@dataclass
class ActionData(DataclassFieldCaster):
    """
    Represents an action from the Epic Kitchen dataset.

    This class encapsulates information about actions performed in the dataset,
    including participant ID, video ID, narration, timestamps, verb, noun, and more.

    Attributes:
        participant_id (str): The unique identifier of the participant.
        video_id (str): The identifier of the video containing the action.
        narration (str): The textual narration of the action.
        start_timestamp (str): The start timestamp of the action in HH:MM:SS format.
        stop_timestamp (str): The stop timestamp of the action in HH:MM:SS format.
        start_frame (int): The starting frame of the action.
        stop_frame (int): The ending frame of the action.
        verb (str): The verb describing the action.
        verb_class (int): The class label for the verb.
        noun (str): The noun associated with the action.
        noun_class (int): The class label for the noun.
        all_nouns (list): A list of all nouns related to the action.
        all_noun_classes (list): A list of class labels for all related nouns.

    Properties:
        start_time (float): The start time of the action in seconds.
        stop_time (float): The stop time of the action in seconds.

    Methods:
        None

    Note:
        This class is designed to represent actions from the Epic Kitchen dataset.
    """
    @property
    def start_time(self) -> float:
        """
        Get the start time of the action in seconds.

        Returns:
            float: The start time in seconds.
        """
        return get_seconds_from_hms_time(self.start_timestamp)

    @property
    def stop_time(self) -> float:
        """
        Get the stop time of the action in seconds.

        Returns:
            float: The stop time in seconds.
        """
        return get_seconds_from_hms_time(self.stop_timestamp)


class EpicKitchenDataset(torch.utils.data.Dataset):
    """
    Video dataset for EpicKitchen-55 Dataset
    <https://epic-kitchens.github.io/2019/>

    This dataset handles the loading, decoding, and configurable clip
    sampling for the videos.
    """

    def __init__(
        self,
        video_info_file_path: str,
        actions_file_path: str,
        clip_sampler: Callable[
            [Dict[str, Video], Dict[str, List[ActionData]]], List[VideoClipInfo]
        ],
        video_data_manifest_file_path: str,
        dataset_type: VideoDatasetType = VideoDatasetType.Frame,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        frame_filter: Optional[Callable[[List[int]], List[int]]] = None,
        multithreaded_io: bool = True,
    ) -> None:
        f"""
        Args:
            video_info_file_path (str):
                Path or URI to manifest with basic metadata of each video.
                File must be a csv (w/header) with columns:
                {[f.name for f in dataclass_fields(VideoInfo)]}

            actions_file_path (str):
                Path or URI to manifest with action annotations for each video.
                File must ber a csv (w/header) with columns:
                {[f.name for f in dataclass_fields(ActionData)]}

            clip_sampler (Callable[[Dict[str, Video]], List[VideoClipInfo]]):
                This callable takes as input all available videos and outputs a list of clips to
                be loaded by the dataset.

            video_data_manifest_file_path (str):
                The path to a json file outlining the available video data for the
                associated videos.  File must be a csv (w/header) with columns:
                {[f.name for f in dataclass_fields(VideoFrameInfo)]}

                or
                {[f.name for f in dataclass_fields(EncodedVideoInfo)]}

                To generate this file from a directory of video frames, see helper
                functions in Module: pytorchvideo.data.epic_kitchen.utils

            dataset_type (VideoDatasetType): The dataformat in which dataset
                video data is store (e.g. video frames, encoded video etc).

            transform (Optional[Callable[[Dict[str, Any]], Any]]):
                This callable is evaluated on the clip output before the clip is returned.
                It can be used for user-defined preprocessing and augmentations to the clips.

                    The clip input is a dictionary with the following format:
                        {{
                            'video': <video_tensor>,
                            'audio': <audio_tensor>,
                            'actions': <List[ActionData]>,
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

        torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.EpicKitchenDataset.__init__")

        assert video_info_file_path
        assert actions_file_path
        assert video_data_manifest_file_path
        assert clip_sampler

        # Populate video and metadata data providers
        self._videos: Dict[str, Video] = VideoDataset._load_videos(
            video_data_manifest_file_path,
            video_info_file_path,
            multithreaded_io,
            dataset_type,
        )

        self._actions: Dict[str, List[ActionData]] = load_dataclass_dict_from_csv(
            actions_file_path, ActionData, "video_id", list_per_key=True
        )
        # Sample datapoints
        self._clips: List[VideoClipInfo] = clip_sampler(self._videos, self._actions)

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
                    'actions': <df[ActionData]>,
                    'start_time': <float>,
                    'stop_time': <float>
                }}
            Otherwise, the transform defines the clip output.
        """
        clip = self._clips[index]
        video = self._videos[clip.video_id]

        if isinstance(video, FrameVideo):
            clip_dict = video.get_clip(
                clip.start_time, clip.stop_time, self._frame_filter
            )
        else:
            clip_dict = video.get_clip(clip.start_time, clip.stop_time)

        clip_data = {
            "video_id": clip.video_id,
            **clip_dict,
            "actions": self._actions[clip.video_id],
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
