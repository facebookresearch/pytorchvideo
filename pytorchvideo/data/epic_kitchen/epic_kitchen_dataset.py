# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import ast
import datetime
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.frame_video import FrameVideo
from pytorchvideo.data.utils import DataclassFieldCaster, load_dataclass_dict_from_csv
from pytorchvideo.data.video import Video


@dataclass
class EpicKitchenClip(DataclassFieldCaster):
    video_id: str
    start_time: float
    stop_time: float


@dataclass
class VideoInfo(DataclassFieldCaster):
    """
    Class representing the video-level metadata of a video from the Epic Kitchen dataset.
    """

    video: str
    resolution: str
    duration: float
    fps: float

    @property
    def participant_id(self) -> str:
        return self.video[:3]

    @property
    def video_id(self) -> str:
        return self.video


@dataclass
class VideoFrameInfo(DataclassFieldCaster):
    """
    Class representing the locations of all frames that compose a video.
    """

    video_id: str
    location: str
    frame_file_stem: str
    frame_string_length: int
    min_frame_number: int
    max_frame_number: int
    file_extension: str


@dataclass
class EncodedVideoInfo(DataclassFieldCaster):
    """
    Class representing the location of an available encoded video.
    """

    video_id: str
    file_path: str


def _get_seconds_from_hms_time(time_str: str) -> float:
    """
    Get Seconds from timestamp of form 'HH:MM:SS'.

    Args:
        time_str (str)

    Returns:
        float of seconds

    """
    time_since_min_time = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    min_time = datetime.datetime.strptime("", "")
    return float((time_since_min_time - min_time).total_seconds())


@dataclass
class ActionData(DataclassFieldCaster):
    """
    Class representing an action from the Epic Kitchen dataset.
    """

    participant_id: str
    video_id: str
    narration: str
    start_timestamp: str
    stop_timestamp: str
    start_frame: int
    stop_frame: int
    verb: str
    verb_class: int
    noun: str
    noun_class: int
    all_nouns: list = DataclassFieldCaster.complex_initialized_dataclass_field(
        ast.literal_eval
    )
    all_noun_classes: list = DataclassFieldCaster.complex_initialized_dataclass_field(
        ast.literal_eval
    )

    @property
    def start_time(self) -> float:
        return _get_seconds_from_hms_time(self.start_timestamp)

    @property
    def stop_time(self) -> float:
        return _get_seconds_from_hms_time(self.stop_timestamp)


class EpicKitchenDatasetType(Enum):
    Frame = 1
    EncodedVideo = 2


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
            [Dict[str, Video], Dict[str, List[ActionData]]], List[EpicKitchenClip]
        ],
        video_data_manifest_file_path: str,
        dataset_type: EpicKitchenDatasetType = EpicKitchenDatasetType.Frame,
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

            clip_sampler (Callable[[Dict[str, Video]], List[EpicKitchenClip]]):
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

            dataset_type (EpicKitchenDatasetType): The dataformat in which dataset
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
        assert video_info_file_path
        assert actions_file_path
        assert video_data_manifest_file_path
        assert clip_sampler

        # Populate video and metadata data providers
        self._videos: Dict[str, Video] = EpicKitchenDataset._load_videos(
            video_data_manifest_file_path,
            video_info_file_path,
            multithreaded_io,
            dataset_type,
        )

        self._actions: Dict[str, List[ActionData]] = load_dataclass_dict_from_csv(
            actions_file_path, ActionData, "video_id", list_per_key=True
        )
        # Sample datapoints
        self._clips: List[EpicKitchenClip] = clip_sampler(self._videos, self._actions)

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

        clip_data = {
            "video_id": clip.video_id,
            **self._videos[clip.video_id].get_clip(
                clip.start_time, clip.stop_time, self._frame_filter
            ),
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

    @staticmethod
    def _load_videos(
        video_data_manifest_file_path: Optional[str],
        video_info_file_path: str,
        multithreaded_io: bool,
        dataset_type: EpicKitchenDatasetType,
    ) -> Dict[str, Video]:
        video_infos: Dict[str, VideoInfo] = load_dataclass_dict_from_csv(
            video_info_file_path, VideoInfo, "video_id"
        )
        if dataset_type == EpicKitchenDatasetType.Frame:
            return EpicKitchenDataset._load_frame_videos(
                video_data_manifest_file_path, video_infos, multithreaded_io
            )
        elif dataset_type == EpicKitchenDatasetType.EncodedVideo:
            return EpicKitchenDataset._load_encoded_videos(
                video_data_manifest_file_path, video_infos
            )

    @staticmethod
    def _load_frame_videos(
        frame_manifest_file_path: str,
        video_infos: Dict[str, VideoInfo],
        multithreaded_io: bool,
    ):
        video_frames: Dict[str, VideoFrameInfo] = load_dataclass_dict_from_csv(
            frame_manifest_file_path, VideoFrameInfo, "video_id"
        )
        EpicKitchenDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames, video_infos
        )
        return {
            video_id: FrameVideo(
                video_frame_to_path_fn=EpicKitchenDataset._frame_number_to_filepath_generator(
                    video_id, video_frames, video_infos
                ),
                duration=video_infos[video_id].duration,
                fps=video_infos[video_id].fps,
                multithreaded_io=multithreaded_io,
            )
            for video_id in video_infos
        }

    @staticmethod
    def _load_encoded_videos(
        encoded_video_manifest_file_path: str, video_infos: Dict[str, VideoInfo]
    ):
        encoded_video_infos: Dict[str, EncodedVideoInfo] = load_dataclass_dict_from_csv(
            encoded_video_manifest_file_path, EncodedVideoInfo, "video_id"
        )
        EpicKitchenDataset._remove_video_info_missing_or_incomplete_videos(
            encoded_video_infos, video_infos
        )

        return {
            video_id: EncodedVideo(encoded_video_info.file_path)
            for video_id, encoded_video_info in encoded_video_infos.items()
        }

    @staticmethod
    def _frame_number_to_filepath_generator(
        video_id: str,
        video_frames: Dict[str, VideoFrameInfo],
        video_infos: Dict[str, VideoInfo],
    ) -> Optional[str]:
        video_info = video_infos[video_id]
        video_frame_info = video_frames[video_info.video_id]

        def frame_number_to_file_path(frame_index: int):
            frame_number = frame_index + video_frame_info.min_frame_number
            if (
                frame_number < video_frame_info.min_frame_number
                or frame_number > video_frame_info.max_frame_number
            ):
                return None

            frame_path_index = str(frame_number)
            frame_prefix = video_frame_info.frame_file_stem
            num_zero_pad = (
                video_frame_info.frame_string_length
                - len(frame_path_index)
                - len(frame_prefix)
            )
            zero_padding = "0" * num_zero_pad
            frame_component = (
                f"{frame_prefix}{zero_padding}{frame_path_index}"
                f".{video_frame_info.file_extension}"
            )
            return f"{video_frame_info.location}/{frame_component}"

        return frame_number_to_file_path

    @staticmethod
    def _remove_video_info_missing_or_incomplete_videos(
        video_data_infos: Dict[str, Union[VideoFrameInfo, EncodedVideoInfo]],
        video_infos: Dict[str, VideoInfo],
    ) -> None:
        # Avoid deletion keys from dict during iteration over keys
        video_ids = list(video_infos)
        for video_id in video_ids:
            video_info = video_infos[video_id]

            # Remove videos we have metadata for but don't have video data
            if video_id not in video_data_infos:
                del video_infos[video_id]
                continue

            # Remove videos we have metadata for but don't have the right number of frames
            if type(video_data_infos[video_id]) == VideoFrameInfo:
                video_frames_info = video_data_infos[video_id]
                expected_frames = round(video_info.duration * video_info.fps)
                num_frames = (
                    video_frames_info.max_frame_number
                    - video_frames_info.min_frame_number
                )
                if abs(num_frames - expected_frames) > video_info.fps:
                    del video_data_infos[video_id]
                    del video_infos[video_id]

        video_ids = list(video_data_infos)  # Avoid modifying dict during iteration
        for video_id in video_ids:
            # Remove videos we have video data for but don't have metadata
            if video_id not in video_infos:

                del video_data_infos[video_id]
