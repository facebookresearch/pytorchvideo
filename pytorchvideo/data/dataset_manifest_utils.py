# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.frame_video import FrameVideo
from pytorchvideo.data.utils import (
    DataclassFieldCaster,
    load_dataclass_dict_from_csv,
    save_dataclass_objs_to_headered_csv,
)
from pytorchvideo.data.video import Video


@dataclass
class EncodedVideoInfo(DataclassFieldCaster):
    """
    Class representing the location of an available encoded video.
    """

    video_id: str
    file_path: str


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
class VideoInfo(DataclassFieldCaster):
    """
    Class representing the video-level metadata of a video from an arbitrary video dataset.
    """

    video_id: str
    resolution: str
    duration: float
    fps: float


@dataclass
class VideoClipInfo(DataclassFieldCaster):
    video_id: str
    start_time: float
    stop_time: float


class VideoDatasetType(Enum):
    Frame = 1
    EncodedVideo = 2


class VideoDataset:
    @staticmethod
    def _load_videos(
        video_data_manifest_file_path: Optional[str],
        video_info_file_path: str,
        multithreaded_io: bool,
        dataset_type: VideoDatasetType,
    ) -> Dict[str, Video]:
        video_infos: Dict[str, VideoInfo] = load_dataclass_dict_from_csv(
            video_info_file_path, VideoInfo, "video_id"
        )
        if dataset_type == VideoDatasetType.Frame:
            return VideoDataset._load_frame_videos(
                video_data_manifest_file_path, video_infos, multithreaded_io
            )
        elif dataset_type == VideoDatasetType.EncodedVideo:
            return VideoDataset._load_encoded_videos(
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
        VideoDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames, video_infos
        )
        return {
            video_id: FrameVideo(
                video_frame_paths=VideoDataset._frame_number_to_filepaths(
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
        encoded_video_manifest_file_path: str,
        video_infos: Dict[str, VideoInfo],
    ):
        encoded_video_infos: Dict[str, EncodedVideoInfo] = load_dataclass_dict_from_csv(
            encoded_video_manifest_file_path, EncodedVideoInfo, "video_id"
        )
        VideoDataset._remove_video_info_missing_or_incomplete_videos(
            encoded_video_infos, video_infos
        )

        return {
            video_id: EncodedVideo.from_path(encoded_video_info.file_path)
            for video_id, encoded_video_info in encoded_video_infos.items()
        }

    @staticmethod
    def _frame_number_to_filepaths(
        video_id: str,
        video_frames: Dict[str, VideoFrameInfo],
        video_infos: Dict[str, VideoInfo],
    ) -> Optional[str]:
        video_info = video_infos[video_id]
        video_frame_info = video_frames[video_info.video_id]

        frame_filepaths = []
        num_frames = (
            video_frame_info.max_frame_number - video_frame_info.min_frame_number + 1
        )
        for frame_index in range(num_frames):
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
            frame_filepaths.append(f"{video_frame_info.location}/{frame_component}")
        return frame_filepaths

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


def get_seconds_from_hms_time(time_str: str) -> float:
    """
    Get Seconds from timestamp of form 'HH:MM:SS'.

    Args:
        time_str (str)

    Returns:
        float of seconds

    """
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            time_since_min_time = datetime.datetime.strptime(time_str, fmt)
            min_time = datetime.datetime.strptime("", "")
            return float((time_since_min_time - min_time).total_seconds())
        except ValueError:
            pass
    raise ValueError(f"No valid data format found for provided string {time_str}.")


def save_encoded_video_manifest(
    encoded_video_infos: Dict[str, EncodedVideoInfo], file_name: str = None
) -> str:
    """
    Saves the encoded video dictionary as a csv file that can be read for future usage.

    Args:
        video_frames (Dict[str, EncodedVideoInfo]):
            Dictionary mapping video_ids to metadata about the location of
            their video data.

        file_name (str):
            location to save file (will be automatically generated if None).

    Returns:
        string of the filename where the video info is stored.
    """
    file_name = (
        f"{os.getcwd()}/encoded_video_manifest.csv" if file_name is None else file_name
    )
    save_dataclass_objs_to_headered_csv(list(encoded_video_infos.values()), file_name)
    return file_name


def save_video_frame_info(
    video_frames: Dict[str, VideoFrameInfo], file_name: str = None
) -> str:
    """
    Saves the video frame dictionary as a csv file that can be read for future usage.

    Args:
        video_frames (Dict[str, VideoFrameInfo]):
            Dictionary mapping video_ids to metadata about the location of
            their video frame files.

        file_name (str):
            location to save file (will be automatically generated if None).

    Returns:
        string of the filename where the video info is stored.
    """
    file_name = (
        f"{os.getcwd()}/video_frame_metadata.csv" if file_name is None else file_name
    )
    save_dataclass_objs_to_headered_csv(list(video_frames.values()), file_name)
    return file_name
