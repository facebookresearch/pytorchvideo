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


@dataclass
class ImageFrameInfo(DataclassFieldCaster):
    """
    Class representing the metadata (and labels) for a single frame
    """

    video_id: str
    frame_id: str
    frame_number: int
    frame_file_path: str


class VideoDatasetType(Enum):
    Frame = 1
    EncodedVideo = 2


class ImageDataset:
    @staticmethod
    def _load_images(
        frame_manifest_file_path: Optional[str],
        video_info_file_path: str,
        multithreaded_io: bool,
    ) -> Dict[str, ImageFrameInfo]:
        """
        Load image frame information from data files and create a dictionary of ImageFrameInfo objects.

        This static method reads information about image frames from data files specified by
        'frame_manifest_file_path' and 'video_info_file_path' and organizes it into a dictionary
        of ImageFrameInfo objects. It ensures consistency and completeness of data between video
        information and frame information.

        Args:
            frame_manifest_file_path (str or None): The file path to the manifest containing frame information.
                If None, frame information will not be loaded.
            video_info_file_path (str): The file path to the CSV file containing video information.
            multithreaded_io (bool): A flag indicating whether to use multithreaded I/O operations.

        Returns:
            Dict[str, ImageFrameInfo]: A dictionary where the keys are frame IDs, and the values
            are ImageFrameInfo objects containing information about each image frame.

        Note:
            - If 'frame_manifest_file_path' is None, frame information will not be loaded.
            - The 'frame_manifest_file_path' and 'video_info_file_path' CSV files must have a common
              key for matching video and frame data.

        """
        video_infos: Dict[str, VideoInfo] = load_dataclass_dict_from_csv(
            video_info_file_path, VideoInfo, "video_id"
        )
        video_frames: Dict[str, VideoFrameInfo] = load_dataclass_dict_from_csv(
            frame_manifest_file_path, VideoFrameInfo, "video_id"
        )
        VideoDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames, video_infos
        )

        image_infos = {}
        for video_id in video_infos:
            frame_filepaths = VideoDataset._frame_number_to_filepaths(
                video_id, video_frames, video_infos
            )
            video_info = video_infos[video_id]
            video_frame_info = video_frames[video_info.video_id]
            for frame_filepath, frame_number in zip(
                frame_filepaths,
                range(
                    video_frame_info.min_frame_number, video_frame_info.max_frame_number
                ),
            ):
                frame_id = os.path.splitext(os.path.basename(frame_filepath))[0]
                image_infos[frame_id] = ImageFrameInfo(
                    video_id, frame_id, frame_number, frame_filepath
                )
        return image_infos


class VideoDataset:
    @staticmethod
    def _load_videos(
        video_data_manifest_file_path: Optional[str],
        video_info_file_path: str,
        multithreaded_io: bool,
        dataset_type: VideoDatasetType,
    ) -> Dict[str, Video]:
        """
        Load videos or frame data and create a dictionary of Video objects.

        This static method loads video data or frame information from specified data files and organizes
        it into a dictionary of Video objects. The type of dataset loaded depends on the 'dataset_type'
        parameter.

        Args:
            video_data_manifest_file_path (str or None): The file path to the manifest containing video or
                frame data. If None, video data or frame data will not be loaded.
            video_info_file_path (str): The file path to the CSV file containing video information.
            multithreaded_io (bool): A flag indicating whether to use multithreaded I/O operations.
            dataset_type (VideoDatasetType): The type of dataset to load, either Frame or EncodedVideo.

        Returns:
            Dict[str, Video]: A dictionary where the keys are video IDs, and the values are Video objects.

        Note:
            - If 'video_data_manifest_file_path' is None, video data or frame data will not be loaded.
            - The 'video_data_manifest_file_path' and 'video_info_file_path' CSV files must have a common
              key for matching video and frame data.
        """

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
        """
        Load frame videos and create a dictionary of FrameVideo objects.

        This static method loads frame video data from the specified frame manifest file and organizes it
        into a dictionary of FrameVideo objects. It ensures consistency and completeness of data between
        video information and frame information.

        Args:
            frame_manifest_file_path (str): The file path to the manifest containing frame information.
            video_infos (Dict[str, VideoInfo]): A dictionary of video information keyed by video ID.
            multithreaded_io (bool): A flag indicating whether to use multithreaded I/O operations.

        Returns:
            Dict[str, FrameVideo]: A dictionary where the keys are video IDs, and the values are FrameVideo
            objects containing frame video data.
        """
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
        """
        Load encoded videos and create a dictionary of EncodedVideo objects.

        This static method loads encoded video data from the specified encoded video manifest file and
        organizes it into a dictionary of EncodedVideo objects. It ensures consistency and completeness of
        data between video information and encoded video information.

        Args:
            encoded_video_manifest_file_path (str): The file path to the manifest containing encoded video
                information.
            video_infos (Dict[str, VideoInfo]): A dictionary of video information keyed by video ID.

        Returns:
            Dict[str, EncodedVideo]: A dictionary where the keys are video IDs, and the values are EncodedVideo
            objects containing encoded video data.
        """
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
        """
        Convert frame numbers to file paths.

        This static method generates file paths for frame numbers based on video frame information and video
        information.

        Args:
            video_id (str): The ID of the video.
            video_frames (Dict[str, VideoFrameInfo]): A dictionary of video frame information keyed by video ID.
            video_infos (Dict[str, VideoInfo]): A dictionary of video information keyed by video ID.

        Returns:
            Optional[str]: A list of file paths for frames or None if frame numbers are invalid.
        """
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
        """
        Remove video information for missing or incomplete videos.

        This static method removes video information for videos that are missing corresponding video data
        or do not have the correct number of frames.

        Args:
            video_data_infos (Dict[str, Union[VideoFrameInfo, EncodedVideoInfo]]): A dictionary of video
                data information keyed by video ID.
            video_infos (Dict[str, VideoInfo]): A dictionary of video information keyed by video ID.
        """
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
    Convert a timestamp of the form 'HH:MM:SS' or 'HH:MM:SS.sss' to seconds.

    Args:
        time_str (str): A string representing a timestamp in the format 'HH:MM:SS' or 'HH:MM:SS.sss'.

    Returns:
        float: The equivalent time in seconds.

    Raises:
        ValueError: If the provided string is not in a valid time format.

    This function parses the input 'time_str' as a timestamp in either 'HH:MM:SS' or 'HH:MM:SS.sss' format.
    It then calculates and returns the equivalent time in seconds as a floating-point number.

    Example:
        - Input: '01:23:45'
          Output: 5025.0 seconds
        - Input: '00:00:01.234'
          Output: 1.234 seconds

    Note:
        - The function supports both fractional seconds and integer seconds.
        - If the input string is not in a valid time format, a ValueError is raised.
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
    Save a dictionary of encoded video information as a CSV file.

    This function takes a dictionary of encoded video information, where keys are video IDs and values
    are EncodedVideoInfo objects, and saves it as a CSV file. The CSV file can be used for future
    reference and data retrieval.

    Args:
        encoded_video_infos (Dict[str, EncodedVideoInfo]):
            A dictionary mapping video IDs to metadata about the location of their encoded video data.
        file_name (str, optional):
            The file name or path where the CSV file will be saved. If not provided, a file name
            will be automatically generated in the current working directory.

    Returns:
        str: The filename where the encoded video information is stored.

    Note:
        - The CSV file will have a header row with column names based on the EncodedVideoInfo data class.
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
    Save a dictionary of video frame information as a CSV file.

    This function takes a dictionary of video frame information, where keys are video IDs and values
    are VideoFrameInfo objects, and saves it as a CSV file. The CSV file can be used for future
    reference and data retrieval.

    Args:
        video_frames (Dict[str, VideoFrameInfo]):
            A dictionary mapping video IDs to metadata about the location of their video frame files.
        file_name (str, optional):
            The file name or path where the CSV file will be saved. If not provided, a file name
            will be automatically generated in the current working directory.

    Returns:
        str: The filename where the video frame information is stored.

    Note:
        - The CSV file will have a header row with column names based on the VideoFrameInfo data class.
    """
    file_name = (
        f"{os.getcwd()}/video_frame_metadata.csv" if file_name is None else file_name
    )
    save_dataclass_objs_to_headered_csv(list(video_frames.values()), file_name)
    return file_name
