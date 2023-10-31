# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image
from pytorchvideo.data.dataset_manifest_utils import (
    ImageDataset,
    ImageFrameInfo,
    VideoClipInfo,
    VideoDataset,
    VideoDatasetType,
)
from pytorchvideo.data.utils import DataclassFieldCaster, load_dataclass_dict_from_csv
from pytorchvideo.data.video import Video


try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True


USER_ENVIRONMENT_MAP = {
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


class LabelType(Enum):
    Environment = 1
    Activity = 2
    UserAttention = 3


LABEL_TYPE_2_MAP = {
    LabelType.Environment: USER_ENVIRONMENT_MAP,
    LabelType.Activity: USER_ACTIVITY_MAP,
    LabelType.UserAttention: USER_ATTENTION_MAP,
}


@dataclass
class LabelData(DataclassFieldCaster):
    """
    Represents a continuous label for a video segment from the DoMSEV dataset.

    Attributes:
        video_id (str): The unique identifier of the video.
        start_time (float): The start time of the label, in seconds.
        stop_time (float): The stop time of the label, in seconds.
        start_frame (int): The 0-indexed ID of the start frame (inclusive).
        stop_frame (int): The 0-indexed ID of the stop frame (inclusive).
        label_id (int): The unique identifier of the label.
        label_name (str): The name of the label.
    """

    video_id: str
    start_time: float  # Start time of the label, in seconds
    stop_time: float  # Stop time of the label, in seconds
    start_frame: int  # 0-indexed ID of the start frame (inclusive)
    stop_frame: int  # 0-index ID of the stop frame (inclusive)
    label_id: int
    label_name: str


# Utility functions
def _seconds_to_frame_index(
    time_in_seconds: float, fps: int, zero_indexed: Optional[bool] = True
) -> int:
    """
    Converts a point in time (in seconds) within a video clip to its closest
    frame index (rounding down), based on a specified frame rate.

    Args:
        time_in_seconds (float): The point in time within the video.
        fps (int): The frame rate (frames per second) of the video.
        zero_indexed (Optional[bool]): Whether the returned frame should be
            zero-indexed (if True) or one-indexed (if False).

    Returns:
        int: The index of the nearest frame (rounding down to the nearest integer).
    """
    frame_idx = math.floor(time_in_seconds * fps)
    if not zero_indexed:
        frame_idx += 1
    return frame_idx


def _get_overlap_for_time_range_pair(
    t1_start: float, t1_stop: float, t2_start: float, t2_stop: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the overlap between two time ranges, if one exists.

    Args:
        t1_start (float): The start time of the first time range.
        t1_stop (float): The stop time of the first time range.
        t2_start (float): The start time of the second time range.
        t2_stop (float): The stop time of the second time range.

    Returns:
        Optional[Tuple[float, float]]: A tuple of <overlap_start_time, overlap_stop_time> if
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


class DomsevFrameDataset(torch.utils.data.Dataset):
    """
    Dataset for frame-based egocentric video classification using the DoMSEV dataset.
    `DoMSEV <https://www.verlab.dcc.ufmg.br/semantic-hyperlapse/cvpr2018-dataset/>`_

    This dataset handles loading, decoding, and configurable sampling of image frames.

    Args:
        video_data_manifest_file_path (str): Path to a JSON file outlining available video data
            for associated videos.
        video_info_file_path (str): Path or URI to a manifest with basic metadata for each video.
        labels_file_path (str): Path or URI to a manifest with temporal annotations for each video.
        transform (Optional[Callable[[Dict[str, Any]], Any]]): A callable for custom preprocessing
            and augmentations to apply to the clips. Default is None.
        multithreaded_io (bool): Control whether IO operations are performed across multiple threads.
            Default is False.

    Attributes:
        _labels_per_frame (Dict[str, int]): A mapping of frame IDs to their corresponding label IDs.
        _user_transform (Optional[Callable[[Dict[str, Any]], Any]]): User-defined transform function.
        _transform (Callable[[Dict[str, Any]], Dict[str, Any]]): Default transformation function.
        _frames (List[ImageFrameInfo]): List of image frame information.

    Methods:
        __getitem__(self, index) -> Dict[str, Any]: Sample an image frame associated with the given index.
        __len__(self) -> int: Get the number of frames in the dataset.
    """

    def __init__(
        self,
        video_data_manifest_file_path: str,
        video_info_file_path: str,
        labels_file_path: str,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        multithreaded_io: bool = False,
    ) -> None:
        """
        Args:
            video_data_manifest_file_path (str):
                The path to a json file outlining the available video data for the
                associated videos.  File must be a csv (w/header) with columns:
                ``{[f.name for f in dataclass_fields(EncodedVideoInfo)]}``

                To generate this file from a directory of video frames, see helper
                functions in module: ``pytorchvideo.data.domsev.utils``

            video_info_file_path (str):
                Path or URI to manifest with basic metadata of each video.
                File must be a csv (w/header) with columns:
                ``{[f.name for f in dataclass_fields(VideoInfo)]}``

            labels_file_path (str):
                Path or URI to manifest with temporal annotations for each video.
                File must be a csv (w/header) with columns:
                ``{[f.name for f in dataclass_fields(LabelData)]}``

            dataset_type (VideoDatasetType): The data format in which dataset
                video data is stored (e.g. video frames, encoded video etc).

            transform (Optional[Callable[[Dict[str, Any]], Any]]):
                This callable is evaluated on the clip output before the clip is returned.
                It can be used for user-defined preprocessing and augmentations to the clips.
                The clip output format is described in __next__().

            multithreaded_io (bool):
                Boolean to control whether io operations are performed across multiple
                threads.
        """
        assert video_info_file_path
        assert labels_file_path
        assert video_data_manifest_file_path

        ## Populate image frame and metadata data providers ##
        # Maps a image frame ID to an `ImageFrameInfo`
        frames_dict: Dict[str, ImageFrameInfo] = ImageDataset._load_images(
            video_data_manifest_file_path,
            video_info_file_path,
            multithreaded_io,
        )
        video_labels: Dict[str, List[LabelData]] = load_dataclass_dict_from_csv(
            labels_file_path, LabelData, "video_id", list_per_key=True
        )
        # Maps an image frame ID to the singular frame label
        self._labels_per_frame: Dict[
            str, int
        ] = DomsevFrameDataset._assign_labels_to_frames(frames_dict, video_labels)

        self._user_transform = transform
        self._transform = self._transform_frame

        # Shuffle the frames order for iteration
        self._frames = list(frames_dict.values())
        random.shuffle(self._frames)

    @staticmethod
    def _assign_labels_to_frames(
        frames_dict: Dict[str, ImageFrameInfo],
        video_labels: Dict[str, List[LabelData]],
    ):
        """
        Assign labels to frames based on temporal annotations.

        Args:
            frames_dict (Dict[str, ImageFrameInfo]): Mapping of frame_id to ImageFrameInfo.
            video_labels (Dict[str, List[LabelData]]): Temporal annotations for each video.

        Also unpacks one label per frame.
        Also converts them to class IDs and then a tensor.
        """
        labels_per_frame: Dict[str, int] = {}
        for frame_id, image_info in frames_dict.items():
            # Filter labels by only the ones that appear within the clip boundaries,
            # and unpack the labels so there is one per frame in the clip
            labels_in_video = video_labels[image_info.video_id]
            for label in labels_in_video:
                if (image_info.frame_number >= label.start_frame) and (
                    image_info.frame_number <= label.stop_frame
                ):
                    labels_per_frame[frame_id] = label.label_id

        return labels_per_frame

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Get an image frame and associated information at the specified index.

        Args:
            index (int): Index for the image frame.

        Returns:
            Dict[str, Any]: Information about the image frame and its label.

            .. code-block:: text

                {{
                    'frame_id': <str>,
                    'image': <image_tensor>,
                    'label': <label_tensor>,
                }}
        """
        frame = self._frames[index]
        label_in_frame = self._labels_per_frame[frame.frame_id]

        image_data = _load_image_from_path(frame.frame_file_path)

        frame_data = {
            "frame_id": frame.frame_id,
            "image": image_data,
            "label": label_in_frame,
        }

        if self._transform:
            frame_data = self._transform(frame_data)

        return frame_data

    def __len__(self) -> int:
        """
        Get the number of frames in the dataset.

        Returns:
            int: The number of frames.
        """
        return len(self._frames)

    def _transform_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transformations to a given image frame.

        Args:
            frame (Dict[str, Any]): Information about the image frame.

        Returns:
            Dict[str, Any]: Transformed information about the image frame.
        """
        for key in frame:
            if frame[key] is None:
                frame[key] = torch.tensor([])

        if self._user_transform:
            frame = self._user_transform(frame)

        return frame


class DomsevVideoDataset(torch.utils.data.Dataset):
    """
    Egocentric classification video clip-based dataset for
    `DoMSEV <https://www.verlab.dcc.ufmg.br/semantic-hyperlapse/cvpr2018-dataset/>`_
    stored as an encoded video (with frame-level labels).

    This dataset handles the loading, decoding, and configurable clip
    sampling for the videos.
    """

    def __init__(
        self,
        video_data_manifest_file_path: str,
        video_info_file_path: str,
        labels_file_path: str,
        clip_sampler: Callable[
            [Dict[str, Video], Dict[str, List[LabelData]]], List[VideoClipInfo]
        ],
        dataset_type: VideoDatasetType = VideoDatasetType.Frame,
        frames_per_second: int = 1,
        transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        frame_filter: Optional[Callable[[List[int]], List[int]]] = None,
        multithreaded_io: bool = False,
    ) -> None:
        """
        Args:
            video_data_manifest_file_path (str):
                The path to a json file outlining the available video data for the
                associated videos.  File must be a csv (w/header) with columns:
                ``{[f.name for f in dataclass_fields(EncodedVideoInfo)]}``

                To generate this file from a directory of video frames, see helper
                functions in module: ``pytorchvideo.data.domsev.utils``

            video_info_file_path (str):
                Path or URI to manifest with basic metadata of each video.
                File must be a csv (w/header) with columns:
                ``{[f.name for f in dataclass_fields(VideoInfo)]}``

            labels_file_path (str):
                Path or URI to manifest with annotations for each video.
                File must be a csv (w/header) with columns:
                ``{[f.name for f in dataclass_fields(LabelData)]}``

            clip_sampler (Callable[[Dict[str, Video], Dict[str, List[LabelData]]],
                List[VideoClipInfo]]):
                Defines how clips should be sampled from each video. See the clip
                sampling documentation for more information.

            dataset_type (VideoDatasetType): The data format in which dataset
                video data is stored (e.g. video frames, encoded video etc).

            frames_per_second (int): The FPS of the stored videos. (NOTE:
                this is variable and may be different than the original FPS
                reported on the DoMSEV dataset website -- it depends on the
                preprocessed subsampling and frame extraction).

            transform (Optional[Callable[[Dict[str, Any]], Any]]):
                This callable is evaluated on the clip output before the clip is returned.
                It can be used for user-defined preprocessing and augmentations to the clips.
                The clip output format is described in __next__().

            frame_filter (Optional[Callable[[List[int]], List[int]]]):
                This callable is evaluated on the set of available frame indices to be
                included in a sampled clip. This can be used to subselect frames within
                a clip to be loaded.

            multithreaded_io (bool):
                Boolean to control whether io operations are performed across multiple
                threads.
        """
        assert video_info_file_path
        assert labels_file_path
        assert video_data_manifest_file_path

        # Populate video and metadata data providers
        self._videos: Dict[str, Video] = VideoDataset._load_videos(
            video_data_manifest_file_path,
            video_info_file_path,
            multithreaded_io,
            dataset_type,
        )

        self._labels_per_video: Dict[
            str, List[LabelData]
        ] = load_dataclass_dict_from_csv(
            labels_file_path, LabelData, "video_id", list_per_key=True
        )

        # Sample datapoints
        self._clips: List[VideoClipInfo] = clip_sampler(
            self._videos, self._labels_per_video
        )

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
            A video clip with the following format if transform is None.

            .. code-block:: text

                {{
                    'video_id': <str>,
                    'video': <video_tensor>,
                    'audio': <audio_tensor>,
                    'labels': <labels_tensor>,
                    'start_time': <float>,
                    'stop_time': <float>
                }}
        """
        clip = self._clips[index]

        # Filter labels by only the ones that appear within the clip boundaries,
        # and unpack the labels so there is one per frame in the clip
        labels_in_video = self._labels_per_video[clip.video_id]
        labels_in_clip = []
        for label_data in labels_in_video:
            overlap_period = _get_overlap_for_time_range_pair(
                clip.start_time,
                clip.stop_time,
                label_data.start_time,
                label_data.stop_time,
            )
            if overlap_period is not None:
                overlap_start_time, overlap_stop_time = overlap_period

                # Convert the overlapping period between clip and label to
                # 0-indexed start and stop frame indexes, so we can unpack 1
                # label per frame.
                overlap_start_frame = _seconds_to_frame_index(
                    overlap_start_time, self._frames_per_second
                )
                overlap_stop_frame = _seconds_to_frame_index(
                    overlap_stop_time, self._frames_per_second
                )

                # Append 1 label per frame
                for _ in range(overlap_start_frame, overlap_stop_frame):
                    labels_in_clip.append(label_data)

        # Convert the list of LabelData objects to a tensor of just the label IDs
        label_ids = [labels_in_clip[i].label_id for i in range(len(labels_in_clip))]
        label_ids_tensor = torch.tensor(label_ids)

        clip_data = {
            "video_id": clip.video_id,
            **self._videos[clip.video_id].get_clip(clip.start_time, clip.stop_time),
            "labels": label_ids_tensor,
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
        """
        Transforms a given video clip, according to some pre-defined transforms
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


def _load_image_from_path(image_path: str, num_retries: int = 10) -> Image:
    """
    Load an image from the given file path and decode it as an RGB image.

    Args:
        image_path (str): The path to the image file.
        num_retries (int): The number of times to retry image reading to handle transient errors.

    Returns:
        Image: A PIL Image representing the loaded image in RGB format.
            The image has the shape (channel, height, width) and pixel values in the range [0, 255].

    Raises:
        Exception: If unable to load the image after the specified number of retries.
    """
    if not _HAS_CV2:
        raise ImportError(
            "opencv2 is required to use FrameVideo. Please "
            "install with 'pip install opencv-python'"
        )

    img_arr = None

    for i in range(num_retries):
        with g_pathmgr.open(image_path, "rb") as f:
            img_str = np.frombuffer(f.read(), np.uint8)
            img_bgr = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb is not None:
            img_arr = img_rgb
            break
        else:
            logging.warning(f"Reading attempt {i}/{num_retries} failed.")
            time.sleep(1e-6)

    if img_arr is None:
        raise Exception("Failed to load image from {}".format(image_path))

    pil_image = Image.fromarray(img_arr)
    return pil_image
