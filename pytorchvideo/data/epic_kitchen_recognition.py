# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch
from pytorchvideo.data.dataset_manifest_utils import (
    EncodedVideoInfo,
    VideoClipInfo,
    VideoDatasetType,
    VideoFrameInfo,
    VideoInfo,
)
from pytorchvideo.data.epic_kitchen import ActionData, EpicKitchenDataset
from pytorchvideo.data.video import Video


class ClipSampling(Enum):
    RandomOffsetUniform = 1


class EpicKitchenRecognition(EpicKitchenDataset):
    """
    Action recognition video data set for EpicKitchen-55 Dataset.
    <https://epic-kitchens.github.io/2019/>

    This dataset handles the loading, decoding, and clip sampling for the videos.
    """

    def __init__(
        self,
        video_info_file_path: str,
        actions_file_path: str,
        video_data_manifest_file_path: str,
        clip_sampling: ClipSampling = ClipSampling.RandomOffsetUniform,
        dataset_type: VideoDatasetType = VideoDatasetType.Frame,
        seconds_per_clip: float = 2.0,
        frames_per_clip: Optional[int] = None,
        transform: Callable[[Dict[str, Any]], Any] = None,
        multithreaded_io: bool = True,
    ):
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

            video_data_manifest_file_path (str):
                The path to a json file outlining the available video data for the
                associated videos. File must be a csv (w/header) with columns either:

                For Frame Videos:
                {[f.name for f in dataclass_fields(VideoFrameInfo)]}

                For Encoded Videos:
                {[f.name for f in dataclass_fields(EncodedVideoInfo)]}

                To generate this file from a directory of video frames, see helper
                functions in Module: pytorchvideo.data.epic_kitchen.utils

            clip_sampling (ClipSampling):
                The type of sampling to perform to perform on the videos of the dataset.

            dataset_type (VideoDatasetType): The dataformat in which dataset
                video data is store (e.g. video frames, encoded video etc).

            seconds_per_clip (float): The length of each sampled clip in seconds.

            frames_per_clip (Optional[int]): The number of frames per clip to sample.

            transform (Callable[[Dict[str, Any]], Any]):
                This callable is evaluated on the clip output before the clip is returned.
                It can be used for user-defined preprocessing and augmentations to the clips.
                The clip input is a dictionary with the following format:
                    {{
                        'video_id': <str>,
                        'video': <video_tensor>,
                        'audio': <audio_tensor>,
                        'label': <List[ActionData]>,
                        'start_time': <float>,
                        'stop_time': <float>
                    }}

                If transform is None, the raw clip output in the above format is
                    returned unmodified.

            multithreaded_io (bool):
                Boolean to control whether parllelizable io operations are performed across
                multiple threads.
        """
        define_clip_structure_fn = (
            EpicKitchenRecognition._define_clip_structure_generator(
                seconds_per_clip, clip_sampling
            )
        )
        transform = EpicKitchenRecognition._transform_generator(transform)
        frame_filter = (
            EpicKitchenRecognition._frame_filter_generator(frames_per_clip)
            if frames_per_clip is not None
            else None
        )

        super().__init__(
            video_info_file_path=video_info_file_path,
            actions_file_path=actions_file_path,
            dataset_type=dataset_type,
            video_data_manifest_file_path=video_data_manifest_file_path,
            transform=transform,
            frame_filter=frame_filter,
            clip_sampler=define_clip_structure_fn,
            multithreaded_io=multithreaded_io,
        )

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
            actions_in_clip: List[ActionData] = [
                a
                for a in clip["actions"]
                if (
                    a.start_time <= clip["stop_time"]
                    and a.stop_time >= clip["start_time"]
                )
            ]
            clip["actions"] = actions_in_clip

            for key in clip:
                if clip[key] is None:
                    clip[key] = torch.tensor([])

            if transform:
                clip = transform(clip)

            return clip

        return transform_clip

    @staticmethod
    def _frame_filter_generator(
        frames_per_clip: int,
    ) -> Callable[[List[int]], List[int]]:
        """
        Args:
            frames_per_clip (int): The number of frames per clip to sample.

        Returns:
            A function that takes in a list of frame indicies and outputs a subsampled list.
        """

        def frame_filer(frame_indices: List[int]) -> List[int]:
            num_frames = len(frame_indices)
            frame_step = int(num_frames // frames_per_clip)
            selected_frames = set(range(0, num_frames, frame_step))
            return [x for i, x in enumerate(frame_indices) if i in selected_frames]

        return frame_filer

    @staticmethod
    def _define_clip_structure_generator(
        seconds_per_clip: float, clip_sampling: ClipSampling
    ) -> Callable[[Dict[str, Video], Dict[str, List[ActionData]]], List[VideoClipInfo]]:
        """
        Args:
            seconds_per_clip (float): The length of each sampled clip in seconds.
            clip_sampling (ClipSampling):
                The type of sampling to perform to perform on the videos of the dataset.

        Returns:
            A function that takes a dictionary of videos and a dictionary of the actions
            for each video and outputs a list of sampled clips.
        """
        if not clip_sampling == ClipSampling.RandomOffsetUniform:
            raise NotImplementedError(
                f"Only {ClipSampling.RandomOffsetUniform} is implemented. "
                f"{clip_sampling} not implemented."
            )

        def define_clip_structure(
            videos: Dict[str, Video], actions: Dict[str, List[ActionData]]
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
