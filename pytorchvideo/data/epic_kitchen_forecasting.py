# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
    Random = 1


class EpicKitchenForecasting(EpicKitchenDataset):
    """
    Action forecasting video data set for EpicKitchen-55 Dataset.
    <https://epic-kitchens.github.io/2019/>

    This dataset handles the loading, decoding, and clip sampling for the videos.
    """

    def __init__(
        self,
        video_info_file_path: str,
        actions_file_path: str,
        video_data_manifest_file_path: str,
        clip_sampling: ClipSampling = ClipSampling.Random,
        dataset_type: VideoDatasetType = VideoDatasetType.Frame,
        seconds_per_clip: float = 2.0,
        clip_time_stride: float = 10.0,
        num_input_clips: int = 1,
        frames_per_clip: Optional[int] = None,
        num_forecast_actions: int = 1,
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

            seconds_per_clip (float): The length of each sampled subclip in seconds.

            clip_time_stride (float): The time difference in seconds between the start of
                each input subclip.

            num_input_clips (int): The number of subclips to be included in the input
                video data.

            frames_per_clip (Optional[int]): The number of frames per clip to sample.
                If None, all frames in the clip will be included.

            num_forecast_actions (int): The number of actions to be included in the
                action vector.

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
            EpicKitchenForecasting._define_clip_structure_generator(
                clip_sampling,
                seconds_per_clip,
                clip_time_stride,
                num_input_clips,
                num_forecast_actions,
            )
        )
        frame_filter = (
            EpicKitchenForecasting._frame_filter_generator(
                frames_per_clip, seconds_per_clip, clip_time_stride, num_input_clips
            )
            if frames_per_clip is not None
            else None
        )
        transform = EpicKitchenForecasting._transform_generator(
            transform, num_forecast_actions, frames_per_clip, num_input_clips
        )

        super().__init__(
            video_info_file_path=video_info_file_path,
            actions_file_path=actions_file_path,
            video_data_manifest_file_path=video_data_manifest_file_path,
            dataset_type=dataset_type,
            transform=transform,
            frame_filter=frame_filter,
            clip_sampler=define_clip_structure_fn,
            multithreaded_io=multithreaded_io,
        )

    @staticmethod
    def _transform_generator(
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
        num_forecast_actions: int,
        frames_per_clip: int,
        num_input_clips: int,
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Args:
            transform (Callable[[Dict[str, Any]], Dict[str, Any]]): A function that performs
            any operation on a clip before it is returned in the default transform function.
            num_forecast_actions: (int) The number of actions to be included in the
                action vector.
            frames_per_clip (int): The number of frames per clip to sample.
            num_input_clips (int): The number of subclips to be included in the video data.

        Returns:
            A function that performs any operation on a clip and returns the transformed clip.
        """

        def transform_clip(clip: Dict[str, Any]) -> Dict[str, Any]:
            assert all(
                clip["actions"][i].start_time <= clip["actions"][i + 1].start_time
                for i in range(len(clip["actions"]) - 1)
            ), "Actions must be sorted"
            next_k_actions: List[ActionData] = [
                a for a in clip["actions"] if (a.start_time > clip["stop_time"])
            ][:num_forecast_actions]
            clip["actions"] = next_k_actions

            assert clip["video"].size()[1] == num_input_clips * frames_per_clip
            clip_video_tensor = torch.stack(
                [
                    clip["video"][
                        :, (i * frames_per_clip) : ((i + 1) * frames_per_clip), :, :
                    ]
                    for i in range(num_input_clips)
                ]
            )
            clip["video"] = clip_video_tensor

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
        seconds_per_clip: float,
        clip_time_stride: float,
        num_input_clips: int,
    ) -> Callable[[List[int]], List[int]]:
        """
        Args:
            frames_per_clip (int): The number of frames per clip to sample.
            seconds_per_clip (float): The length of each sampled subclip in seconds.
            clip_time_stride (float): The time difference in seconds between the start of
                each input subclip.
            num_input_clips (int): The number of subclips to be included in the video data.

        Returns:
            A function that takes in a list of frame indicies and outputs a subsampled list.
        """
        time_window_length = seconds_per_clip + (num_input_clips - 1) * clip_time_stride
        desired_frames_per_second = frames_per_clip / seconds_per_clip

        def frame_filter(frame_indices: List[int]) -> List[int]:
            num_available_frames_for_all_clips = len(frame_indices)
            available_frames_per_second = (
                num_available_frames_for_all_clips / time_window_length
            )
            intra_clip_sampling_stride = int(
                available_frames_per_second // desired_frames_per_second
            )
            selected_frames = set()
            for i in range(num_input_clips):
                clip_start_index = int(
                    i * clip_time_stride * available_frames_per_second
                )
                for j in range(frames_per_clip):
                    selected_frames.add(
                        clip_start_index + j * intra_clip_sampling_stride
                    )
            return [x for i, x in enumerate(frame_indices) if i in selected_frames]

        return frame_filter

    @staticmethod
    def _define_clip_structure_generator(
        clip_sampling: str,
        seconds_per_clip: float,
        clip_time_stride: float,
        num_input_clips: int,
        num_forecast_actions: int,
    ) -> Callable[[Dict[str, Video], Dict[str, List[ActionData]]], List[VideoClipInfo]]:
        """
        Args:
            clip_sampling (ClipSampling):
                The type of sampling to perform to perform on the videos of the dataset.
            seconds_per_clip (float): The length of each sampled clip in seconds.
            clip_time_stride: The time difference in seconds between the start of
                each input subclip.
            num_input_clips (int):  The number of subclips to be included in the video data.
            num_forecast_actions (int): The number of actions to be included in the
                action vector.

        Returns:
            A function that takes a dictionary of videos and outputs a list of sampled
            clips.
        """
        # TODO(T77683480)
        if not clip_sampling == ClipSampling.Random:
            raise NotImplementedError(
                f"Only {ClipSampling.Random} is implemented. "
                f"{clip_sampling} not implemented."
            )

        time_window_length = seconds_per_clip + (num_input_clips - 1) * clip_time_stride

        def define_clip_structure(
            videos: Dict[str, Video], video_actions: Dict[str, List[ActionData]]
        ) -> List[VideoClipInfo]:
            candidate_sample_clips = []
            for video_id, actions in video_actions.items():
                for i, action in enumerate(actions[: (-1 * num_forecast_actions)]):
                    # Only actions with num_forecast_actions after to predict
                    # Confirm there are >= num_forecast_actions available
                    # (it is possible for actions to overlap)
                    number_valid_actions = 0
                    for j in range(i + 1, len(actions)):
                        if actions[j].start_time > action.stop_time:
                            number_valid_actions += 1
                        if number_valid_actions == num_forecast_actions:
                            if (
                                action.start_time - time_window_length >= 0
                            ):  # Only add clips that have the full input video available
                                candidate_sample_clips.append(
                                    VideoClipInfo(
                                        video_id,
                                        action.stop_time - time_window_length,
                                        action.stop_time,
                                    )
                                )
                            break
            return candidate_sample_clips

        return define_clip_structure
