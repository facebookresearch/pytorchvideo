# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Type

import torch
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import (
    ClipInfo,
)
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset


logger = logging.getLogger(__name__)


def video_only_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
):
    """
    Builds a LabeledVideoDataset with no annotations from a json file with the following
    format:

        .. code-block:: text

            {
              "video_name1": {...}
              "video_name2": {...}
              ....
              "video_nameN": {...}
            }

    Args:
        labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                video file paths and associated labels. If video paths are a folder
                it's interpreted as a frame video, otherwise it must be an encoded
                video.

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
            video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
            video container. This defines the order videos are decoded and,
            if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
            the clip is returned. It can be used for user defined preprocessing and
            augmentations on the clips. The clip output format is described in __next__().

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video. Not used for
            frame videos.
    """

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.json_dataset.video_only_dataset")

    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                annotations = json.load(f)
        except Exception:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset")

        # LabeledVideoDataset requires the data to be list of tuples with format:
        # (video_paths, annotation_dict), for no annotations we just pass in an empty dict.
        video_paths = [
            (os.path.join(video_path_prefix, x), {}) for x in annotations.keys()
        ]
    else:
        raise FileNotFoundError(f"{data_path} not found.")

    dataset = LabeledVideoDataset(
        video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset


def clip_recognition_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
):
    """
    Builds a LabeledVideoDataset with noun, verb annotations from a json file with the following
    format:

        .. code-block:: text

            {
              "video_name1": {
                  {
                    "benchmarks": {
                        "forecasting_hands_objects": [
                            {
                                "critical_frame_selection_parent_start_sec": <start_sec>
                                "critical_frame_selection_parent_end_sec": <end_sec>
                                {
                                    "taxonomy: {
                                        "noun": <label>,
                                        "verb": <label>,
                                    }
                                }
                            },
                            {
                                ...
                            }
                        ]
                    }
                  }
              }
              "video_name2": {...}
              ....
              "video_nameN": {...}
            }

    Args:
        labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                video file paths and associated labels. If video paths are a folder
                it's interpreted as a frame video, otherwise it must be an encoded
                video.

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
            video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
            video container. This defines the order videos are decoded and,
            if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
            the clip is returned. It can be used for user defined preprocessing and
            augmentations on the clips. The clip output format is described in __next__().

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video. Not used for
            frame videos.
    """
    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                annotations = json.load(f)
        except Exception:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset")

        # LabeledVideoDataset requires the data to be list of tuples with format:
        # (video_paths, annotation_dict), for no annotations we just pass in an empty dict.
        untrimmed_clip_annotations = []
        for video_name, child in annotations.items():
            video_path = os.path.join(video_path_prefix, video_name)
            for clip_annotation in child["benchmarks"]["forecasting_hands_objects"]:
                clip_start = clip_annotation[
                    "critical_frame_selection_parent_start_sec"
                ]
                clip_end = clip_annotation["critical_frame_selection_parent_end_sec"]
                taxonomy = clip_annotation["taxonomy"]
                noun_label = taxonomy["noun"]
                verb_label = taxonomy["verb"]
                verb_unsure = taxonomy["verb_unsure"]
                noun_unsure = taxonomy["noun_unsure"]
                if (
                    noun_label is None
                    or verb_label is None
                    or verb_unsure
                    or noun_unsure
                ):
                    continue

                untrimmed_clip_annotations.append(
                    (
                        video_path,
                        {
                            "clip_start_sec": clip_start,
                            "clip_end_sec": clip_end,
                            "noun_label": noun_label,
                            "verb_label": verb_label,
                        },
                    )
                )
    else:
        raise FileNotFoundError(f"{data_path} not found.")

    # Map noun and verb key words to unique index.
    def map_labels_to_index(label_name):
        labels = list({info[label_name] for _, info in untrimmed_clip_annotations})
        label_to_idx = {label: i for i, label in enumerate(labels)}
        for i in range(len(untrimmed_clip_annotations)):
            label = untrimmed_clip_annotations[i][1][label_name]
            untrimmed_clip_annotations[i][1][label_name] = label_to_idx[label]

    map_labels_to_index("noun_label")
    map_labels_to_index("verb_label")

    dataset = LabeledVideoDataset(
        untrimmed_clip_annotations,
        UntrimmedClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset


class UntrimmedClipSampler:
    """
    A wrapper for adapting untrimmed annotated clips from the json_dataset to the
    standard `pytorchvideo.data.ClipSampler` expected format. Specifically, for each
    clip it uses the provided `clip_sampler` to sample between "clip_start_sec" and
    "clip_end_sec" from the json_dataset clip annotation.
    """

    def __init__(self, clip_sampler: ClipSampler) -> None:
        """
        Args:
            clip_sampler (`pytorchvideo.data.ClipSampler`): Strategy used for sampling
                between the untrimmed clip boundary.
        """
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> ClipInfo:
        clip_start_boundary = clip_info["clip_start_sec"]
        clip_end_boundary = clip_info["clip_end_sec"]
        duration = clip_start_boundary - clip_end_boundary

        # Sample between 0 and duration of untrimmed clip, then add back start boundary.
        clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info)
        return ClipInfo(
            clip_info.clip_start_sec + clip_start_boundary,
            clip_info.clip_end_sec + clip_start_boundary,
            clip_info.clip_index,
            clip_info.aug_index,
            clip_info.is_last_clip,
        )
