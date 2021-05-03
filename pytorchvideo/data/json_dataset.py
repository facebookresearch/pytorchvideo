# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import os
import pathlib
from typing import Any, Callable, Dict, Optional, Type

import torch
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset


logger = logging.getLogger(__name__)


def video_only_dataset(
    data_path: pathlib.Path,
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
