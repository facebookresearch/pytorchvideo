# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.utils import optional_threaded_foreach

from .utils import thwc_to_cthw
from .video import Video


logger = logging.getLogger(__name__)


class FrameVideo(Video):
    """
    FrameVideo is an abstractions for accessing clips based on their start and end
    time for a video where each frame is stored as an image. PathManager is used for
    frame image reading, allowing non-local uri's to be used.
    """

    def __init__(
        self,
        duration: float,
        fps: float,
        video_frame_to_path_fn: Callable[[int], str] = None,
        video_frame_paths: List[str] = None,
        multithreaded_io: bool = False,
    ) -> None:
        """
        Args:
            duration (float): the duration of the video in seconds.
            fps (float): the target fps for the video. This is needed to link the frames
                to a second timestamp in the video.
            video_frame_to_path_fn (Callable[[int], str]): a function that maps from a frame
                index integer to the file path where the frame is located.
            video_frame_paths (List[str]): Dictionary of frame paths for each index of a video.
            multithreaded_io (bool):  controls whether parllelizable io operations are
                performed across multiple threads.
        """
        self._duration = duration
        self._fps = fps

        assert (video_frame_to_path_fn is None) != (
            video_frame_paths is None
        ), "Only one of video_frame_to_path_fn or video_frame_paths can be provided"
        self._video_frame_to_path_fn = video_frame_to_path_fn
        self._video_frame_paths = video_frame_paths

        self._multithreaded_io = multithreaded_io

    @classmethod
    def from_frame_paths(
        cls,
        video_frame_paths: List[str],
        fps: float = 30.0,
        multithreaded_io: bool = False,
    ):
        """
        Args:
            video_frame_paths (List[str]): a list of paths to each frames in the video.
            fps (float): the target fps for the video. This is needed to link the frames
                to a second timestamp in the video.
            multithreaded_io (bool):  controls whether parllelizable io operations are
                performed across multiple threads.
        """
        assert len(video_frame_paths) != 0, "video_frame_paths is empty"

        return cls(
            len(video_frame_paths) / fps,
            fps,
            video_frame_paths=video_frame_paths,
            multithreaded_io=multithreaded_io,
        )

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def _get_frame_index_for_time(self, time_sec: float) -> int:
        return int(np.round(self._fps * time_sec))

    def get_clip(
        self,
        start_sec: float,
        end_sec: float,
        frame_filter: Optional[Callable[[List[int]], List[int]]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the stored video at the specified start and end times
        in seconds (the video always starts at 0 seconds). Given that PathManager may
        be fetching the frames from network storage, to handle transient errors, frame
        reading is retried N times.

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
            frame_filter (Optional[Callable[List[int], List[int]]]):
                function to subsample frames in a clip before loading.
                If None, no subsampling is peformed.
        Returns:
            clip_frames: A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.float32 and
                in the range [0 - 255]. Raises an exception if unable to load images.

            clip_data:
                "video": A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.float32 and
                in the range [0 - 255]. Raises an exception if unable to load images.

                "frame_indices": A list of indices for each frame relative to all frames in the
                video.

            Returns None if no frames are found.
        """
        if start_sec < 0 or start_sec > self._duration:
            logger.warning(
                f"No frames found within {start_sec} and {end_sec} seconds. Video starts"
                f"at time 0 and ends at {self._duration}."
            )
            return None

        end_sec = min(end_sec, self._duration)

        start_frame_index = self._get_frame_index_for_time(start_sec)
        end_frame_index = self._get_frame_index_for_time(end_sec)
        frame_indices = list(range(start_frame_index, end_frame_index))
        # Frame filter function to allow for subsampling before loading
        if frame_filter:
            frame_indices = frame_filter(frame_indices)

        clip_paths = [self._video_frame_to_path(i) for i in frame_indices]
        clip_frames = _load_images_with_retries(
            clip_paths, multithreaded=self._multithreaded_io
        )
        clip_frames = thwc_to_cthw(clip_frames).to(torch.float32)
        return {"video": clip_frames, "frame_indices": frame_indices}

    def _video_frame_to_path(self, frame_index: int) -> str:
        if self._video_frame_to_path_fn:
            return self._video_frame_to_path_fn(frame_index)
        elif self._video_frame_paths:
            return self._video_frame_paths[frame_index]
        else:
            raise Exception(
                "One of _video_frame_to_path_fn or _video_frame_paths must be set"
            )


def _load_images_with_retries(
    image_paths: List[str], num_retries: int = 10, multithreaded: bool = True
) -> torch.Tensor:
    """
    Loads the given image paths using PathManager, decodes them as RGB images and
    returns them as a stacked tensors.
    Args:
        image_paths (List[str]): a list of paths to images.
        num_retries (int): number of times to retry image reading to handle transient error.
        multithreaded (bool): if images are fetched via multiple threads in parallel.
    Returns:
        A tensor of the clip's RGB frames with shape:
        (time, height, width, channel). The frames are of type torch.uint8 and
        in the range [0 - 255]. Raises an exception if unable to load images.
    """
    imgs = [None for i in image_paths]

    def fetch_image(image_index: int, image_path: str) -> None:
        for i in range(num_retries):
            with g_pathmgr.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img_bgr = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb is not None:
                imgs[image_index] = img_rgb
                return
            else:
                logging.warning(f"Reading attempt {i}/{num_retries} failed.")
                time.sleep(1e-6)

    optional_threaded_foreach(fetch_image, enumerate(image_paths), multithreaded)

    if any((img is None for img in imgs)):
        raise Exception("Failed to load images from {}".format(image_paths))

    return torch.as_tensor(np.stack(imgs))
