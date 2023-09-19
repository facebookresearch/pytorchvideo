# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.utils import optional_threaded_foreach

from .utils import thwc_to_cthw
from .video import Video


try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True


logger = logging.getLogger(__name__)


class FrameVideo(Video):
    """
    FrameVideo is an abstraction for accessing clips based on their start and end
    time for a video where each frame is stored as an image. PathManager is used for
    frame image reading, allowing non-local URIs to be used.
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
        Initialize a FrameVideo object.

        Args:
            duration (float):
                The duration of the video in seconds.
            fps (float):
                The target FPS for the video. This is needed to link the frames to a second
                timestamp in the video.
            video_frame_to_path_fn (Callable[[int], str], optional):
                A function that maps from a frame index integer to the file path where the
                frame is located.
            video_frame_paths (List[str], optional):
                List of frame paths for each index of a video.
            multithreaded_io (bool, optional):
                Controls whether parallelizable IO operations are performed across multiple
                threads.
        """
        if not _HAS_CV2:
            raise ImportError(
                "opencv2 is required to use FrameVideo. Please "
                "install with 'pip install opencv-python'"
            )

        self._duration = duration
        self._fps = fps
        self._multithreaded_io = multithreaded_io

        assert (video_frame_to_path_fn is None) != (
            video_frame_paths is None
        ), "Only one of video_frame_to_path_fn or video_frame_paths can be provided"
        self._video_frame_to_path_fn = video_frame_to_path_fn
        self._video_frame_paths = video_frame_paths

        # Set the pathname to the parent directory of the first frame.
        self._name = os.path.basename(
            os.path.dirname(self._video_frame_to_path(frame_index=0))
        )

    @classmethod
    def from_directory(
        cls,
        path: str,
        fps: float = 30.0,
        multithreaded_io=False,
        path_order_cache: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Create a FrameVideo object from a directory containing frame images.

        Args:
            path (str):
                Path to the frame video directory.
            fps (float, optional):
                The target FPS for the video. This is needed to link the frames to a second
                timestamp in the video.
            multithreaded_io (bool, optional):
                Controls whether parallelizable IO operations are performed across multiple
                threads.
            path_order_cache (dict, optional):
                An optional mapping from directory path to list of frames in the directory
                in numerical order. Used for speedup by caching the frame paths.

        Returns:
            FrameVideo:
                A FrameVideo object created from the provided frame directory.
        """
        if path_order_cache is not None and path in path_order_cache:
            return cls.from_frame_paths(path_order_cache[path], fps, multithreaded_io)

        assert g_pathmgr.isdir(path), f"{path} is not a directory"
        rel_frame_paths = g_pathmgr.ls(path)

        def natural_keys(text):
            return [int(c) if c.isdigit() else c for c in re.split("(\d+)", text)]

        rel_frame_paths.sort(key=natural_keys)
        frame_paths = [os.path.join(path, f) for f in rel_frame_paths]
        if path_order_cache is not None:
            path_order_cache[path] = frame_paths
        return cls.from_frame_paths(frame_paths, fps, multithreaded_io)

    @classmethod
    def from_frame_paths(
        cls,
        video_frame_paths: List[str],
        fps: float = 30.0,
        multithreaded_io: bool = False,
    ):
        """
        Create a FrameVideo object from a list of frame image paths.

        Args:
            video_frame_paths (List[str]):
                A list of paths to each frame in the video.
            fps (float, optional):
                The target FPS for the video. This is needed to link the frames to a second
                timestamp in the video.
            multithreaded_io (bool, optional):
                Controls whether parallelizable IO operations are performed across multiple
                threads.

        Returns:
            FrameVideo:
                A FrameVideo object created from the provided frame image paths.
        """
        assert len(video_frame_paths) != 0, "video_frame_paths is empty"
        return cls(
            len(video_frame_paths) / fps,
            fps,
            video_frame_paths=video_frame_paths,
            multithreaded_io=multithreaded_io,
        )

    @property
    def name(self) -> float:
        """
        Returns the name of the FrameVideo.

        Returns:
            str: The name of the FrameVideo.
        """
        return self._name

    @property
    def duration(self) -> float:
        """
        Returns the duration of the FrameVideo.

        Returns:
            float: The duration of the FrameVideo in seconds.
        """
        return self._duration

    def _get_frame_index_for_time(self, time_sec: float) -> int:
        return math.ceil(self._fps * time_sec)

    def get_clip(
        self,
        start_sec: float,
        end_sec: float,
        frame_filter: Optional[Callable[[List[int]], List[int]]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the stored video at the specified start and end times
        in seconds (the video always starts at 0 seconds). Returned frames will be
        in [start_sec, end_sec). Given that PathManager may be fetching the frames
        from network storage, to handle transient errors, frame reading is retried N times.
        Note that as end_sec is exclusive, so you may need to use `get_clip(start_sec, duration + EPS)`
        to get the last frame.

        Args:
            start_sec (float):
                The clip start time in seconds.
            end_sec (float):
                The clip end time in seconds.
            frame_filter (Optional[Callable[List[int], List[int]]], optional):
                Function to subsample frames in a clip before loading.
                If None, no subsampling is performed.

        Returns:
            Dict[str, Optional[torch.Tensor]]:
                A dictionary containing the following keys:
                - "video": A tensor of the clip's RGB frames with shape:
                  (channel, time, height, width). The frames are of type torch.float32 and
                  in the range [0 - 255].
                - "frame_indices": A list of indices for each frame relative to all frames in the
                  video.
                - "audio": None (audio is not supported in FrameVideo).

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
        end_frame_index = min(
            self._get_frame_index_for_time(end_sec), len(self._video_frame_paths)
        )
        frame_indices = list(range(start_frame_index, end_frame_index))
        # Frame filter function to allow for subsampling before loading
        if frame_filter:
            frame_indices = frame_filter(frame_indices)

        clip_paths = [self._video_frame_to_path(i) for i in frame_indices]
        clip_frames = _load_images_with_retries(
            clip_paths, multithreaded=self._multithreaded_io
        )
        clip_frames = thwc_to_cthw(clip_frames).to(torch.float32)
        return {"video": clip_frames, "frame_indices": frame_indices, "audio": None}

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
    Loads the given image paths using PathManager, decodes them as RGB images, and
    returns them as a stacked tensor.

    Args:
        image_paths (List[str]):
            A list of paths to images.
        num_retries (int, optional):
            Number of times to retry image reading to handle transient errors.
        multithreaded (bool, optional):
            If images are fetched via multiple threads in parallel.

    Returns:
        torch.Tensor:
            A tensor of the clip's RGB frames with shape: (time, height, width, channel).
            The frames are of type torch.uint8 and in the range [0 - 255].

    Raises:
        Exception: If unable to load images.
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
