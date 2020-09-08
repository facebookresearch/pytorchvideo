# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

from .utils import thwc_to_cthw


logger = logging.getLogger(__name__)


class FrameVideo:
    """
    FrameVideo is an abstractions for accessing clips based on their start and end
    time for a video where each frame is stored as an image. PathManager is used for
    frame image reading, allowing non-local uri's to be used.
    """

    def __init__(self, video_frame_paths: List[str], fps: int = 30) -> None:
        """
        Args:
            video_frame_paths (List[str]): a list of paths to each frames in the video.
            fps (int): the target fps for the video. This is needed to link the frames
                to a second timestamp in the video.
        """
        assert len(video_frame_paths) != 0, "video_frame_paths is empty"
        self._fps = fps
        self._video_frame_paths = video_frame_paths
        self._duration = len(video_frame_paths) / self._fps

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Retrieves frames from the stored video at the specified start and end times
        in seconds (the video always starts at 0 seconds). Given that PathManager may
        be fetching the frames from network storage, to handle transient errors, frame
        reading is retried N times.

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_frames: A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.uint8 and
                in the range [0 - 255]. Raises an exception if unable to load images.

            clip_indices: A list of indices for each frame relative to all frames in the
                video.

            Returns None if no frames are found.
        """
        if start_sec < 0 or end_sec > self._duration:
            logger.warning(
                f"No frames found within {start_sec} and {end_sec} seconds. Video starts "
                f"at time 0 and ends at {self.duration}."
            )
            return None

        start_frame_index = self._get_frame_index_for_time(start_sec)
        end_frame_index = self._get_frame_index_for_time(end_sec)
        frame_indices = list(range(start_frame_index, end_frame_index))
        clip_paths = [self._video_frame_paths[i] for i in frame_indices]

        # TODO(Tullie): Add efficient temporal subsampling.
        clip_frames = _load_images_with_retries(clip_paths)
        clip_frames = thwc_to_cthw(clip_frames)
        return clip_frames, frame_indices

    def _get_frame_index_for_time(self, time_sec: float):
        return int(np.round(self._fps * time_sec))


def _load_images_with_retries(
    image_paths: List[str], num_retries: int = 10
) -> torch.Tensor:
    """
    Loads the given image paths using PathManager, decodes them as RGB images and
    returns them as a stacked tensors.
    Args:
        image_paths (List[str]): a list of paths to images.
        num_retries (int): number of times to retry image reading to handle transient error.
    Returns:
        A tensor of the clip's RGB frames with shape:
        (time, height, width, channel). The frames are of type torch.uint8 and
        in the range [0 - 255]. Raises an exception if unable to load images.
    """
    for i in range(num_retries):
        imgs = []
        try:
            for image_path in image_paths:
                with PathManager.open(image_path, "rb") as f:
                    img_str = np.frombuffer(f.read(), np.uint8)
                    img_bgr = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                imgs.append(img_rgb)
        except Exception:
            imgs = None

        if imgs is not None and all(img is not None for img in imgs):
            imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logging.warning(f"Reading attempt {i}/{num_retries} failed.")
            time.sleep(1e-6)

    raise Exception("Failed to load images from {}".format(image_paths))
