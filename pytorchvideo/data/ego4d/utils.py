# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from iopath.common.file_io import g_pathmgr

from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler
from pytorchvideo.data.utils import get_logger

log: logging.Logger = get_logger("Ego4dDatasetUtils")


# TODO: Round to fps (and ideally frame align)
def check_window_len(
    s_time: float, e_time: float, w_len: float, video_dur: float
) -> Tuple[float, float]:
    """
    Constrain or slide the given time window to match a specified length  `w_len` while
    considering the video or clip duration.

    Args:
        s_time (float): The start time of the original time window.
        e_time (float): The end time of the original time window.
        w_len (float): The desired length of the time window.
        video_dur (float): The duration of the video or clip.

    Returns:
        Tuple[float, float]: A tuple containing the adjusted start and end times.

    This function adjusts the time window defined by `s_time` and `e_time` to match
    a specified length `w_len`. If the time window is larger or smaller than `w_len`,
    it is adjusted by equally extending or trimming the interior. If the adjusted
    time window exceeds the duration of the video or clip, it is further adjusted to
    stay within the video duration.

    Note: The function ensures that the adjusted time window has a length close to `w_len`.
    """
    # adjust to match w_len
    interval = e_time - s_time
    if abs(interval - w_len) > 0.001:
        # TODO: Do we want to sample rather than trim the interior when larger?
        delta = w_len - (e_time - s_time)
        s_time = s_time - (delta / 2)
        e_time = e_time + (delta / 2)
        if s_time < 0:
            e_time += -s_time
            s_time = 0
    if video_dur:
        if e_time > video_dur:
            overlap = e_time - video_dur
            assert s_time >= overlap, "Incompatible w_len / video_dur"
            s_time -= overlap
            e_time -= overlap
            log.info(
                f"check_window_len: video overlap ({overlap}) adjusted -> ({s_time:.2f}, {e_time:.2f}) video: {video_dur}"  # noqa
            )
    if abs((e_time - s_time) - w_len) > 0.01:
        log.error(
            f"check_window_len: invalid time interval: {s_time}, {e_time}",
            stack_info=True,
        )
    return s_time, e_time


# TODO: Move to FixedClipSampler?
class MomentsClipSampler(ClipSampler):
    """
    ClipSampler for Ego4d moments. This sampler returns a fixed-duration `window_sec`
    window around a given annotation, adjusting for the end of the clip/video if necessary.

    The `clip_start` and `clip_end` fields are added to the annotation dictionary to
    facilitate future lookups.

    Args:
        window_sec (float): The duration (in seconds) of the fixed window to sample.

    This ClipSampler is designed for Ego4d moments and ensures that clips are sampled
    with a fixed duration specified by `window_sec`. It adjusts the window's position
    if needed to account for the end of the clip or video.
    """

    def __init__(self, window_sec: float = 0) -> None:
        self.window_sec = window_sec

    def __call__(
        self,
        last_clip_end_time: float,
        video_duration: float,
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        assert (
            last_clip_end_time is None or last_clip_end_time <= video_duration
        ), f"last_clip_end_time ({last_clip_end_time}) > video_duration ({video_duration})"
        start = annotation["label_video_start_sec"]
        end = annotation["label_video_end_sec"]
        if video_duration is not None and end > video_duration:
            log.error(f"Invalid video_duration/end_sec: {video_duration} / {end}")
            # If it's small, proceed anyway
            if end > video_duration + 0.1:
                raise Exception(
                    f"Invalid video_duration/end_sec: {video_duration} / {end} ({annotation['video_name']})"  # noqa
                )
        assert end >= start, f"end < start: {end:.2f} / {start:.2f}"
        if self.window_sec > 0:
            s, e = check_window_len(start, end, self.window_sec, video_duration)
            if s != start or e != end:
                # log.info(
                #     f"clip window slid ({start:.2f}|{end:.2f}) -> ({s:.2f}|{e:.2f})"
                # )
                start = s
                end = e
        annotation["clip_start"] = start
        annotation["clip_end"] = end
        return ClipInfo(start, end, 0, 0, True)


def get_label_id_map(label_id_map_path: str) -> Dict[str, int]:
    """
    Reads a label-to-ID mapping from a JSON file.

    Args:
        label_id_map_path (str): The path to the label ID mapping JSON file.

    Returns:
        Dict[str, int]: A dictionary mapping label names to their corresponding IDs.

    This function reads a JSON file containing label-to-ID mapping and returns it as a dictionary.
    """
    label_name_id_map: Dict[str, int]

    try:
        with g_pathmgr.open(label_id_map_path, "r") as f:
            label_json = json.load(f)

            # TODO: Verify?
            return label_json
    except Exception:
        raise FileNotFoundError(f"{label_id_map_path} must be a valid label id json")


class Ego4dImuDataBase(ABC):
    """
    Base class placeholder for Ego4d IMU data.

    This is a base class for handling Ego4d IMU data. It defines the required interface for
    checking if IMU data is available for a video and retrieving IMU samples.
    """

    def __init__(self, basepath: str):
        """
        Initializes an instance of Ego4dImuDataBase.

        Args:
            basepath (str): The base path for Ego4d IMU data.
        """
        self.basepath = basepath

    @abstractmethod
    def has_imu(self, video_uid: str) -> bool:
        """
        Checks if IMU data is available for a video.

        Args:
            video_uid (str): The unique identifier of the video.

        Returns:
            bool: True if IMU data is available, False otherwise.

        This method should be implemented to check if IMU data exists for a specific video
        identified by its unique ID.
        """
        pass

    @abstractmethod
    def get_imu_sample(
        self, video_uid: str, video_start: float, video_end: float
    ) -> Dict[str, Any]:
        """
        Retrieves an IMU sample for a video segment.

        Args:
            video_uid (str): The unique identifier of the video.
            video_start (float): The start time of the video segment.
            video_end (float): The end time of the video segment.

        Returns:
            Dict[str, Any]: A dictionary containing IMU data.

        This method should be implemented to retrieve IMU data for a specific video segment
        identified by its unique ID and time range.
        """
        pass
