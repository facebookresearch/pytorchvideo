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
    Constrain/slide the give time window to `w_len` size and the video/clip length.
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
    ClipSampler for Ego4d moments. Will return a fixed `window_sec` window
    around the given annotation, shifting where relevant to account for the end
    of the clip/video.

    clip_start/clip_end is added to the annotation dict to facilitate future lookups.
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
    """

    def __init__(self, basepath: str):
        self.basepath = basepath

    @abstractmethod
    def has_imu(self, video_uid: str) -> bool:
        pass

    @abstractmethod
    def get_imu_sample(
        self, video_uid: str, video_start: float, video_end: float
    ) -> Dict[str, Any]:
        pass
