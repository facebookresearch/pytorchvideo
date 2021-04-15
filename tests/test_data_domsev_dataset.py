# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import tempfile
import unittest
import unittest.mock
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import torch
from parameterized import parameterized
from pytorchvideo.data.dataset_manifest_utils import VideoClipInfo, VideoDatasetType
from pytorchvideo.data.domsev import (
    ActivityData,
    DomsevDataset,
    _get_overlap_for_time_range_pair,
    _seconds_to_frame_index,
)
from pytorchvideo.data.utils import save_dataclass_objs_to_headered_csv
from utils import (
    MOCK_VIDEO_IDS,
    MOCK_VIDEO_INFOS,
    get_encoded_video_infos,
    get_flat_video_frames,
)


def _frame_index_to_seconds(
    frame_index: int, fps: int, zero_indexed: Optional[bool] = True
) -> float:
    """
    Converts a frame index within a video clip to the corresponding
    point in time (in seconds) within the video, based on a specified frame rate.

    Args:
        frame_index (int): The index of the frame within the video.
        fps (int): The frame rate (frames per second) of the video.
        zero_indexed (Optional[bool]): Whether the specified frame is zero-indexed
            (if True) or one-indexed (if False).

    Returns:
        (float) The point in time within the video.
    """
    if not zero_indexed:
        frame_index -= 1
    time_in_seconds = frame_index / fps
    return time_in_seconds


class TestDomsevDataset(unittest.TestCase):

    # video_id: str
    # start_time: float  # Start time of the activity, in seconds
    # stop_time: float  # Stop time of the activity, in seconds
    # start_frame: int  # 0-indexed ID of the start frame (inclusive)
    # stop_frame: int  # 0-index ID of the stop frame (inclusive)
    # activity_id: int
    # activity_name: str
    ACTIVITIES_DATA = {
        MOCK_VIDEO_IDS[0]: [
            ActivityData(
                MOCK_VIDEO_IDS[0],
                0.0,
                6.0,
                1,
                181,
                1,
                "walking",
            ),
            ActivityData(
                MOCK_VIDEO_IDS[0],
                6.0333333,
                10.0,
                182,
                301,
                2,
                "running",
            ),
            ActivityData(
                MOCK_VIDEO_IDS[0],
                10.033333,
                20.0,
                302,
                601,
                0,
                "none",
            ),
        ],
        MOCK_VIDEO_IDS[1]: [
            ActivityData(
                MOCK_VIDEO_IDS[1],
                3.0,
                5.0,
                181,
                301,
                7,
                "cooking",
            ),
        ],
        MOCK_VIDEO_IDS[2]: [
            ActivityData(
                MOCK_VIDEO_IDS[2],
                100.0,
                200.0,
                3001,
                6001,
                9,
                "observing",
            ),
        ],
        MOCK_VIDEO_IDS[3]: [
            ActivityData(
                MOCK_VIDEO_IDS[3],
                10.0,
                20.0,
                901,
                1801,
                5,
                "driving",
            ),
        ],
    }

    def setUp(self):
        pass

    def test_seconds_to_frame_index(self):
        self.assertEqual(_seconds_to_frame_index(10.56, 1, zero_indexed=True), 10)
        self.assertEqual(_seconds_to_frame_index(10.56, 1, zero_indexed=False), 11)

        self.assertEqual(_seconds_to_frame_index(9.99, 1, zero_indexed=True), 9)
        self.assertEqual(_seconds_to_frame_index(9.99, 1, zero_indexed=False), 10)

        self.assertEqual(_seconds_to_frame_index(1.01, 10, zero_indexed=True), 10)
        self.assertEqual(_seconds_to_frame_index(1.01, 10, zero_indexed=False), 11)

    def test_get_overlap_for_time_range_pair(self):
        self.assertEqual(_get_overlap_for_time_range_pair(0, 1, 0.1, 0.2), (0.1, 0.2))
        self.assertEqual(_get_overlap_for_time_range_pair(0.1, 0.2, 0, 1), (0.1, 0.2))
        self.assertEqual(_get_overlap_for_time_range_pair(0, 1, 0.9, 1.1), (0.9, 1.0))
        self.assertEqual(_get_overlap_for_time_range_pair(0, 0.2, 0.1, 1), (0.1, 0.2))

    @parameterized.expand([(VideoDatasetType.Frame,), (VideoDatasetType.EncodedVideo,)])
    def test__len__(self, dataset_type):
        with tempfile.TemporaryDirectory(prefix=f"{TestDomsevDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            save_dataclass_objs_to_headered_csv(
                list(MOCK_VIDEO_INFOS.values()), video_info_file
            )
            activity_file = tempdir / "activity_video_info.csv"
            activities = []
            for activity_list in self.ACTIVITIES_DATA.values():
                for activity in activity_list:
                    activities.append(activity)
            save_dataclass_objs_to_headered_csv(activities, activity_file)

            video_data_manifest_file_path = (
                tempdir / "video_data_manifest_file_path.json"
            )
            with ExitStack() as stack:
                if dataset_type == VideoDatasetType.Frame:
                    video_data_dict = get_flat_video_frames(tempdir, "jpg")
                elif dataset_type == VideoDatasetType.EncodedVideo:
                    video_data_dict = get_encoded_video_infos(tempdir, stack)

                save_dataclass_objs_to_headered_csv(
                    list(video_data_dict.values()), video_data_manifest_file_path
                )
                video_ids = list(self.ACTIVITIES_DATA)
                dataset = DomsevDataset(
                    video_data_manifest_file_path=str(video_data_manifest_file_path),
                    video_info_file_path=str(video_info_file),
                    activities_file_path=str(activity_file),
                    dataset_type=dataset_type,
                    clip_sampler=lambda x, y: [
                        VideoClipInfo(video_ids[i // 2], i * 2.0, i * 2.0 + 0.9)
                        for i in range(0, 7)
                    ],
                )

                self.assertEqual(len(dataset._videos), 4)
                total_activities = [
                    activity
                    for video_activities in list(dataset._activities.values())
                    for activity in video_activities
                ]
                self.assertEqual(len(total_activities), 6)
                self.assertEqual(len(dataset), 7)  # Num clips

    @parameterized.expand([(VideoDatasetType.Frame,), (VideoDatasetType.EncodedVideo,)])
    def test__getitem__(self, dataset_type):
        with tempfile.TemporaryDirectory(prefix=f"{TestDomsevDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            save_dataclass_objs_to_headered_csv(
                list(MOCK_VIDEO_INFOS.values()), video_info_file
            )
            activity_file = tempdir / "activity_video_info.csv"
            activities = []
            for activity_list in self.ACTIVITIES_DATA.values():
                for activity in activity_list:
                    activities.append(activity)
            save_dataclass_objs_to_headered_csv(activities, activity_file)

            video_data_manifest_file_path = (
                tempdir / "video_data_manifest_file_path.json"
            )
            with ExitStack() as stack:
                if dataset_type == VideoDatasetType.Frame:
                    video_data_dict = get_flat_video_frames(tempdir, "jpg")
                elif dataset_type == VideoDatasetType.EncodedVideo:
                    video_data_dict = get_encoded_video_infos(tempdir, stack)

                save_dataclass_objs_to_headered_csv(
                    list(video_data_dict.values()), video_data_manifest_file_path
                )
                video_ids = list(self.ACTIVITIES_DATA)
                dataset = DomsevDataset(
                    video_data_manifest_file_path=str(video_data_manifest_file_path),
                    video_info_file_path=str(video_info_file),
                    activities_file_path=str(activity_file),
                    dataset_type=dataset_type,
                    clip_sampler=lambda x, y: [
                        VideoClipInfo(video_ids[i // 2], i * 2.0, i * 2.0 + 0.9)
                        for i in range(0, 7)
                    ],
                )

                get_clip_string = (
                    "pytorchvideo.data.frame_video.FrameVideo.get_clip"
                    if dataset_type == VideoDatasetType.Frame
                    else "pytorchvideo.data.encoded_video.EncodedVideo.get_clip"
                )
                with unittest.mock.patch(
                    get_clip_string,
                    return_value=({"video": torch.rand(3, 5, 10, 20), "audio": []}),
                ) as _:
                    clip_1 = dataset.__getitem__(1)
                    for i, a in enumerate(clip_1["activities"]):
                        self.assertEqual(a, self.ACTIVITIES_DATA[video_ids[0]][i])
                    self.assertEqual(clip_1["start_time"], 2.0)
                    self.assertEqual(clip_1["stop_time"], 2.9)
                    self.assertEqual(clip_1["video_id"], MOCK_VIDEO_IDS[0])

                    clip_2 = dataset.__getitem__(2)
                    for i, a in enumerate(clip_2["activities"]):
                        self.assertEqual(a, self.ACTIVITIES_DATA[video_ids[1]][i])
                    self.assertEqual(clip_2["start_time"], 4.0)
                    self.assertEqual(clip_2["stop_time"], 4.9)
                    self.assertEqual(clip_2["video_id"], MOCK_VIDEO_IDS[1])
