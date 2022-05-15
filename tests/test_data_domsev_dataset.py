# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import tempfile
import unittest
import unittest.mock
from contextlib import ExitStack
from pathlib import Path

import torch
from parameterized import parameterized
from pytorchvideo.data.dataset_manifest_utils import VideoClipInfo, VideoDatasetType
from pytorchvideo.data.domsev import (
    _get_overlap_for_time_range_pair,
    _seconds_to_frame_index,
    DomsevVideoDataset,
    LabelData,
)
from pytorchvideo.data.utils import save_dataclass_objs_to_headered_csv
from utils import (
    get_encoded_video_infos,
    get_flat_video_frames,
    MOCK_VIDEO_IDS,
    MOCK_VIDEO_INFOS,
)


class TestDomsevVideoDataset(unittest.TestCase):

    # video_id: str
    # start_time: float  # Start time of the label, in seconds
    # stop_time: float  # Stop time of the label, in seconds
    # start_frame: int  # 0-indexed ID of the start frame (inclusive)
    # stop_frame: int  # 0-index ID of the stop frame (inclusive)
    # label_id: int
    # label_name: str
    LABELS_DATA = {
        MOCK_VIDEO_IDS[0]: [
            LabelData(
                MOCK_VIDEO_IDS[0],
                0.0,
                6.0,
                1,
                181,
                1,
                "walking",
            ),
            LabelData(
                MOCK_VIDEO_IDS[0],
                6.0333333,
                10.0,
                182,
                301,
                2,
                "running",
            ),
            LabelData(
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
            LabelData(
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
            LabelData(
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
            LabelData(
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
        with tempfile.TemporaryDirectory(prefix=f"{TestDomsevVideoDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            save_dataclass_objs_to_headered_csv(
                list(MOCK_VIDEO_INFOS.values()), video_info_file
            )
            label_file = tempdir / "activity_video_info.csv"
            labels = []
            for label_list in self.LABELS_DATA.values():
                for label_data in label_list:
                    labels.append(label_data)
            save_dataclass_objs_to_headered_csv(labels, label_file)

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
                video_ids = list(self.LABELS_DATA)
                dataset = DomsevVideoDataset(
                    video_data_manifest_file_path=str(video_data_manifest_file_path),
                    video_info_file_path=str(video_info_file),
                    labels_file_path=str(label_file),
                    dataset_type=dataset_type,
                    clip_sampler=lambda x, y: [
                        VideoClipInfo(video_ids[i // 2], i * 2.0, i * 2.0 + 0.9)
                        for i in range(0, 7)
                    ],
                )

                self.assertEqual(len(dataset._videos), 4)
                total_labels = [
                    label_data
                    for video_labels in list(dataset._labels_per_video.values())
                    for label_data in video_labels
                ]
                self.assertEqual(len(total_labels), 6)
                self.assertEqual(len(dataset), 7)  # Num clips

    @parameterized.expand([(VideoDatasetType.Frame,), (VideoDatasetType.EncodedVideo,)])
    def test__getitem__(self, dataset_type):
        with tempfile.TemporaryDirectory(prefix=f"{TestDomsevVideoDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            save_dataclass_objs_to_headered_csv(
                list(MOCK_VIDEO_INFOS.values()), video_info_file
            )
            label_file = tempdir / "activity_video_info.csv"
            labels = []
            for label_list in self.LABELS_DATA.values():
                for label_data in label_list:
                    labels.append(label_data)
            save_dataclass_objs_to_headered_csv(labels, label_file)

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
                video_ids = list(self.LABELS_DATA)
                dataset = DomsevVideoDataset(
                    video_data_manifest_file_path=str(video_data_manifest_file_path),
                    video_info_file_path=str(video_info_file),
                    labels_file_path=str(label_file),
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
                    for i, a in enumerate(clip_1["labels"]):
                        self.assertEqual(a, self.LABELS_DATA[video_ids[0]][i])
                    self.assertEqual(clip_1["start_time"], 2.0)
                    self.assertEqual(clip_1["stop_time"], 2.9)
                    self.assertEqual(clip_1["video_id"], MOCK_VIDEO_IDS[0])

                    clip_2 = dataset.__getitem__(2)
                    for i, a in enumerate(clip_2["labels"]):
                        self.assertEqual(a, self.LABELS_DATA[video_ids[1]][i])
                    self.assertEqual(clip_2["start_time"], 4.0)
                    self.assertEqual(clip_2["stop_time"], 4.9)
                    self.assertEqual(clip_2["video_id"], MOCK_VIDEO_IDS[1])
