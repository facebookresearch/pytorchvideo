# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import tempfile
import unittest
import unittest.mock
from contextlib import ExitStack
from pathlib import Path

import torch
from parameterized import parameterized
from pytorchvideo.data.dataset_manifest_utils import VideoClipInfo, VideoDatasetType
from pytorchvideo.data.epic_kitchen import ActionData, EpicKitchenDataset
from pytorchvideo.data.utils import save_dataclass_objs_to_headered_csv
from utils import (
    MOCK_VIDEO_IDS,
    MOCK_VIDEO_INFOS,
    get_encoded_video_infos,
    get_flat_video_frames,
)


class TestEpicKitchenDataset(unittest.TestCase):

    ACTIONS_DATAS = {
        MOCK_VIDEO_IDS[0]: [
            ActionData(
                "P01",
                "P01_01",
                "turn on light",
                "00:00:04.00",
                "00:00:06.00",
                262,
                370,
                "turn-on",
                12,
                "light",
                113,
                "['light']",
                "[113]",
            ),
            ActionData(
                "P01",
                "P01_01",
                "close door",
                "00:00:05.00",
                "00:00:07.00",
                418,
                569,
                "close",
                3,
                "door",
                8,
                "['door']",
                "[8]",
            ),
            ActionData(
                "P01",
                "P01_01",
                "close fridge",
                "00:01:1.91",
                "01:00:5.33",
                1314,
                1399,
                "close",
                3,
                "fridge",
                10,
                "['fridge']",
                "[10]",
            ),
        ],
        MOCK_VIDEO_IDS[1]: [
            ActionData(
                "P02",
                "P02_002",
                "turn on light",
                "00:00:04.00",
                "00:00:06.00",
                262,
                370,
                "turn-on",
                12,
                "light",
                113,
                "['light']",
                "[113]",
            )
        ],
        MOCK_VIDEO_IDS[2]: [
            ActionData(
                "P02",
                "P02_005",
                "turn on light",
                "00:00:04.00",
                "00:00:06.00",
                262,
                370,
                "turn-on",
                12,
                "light",
                113,
                "['light']",
                "[113]",
            )
        ],
        MOCK_VIDEO_IDS[3]: [
            ActionData(
                "P07",
                "P07_002",
                "turn on light",
                "00:00:04.00",
                "00:00:06.00",
                262,
                370,
                "turn-on",
                12,
                "light",
                113,
                "['light']",
                "[113]",
            )
        ],
    }

    def test_ActionData(self):

        action = ActionData(
            # This is a key-mapping as the underlying epic-kitchen
            # annotation files are of these string columns
            **{
                "participant_id": "P07",
                "video_id": "P07_002",
                "narration": "turn on light",
                "start_timestamp": "00:00:04.00",
                "stop_timestamp": "00:00:06.50",
                "start_frame": "262",
                "stop_frame": "370",
                "verb": "turn-on",
                "verb_class": "12",
                "noun": "light",
                "noun_class": "113",
                "all_nouns": "['light', 'finger', 'wall']",
                "all_noun_classes": "[113, 1232, 1]",
            }
        )
        self.assertEqual(action.video_id, "P07_002")
        self.assertEqual(action.start_time, 4.0)
        self.assertEqual(action.stop_time, 6.5)
        self.assertEqual(action.verb_class, 12)
        self.assertEqual(action.noun_class, 113)
        self.assertEqual(action.all_nouns, ["light", "finger", "wall"])

        self.assertEqual(action.all_noun_classes, [113, 1232, 1])

    @parameterized.expand([(VideoDatasetType.Frame,), (VideoDatasetType.EncodedVideo,)])
    def test__len__(self, dataset_type):
        with tempfile.TemporaryDirectory(prefix=f"{TestEpicKitchenDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            save_dataclass_objs_to_headered_csv(
                list(MOCK_VIDEO_INFOS.values()), video_info_file
            )
            action_file = tempdir / "action_video_info.csv"
            actions = []
            for action_list in self.ACTIONS_DATAS.values():
                for action in action_list:
                    actions.append(action)
            save_dataclass_objs_to_headered_csv(actions, action_file)

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

                dataset = EpicKitchenDataset(
                    video_info_file_path=str(video_info_file),
                    actions_file_path=str(action_file),
                    clip_sampler=lambda x, y: [
                        VideoClipInfo(str(i), i * 2.0, i * 2.0 + 0.9)
                        for i in range(0, 7)
                    ],
                    video_data_manifest_file_path=str(video_data_manifest_file_path),
                    dataset_type=dataset_type,
                )

                self.assertEqual(len(dataset), 7)

    @parameterized.expand([(VideoDatasetType.Frame,), (VideoDatasetType.EncodedVideo,)])
    def test__getitem__(self, dataset_type):
        with tempfile.TemporaryDirectory(prefix=f"{TestEpicKitchenDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            save_dataclass_objs_to_headered_csv(
                list(MOCK_VIDEO_INFOS.values()), video_info_file
            )
            action_file = tempdir / "action_video_info.csv"
            actions = []
            for action_list in self.ACTIONS_DATAS.values():
                for action in action_list:
                    actions.append(action)
            save_dataclass_objs_to_headered_csv(actions, action_file)

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
                video_ids = list(self.ACTIONS_DATAS)
                dataset = EpicKitchenDataset(
                    video_info_file_path=str(video_info_file),
                    actions_file_path=str(action_file),
                    clip_sampler=lambda x, y: [
                        VideoClipInfo(video_ids[i // 2], i * 2.0, i * 2.0 + 0.9)
                        for i in range(0, 7)
                    ],
                    video_data_manifest_file_path=str(video_data_manifest_file_path),
                    dataset_type=dataset_type,
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
                    for i, a in enumerate(clip_1["actions"]):
                        self.assertEqual(a, self.ACTIONS_DATAS[video_ids[0]][i])
                    self.assertEqual(clip_1["start_time"], 2.0)
                    self.assertEqual(clip_1["stop_time"], 2.9)
                    self.assertEqual(clip_1["video_id"], MOCK_VIDEO_IDS[0])

                    clip_2 = dataset.__getitem__(2)
                    for i, a in enumerate(clip_2["actions"]):
                        self.assertEqual(a, self.ACTIONS_DATAS[video_ids[1]][i])
                    self.assertEqual(clip_2["start_time"], 4.0)
                    self.assertEqual(clip_2["stop_time"], 4.9)
                    self.assertEqual(clip_2["video_id"], MOCK_VIDEO_IDS[1])
