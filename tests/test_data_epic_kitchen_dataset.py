# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import tempfile
import unittest
import unittest.mock
from contextlib import ExitStack
from pathlib import Path

import torch
from parameterized import parameterized
from pytorchvideo.data.dataset_manifest_utils import (
    EncodedVideoInfo,
    VideoClipInfo,
    VideoDatasetType,
    VideoFrameInfo,
    VideoInfo,
    VideoDataset,
)
from pytorchvideo.data.epic_kitchen import (
    ActionData,
    EpicKitchenDataset,
)
from pytorchvideo.data.utils import save_dataclass_objs_to_headered_csv
from pytorchvideo.tests.utils import temp_encoded_video


def get_flat_video_frames(directory, file_extension):
    return {
        "P02_001": VideoFrameInfo(
            video_id="P02_001",
            location=f"{directory}/P02_001",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=3000,
            file_extension=file_extension,
        ),
        "P02_002": VideoFrameInfo(
            video_id="P02_002",
            location=f"{directory}/P02_002",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=3001,
            file_extension=file_extension,
        ),
        "P02_005": VideoFrameInfo(
            video_id="P02_005",
            location=f"{directory}/P02_005",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=30003,
            file_extension=file_extension,
        ),
        "P07_002": VideoFrameInfo(
            video_id="P07_002",
            location=f"{directory}/P07_002",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=1530,
            file_extension=file_extension,
        ),
    }


def get_encoded_video_infos(directory, exit_stack=None):
    video_ids = ["P02_001", "P02_002", "P02_005", "P07_002"]
    encoded_video_infos = {}
    for video_id in video_ids:
        file_path, _ = (
            exit_stack.enter_context(temp_encoded_video(10, 10))
            if exit_stack
            else (f"{directory}/{video_id}.mp4", None)
        )
        encoded_video_infos[video_id] = EncodedVideoInfo(video_id, file_path)
    return encoded_video_infos


class TestEpicKitchenDataset(unittest.TestCase):

    VIDEO_INFOS_A = {
        "P02_001": VideoInfo(
            video_id="P02_001", resolution="1080x1920", duration=100, fps=30
        ),
        "P02_002": VideoInfo(
            video_id="P02_002", resolution="1080x1920", duration=50, fps=60
        ),
        "P02_005": VideoInfo(
            video_id="P02_005", resolution="720x1280", duration=1000.09, fps=30
        ),
        "P07_002": VideoInfo(
            video_id="P07_002", resolution="720x1280", duration=17.001, fps=90
        ),
    }
    ACTIONS_DATAS = {
        "P02_001": [
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
        "P02_002": [
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
        "P02_005": [
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
        "P07_002": [
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

    def test_VideoFrameInfo(self):
        video_frame_info = VideoFrameInfo(
            # This is a key-mapping as the underlying epic-kitchen
            # annotation files are of these string columns
            **{
                "video_id": "P01_012",
                "location": "c:/",
                "frame_file_stem": "P01_012_",
                "frame_string_length": "20",
                "min_frame_number": "0",
                "max_frame_number": "22",
                "file_extension": "png",
            }
        )
        self.assertEqual(video_frame_info.video_id, "P01_012")
        self.assertEqual(video_frame_info.location, "c:/")
        self.assertEqual(video_frame_info.frame_file_stem, "P01_012_")
        self.assertEqual(video_frame_info.frame_string_length, 20)
        self.assertEqual(video_frame_info.min_frame_number, 0)
        self.assertEqual(video_frame_info.max_frame_number, 22)
        self.assertEqual(video_frame_info.file_extension, "png")

    def test_EncodedVideoInfo(self):
        encoded_video_info = EncodedVideoInfo(
            # This is a key-mapping as the underlying epic-kitchen
            # annotation files are of these string columns
            **{"video_id": "P01_12", "file_path": "c:/P01_12.mp4"}
        )
        self.assertEqual(encoded_video_info.video_id, "P01_12")
        self.assertEqual(encoded_video_info.file_path, "c:/P01_12.mp4")

    def test_VideoInfo(self):
        video_info = VideoInfo(
            # This is a key-mapping as the underlying epic-kitchen
            # annotation files are of these string columns
            **{
                "video_id": "P01_01",
                "resolution": "1000x200",
                "duration": "123.45",
                "fps": "59.9",
            }
        )
        self.assertEqual(video_info.video_id, "P01_01")
        self.assertEqual(video_info.resolution, "1000x200")
        self.assertEqual(video_info.duration, 123.45)
        self.assertEqual(video_info.fps, 59.9)

    def test_frame_number_to_filepath(self):
        file_name_fn_P07_002 = VideoDataset._frame_number_to_filepath_generator(
            "P07_002", get_flat_video_frames("testdirectory", "jpg"), self.VIDEO_INFOS_A
        )
        file_path = file_name_fn_P07_002(100)
        self.assertEqual(file_path, "testdirectory/P07_002/frame_0000000101.jpg")
        file_path = file_name_fn_P07_002(10000)
        self.assertIsNone(file_path)
        file_path = file_name_fn_P07_002(-1)
        self.assertIsNone(file_path)

        file_name_fn_P02_002 = VideoDataset._frame_number_to_filepath_generator(
            "P02_002",
            get_flat_video_frames("testdirectory2", "png"),
            self.VIDEO_INFOS_A,
        )
        file_path = file_name_fn_P02_002(0)
        self.assertEqual(file_path, "testdirectory2/P02_002/frame_0000000002.png")
        file_path = file_name_fn_P02_002(2999)
        self.assertEqual(file_path, "testdirectory2/P02_002/frame_0000003001.png")
        file_path = file_name_fn_P02_002(3000)
        self.assertIsNone(file_path)

    def test_remove_video_info_missing_or_incomplete_videos(self):
        video_infos_a = self.VIDEO_INFOS_A.copy()
        video_frames_a = get_flat_video_frames("testdirectory2", "jpg")
        video_frames_a_copy = video_frames_a.copy()

        # No-Op
        VideoDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames_a, video_infos_a
        )

        self.assertEqual(len(video_infos_a), len(self.VIDEO_INFOS_A))
        for video_id in video_infos_a:
            self.assertEqual(video_infos_a[video_id], self.VIDEO_INFOS_A[video_id])

        self.assertEqual(len(video_frames_a), len(video_frames_a_copy))
        for video_id in video_frames_a:
            self.assertEqual(video_frames_a[video_id], video_frames_a_copy[video_id])

        video_infos_b = self.VIDEO_INFOS_A.copy()
        video_frames_b = video_frames_a_copy.copy()

        # Unmatched video info, should be removed
        video_infos_b["P07_001"] = VideoInfo(
            video_id="P07_001", resolution="720x1280", duration=17.001, fps=30
        )

        # Unmatched video frame entry, should be removed
        video_frames_b["P07_002"]: VideoFrameInfo(
            min_frame_number=1, max_frame_number=1530, frame_string_length=8
        )

        # Video info that defines approximately 6000 frames with 600 present from frame manifest
        # Should be dropped
        video_frames_b["P08_001"]: VideoFrameInfo(
            min_frame_number=1, max_frame_number=600, frame_string_length=8
        )

        video_infos_b["P08_001"] = VideoInfo(
            video_id="P08_001", resolution="720x1280", duration=100, fps=60
        )

        VideoDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames_b, video_infos_b
        )

        # All newly added fields should be removed
        self.assertEqual(len(video_infos_b), len(self.VIDEO_INFOS_A))
        for video_id in video_infos_b:
            self.assertEqual(video_infos_b[video_id], self.VIDEO_INFOS_A[video_id])

        self.assertEqual(len(video_frames_b), len(video_frames_a_copy))
        for video_id in video_frames_b:
            self.assertEqual(video_frames_b[video_id], video_frames_a_copy[video_id])

    @parameterized.expand([(VideoDatasetType.Frame,), (VideoDatasetType.EncodedVideo,)])
    def test__len__(self, dataset_type):
        with tempfile.TemporaryDirectory(prefix=f"{TestEpicKitchenDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            save_dataclass_objs_to_headered_csv(
                list(self.VIDEO_INFOS_A.values()), video_info_file
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
                list(self.VIDEO_INFOS_A.values()), video_info_file
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
                    self.assertEqual(clip_1["video_id"], "P02_001")

                    clip_2 = dataset.__getitem__(2)
                    for i, a in enumerate(clip_2["actions"]):
                        self.assertEqual(a, self.ACTIONS_DATAS[video_ids[1]][i])
                    self.assertEqual(clip_2["start_time"], 4.0)
                    self.assertEqual(clip_2["stop_time"], 4.9)
                    self.assertEqual(clip_2["video_id"], "P02_002")
