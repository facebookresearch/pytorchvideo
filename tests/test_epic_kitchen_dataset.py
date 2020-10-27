import csv
import tempfile
import unittest
import unittest.mock
from dataclasses import fields as dataclass_fields
from pathlib import Path

import torch
from pytorchvideo.data.epic_kitchen import (
    ActionData,
    EpicKitchenClip,
    EpicKitchenDataset,
    VideoFrameInfo,
    VideoInfo,
)


def write_video_info_file(video_infos, file_path):
    with open(file_path, "w") as f:
        f.write("video,resolution,duration,fps\n")
        for _, video_info in video_infos.items():
            f.write(
                f"{video_info.video_id},{video_info.resolution},\
                {str(video_info.duration)},{str(video_info.fps)}\n"
            )


def write_frame_manifest_file(video_frames, file_path):
    field_names = [f.name for f in dataclass_fields(VideoFrameInfo)]
    with open(file_path, "w") as output_file:
        writer = csv.writer(output_file, delimiter=",", quotechar='"')
        writer.writerow(field_names)
        for _, video_frame_info in video_frames.items():
            writer.writerow((getattr(video_frame_info, f) for f in field_names))


def write_actions_file(file_path, actions):
    field_names = [f.name for f in dataclass_fields(ActionData)]
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerow(field_names)
        for video_id in actions:
            video_actions = actions[video_id]
            for a in video_actions:
                writer.writerow((getattr(a, f) for f in field_names))


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


class TestEpicKitchenDataset(unittest.TestCase):

    VIDEO_INFOS_A = {
        "P02_001": VideoInfo(
            video="P02_001", resolution="1080x1920", duration=100, fps=30
        ),
        "P02_002": VideoInfo(
            video="P02_002", resolution="1080x1920", duration=50, fps=60
        ),
        "P02_005": VideoInfo(
            video="P02_005", resolution="720x1280", duration=1000.09, fps=30
        ),
        "P07_002": VideoInfo(
            video="P07_002", resolution="720x1280", duration=17.001, fps=90
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

    def test_VideoInfo(self):
        video_info = VideoInfo(
            # This is a key-mapping as the underlying epic-kitchen
            # annotation files are of these string columns
            **{
                "video": "P01_01",
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
        file_name_fn_P07_002 = EpicKitchenDataset._frame_number_to_filepath_generator(
            "P07_002", get_flat_video_frames("testdirectory", "jpg"), self.VIDEO_INFOS_A
        )
        file_path = file_name_fn_P07_002(100)
        self.assertEqual(file_path, "testdirectory/P07_002/frame_0000000101.jpg")
        file_path = file_name_fn_P07_002(10000)
        self.assertIsNone(file_path)
        file_path = file_name_fn_P07_002(-1)
        self.assertIsNone(file_path)

        file_name_fn_P02_002 = EpicKitchenDataset._frame_number_to_filepath_generator(
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
        EpicKitchenDataset._remove_video_info_missing_or_incomplete_videos(
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
            video="P07_001", resolution="720x1280", duration=17.001, fps=30
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
            video="P08_001", resolution="720x1280", duration=100, fps=60
        )

        EpicKitchenDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames_b, video_infos_b
        )

        # All newly added fields should be removed
        self.assertEqual(len(video_infos_b), len(self.VIDEO_INFOS_A))
        for video_id in video_infos_b:
            self.assertEqual(video_infos_b[video_id], self.VIDEO_INFOS_A[video_id])

        self.assertEqual(len(video_frames_b), len(video_frames_a_copy))
        for video_id in video_frames_b:
            self.assertEqual(video_frames_b[video_id], video_frames_a_copy[video_id])

    def test__len__(self):
        with tempfile.TemporaryDirectory(prefix=f"{TestEpicKitchenDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            action_file = tempdir / "action_video_info.csv"
            video_frames_file = tempdir / "test_frame_manifest.json"
            write_video_info_file(self.VIDEO_INFOS_A, video_info_file)
            write_actions_file(action_file, self.ACTIONS_DATAS)
            write_frame_manifest_file(
                get_flat_video_frames("test_dir", "jpg"), video_frames_file
            )

            dataset = EpicKitchenDataset(
                video_info_file_path=str(video_info_file),
                actions_file_path=str(action_file),
                clip_sampler=lambda x, y: [
                    EpicKitchenClip(str(i), i * 2.0, i * 2.0 + 0.9) for i in range(0, 7)
                ],
                frame_manifest_file_path=str(video_frames_file),
            )

            self.assertEqual(len(dataset), 7)

    def test__getitem__(self):
        with tempfile.TemporaryDirectory(prefix=f"{TestEpicKitchenDataset}") as tempdir:
            tempdir = Path(tempdir)

            video_info_file = tempdir / "test_video_info.csv"
            action_file = tempdir / "action_video_info.csv"
            video_frames_file = tempdir / "test_frame_manifest.json"
            write_video_info_file(self.VIDEO_INFOS_A, video_info_file)
            write_actions_file(action_file, self.ACTIONS_DATAS)
            write_frame_manifest_file(
                get_flat_video_frames("test_dir", "png"), video_frames_file
            )

            video_ids = list(self.ACTIONS_DATAS)
            dataset = EpicKitchenDataset(
                video_info_file_path=str(video_info_file),
                actions_file_path=str(action_file),
                clip_sampler=lambda x, y: [
                    EpicKitchenClip(video_ids[i // 2], i * 2.0, i * 2.0 + 0.9)
                    for i in range(0, 7)
                ],
                frame_manifest_file_path=str(video_frames_file),
            )
            expected_actions = self.ACTIONS_DATAS
            with unittest.mock.patch(
                "pytorchvideo.data.frame_video.FrameVideo.get_clip",
                return_value=({"video": torch.rand(3, 5, 10, 20), "audio": []}),
            ) as _:
                clip_1 = dataset.__getitem__(1)
                for i, a in enumerate(clip_1["actions"]):
                    self.assertEqual(a, expected_actions[video_ids[0]][i])
                self.assertEqual(clip_1["start_time"], 2.0)
                self.assertEqual(clip_1["stop_time"], 2.9)
                self.assertEqual(clip_1["video_id"], "P02_001")

                clip_2 = dataset.__getitem__(2)
                for i, a in enumerate(clip_2["actions"]):
                    self.assertEqual(a, expected_actions[video_ids[1]][i])
                self.assertEqual(clip_2["start_time"], 4.0)
                self.assertEqual(clip_2["stop_time"], 4.9)
                self.assertEqual(clip_2["video_id"], "P02_002")
