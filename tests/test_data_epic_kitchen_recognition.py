# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import unittest.mock

import torch
from pytorchvideo.data import EpicKitchenRecognition
from pytorchvideo.data.epic_kitchen import ActionData
from pytorchvideo.data.epic_kitchen_recognition import ClipSampling
from pytorchvideo.data.frame_video import FrameVideo


class TestEpicKitchenRecognition(unittest.TestCase):
    def test_transform_generator(self):
        clip = {
            "start_time": 2.5,
            "stop_time": 6.5,
            "video": torch.rand(3, 4, 10, 20),
            "actions": [
                ActionData(
                    "P01",
                    "P01_01",
                    "turn off light",
                    "00:00:01.00",
                    "00:00:02.00",
                    262,
                    370,
                    "turn-off",
                    12,
                    "light",
                    113,
                    "['light']",
                    "[113]",
                ),
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
                    "00:00:06.00",
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
                    "slam door",
                    "00:00:10.00",
                    "00:00:11.00",
                    408,
                    509,
                    "slam",
                    3,
                    "door",
                    8,
                    "['door']",
                    "[8]",
                ),
            ],
        }

        def additional_transform(clip):
            clip["video"] = clip["video"].permute(1, 2, 3, 0)
            return clip

        transform_fn = EpicKitchenRecognition._transform_generator(additional_transform)

        transformed_clip = transform_fn(clip)

        self.assertEqual(len(transformed_clip["actions"]), 2)
        # Sort for stability
        sorted_actions = sorted(transformed_clip["actions"], key=lambda a: a.start_time)

        self.assertEqual(sorted_actions[0].narration, "turn on light")
        self.assertEqual(sorted_actions[1].narration, "close door")

        self.assertEqual(transformed_clip["start_time"], 2.5)
        self.assertEqual(transformed_clip["stop_time"], 6.5)

        self.assertEqual(transformed_clip["video"].size(), torch.Size([4, 10, 20, 3]))

    def test_frame_filter_generator(self):
        input_list = list(range(10))

        frame_filter_fn = EpicKitchenRecognition._frame_filter_generator(10)
        all_elements = frame_filter_fn(input_list)
        self.assertEqual(all_elements, input_list)

        frame_filter_fn = EpicKitchenRecognition._frame_filter_generator(5)
        half_elements = frame_filter_fn(input_list)
        self.assertEqual(len(half_elements), 5)
        self.assertEqual(half_elements, [i for i in input_list if not i % 2])

        frame_filter_fn = EpicKitchenRecognition._frame_filter_generator(1)
        half_elements = frame_filter_fn(input_list)
        self.assertEqual(len(half_elements), 1)
        self.assertEqual(half_elements[0], 0)

    def test_define_clip_structure_generator(self):
        seconds_per_clip = 5
        define_clip_structure_fn = (
            EpicKitchenRecognition._define_clip_structure_generator(
                seconds_per_clip=5, clip_sampling=ClipSampling.RandomOffsetUniform
            )
        )
        frame_videos = {
            "P01_003": FrameVideo.from_frame_paths(
                [f"root/P01_003/frame_{i}" for i in range(100)], 10
            ),
            "P02_004": FrameVideo.from_frame_paths(
                [f"root/P02_004/frame_{i}" for i in range(300)], 10
            ),
            "P11_010": FrameVideo.from_frame_paths(
                [f"root/P11_010/frame_{i}" for i in range(600)], 30
            ),
        }
        actions = {video_id: [] for video_id in frame_videos}
        random_value = 0.5
        with unittest.mock.patch("random.random", return_value=random_value) as _:
            clips = define_clip_structure_fn(frame_videos, actions)
            sorted_clips = sorted(clips, key=lambda c: c.start_time)  # For stability

            for clip in sorted_clips:
                self.assertEqual(clip.stop_time - clip.start_time, seconds_per_clip)

            clips_P01_003 = [c for c in sorted_clips if c.video_id == "P01_003"]
            self.assertEqual(len(clips_P01_003), 1)
            for i in range(len(clips_P01_003)):
                self.assertEqual(
                    clips_P01_003[i].start_time, seconds_per_clip * (i + random_value)
                )

            clips_P02_004 = [c for c in sorted_clips if c.video_id == "P02_004"]
            self.assertEqual(len(clips_P02_004), 5)
            for i in range(len(clips_P02_004)):
                self.assertEqual(
                    clips_P02_004[i].start_time, seconds_per_clip * (i + random_value)
                )

            clips_P11_010 = [c for c in sorted_clips if c.video_id == "P11_010"]
            self.assertEqual(len(clips_P11_010), 3)
            for i in range(len(clips_P11_010)):
                self.assertEqual(
                    clips_P11_010[i].start_time, seconds_per_clip * (i + random_value)
                )
