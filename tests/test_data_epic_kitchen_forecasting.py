# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import unittest.mock

import torch
from pytorchvideo.data import EpicKitchenForecasting
from pytorchvideo.data.epic_kitchen import ActionData
from pytorchvideo.data.epic_kitchen_forecasting import ClipSampling
from pytorchvideo.data.frame_video import FrameVideo


class TestEpicKitchenForecasting(unittest.TestCase):
    def test_transform_generator(self):
        clip = {
            "start_time": 2.5,
            "stop_time": 6.5,
            "video": torch.rand(3, 8, 10, 20),
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
                ActionData(
                    "P01",
                    "P01_01",
                    "slam door",
                    "00:00:11.00",
                    "00:00:12.00",
                    408,
                    509,
                    "slam",
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
                    "00:00:12.00",
                    "00:00:13.00",
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
            clip["video"] = clip["video"].permute(1, 2, 3, 4, 0)
            return clip

        transform_fn = EpicKitchenForecasting._transform_generator(
            additional_transform,
            num_forecast_actions=3,
            num_input_clips=2,
            frames_per_clip=4,
        )

        transformed_clip = transform_fn(clip)

        self.assertEqual(len(transformed_clip["actions"]), 3)

        self.assertEqual(transformed_clip["actions"][0].narration, "slam door")
        self.assertEqual(transformed_clip["actions"][1].narration, "slam door")
        self.assertEqual(transformed_clip["actions"][2].narration, "slam door")

        self.assertEqual(transformed_clip["actions"][0].start_time, 10.0)
        self.assertEqual(transformed_clip["actions"][1].start_time, 11.0)
        self.assertEqual(transformed_clip["actions"][2].start_time, 12.0)

        self.assertEqual(transformed_clip["start_time"], 2.5)
        self.assertEqual(transformed_clip["stop_time"], 6.5)

        self.assertEqual(
            transformed_clip["video"].size(), torch.Size([3, 4, 10, 20, 2])
        )

    def test_frame_filter_generator(self):
        # 11 seconds of video at 4 fps
        input_list = list(range(44))

        # 11 second clip at 4 fps, all frames are included
        frame_filter_fn = EpicKitchenForecasting._frame_filter_generator(
            seconds_per_clip=1,
            num_input_clips=11,
            frames_per_clip=4,
            clip_time_stride=1,
        )

        all_elements = frame_filter_fn(input_list)

        self.assertEqual(all_elements, input_list)

        # 11 second clip at 4 fps, seconds 0-1 and 10-11 are included
        frame_filter_fn = EpicKitchenForecasting._frame_filter_generator(
            seconds_per_clip=1,
            num_input_clips=2,
            frames_per_clip=4,
            clip_time_stride=10,
        )
        elements_2_clips = frame_filter_fn(input_list)
        self.assertEqual(len(elements_2_clips), 8)
        self.assertEqual(elements_2_clips, input_list[:4] + input_list[-4:])

        # 11 second clip at 2 fps, seconds 0-1 and 10-11 are included
        frame_filter_fn = EpicKitchenForecasting._frame_filter_generator(
            seconds_per_clip=1,
            num_input_clips=2,
            frames_per_clip=2,
            clip_time_stride=10,
        )
        elements_2_clips_2fps = frame_filter_fn(input_list)
        self.assertEqual(len(elements_2_clips_2fps), 4)
        self.assertEqual(elements_2_clips_2fps, [0, 2, 40, 42])

    def test_define_clip_structure_generator(self):
        frame_videos = {
            "P01_003": FrameVideo.from_frame_paths(
                [f"root/P01_003/frame_{i}" for i in range(200)], 10
            ),
            "P02_004": FrameVideo.from_frame_paths(
                [f"root/P02_004/frame_{i}" for i in range(300)], 10
            ),
            "P11_010": FrameVideo.from_frame_paths(
                [f"root/P11_010/frame_{i}" for i in range(600)], 30
            ),
        }
        actions = {
            "P01_003": [
                ActionData(
                    "P01",
                    "P01_003",
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
                    "P01_003",
                    "turn on light",
                    "00:00:04.00",
                    "00:00:05.00",
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
                    "P01_003",
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
                    "P01_003",
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
            "P02_004": [
                ActionData(
                    "P02",
                    "P02_004",
                    "turn off light",
                    "00:00:04.00",
                    "00:00:05.00",
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
                    "P02",
                    "P02_004",
                    "turn on light",
                    "00:00:05.00",
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
                    "P02",
                    "P02_004",
                    "close door",
                    "00:00:08.00",
                    "00:00:09.00",
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
                    "P02",
                    "P02_004",
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
            "P11_010": [
                ActionData(
                    "P11",
                    "P11_010",
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
                    "P11",
                    "P11_010",
                    "turn on light",
                    "00:00:04.00",
                    "00:00:05.50",
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
                    "P11",
                    "P11_010",
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
                    "P11",
                    "P11_010",
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
                    "P11",
                    "P11_010",
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
        random_value = 0.5
        with unittest.mock.patch("random.random", return_value=random_value) as _:
            define_clip_structure_fn = (
                EpicKitchenForecasting._define_clip_structure_generator(
                    seconds_per_clip=1,
                    clip_time_stride=3,
                    num_input_clips=2,
                    num_forecast_actions=2,
                    clip_sampling=ClipSampling.Random,
                )
            )
            clips = define_clip_structure_fn(frame_videos, actions)
            sorted_clips = sorted(clips, key=lambda c: c.start_time)  # For stability
            for clip in sorted_clips:
                self.assertEqual(clip.stop_time - clip.start_time, 4.0)

            clips_P01_003 = [c for c in sorted_clips if c.video_id == "P01_003"]
            self.assertEqual(len(clips_P01_003), 1)

            clips_P01_003[0].start_time == actions["P01_003"][1].stop_time

            clips_P02_004 = [c for c in sorted_clips if c.video_id == "P02_004"]
            self.assertEqual(len(clips_P02_004), 2)
            clips_P02_004[0].start_time == actions["P02_004"][0].stop_time
            clips_P02_004[1].start_time == actions["P02_004"][1].stop_time

            clips_P11_010 = [c for c in sorted_clips if c.video_id == "P11_010"]
            self.assertEqual(len(clips_P11_010), 1)
            clips_P11_010[0].start_time == actions["P11_010"][1].stop_time
