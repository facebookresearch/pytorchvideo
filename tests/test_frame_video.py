#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import pytest
from pytorchvideo.data.frame_video import FrameVideo
from utils import temp_frame_video


class TestFrameVideo(unittest.TestCase):
    def test_frame_video_works(self):
        frame_names = [f"{str(i)}.png" for i in range(3)]
        with temp_frame_video(frame_names) as (f_name, data):
            frame_paths = [f_name / x for x in frame_names]
            test_video = FrameVideo(frame_paths)
            expected_duration = (
                0.1  # Total duration of 3 frames at 30fps is 0.1 seconds.
            )
            self.assertEqual(test_video.duration, expected_duration)

            # All frames (0 - 0.1 seconds)
            frames, indices = test_video.get_clip(0, 0.1)
            self.assertTrue(frames.equal(data))
            self.assertEqual(indices, [0, 1, 2])

            # 2 frames (0 - 0.066 seconds)
            frames, indices = test_video.get_clip(0, 0.066)
            self.assertTrue(frames.equal(data[:, :2]))
            self.assertEqual(indices, [0, 1])

            # No frames (3 - 5 seconds)
            result = test_video.get_clip(3, 5)
            self.assertEqual(result, None)

    def test_open_video_failure(self):
        test_video = FrameVideo(["non_existent_file.txt"])
        with pytest.raises(Exception):
            test_video.get_clip(0, 0.01)  # duration is 1 / 30 because one frame

    def test_empty_frames_failure(self):
        with pytest.raises(AssertionError):
            FrameVideo([])
