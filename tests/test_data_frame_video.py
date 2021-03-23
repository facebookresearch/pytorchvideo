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
            test_video = FrameVideo.from_frame_paths(frame_paths)
            expected_duration = (
                0.1  # Total duration of 3 frames at 30fps is 0.1 seconds.
            )
            self.assertEqual(test_video.duration, expected_duration)

            # All frames (0 - 0.1 seconds)
            clip = test_video.get_clip(0, 0.1)
            frames, indices = clip["video"], clip["frame_indices"]
            self.assertTrue(frames.equal(data))
            self.assertEqual(indices, [0, 1, 2])

            # All frames (0 - 0.1 seconds), filtred to middle frame
            clip = test_video.get_clip(0, 0.1, lambda lst: lst[1:2])
            frames, indices = clip["video"], clip["frame_indices"]
            self.assertTrue(frames.equal(data[:, 1:2]))
            self.assertEqual(indices, [1])

            # 2 frames (0 - 0.066 seconds)
            clip = test_video.get_clip(0, 0.066)
            frames, indices = clip["video"], clip["frame_indices"]
            self.assertTrue(frames.equal(data[:, :2]))
            self.assertEqual(indices, [0, 1])

            # No frames (3 - 5 seconds)
            result = test_video.get_clip(3, 5)
            self.assertEqual(result, None)

    def test_open_video_failure(self):
        test_video = FrameVideo.from_frame_paths(["non_existent_file.txt"])
        with pytest.raises(Exception):
            test_video.get_clip(0, 0.01)  # duration is 1 / 30 because one frame

    def test_empty_frames_failure(self):
        with pytest.raises(AssertionError):
            FrameVideo.from_frame_paths([])
