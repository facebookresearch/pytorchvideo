#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import tempfile
import unittest

import av
import pytest
from pytorchvideo.data.encoded_video import EncodedVideo
from utils import temp_encoded_video


class TestEncodedVideo(unittest.TestCase):
    def test_video_with_header_info_works(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (f_name, data):

            test_video = EncodedVideo(f_name)
            self.assertEqual(test_video.duration, num_frames / fps)

            # All frames
            frames = test_video.get_clip(0, test_video.duration)
            self.assertTrue(frames.equal(data))

            # Half frames. eps(1e-6) is subtracted from half duration because clip
            # sampling is start_time inclusive.
            frames = test_video.get_clip(0, test_video.duration / 2 - 1e-6)
            self.assertTrue(frames.equal(data[:, : num_frames // 2]))

            # No frames (3 - 5 seconds)
            frames = test_video.get_clip(
                test_video.duration + 1, test_video.duration + 2
            )
            self.assertEqual(frames, None)

            test_video.close()

    def test_open_video_failure(self):
        with pytest.raises(av.AVError):
            test_video = EncodedVideo("non_existent_file.txt")
            test_video.close()

    def test_decode_video_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"This is not an mp4 file")
            with pytest.raises(av.AVError):
                test_video = EncodedVideo(f.name)
                test_video.close()
