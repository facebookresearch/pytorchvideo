#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import tempfile
import unittest

import av
import pytest
from pytorchvideo.data.encoded_video import EncodedVideo
from utils import temp_encoded_video, temp_encoded_video_with_audio


class TestEncodedVideo(unittest.TestCase):
    def test_video_works(self):
        num_frames = 11
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (file_name, data):
            test_video = EncodedVideo(file_name)
            self.assertAlmostEqual(test_video.duration, num_frames / fps)

            # All frames
            frames, audio_samples = test_video.get_clip(0, test_video.duration)
            self.assertTrue(frames.equal(data))
            self.assertEqual(audio_samples, None)

            # Half frames
            frames, audio_samples = test_video.get_clip(0, test_video.duration / 2)
            self.assertTrue(frames.equal(data[:, : round(num_frames / 2)]))
            self.assertEqual(audio_samples, None)

            # No frames
            error = test_video.get_clip(
                test_video.duration + 1, test_video.duration + 3
            )
            self.assertEqual(error, None)
            self.assertEqual(error, None)

            test_video.close()

    def test_video_with_shorter_audio_works(self):
        num_audio_samples = 8000
        num_frames = 5
        fps = 5
        audio_rate = 8000
        with temp_encoded_video_with_audio(
            num_frames=num_frames,
            fps=fps,
            num_audio_samples=num_audio_samples,
            audio_rate=audio_rate,
        ) as (file_name, video_data, audio_data):
            test_video = EncodedVideo(file_name)

            # Duration is max of both streams, therefore, the video duration will be expected.
            self.assertEqual(test_video.duration, num_frames / fps)

            # All audio
            frames, audio_samples = test_video.get_clip(0, test_video.duration)
            self.assertTrue(frames.equal(video_data))
            self.assertTrue(audio_samples.equal(audio_data))

            # Half frames
            frames, audio_samples = test_video.get_clip(0, test_video.duration / 2)
            self.assertTrue(frames.equal(video_data[:, : num_frames // 2]))
            self.assertTrue(audio_samples.equal(audio_data))

            test_video.close()

    def test_video_with_longer_audio_works(self):
        audio_rate = 10000
        fps = 5
        num_frames = 5
        num_audio_samples = 40000
        with temp_encoded_video_with_audio(
            num_frames=num_frames,
            fps=fps,
            num_audio_samples=num_audio_samples,
            audio_rate=audio_rate,
        ) as (file_name, video_data, audio_data):
            test_video = EncodedVideo(file_name)

            # All audio
            frames, audio_samples = test_video.get_clip(0, test_video.duration)
            self.assertTrue(frames.equal(video_data))
            self.assertTrue(audio_samples.equal(audio_data))

            # No frames (3 - 5 seconds)
            frames = test_video.get_clip(
                test_video.duration + 1, test_video.duration + 2
            )
            self.assertEqual(frames, None)

            test_video.close()

    def test_open_video_failure(self):
        with pytest.raises(FileNotFoundError):
            test_video = EncodedVideo("non_existent_file.txt")
            test_video.close()

    def test_decode_video_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"This is not an mp4 file")
            with pytest.raises(av.AVError):
                test_video = EncodedVideo(f.name)
                test_video.close()
