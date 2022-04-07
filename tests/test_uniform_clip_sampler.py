# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import unittest
from typing import Optional

import numpy as np
from parameterized import parameterized
from pytorchvideo.data.clip_sampling import UniformClipSampler


def _num_clips(
    duration_sec: float,
    fps: float,
    stride_frames: int,
    window_size_frames: int,
    backpad_last: bool = True,
) -> int:
    """
    Utility to calculate the number of clips for a given duration, fps, stride & window_size
    """
    num_frames = round(duration_sec * fps)
    N = num_frames - window_size_frames
    if N < 0:
        return 1
    result = int(N / stride_frames + 1)
    pad = backpad_last and N % stride_frames != 0
    return result + pad


class TestUniformClipSampler(unittest.TestCase):
    @parameterized.expand(
        [
            (False, 30, 32, 16, 32 / 30, 1),
            (True, 30, 32, 16, 32 / 30, 1),
            (True, 30, 32, 16, 33 / 30, 2),
            (False, 30, 32, 16, 34 / 30, 1),
            (True, 30, 32, 16, 34 / 30, 2),
            (False, 30, 32, 16, 47 / 30, 1),
            (True, 30, 32, 16, 47 / 30, 2),
            (False, 30, 32, 16, 48 / 30, 2),
            (True, 30, 32, 16, 48 / 30, 2),
            (False, 30, 32, 16, 72 / 30, 3),
            (True, 30, 32, 16, 72 / 30, 4),
            (False, 30, 32, 16, 109 / 30, 5),
            (True, 30, 32, 16, 109 / 30, 6),
            (False, 30, 32, 3, 35 / 30, 2),
            (True, 30, 32, 3, 35 / 30, 2),
            (False, 30, 32, 3, 36 / 30, 2),
            (True, 30, 32, 3, 36 / 30, 3),
            (True, 30, 32, 3, 35 / 30, 2),
            (False, 30, 32, 3, 36 / 30, 2),
            (True, 30, 32, 3, 36 / 30, 3),
            # no stride => window size
            (False, 30, 32, 32, 32 / 30, 1),
            (True, 30, 32, 32, 32 / 30, 1),
            (False, 30, 32, 32, 54 / 30, 1),
            (True, 30, 32, 32, 54 / 30, 2),
            (False, 30, 32, 32, 64 / 30, 2),
            (True, 30, 32, 32, 64 / 30, 2),
            # test None for stride
            (False, 30, 32, None, 64 / 30, 2),
            (True, 30, 32, None, 64 / 30, 2),
            # stride = {1, 2}
            (False, 30, 2, 1, 32 / 30, 31),
            (True, 30, 2, 1, 32 / 30, 31),
            # > half stride
            (False, 30, 32, 24, 107 / 30, 4),
            (True, 30, 32, 24, 107 / 30, 5),
            (False, 30, 5, 1, 11 / 30, 7),
            (True, 30, 5, 1, 11 / 30, 7),
            # stride > window size
            (False, 30, 1, 5, 11 / 30, 3),
            (True, 30, 1, 5, 11 / 30, 3),
            (True, 30, 1, 5, 1759 / 30, 353),
            (False, 30, 3, 10, 132 / 30, 13),
            (True, 30, 3, 10, 132 / 30, 14),
            (False, 30, 6, 10, 111 / 30, 11),
            (True, 30, 6, 10, 111 / 30, 12),
            # stride <= window size
            (False, 30, 10, 3, 132 / 30, 41),
            (True, 30, 10, 3, 132 / 30, 42),
            (False, 30, 10, 6, 111 / 30, 17),
            (True, 30, 10, 6, 111 / 30, 18),
            (True, 30, 1, 1, 132 / 30, 132),
        ]
    )
    def test_uniform_clip_sampler(
        self,
        backpad_last: bool,
        fps: int,
        window_size: int,
        stride_frames: Optional[int],
        video_length: float,
        expected_number_of_clips: int,
    ):
        """
        Utility to test the uniform clip sampler
        """
        sampler = UniformClipSampler(
            window_size / fps,
            stride_frames / fps if stride_frames is not None else None,
            backpad_last=backpad_last,
        )
        predicted_n_clips = _num_clips(
            video_length,
            fps,
            stride_frames=stride_frames if stride_frames is not None else window_size,
            window_size_frames=window_size,
            backpad_last=backpad_last,
        )
        self.assertEqual(predicted_n_clips, expected_number_of_clips)

        s_prime = stride_frames if stride_frames is not None else window_size
        expected_start_end_times = [
            ((i * s_prime) / fps, ((i * s_prime + window_size) / fps))
            for i in range(expected_number_of_clips)
        ]
        if expected_start_end_times[-1][1] - video_length > 1e-6:
            expected_start_end_times[-1] = (
                video_length - window_size / fps,
                video_length,
            )

        self.assertTrue(
            (
                expected_start_end_times[-1][0] + (s_prime / fps) > video_length
                or expected_start_end_times[-1][-1] + (s_prime / fps) > video_length
            )
        )
        if len(expected_start_end_times) >= 2:
            self.assertNotAlmostEqual(
                expected_start_end_times[-2][0], expected_start_end_times[-1][0]
            )
            self.assertNotAlmostEqual(
                expected_start_end_times[-2][1], expected_start_end_times[-1][1]
            )

        start_end_times = []

        last_clip_time = None
        annotation = {}
        while True:
            clip = sampler(last_clip_time, video_length, annotation)
            last_clip_time = copy.deepcopy(clip.clip_end_sec)
            n_frames = (clip.clip_end_sec - clip.clip_start_sec) * fps
            int_n_frames = int(np.round(float(n_frames)))
            self.assertAlmostEqual(float(int_n_frames), float(n_frames))
            self.assertEqual(int_n_frames, window_size)

            start_end_times.append(
                (float(clip.clip_start_sec), float(clip.clip_end_sec))
            )
            if clip.is_last_clip:
                break

            # just in case we get an infinite loop
            if len(start_end_times) > 2 * expected_number_of_clips:
                break

        self.assertEqual(len(start_end_times), expected_number_of_clips)
        for (start, end), (expected_start, expected_end) in zip(
            start_end_times, expected_start_end_times
        ):
            self.assertAlmostEqual(float(start), expected_start)
            self.assertAlmostEqual(float(end), expected_end)

    @parameterized.expand(
        [
            (60 / 30, 30, 16, 32, True, 3),
            (60 / 30, 30, 16, 32, True, 3),
            (5 / 30, 30, 2, 3, True, 2),
            (39 / 30, 30, 3, 32, True, 4),
            (9 / 30, 30, 2, 2, True, 5),
            (10 / 30, 30, 2, 2, True, 5),
            (39 / 30, 30, 16, 32, True, 2),
            (39 / 30, 30, 31, 32, True, 2),
            (203 / 30, 30, 2, 32, True, 87),
            (203 / 30, 30, 3, 32, True, 58),
            (203 / 30, 30, 31, 32, True, 7),
            (60 / 30, 30, 16, 32, False, 2),
            (60 / 30, 30, 16, 32, False, 2),
            (5 / 30, 30, 2, 3, False, 2),
            (39 / 30, 30, 3, 32, False, 3),
            (9 / 30, 30, 2, 2, False, 4),
            (10 / 30, 30, 2, 2, False, 5),
            (39 / 30, 30, 16, 32, False, 1),
            (39 / 30, 30, 31, 32, False, 1),
            (203 / 30, 30, 2, 32, False, 86),
            (203 / 30, 30, 3, 32, False, 58),
            (203 / 30, 30, 31, 32, False, 6),
            (203 / 30, 30, 1, 32, False, 203 - 32 + 1),
            (19 / 30, 30, 1, 32, False, 1),
            (19 / 30, 30, 1, 32, True, 1),
            (33 / 30, 30, 1, 32, False, 2),
            (33 / 30, 30, 1, 32, True, 2),
            (11 / 30, 30, 1, 5, False, 7),
            (11 / 30, 30, 1, 5, True, 7),
            (11 / 30, 30, 5, 1, False, 3),
            (11 / 30, 30, 5, 1, True, 3),
            (1759 / 30, 30, 5, 1, True, 353),
        ]
    )
    def test_num_clips(
        self,
        duration_sec: float,
        fps: int,
        stride_frames: int,
        window_size_frames: int,
        backpad_last: bool,
        expected: int,
    ):
        self.assertEqual(
            _num_clips(
                duration_sec, fps, stride_frames, window_size_frames, backpad_last
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
