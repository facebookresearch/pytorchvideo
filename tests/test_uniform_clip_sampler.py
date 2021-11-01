# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
from typing import Optional

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

    result = N // stride_frames + 1

    # handle padded frame
    if backpad_last and N % stride_frames != 0:
        result += 1
    return result


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

        last_clip_time = 0
        annotation = {}
        n_clips = 0
        while True:
            clip = sampler(last_clip_time, video_length, annotation)
            last_clip_time = clip.clip_end_sec
            n_clips += 1
            if clip.is_last_clip:
                break

            # just in case we get an infinite loop
            if n_clips > 2 * expected_number_of_clips:
                break

        predicted_n_clips = _num_clips(
            video_length,
            fps,
            stride_frames=stride_frames if stride_frames is not None else window_size,
            window_size_frames=window_size,
            backpad_last=backpad_last,
        )
        self.assertEqual(predicted_n_clips, expected_number_of_clips)
        self.assertEqual(n_clips, expected_number_of_clips)

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
