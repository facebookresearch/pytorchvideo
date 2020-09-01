#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pathlib
import tempfile
import unittest
import unittest.mock
from typing import List, Tuple
from unittest.mock import Mock, patch

# av import has to be added for `buck test` to work.
import av  # noqa: F401
import torch
from pytorchvideo.data import Kinetics
from pytorchvideo.data.clip_sampling import make_clip_sampler
from pytorchvideo.data.utils import MultiProcessSampler, thwc_to_cthw
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils import create_video_frames, temp_encoded_video


class TestKineticsDataset(unittest.TestCase):
    # Clip sampling is start time inclusive so we need to subtract _EPS from
    # total_duration / 2 to sample half of the frames of a video.
    _EPS = 1e-9

    def test_single_clip_per_video_works(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            total_duration = num_frames / fps
            clip_sampler = make_clip_sampler("uniform", total_duration)
            dataset = Kinetics(
                f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            expected = [(0, data), (1, data)]

            for i, sample in enumerate(dataset):
                self.assertTrue(sample["video"].equal(expected[i][1]))
                self.assertEqual(sample["label"], expected[i][0])

    def test_multiple_clips_per_video_works(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            total_duration = num_frames / fps
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("uniform", half_duration)
            dataset = Kinetics(
                f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            half_frames = num_frames // 2
            first_half_data = data[:, :half_frames]
            second_half_data = data[:, half_frames:]
            expected = [
                (0, first_half_data),
                (0, second_half_data),
                (1, first_half_data),
                (1, second_half_data),
            ]

            for i, sample in enumerate(dataset):
                self.assertEqual(sample["label"], expected[i][0])
                self.assertTrue(sample["video"].equal(expected[i][1]))

    def test_video_name_with_whitespace_works(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps, prefix="pre fix") as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            total_duration = num_frames / fps
            clip_sampler = make_clip_sampler("uniform", total_duration)
            dataset = Kinetics(
                f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            expected = [(0, data), (1, data)]

            for i, sample in enumerate(dataset):
                self.assertTrue(sample["video"].equal(expected[i][1]))
                self.assertEqual(sample["label"], expected[i][0])

    def test_random_clip_sampling_works(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            total_duration = num_frames / fps
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("random", half_duration)
            dataset = Kinetics(
                f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            # [(expected_label, expected_t_shape), ...]
            expected = [(0, 5), (1, 5)]

            for i, sample in enumerate(dataset):
                self.assertEqual(sample["video"].shape[1], expected[i][1])
                self.assertEqual(sample["label"], expected[i][0])

    def test_reading_from_directory_structure(self):
        # For an unknown reason this import has to be here for `buck test` to work.
        import torchvision.io as io

        with tempfile.TemporaryDirectory() as root_dir:

            # Create test directory structure with two classes and a video in each.
            root_dir_name = pathlib.Path(root_dir)
            test_class_1 = root_dir_name / "running"
            test_class_1.mkdir()
            data_1 = create_video_frames(15, 10, 10)
            test_class_2 = root_dir_name / "cleaning windows"
            test_class_2.mkdir()
            data_2 = create_video_frames(20, 15, 15)
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", dir=test_class_1
            ) as f_1, tempfile.NamedTemporaryFile(
                suffix=".mp4", dir=test_class_2
            ) as f_2:
                f_1.close()
                f_2.close()

                # Write lossless video for each class.
                io.write_video(
                    f_1.name,
                    data_1,
                    fps=30,
                    video_codec="libx264rgb",
                    options={"crf": "0"},
                )
                io.write_video(
                    f_2.name,
                    data_2,
                    fps=30,
                    video_codec="libx264rgb",
                    options={"crf": "0"},
                )

                clip_sampler = make_clip_sampler("uniform", 3)
                dataset = Kinetics(
                    root_dir, clip_sampler=clip_sampler, video_sampler=SequentialSampler
                )

                # Videos are sorted alphabetically so "cleaning windows" (i.e. data_2)
                # will be first.
                sample_1 = next(dataset)
                self.assertEqual(sample_1["label"], 0)
                self.assertTrue(sample_1["video"].equal(thwc_to_cthw(data_2)))

                sample_2 = next(dataset)
                self.assertEqual(sample_2["label"], 1)
                self.assertTrue(sample_2["video"].equal(thwc_to_cthw(data_1)))

    def test_sampling_with_multiple_processes(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            total_duration = num_frames / fps
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("uniform", half_duration)
            dataset = Kinetics(
                f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            half_frames = num_frames // 2
            first_half_data = data[:, :half_frames]
            second_half_data = data[:, half_frames:]
            expected = [
                (0, first_half_data),
                (0, second_half_data),
                (1, first_half_data),
                (1, second_half_data),
            ]

            test_dataloader = DataLoader(dataset, batch_size=None, num_workers=2)
            actual = [(sample["label"], sample["video"]) for sample in test_dataloader]
            self.assertTrue(unordered_list_compare(expected, actual))

    def test_sampling_with_non_divisible_processes_by_videos(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())
                f.write(f"{video_file_name} 2\n".encode())

            total_duration = num_frames / fps
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("uniform", half_duration)
            dataset = Kinetics(
                f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            half_frames = num_frames // 2
            first_half_data = data[:, :half_frames]
            second_half_data = data[:, half_frames:]
            expected = [
                (0, first_half_data),
                (0, second_half_data),
                (2, first_half_data),
                (2, second_half_data),
                (1, first_half_data),
                (1, second_half_data),
            ]

            test_dataloader = DataLoader(dataset, batch_size=None, num_workers=2)
            actual = [(sample["label"], sample["video"]) for sample in test_dataloader]
            self.assertTrue(unordered_list_compare(expected, actual))

    def test_sampling_with_more_processes_than_videos(self):
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())
                f.write(f"{video_file_name} 2\n".encode())

            total_duration = num_frames / fps
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("uniform", half_duration)
            dataset = Kinetics(
                f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            half_frames = num_frames // 2
            first_half_data = data[:, :half_frames]
            second_half_data = data[:, half_frames:]
            expected = [
                (0, first_half_data),
                (0, second_half_data),
                (2, first_half_data),
                (2, second_half_data),
                (1, first_half_data),
                (1, second_half_data),
            ]
            test_dataloader = DataLoader(dataset, batch_size=None, num_workers=8)
            actual = [(sample["label"], sample["video"]) for sample in test_dataloader]
            self.assertTrue(unordered_list_compare(expected, actual))

    def test_sampling_with_non_divisible_processes_by_clips(self):

        # Make one video with 15 frames and one with 10 frames, producing 3 clips and 2
        # clips respectively.
        num_frames = 10
        fps = 5
        with temp_encoded_video(num_frames=int(num_frames * 1.5), fps=fps) as (
            video_file_name_1,
            data_1,
        ):
            with temp_encoded_video(num_frames=num_frames, fps=fps) as (
                video_file_name_2,
                data_2,
            ):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                    f.write(f"{video_file_name_1} 0\n".encode())
                    f.write(f"{video_file_name_2} 1\n".encode())

                total_duration = num_frames / fps
                half_duration = total_duration / 2 - self._EPS
                clip_sampler = make_clip_sampler("uniform", half_duration)
                dataset = Kinetics(
                    f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
                )

                half_frames = num_frames // 2
                expected = {
                    (0, data_1[:, half_frames * 2 :]),  # 1/3 clip
                    (0, data_1[:, half_frames : half_frames * 2]),  # 2/3 clip
                    (0, data_1[:, :half_frames]),  # 3/3/ clip
                    (1, data_2[:, :half_frames]),  # First half
                    (1, data_2[:, half_frames:]),  # Second half
                }

                test_dataloader = DataLoader(dataset, batch_size=None, num_workers=2)
                actual = [
                    (sample["label"], sample["video"]) for sample in test_dataloader
                ]
                self.assertTrue(unordered_list_compare(expected, actual))

    def test_multi_process_sampler(self):
        # Test coverage ignores multi-process lines of code so we need to mock out
        # the multiprocess environment information to test in a single process.
        with patch("torch.utils.data.get_worker_info") as get_worker_info:
            get_worker_info.return_value = Mock(id=2, num_workers=3)
            inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            dataset = TensorDataset(inps, tgts)
            sampler = iter(MultiProcessSampler(SequentialSampler(dataset)))

            # Sampler indices will be split into 3. The last worker (id=2) will have the
            # last 3 indices (7, 8, 9).
            self.assertEqual(list(sampler), [7, 8, 9])


def unordered_list_compare(
    expected: List[Tuple[int, torch.Tensor]], actual: List[Tuple[int, torch.Tensor]]
):
    """
    Returns:
        True if all tuple values from expected found in actual and lengths are equal.
    """
    if len(actual) != len(expected):
        return False

    for expected_x in expected:

        # Uses torch comparator for Tensor.
        if not any(
            actual_x[0] == expected_x[0] and actual_x[1].equal(expected_x[1])
            for actual_x in actual
        ):
            return False

    return True
