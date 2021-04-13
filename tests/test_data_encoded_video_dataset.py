# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import collections
import contextlib
import itertools
import math
import multiprocessing
import os
import pathlib
import tempfile
import unittest
import unittest.mock
from typing import List, Tuple
from unittest.mock import Mock, patch

# av import has to be added for `buck test` to work.
import av  # noqa: F401
import torch
import torch.distributed as dist
from parameterized import parameterized
from pytorchvideo.data import Hmdb51
from pytorchvideo.data.clip_sampling import make_clip_sampler
from pytorchvideo.data.encoded_video_dataset import (
    EncodedVideoDataset,
    labeled_encoded_video_dataset,
)
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler, thwc_to_cthw
from torch.multiprocessing import Process
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from utils import create_dummy_video_frames, temp_encoded_video


DECODER_LIST = [("pyav",), ("torchvision",)]


class TestEncodedVideoDataset(unittest.TestCase):
    # Clip sampling is start time inclusive so we need to subtract _EPS from
    # total_duration / 2 to sample half of the frames of a video.
    _EPS = 1e-9

    def setUp(self):
        # Fail fast for tests
        EncodedVideoDataset._MAX_CONSECUTIVE_FAILURES = 1

    @parameterized.expand(DECODER_LIST)
    def test_single_clip_per_video_works(self, decoder):
        with mock_encoded_video_dataset_file() as (mock_csv, expected, total_duration):
            clip_sampler = make_clip_sampler("uniform", total_duration)
            dataset = labeled_encoded_video_dataset(
                data_path=mock_csv,
                clip_sampler=clip_sampler,
                video_sampler=SequentialSampler,
                decode_audio=False,
                decoder=decoder,
            )
            test_dataloader = DataLoader(dataset, batch_size=None, num_workers=2)

            for _ in range(2):
                actual = [
                    (sample["label"], sample["video"]) for sample in test_dataloader
                ]
                assert_unordered_list_compare_true(self, expected, actual)

    @parameterized.expand(DECODER_LIST)
    def test_video_name_with_whitespace_works(self, decoder):
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
            labeled_video_paths = LabeledVideoPaths.from_path(f.name)
            dataset = EncodedVideoDataset(
                labeled_video_paths,
                clip_sampler=clip_sampler,
                video_sampler=SequentialSampler,
                decode_audio=False,
                decoder=decoder,
            )

            expected = [(0, data), (1, data)]
            for i, sample in enumerate(dataset):
                self.assertTrue(sample["video"].equal(expected[i][1]))
                self.assertEqual(sample["label"], expected[i][0])

    @parameterized.expand(DECODER_LIST)
    def test_random_clip_sampling_works(self, decoder):
        with mock_encoded_video_dataset_file() as (
            mock_csv,
            label_videos,
            total_duration,
        ):
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("random", half_duration)
            labeled_video_paths = LabeledVideoPaths.from_path(mock_csv)
            dataset = EncodedVideoDataset(
                labeled_video_paths,
                clip_sampler=clip_sampler,
                video_sampler=SequentialSampler,
                decode_audio=False,
                decoder=decoder,
            )

            expected_labels = [label for label, _ in label_videos]
            for i, sample in enumerate(dataset):
                expected_t_shape = 5
                self.assertEqual(sample["video"].shape[1], expected_t_shape)
                self.assertEqual(sample["label"], expected_labels[i])

    @parameterized.expand(DECODER_LIST)
    def test_reading_from_directory_structure_hmdb51(self, decoder):
        # For an unknown reason this import has to be here for `buck test` to work.
        import torchvision.io as io

        with tempfile.TemporaryDirectory() as root_dir:

            # Create test directory structure with two classes and a video in each.
            root_dir_name = pathlib.Path(root_dir)
            action_1 = "running"
            action_2 = "cleaning_windows"

            videos_root_dir = root_dir_name / "videos"
            videos_root_dir.mkdir()

            test_class_1 = videos_root_dir / action_1
            test_class_1.mkdir()
            data_1 = create_dummy_video_frames(15, 10, 10)
            test_class_2 = videos_root_dir / action_2
            test_class_2.mkdir()
            data_2 = create_dummy_video_frames(20, 15, 15)

            test_splits = root_dir_name / "folds"
            test_splits.mkdir()

            with tempfile.NamedTemporaryFile(
                suffix="_u_nm_np1_ba_goo_19.avi", dir=test_class_1
            ) as f_1, tempfile.NamedTemporaryFile(
                suffix="_u_nm_np1_fr_med_1.avi", dir=test_class_2
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

                _, video_name_1 = os.path.split(f_1.name)
                _, video_name_2 = os.path.split(f_2.name)

                with open(
                    os.path.join(test_splits, action_1 + "_test_split1.txt"), "w"
                ) as f:
                    f.write(f"{video_name_1} 1\n")

                with open(
                    os.path.join(test_splits, action_2 + "_test_split1.txt"), "w"
                ) as f:
                    f.write(f"{video_name_2} 1\n")

                clip_sampler = make_clip_sampler("uniform", 3)
                dataset = Hmdb51(
                    data_path=test_splits,
                    video_path_prefix=root_dir_name / "videos",
                    clip_sampler=clip_sampler,
                    video_sampler=SequentialSampler,
                    split_id=1,
                    split_type="train",
                    decode_audio=False,
                    decoder=decoder,
                )

                # Videos are sorted alphabetically so "cleaning windows" (i.e. data_2)
                # will be first.
                sample_1 = next(dataset)
                sample_2 = next(dataset)

                self.assertTrue(sample_1["label"] in [action_1, action_2])
                if sample_1["label"] == action_2:
                    sample_1, sample_2 = sample_2, sample_1

                self.assertEqual(sample_1["label"], action_1)
                self.assertEqual(5, len(sample_1["meta_tags"]))
                self.assertTrue(
                    sample_1["video"].equal(thwc_to_cthw(data_1).to(torch.float32))
                )

                self.assertEqual(sample_2["label"], action_2)
                self.assertEqual(5, len(sample_2["meta_tags"]))
                self.assertTrue(
                    sample_2["video"].equal(thwc_to_cthw(data_2).to(torch.float32))
                )

    @parameterized.expand(DECODER_LIST)
    def test_constant_clips_per_video_sampling_works(self, decoder):
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

                clip_frames = 2
                duration_for_frames = clip_frames / fps - self._EPS
                clip_sampler = make_clip_sampler(
                    "constant_clips_per_video", duration_for_frames, 2
                )
                labeled_video_paths = LabeledVideoPaths.from_path(f.name)
                dataset = EncodedVideoDataset(
                    labeled_video_paths,
                    clip_sampler=clip_sampler,
                    video_sampler=SequentialSampler,
                    decode_audio=False,
                    decoder=decoder,
                )

                # Dataset has 2 videos. Each video has two evenly spaced clips of size
                # clip_frames sampled. The first clip of each video will always be
                # sampled at second 0. The second clip of the video is the next frame
                # from time: (total_duration - clip_duration) / 2
                half_frames_1 = math.ceil((data_1.shape[1] - clip_frames) / 2)
                half_frames_2 = math.ceil((data_2.shape[1] - clip_frames) / 2)
                expected = [
                    (0, data_1[:, :clip_frames]),
                    (0, data_1[:, half_frames_1 : half_frames_1 + clip_frames]),
                    (1, data_2[:, :clip_frames]),
                    (1, data_2[:, half_frames_2 : half_frames_2 + clip_frames]),
                ]
                for i, sample in enumerate(dataset):
                    self.assertTrue(sample["video"].equal(expected[i][1]))
                    self.assertEqual(sample["label"], expected[i][0])

    @parameterized.expand(DECODER_LIST)
    def test_reading_from_directory_structure(self, decoder):
        # For an unknown reason this import has to be here for `buck test` to work.
        import torchvision.io as io

        with tempfile.TemporaryDirectory() as root_dir:

            # Create test directory structure with two classes and a video in each.
            root_dir_name = pathlib.Path(root_dir)
            test_class_1 = root_dir_name / "running"
            test_class_1.mkdir()
            data_1 = create_dummy_video_frames(15, 10, 10)
            test_class_2 = root_dir_name / "cleaning windows"
            test_class_2.mkdir()
            data_2 = create_dummy_video_frames(20, 15, 15)
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
                labeled_video_paths = LabeledVideoPaths.from_path(root_dir)
                dataset = EncodedVideoDataset(
                    labeled_video_paths,
                    clip_sampler=clip_sampler,
                    video_sampler=SequentialSampler,
                    decode_audio=False,
                    decoder=decoder,
                )

                # Videos are sorted alphabetically so "cleaning windows" (i.e. data_2)
                # will be first.
                sample_1 = next(dataset)
                self.assertEqual(sample_1["label"], 0)
                self.assertTrue(
                    sample_1["video"].equal(thwc_to_cthw(data_2).to(torch.float32))
                )

                sample_2 = next(dataset)
                self.assertEqual(sample_2["label"], 1)
                self.assertTrue(
                    sample_2["video"].equal(thwc_to_cthw(data_1).to(torch.float32))
                )

    @parameterized.expand(DECODER_LIST)
    def test_random_video_sampler(self, decoder):
        with mock_encoded_video_dataset_file() as (mock_csv, expected, total_duration):
            clip_sampler = make_clip_sampler("uniform", total_duration)
            dataset = labeled_encoded_video_dataset(
                data_path=mock_csv,
                clip_sampler=clip_sampler,
                video_sampler=RandomSampler,
                decode_audio=False,
                decoder=decoder,
            )

            for _ in range(2):
                actual = [(sample["label"], sample["video"]) for sample in dataset]
                assert_unordered_list_compare_true(self, expected, actual)

    @parameterized.expand(itertools.product([0, 1, 2], ["pyav", "torchvision"]))
    def test_random_video_sampler_multiprocessing(self, num_workers, decoder):
        with mock_encoded_video_dataset_file() as (mock_csv, expected, total_duration):
            clip_sampler = make_clip_sampler("uniform", total_duration)
            dataset = labeled_encoded_video_dataset(
                data_path=mock_csv,
                clip_sampler=clip_sampler,
                video_sampler=RandomSampler,
                decode_audio=False,
                decoder=decoder,
            )
            test_dataloader = DataLoader(
                dataset, batch_size=None, num_workers=num_workers
            )

            for _ in range(2):
                actual = [
                    (sample["label"], sample["video"]) for sample in test_dataloader
                ]
                assert_unordered_list_compare_true(self, expected, actual)

    @parameterized.expand(DECODER_LIST)
    def test_sampling_with_multiple_processes(self, decoder):
        with mock_encoded_video_dataset_file() as (
            mock_csv,
            label_videos,
            total_duration,
        ):
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("uniform", half_duration)
            labeled_video_paths = LabeledVideoPaths.from_path(mock_csv)
            dataset = EncodedVideoDataset(
                labeled_video_paths,
                clip_sampler=clip_sampler,
                video_sampler=SequentialSampler,
                decode_audio=False,
                decoder=decoder,
            )

            # Split each full video into two clips.
            expected = []
            for label, data in label_videos:
                num_frames = data.shape[0]
                half_frames = num_frames // 2
                first_half_data = data[:, :half_frames]
                second_half_data = data[:, half_frames:]
                expected.append((label, first_half_data))
                expected.append((label, second_half_data))

            test_dataloader = DataLoader(dataset, batch_size=None, num_workers=2)
            actual = [(sample["label"], sample["video"]) for sample in test_dataloader]
            assert_unordered_list_compare_true(self, expected, actual)

    @parameterized.expand(DECODER_LIST)
    def test_sampling_with_non_divisible_processes_by_videos(self, decoder):
        with mock_encoded_video_dataset_file() as (
            mock_csv,
            label_videos,
            total_duration,
        ):
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("uniform", half_duration)
            labeled_video_paths = LabeledVideoPaths.from_path(mock_csv)
            dataset = EncodedVideoDataset(
                labeled_video_paths,
                clip_sampler=clip_sampler,
                video_sampler=SequentialSampler,
                decode_audio=False,
                decoder=decoder,
            )

            # Split each full video into two clips.
            expected = []
            for label, data in label_videos:
                num_frames = data.shape[0]
                half_frames = num_frames // 2
                first_half_data = data[:, :half_frames]
                second_half_data = data[:, half_frames:]
                expected.append((label, first_half_data))
                expected.append((label, second_half_data))

            test_dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
            actual = [(sample["label"], sample["video"]) for sample in test_dataloader]
            assert_unordered_list_compare_true(self, expected, actual)

    @parameterized.expand(DECODER_LIST)
    def test_sampling_with_more_processes_than_videos(self, decoder):
        with mock_encoded_video_dataset_file() as (
            mock_csv,
            label_videos,
            total_duration,
        ):
            half_duration = total_duration / 2 - self._EPS
            clip_sampler = make_clip_sampler("uniform", half_duration)
            labeled_video_paths = LabeledVideoPaths.from_path(mock_csv)
            dataset = EncodedVideoDataset(
                labeled_video_paths,
                clip_sampler=clip_sampler,
                video_sampler=SequentialSampler,
                decode_audio=False,
                decoder=decoder,
            )

            # Split each full video into two clips.
            expected = []
            for label, data in label_videos:
                num_frames = data.shape[0]
                half_frames = num_frames // 2
                first_half_data = data[:, :half_frames]
                second_half_data = data[:, half_frames:]
                expected.append((label, first_half_data))
                expected.append((label, second_half_data))

            test_dataloader = DataLoader(dataset, batch_size=None, num_workers=16)
            actual = [(sample["label"], sample["video"]) for sample in test_dataloader]
            assert_unordered_list_compare_true(self, expected, actual)

    @parameterized.expand(DECODER_LIST)
    def test_sampling_with_non_divisible_processes_by_clips(self, decoder):

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
                labeled_video_paths = LabeledVideoPaths.from_path(f.name)
                dataset = EncodedVideoDataset(
                    labeled_video_paths,
                    clip_sampler=clip_sampler,
                    video_sampler=SequentialSampler,
                    decode_audio=False,
                    decoder=decoder,
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
                assert_unordered_list_compare_true(self, expected, actual)

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

    @parameterized.expand(DECODER_LIST)
    def test_sampling_with_distributed_sampler(self, decoder):

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

                # Create several processes initialized in a PyTorch distributed process
                # group so that distributed sampler is setup correctly when dataset is
                # constructed.
                num_processes = 2
                processes = []
                return_dict = multiprocessing.Manager().dict()
                for rank in range(num_processes):
                    p = Process(
                        target=run_distributed,
                        args=(
                            rank,
                            num_processes,
                            decoder,
                            half_duration,
                            f.name,
                            return_dict,
                        ),
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                # After joining all distributed processes we expect all these label,
                # video pairs to be returned in random order.
                half_frames = num_frames // 2
                expected = {
                    (0, data_1[:, :half_frames]),  # 1/3 clip
                    (0, data_1[:, half_frames : half_frames * 2]),  # 2/3 clip
                    (0, data_1[:, half_frames * 2 :]),  # 3/3 clip
                    (1, data_2[:, :half_frames]),  # First half
                    (1, data_2[:, half_frames:]),  # Second half
                }

                epoch_results = collections.defaultdict(list)
                for v in return_dict.values():
                    for k_2, v_2 in v.items():
                        epoch_results[k_2].extend(v_2)

                assert_unordered_list_compare_true(
                    self, expected, epoch_results["epoch_1"]
                )
                assert_unordered_list_compare_true(
                    self, expected, epoch_results["epoch_2"]
                )


def assert_unordered_list_compare_true(
    self,
    expected: List[Tuple[int, torch.Tensor]],
    actual: List[Tuple[int, torch.Tensor]],
):
    """
    Asserts True if all tuple values from expected found in actual and lengths are equal.
    """
    expected_str = str([(label, clip.shape) for label, clip in expected])
    actual = str([(label, clip.shape) for label, clip in actual])
    failure_str = f"Expected set: {expected_str}\n actual set: {actual}"
    self.assertTrue(unordered_list_compare, msg=failure_str)


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


def run_distributed(rank, size, decoder, clip_duration, data_name, return_dict):
    """
    This function is run by each distributed process. It samples videos
    based on the distributed split (determined by the
    DistributedSampler) and returns the dataset clips in the return_dict.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=size)
    clip_sampler = make_clip_sampler("uniform", clip_duration)
    labeled_video_paths = LabeledVideoPaths.from_path(data_name)
    dataset = EncodedVideoDataset(
        labeled_video_paths,
        clip_sampler=clip_sampler,
        video_sampler=DistributedSampler,
        decode_audio=False,
        decoder=decoder,
    )
    test_dataloader = DataLoader(dataset, batch_size=None, num_workers=1)

    # Run two epochs, simulating use in a training loop
    dataset.video_sampler.set_epoch(0)
    epoch_1 = [(sample["label"], sample["video"]) for sample in test_dataloader]
    dataset.video_sampler.set_epoch(1)
    epoch_2 = [(sample["label"], sample["video"]) for sample in test_dataloader]
    return_dict[rank] = {"epoch_1": epoch_1, "epoch_2": epoch_2}


@contextlib.contextmanager
def mock_encoded_video_dataset_file():
    """
    Creates a temporary mock encoded video dataset with 4 videos labeled from 0 - 4.
    Returns a labeled video file which points to this mock encoded video dataset, the
    ordered label and videos tuples and the video duration in seconds.
    """
    num_frames = 10
    fps = 5
    with temp_encoded_video(num_frames=num_frames, fps=fps) as (
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
                f.write(f"{video_file_name_1} 2\n".encode())
                f.write(f"{video_file_name_2} 3\n".encode())

            label_videos = [
                (0, data_1),
                (1, data_2),
                (2, data_1),
                (3, data_2),
            ]
            video_duration = num_frames / fps
            yield f.name, label_videos, video_duration
