# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import pathlib
import tempfile
import unittest

from pytorchvideo.data import Charades
from pytorchvideo.data.clip_sampling import make_clip_sampler
from torch.utils.data import SequentialSampler
from utils import temp_frame_video, temp_frame_video_dataset


@contextlib.contextmanager
def temp_charades_dataset():

    # Create csv containing 2 test frame videos.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write("original_vido_id video_id frame_id path labels\n".encode())

        with temp_frame_video_dataset() as (video_frames, _):
            for (
                original_video_id,
                video_id,
                frame_id,
                path,
                label,
                _,
            ) in video_frames:
                f.write(
                    f"{original_video_id} {video_id} {frame_id} {path} {label}\n".encode()
                )

            f.close()
            yield f.name, video_frames[0][-1], video_frames[1][-1]


class TestCharadesDataset(unittest.TestCase):
    def test_single_clip_per_video_works(self):
        with temp_charades_dataset() as (filename, video_1, video_2):
            clip_sampler = make_clip_sampler(
                "uniform", 0.1  # Total duration of 3 frames at 30fps is 0.1 seconds.
            )
            dataset = Charades(
                filename, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )
            expected = [([[0], [0], [0]], video_1), ([[1], [1], [1]], video_2)]
            for sample, expected_sample in zip(dataset, expected):
                self.assertEqual(sample["label"], expected_sample[0])
                self.assertTrue(sample["video"].equal(expected_sample[1]))

    def test_multiple_clips_per_video_works(self):
        with temp_charades_dataset() as (filename, video_1, video_2):
            clip_sampler = make_clip_sampler(
                "uniform", 0.033  # Expects each clip to have 1 frame each.
            )
            dataset = Charades(
                filename, clip_sampler=clip_sampler, video_sampler=SequentialSampler
            )

            expected = [
                ([[0]], video_1[:, 0:1]),
                ([[0]], video_1[:, 1:2]),
                ([[0]], video_1[:, 2:3]),
                ([[1]], video_2[:, 0:1]),
                ([[1]], video_2[:, 1:2]),
                ([[1]], video_2[:, 2:3]),
            ]
            for sample, expected_sample in zip(dataset, expected):
                self.assertEqual(sample["label"], expected_sample[0])
                self.assertTrue(sample["video"].equal(expected_sample[1]))

    def test_multiple_labels_per_frame(self):
        frame_names = [f"{str(i)}.png" for i in range(3)]

        # Create csv containing a test frame videos.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write("original_vido_id video_id frame_id path labels\n".encode())
            with temp_frame_video(frame_names) as (frame_1_video_dir, data_1):
                for i, frame_name in enumerate(frame_names):
                    original_video_id = str(frame_1_video_dir)
                    video_id = "1"
                    frame_id = str(i)
                    path = pathlib.Path(frame_1_video_dir) / frame_name
                    label = "0,100"
                    f.write(
                        f"{original_video_id} {video_id} {frame_id} {path} {label}\n".encode()
                    )

                f.close()

                clip_sampler = make_clip_sampler(
                    "random",
                    0.1,  # Total duration of 3 frames at 30fps is 0.1 seconds.
                )
                dataset = Charades(
                    f.name, clip_sampler=clip_sampler, video_sampler=SequentialSampler
                )

                sample = next(dataset)
                self.assertEqual(sample["label"], [[0, 100], [0, 100], [0, 100]])
                self.assertTrue(sample["video"].equal(data_1))
