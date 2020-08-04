import os
import pathlib
import tempfile
import unittest

import torchvision.io as io
from pytorchvideo.data import Kinetics
from utils import create_video_frames, temp_video


def _THWC_to_CTHW(data):
    return data.permute(3, 0, 1, 2)


class TestKineticsDataset(unittest.TestCase):
    def test_single_clip_per_video_works(self):
        with temp_video(num_frames=10, fps=5, lossless=True) as (video_file_name, data):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            dataset = Kinetics(
                f.name, clip_sampling_type="uniform", clip_duration=2, clips_per_video=1
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["label"], 0)
            self.assertTrue(dataset[0]["video"].equal(_THWC_to_CTHW(data)))

            os.unlink(f.name)

    def test_multiple_clips_per_video_works(self):
        with temp_video(num_frames=10, fps=5, lossless=True) as (video_file_name, data):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            dataset = Kinetics(
                f.name,
                clip_sampling_type="uniform",
                clip_duration=0.99,
                clips_per_video=2,
            )

            self.assertEqual(len(dataset), 4)
            data = _THWC_to_CTHW(data)
            self.assertTrue(dataset[0]["video"].equal(data[:, :5]))
            self.assertEqual(dataset[0]["label"], 0)
            self.assertTrue(dataset[1]["video"].equal(data[:, 5:]))
            self.assertEqual(dataset[1]["label"], 0)
            self.assertTrue(dataset[2]["video"].equal(data[:, :5]))
            self.assertEqual(dataset[2]["label"], 1)
            self.assertTrue(dataset[3]["video"].equal(data[:, 5:]))
            self.assertEqual(dataset[3]["label"], 1)

            os.unlink(f.name)

    def test_video_name_with_whitespace_works(self):
        with temp_video(num_frames=10, fps=5, lossless=True, prefix="pre fix") as (
            video_file_name,
            data,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            dataset = Kinetics(
                f.name, clip_sampling_type="uniform", clip_duration=2, clips_per_video=1
            )

            data = _THWC_to_CTHW(data)
            self.assertEqual(len(dataset), 2)
            self.assertTrue(dataset[0]["video"].equal(data))
            self.assertEqual(dataset[0]["label"], 0)

            os.unlink(f.name)

    def test_random_clip_sampling_works(self):
        with temp_video(num_frames=10, fps=5, lossless=True) as (video_file_name, data):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name} 0\n".encode())
                f.write(f"{video_file_name} 1\n".encode())

            dataset = Kinetics(
                f.name, clip_sampling_type="random", clip_duration=1, clips_per_video=2
            )

            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset[0]["video"].shape[1], 5)

            os.unlink(f.name)

    def test_reading_from_directory_structure(self):
        with tempfile.TemporaryDirectory() as root_dir:
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

                dataset = Kinetics(
                    root_dir,
                    clip_sampling_type="uniform",
                    clip_duration=3,
                    clips_per_video=1,
                )

                self.assertEqual(len(dataset), 2)

                # Videos are sorted alphabetically so "cleaning windows" (i.e. data_2)
                # will be first.
                self.assertEqual(dataset[0]["label"], 0)
                self.assertTrue(dataset[0]["video"].equal(_THWC_to_CTHW(data_2)))
                self.assertEqual(dataset[1]["label"], 1)
                self.assertTrue(dataset[1]["video"].equal(_THWC_to_CTHW(data_1)))
