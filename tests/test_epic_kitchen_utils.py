import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from pytorchvideo.data.epic_kitchen.epic_kitchen_dataset import VideoFrameInfo
from pytorchvideo.data.epic_kitchen.utils import (
    build_frame_manifest_from_flat_directory,
    build_frame_manifest_from_nested_directory,
    save_video_frame_info,
)


def write_mock_frame_files(video_frames, tempdir, ext):
    tempdir = Path(tempdir)
    for _, video_frame_info in video_frames.items():
        if not os.path.isdir(video_frame_info.location):
            os.mkdir(video_frame_info.location)

        for frame_num in range(
            video_frame_info.min_frame_number, video_frame_info.max_frame_number + 1
        ):
            frame_num_str = str(frame_num)
            stem = video_frame_info.frame_file_stem
            frame_num_zeros = "0" * (
                video_frame_info.frame_string_length - len(frame_num_str) - len(stem)
            )
            frame_file_name = f"{stem}{frame_num_zeros}{frame_num_str}.{ext}"
            with open(f"{video_frame_info.location}/{frame_file_name}", "w") as f:
                f.write("0")


def get_flat_video_frames(directory, file_extension):
    return {
        "P02_001": VideoFrameInfo(
            video_id="P02_001",
            location=f"{directory}/P02_001",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=3000,
            file_extension=file_extension
        ),
        "P02_002": VideoFrameInfo(
            video_id="P02_002",
            location=f"{directory}/P02_002",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=3001,
            file_extension=file_extension
        ),
        "P02_005": VideoFrameInfo(
            video_id="P02_005",
            location=f"{directory}/P02_005",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=30003,
            file_extension=file_extension
        ),
        "P07_002": VideoFrameInfo(
            video_id="P07_002",
            location=f"{directory}/P07_002",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=1530,
            file_extension=file_extension
        ),
    }


def get_nested_video_frames(directory, file_extension):
    return {
        "P02_001": VideoFrameInfo(
            video_id="P02_001",
            location=f"{directory}/P02",
            frame_file_stem="P02_001_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=3000,
            file_extension=file_extension
        ),
        "P02_002": VideoFrameInfo(
            video_id="P02_002",
            location=f"{directory}/P02",
            frame_file_stem="P02_002_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=3001,
            file_extension=file_extension
        ),
        "P02_005": VideoFrameInfo(
            video_id="P02_005",
            location=f"{directory}/P02",
            frame_file_stem="P02_005_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=30003,
            file_extension=file_extension
        ),
        "P07_002": VideoFrameInfo(
            video_id="P07_002",
            location=f"{directory}/P07",
            frame_file_stem="P07_002_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=1530,
            file_extension=file_extension
        ),
    }


class TestEpicKitchenUtils(unittest.TestCase):
    def test_build_frame_manifest_from_flat_directory_sync(self):
        self.test_build_frame_manifest_from_flat_directory(multithreading=False)

    def test_build_frame_manifest_from_flat_directory(self, multithreading=True):
        with tempfile.TemporaryDirectory(prefix="TestEpicKitchenUtils") as tempdir:
            video_frames_expected = get_flat_video_frames(tempdir, "jpg")
            write_mock_frame_files(video_frames_expected, tempdir, "jpg")

            video_frames = build_frame_manifest_from_flat_directory(
                tempdir, multithreading
            )

            self.assertEqual(len(video_frames_expected), len(video_frames))
            for video_id in video_frames_expected:
                self.assertEqual(
                    video_frames[video_id], video_frames_expected[video_id]
                )

    def test_build_frame_manifest_from_nested_directory_sync(self):
        self.test_build_frame_manifest_from_nested_directory(multithreading=False)

    def test_build_frame_manifest_from_nested_directory(self, multithreading=True):
        with tempfile.TemporaryDirectory(prefix="TestEpicKitchenUtils") as tempdir:
            video_frames_expected = get_nested_video_frames(tempdir, "png")
            write_mock_frame_files(video_frames_expected, tempdir, "png")

            video_frames = build_frame_manifest_from_nested_directory(
                tempdir, multithreading
            )
            self.assertEqual(len(video_frames_expected), len(video_frames))
            for video_id in video_frames_expected:
                self.assertEqual(
                    video_frames[video_id], video_frames_expected[video_id]
                )

    def test_save_video_frame_info(self):
        with tempfile.TemporaryDirectory(prefix="TestEpicKitchenUtils") as tempdir:
            video_frames = get_flat_video_frames(tempdir, "jpg")

            saved_file_name = f"{tempdir}/frame_manifest.csv"
            save_video_frame_info(video_frames, saved_file_name)

            with open(saved_file_name) as f:
                csv_text = f.readlines()
                assert len(csv_text) == len(video_frames) + 1
