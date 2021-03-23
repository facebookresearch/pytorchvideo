# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from pytorchvideo.data.dataset_manifest_utils import EncodedVideoInfo, VideoFrameInfo
from pytorchvideo.data.epic_kitchen.utils import (
    build_encoded_manifest_from_nested_directory,
    build_frame_manifest_from_flat_directory,
    build_frame_manifest_from_nested_directory,
)


def write_mock_frame_files(video_frames, tempdir, ext):
    tempdir = Path(tempdir)
    for _, video_frame_info in video_frames.items():
        if not os.path.isdir(video_frame_info.location):
            os.mkdir(video_frame_info.location)

        for frame_num in reversed(
            range(
                video_frame_info.min_frame_number, video_frame_info.max_frame_number + 1
            )
        ):  # Here we reverse the order of the frames we write to test that code
            # doesn't rely on ls returning frames in order due to
            # frames being written in order temporally.
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
            file_extension=file_extension,
        ),
        "P02_002": VideoFrameInfo(
            video_id="P02_002",
            location=f"{directory}/P02_002",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=3001,
            file_extension=file_extension,
        ),
        "P02_005": VideoFrameInfo(
            video_id="P02_005",
            location=f"{directory}/P02_005",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=30003,
            file_extension=file_extension,
        ),
        "P07_002": VideoFrameInfo(
            video_id="P07_002",
            location=f"{directory}/P07_002",
            frame_file_stem="frame_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=1530,
            file_extension=file_extension,
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
            file_extension=file_extension,
        ),
        "P02_002": VideoFrameInfo(
            video_id="P02_002",
            location=f"{directory}/P02",
            frame_file_stem="P02_002_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=3001,
            file_extension=file_extension,
        ),
        "P02_005": VideoFrameInfo(
            video_id="P02_005",
            location=f"{directory}/P02",
            frame_file_stem="P02_005_",
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=30003,
            file_extension=file_extension,
        ),
        "P07_002": VideoFrameInfo(
            video_id="P07_002",
            location=f"{directory}/P07",
            frame_file_stem="P07_002_",
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=1530,
            file_extension=file_extension,
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

    def test_build_encoded_manifest_from_nested_directory(self):
        file_names = ["P01_01.mp4", "P01_07.mp4", "P23_11.mp4", "P11_00.mp4"]
        with tempfile.TemporaryDirectory(prefix="TestEpicKitchenUtils") as tempdir:

            for file_name in file_names:
                participant_path = Path(tempdir) / file_name[:3]
                if not os.path.isdir(participant_path):
                    os.mkdir(participant_path)

                with open(participant_path / file_name, "w") as f:
                    f.write("0")

            encoded_video_dict = build_encoded_manifest_from_nested_directory(tempdir)

            self.assertEqual(
                sorted(encoded_video_dict), ["P01_01", "P01_07", "P11_00", "P23_11"]
            )
            self.assertEqual(
                encoded_video_dict["P01_01"],
                EncodedVideoInfo("P01_01", str(Path(tempdir) / "P01/P01_01.mp4")),
            )
            self.assertEqual(
                encoded_video_dict["P01_07"],
                EncodedVideoInfo("P01_07", str(Path(tempdir) / "P01/P01_07.mp4")),
            )
            self.assertEqual(
                encoded_video_dict["P11_00"],
                EncodedVideoInfo("P11_00", str(Path(tempdir) / "P11/P11_00.mp4")),
            )
            self.assertEqual(
                encoded_video_dict["P23_11"],
                EncodedVideoInfo("P23_11", str(Path(tempdir) / "P23/P23_11.mp4")),
            )
