# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import unittest.mock

from pytorchvideo.data.dataset_manifest_utils import (
    EncodedVideoInfo,
    VideoDataset,
    VideoFrameInfo,
    VideoInfo,
)
from utils import MOCK_VIDEO_IDS, MOCK_VIDEO_INFOS, get_flat_video_frames


class TestDatasetManifestUtils(unittest.TestCase):
    def test_VideoFrameInfo(self):
        video_frame_info = VideoFrameInfo(
            # This is a key-mapping as the underlying
            # annotation files are of these string columns
            **{
                "video_id": "P01_012",
                "location": "c:/",
                "frame_file_stem": "P01_012_",
                "frame_string_length": "20",
                "min_frame_number": "0",
                "max_frame_number": "22",
                "file_extension": "png",
            }
        )
        self.assertEqual(video_frame_info.video_id, "P01_012")
        self.assertEqual(video_frame_info.location, "c:/")
        self.assertEqual(video_frame_info.frame_file_stem, "P01_012_")
        self.assertEqual(video_frame_info.frame_string_length, 20)
        self.assertEqual(video_frame_info.min_frame_number, 0)
        self.assertEqual(video_frame_info.max_frame_number, 22)
        self.assertEqual(video_frame_info.file_extension, "png")

    def test_EncodedVideoInfo(self):
        encoded_video_info = EncodedVideoInfo(
            # This is a key-mapping as the underlying epic-kitchen
            # annotation files are of these string columns
            **{"video_id": "P01_12", "file_path": "c:/P01_12.mp4"}
        )
        self.assertEqual(encoded_video_info.video_id, "P01_12")
        self.assertEqual(encoded_video_info.file_path, "c:/P01_12.mp4")

    def test_VideoInfo(self):
        video_info = VideoInfo(
            # This is a key-mapping as the underlying epic-kitchen
            # annotation files are of these string columns
            **{
                "video_id": "P01_01",
                "resolution": "1000x200",
                "duration": "123.45",
                "fps": "59.9",
            }
        )
        self.assertEqual(video_info.video_id, "P01_01")
        self.assertEqual(video_info.resolution, "1000x200")
        self.assertEqual(video_info.duration, 123.45)
        self.assertEqual(video_info.fps, 59.9)

    def test_frame_number_to_filepath(self):
        file_names_vid4 = VideoDataset._frame_number_to_filepaths(
            MOCK_VIDEO_IDS[3],
            get_flat_video_frames("testdirectory", "jpg"),
            MOCK_VIDEO_INFOS,
        )
        file_path = file_names_vid4[100]
        self.assertEqual(
            file_path, f"testdirectory/{MOCK_VIDEO_IDS[3]}/frame_0000000101.jpg"
        )
        with self.assertRaises(IndexError):
            file_path = file_names_vid4[10000]
        file_path = file_names_vid4[-1]
        self.assertEqual(
            file_path, f"testdirectory/{MOCK_VIDEO_IDS[3]}/frame_0000001530.jpg"
        )

        file_names_vid2 = VideoDataset._frame_number_to_filepaths(
            MOCK_VIDEO_IDS[1],
            get_flat_video_frames("testdirectory2", "png"),
            MOCK_VIDEO_INFOS,
        )
        file_path = file_names_vid2[0]
        self.assertEqual(
            file_path, f"testdirectory2/{MOCK_VIDEO_IDS[1]}/frame_0000000002.png"
        )
        file_path = file_names_vid2[2999]
        self.assertEqual(
            file_path, f"testdirectory2/{MOCK_VIDEO_IDS[1]}/frame_0000003001.png"
        )
        with self.assertRaises(IndexError):
            file_path = file_names_vid2[3000]

    def test_remove_video_info_missing_or_incomplete_videos(self):
        video_infos_a = MOCK_VIDEO_INFOS.copy()
        video_frames_a = get_flat_video_frames("testdirectory2", "jpg")
        video_frames_a_copy = video_frames_a.copy()

        # No-Op
        VideoDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames_a, video_infos_a
        )

        self.assertEqual(len(video_infos_a), len(MOCK_VIDEO_INFOS))
        for video_id in video_infos_a:
            self.assertEqual(video_infos_a[video_id], MOCK_VIDEO_INFOS[video_id])

        self.assertEqual(len(video_frames_a), len(video_frames_a_copy))
        for video_id in video_frames_a:
            self.assertEqual(video_frames_a[video_id], video_frames_a_copy[video_id])

        video_infos_b = MOCK_VIDEO_INFOS.copy()
        video_frames_b = video_frames_a_copy.copy()

        # Unmatched video info, should be removed
        video_infos_b["P07_001"] = VideoInfo(
            video_id="P07_001", resolution="720x1280", duration=17.001, fps=30
        )

        # Unmatched video frame entry, should be removed
        video_frames_b["P07_002"]: VideoFrameInfo(
            min_frame_number=1, max_frame_number=1530, frame_string_length=8
        )

        # Video info that defines approximately 6000 frames with 600 present from frame manifest
        # Should be dropped
        video_frames_b["P08_001"]: VideoFrameInfo(
            min_frame_number=1, max_frame_number=600, frame_string_length=8
        )

        video_infos_b["P08_001"] = VideoInfo(
            video_id="P08_001", resolution="720x1280", duration=100, fps=60
        )

        VideoDataset._remove_video_info_missing_or_incomplete_videos(
            video_frames_b, video_infos_b
        )

        # All newly added fields should be removed
        self.assertEqual(len(video_infos_b), len(MOCK_VIDEO_INFOS))
        for video_id in video_infos_b:
            self.assertEqual(video_infos_b[video_id], MOCK_VIDEO_INFOS[video_id])

        self.assertEqual(len(video_frames_b), len(video_frames_a_copy))
        for video_id in video_frames_b:
            self.assertEqual(video_frames_b[video_id], video_frames_a_copy[video_id])
