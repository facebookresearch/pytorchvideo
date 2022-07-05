# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import tempfile
import unittest
import unittest.mock
from dataclasses import dataclass
from pathlib import Path

from pytorchvideo.data.utils import (
    DataclassFieldCaster,
    export_video_array,
    load_dataclass_dict_from_csv,
    save_dataclass_objs_to_headered_csv,
)
from pytorchvideo.data.video import VideoPathHandler
from utils import temp_encoded_video


@dataclass
class TestDataclass(DataclassFieldCaster):
    a: str
    b: int
    b_plus_1: int = DataclassFieldCaster.complex_initialized_dataclass_field(
        lambda v: int(v) + 1
    )
    c: float
    d: list
    e: dict = DataclassFieldCaster.complex_initialized_dataclass_field(lambda v: {v: v})


@dataclass
class TestDataclass2(DataclassFieldCaster):
    a: str
    b: int


class TestDataUtils(unittest.TestCase):
    def test_DataclassFieldCaster(self):
        test_obj = TestDataclass("1", "1", "1", "1", "abc", "k")

        self.assertEqual(test_obj.a, "1")
        self.assertEqual(type(test_obj.a), str)

        self.assertEqual(test_obj.b, 1)
        self.assertEqual(type(test_obj.b), int)
        self.assertEqual(test_obj.b_plus_1, 2)

        self.assertEqual(test_obj.c, 1.0)
        self.assertEqual(type(test_obj.c), float)

        self.assertEqual(test_obj.d, ["a", "b", "c"])
        self.assertEqual(type(test_obj.d), list)

        self.assertEqual(test_obj.e, {"k": "k"})
        self.assertEqual(type(test_obj.e), dict)

    def _export_video_array(
        self,
        video_codec="libx264rgb",
        height=10,
        width=10,
        num_frames=10,
        fps=5,
        options=None,
        epsilon=3,
    ):
        with temp_encoded_video(
            num_frames=num_frames, fps=fps, height=height, width=width
        ) as (video_file_name, data,), tempfile.TemporaryDirectory(
            prefix="video_stop_gap_test"
        ) as tempdir:
            exported_video_path = os.path.join(tempdir, "video.mp4")
            export_video_array(
                data,
                output_path=exported_video_path,
                rate=fps,
                video_codec=video_codec,
                options=options,
            )
            vp_handler = VideoPathHandler()
            video = vp_handler.video_from_path(exported_video_path, decode_audio=False)
            reloaded_data = video.get_clip(0, video.duration)["video"]
            self.assertLessEqual((data - reloaded_data).abs().mean(), epsilon)

    def test_export_video_array_mult(self):
        self._export_video_array(
            video_codec="libx264rgb",
            height=10,
            width=10,
            num_frames=10,
            fps=5,
            options={"crf": "0"},
            epsilon=1e-6,
        )
        self._export_video_array(
            video_codec="mpeg4", height=10, width=10, num_frames=10, fps=5
        )
        self._export_video_array(
            video_codec="mpeg4", height=480, width=640, num_frames=30, fps=30
        )

    def test_load_dataclass_dict_from_csv_value_dict(self):
        dataclass_objs = [
            TestDataclass2("a", 1),
            TestDataclass2("b", 2),
            TestDataclass2("c", 3),
            TestDataclass2("d", 4),
        ]
        with tempfile.TemporaryDirectory(prefix=f"{TestDataUtils}") as tempdir:
            csv_file_name = Path(tempdir) / "data.csv"
            save_dataclass_objs_to_headered_csv(dataclass_objs, csv_file_name)

            test_dict = load_dataclass_dict_from_csv(
                csv_file_name, TestDataclass2, "a", list_per_key=False
            )
            self.assertEqual(len(test_dict), 4)
            self.assertEqual(test_dict["c"].b, 3)

    def test_load_dataclass_dict_from_csv_list_dict(self):
        dataclass_objs = [
            TestDataclass2("a", 1),
            TestDataclass2("a", 2),
            TestDataclass2("b", 3),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
        ]
        with tempfile.TemporaryDirectory(prefix=f"{TestDataUtils}") as tempdir:
            csv_file_name = Path(tempdir) / "data.csv"
            save_dataclass_objs_to_headered_csv(dataclass_objs, csv_file_name)
            test_dict = load_dataclass_dict_from_csv(
                csv_file_name, TestDataclass2, "a", list_per_key=True
            )
            self.assertEqual(len(test_dict), 3)
            self.assertEqual([x.b for x in test_dict["a"]], [1, 2])
            self.assertEqual([x.b for x in test_dict["b"]], [3])
            self.assertEqual([x.b for x in test_dict["c"]], [4, 4, 4])

    def test_load_dataclass_dict_from_csv_throws(self):
        dataclass_objs = [
            TestDataclass2("a", 1),
            TestDataclass2("a", 2),
            TestDataclass2("b", 3),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
            TestDataclass2("c", 4),
        ]
        with tempfile.TemporaryDirectory(prefix=f"{TestDataUtils}") as tempdir:
            csv_file_name = Path(tempdir) / "data.csv"
            save_dataclass_objs_to_headered_csv(dataclass_objs, csv_file_name)
            self.assertRaises(
                AssertionError,
                lambda: load_dataclass_dict_from_csv(
                    csv_file_name, TestDataclass2, "a", list_per_key=False
                ),
            )

    def test_save_dataclass_objs_to_headered_csv(self):
        dataclass_objs = [
            TestDataclass2("a", 1),
            TestDataclass2("a", 2),
            TestDataclass2("b", 3),
        ]

        with tempfile.TemporaryDirectory(prefix=f"{TestDataUtils}") as tempdir:
            csv_file_name = Path(tempdir) / "data.csv"
            save_dataclass_objs_to_headered_csv(dataclass_objs, csv_file_name)
            with open(csv_file_name) as f:
                lines = list(f.readlines())
                self.assertEqual(len(lines), 4)
                self.assertEqual(lines[0], "a,b\n")
                self.assertEqual(lines[1], "a,1\n")
                self.assertEqual(lines[2], "a,2\n")
                self.assertEqual(lines[3], "b,3\n")
