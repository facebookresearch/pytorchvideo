# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import json
import tempfile
import unittest

from pytorchvideo.data import json_dataset
from pytorchvideo.data.clip_sampling import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from utils import temp_frame_video_dataset


class TestJsonDatasets(unittest.TestCase):
    def setUp(self):
        LabeledVideoDataset._MAX_CONSECUTIVE_FAILURES = 1

    def test_video_only_frame_video_dataset_works(self):
        total_duration = 2.0
        with mock_json_annotations() as (annotation_json, labels, duration):
            clip_sampler = make_clip_sampler("random", total_duration)
            dataset = json_dataset.video_only_dataset(
                data_path=annotation_json,
                clip_sampler=clip_sampler,
                decode_audio=False,
            )

            self.assertEqual(dataset.num_videos, 2)
            self.assertEqual(len(list(iter(dataset))), 2)


@contextlib.contextmanager
def mock_json_annotations():
    with temp_frame_video_dataset() as (_, videos):
        json_dict = {}
        for video in videos:
            name = str(video[0])
            json_dict[name] = {}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w+") as f:
            json.dump(json_dict, f)

        label_videos = [
            (0, videos[0][-1]),
            (1, videos[1][-1]),
        ]

        total_duration = 3 / 30  # 30 fps, 3 frames
        yield f.name, label_videos, total_duration
