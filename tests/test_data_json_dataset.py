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

    def test_recognition_random_clip_sampler(self):
        total_duration = 0.05
        with mock_json_annotations() as (annotation_json, labels, duration):
            clip_sampler = make_clip_sampler("random", total_duration)
            dataset = json_dataset.clip_recognition_dataset(
                data_path=annotation_json,
                clip_sampler=clip_sampler,
                decode_audio=False,
            )

            self.assertEqual(dataset.num_videos, 4)
            self.assertEqual(len(list(iter(dataset))), 4)

    def test_recognition_uniform_clip_sampler(self):
        total_duration = 0.05
        with mock_json_annotations() as (annotation_json, labels, duration):
            clip_sampler = make_clip_sampler("uniform", total_duration)
            dataset = json_dataset.clip_recognition_dataset(
                data_path=annotation_json,
                clip_sampler=clip_sampler,
                decode_audio=False,
            )

            self.assertEqual(dataset.num_videos, 4)
            self.assertEqual(len(list(iter(dataset))), 4)

    def test_video_only_frame_video_dataset(self):
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
        label_videos = []
        json_dict = {}
        for video in videos:
            label_videos.append((video[-3], video[-2]))
            name = str(video[0])
            json_dict[name] = {
                "benchmarks": {
                    "forecasting_hands_objects": [
                        {
                            "critical_frame_selection_parent_start_sec": 0.001,
                            "critical_frame_selection_parent_end_sec": 0.012,
                            "taxonomy": {
                                "noun": video[-3],
                                "verb": video[-3],
                                "noun_unsure": False,
                                "verb_unsure": False,
                            },
                        },
                        {
                            "critical_frame_selection_parent_start_sec": 0.01,
                            "critical_frame_selection_parent_end_sec": 0.05,
                            "taxonomy": {
                                "noun": video[-3],
                                "verb": video[-3],
                                "noun_unsure": False,
                                "verb_unsure": False,
                            },
                        },
                    ]
                }
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wt") as f:
            json.dump(json_dict, f)
            f.close()

        min_duration = min(videos[0][-1], videos[1][-1])
        yield f.name, label_videos, min_duration
