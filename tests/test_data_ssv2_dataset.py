# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import json
import pathlib
import tempfile
import unittest

from pytorchvideo.data import SSv2
from pytorchvideo.data.clip_sampling import make_clip_sampler
from torch.utils.data import SequentialSampler
from utils import temp_frame_video


@contextlib.contextmanager
def temp_ssv2_dataset():
    frame_names = [f"{str(i)}.png" for i in range(7)]

    # Create json file for label names.
    labels = [
        "Approaching something with your camera",
        "Attaching something to something",
    ]
    label_names = {labels[0]: "0", labels[1]: "1"}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(label_names, f)
        label_name_file = f.name

    # Create csv containing 2 test frame videos.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write("original_vido_id video_id frame_id path labels\n".encode())

        # Frame video 1
        with temp_frame_video(frame_names) as (frame_1_video_dir, data_1):
            for i, frame_name in enumerate(frame_names):
                original_video_id = str(frame_1_video_dir)
                video_id = "1"
                frame_id = str(i)
                path = pathlib.Path(frame_1_video_dir) / frame_name
                f.write(
                    f"{original_video_id} {video_id} {frame_id} {path} ''\n".encode()
                )

            # Frame video 2
            with temp_frame_video(frame_names) as (frame_2_video_dir, data_2):
                for i, frame_name in enumerate(frame_names):
                    original_video_id = str(frame_2_video_dir)
                    video_id = "2"
                    frame_id = str(i)
                    path = pathlib.Path(frame_2_video_dir) / frame_name
                    f.write(
                        f"{original_video_id} {video_id} {frame_id} {path} ''\n".encode()
                    )

                f.close()
                video_path_file = f.name

                # Create json file for lable names.
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".json"
                ) as f:
                    videos = [
                        {"id": str(frame_1_video_dir), "template": labels[0]},
                        {"id": str(frame_2_video_dir), "template": labels[1]},
                    ]
                    json.dump(videos, f)
                    video_label_file = f.name

                yield label_name_file, video_label_file, video_path_file, data_1, data_2


class TestSSv2Dataset(unittest.TestCase):
    def test_single_clip_per_video_works(self):
        with temp_ssv2_dataset() as (
            label_name_file,
            video_label_file,
            video_path_file,
            video_1,
            video_2,
        ):

            # Put arbitrary duration as ssv2 always needs full video clip.
            clip_sampler = make_clip_sampler("constant_clips_per_video", 1.0, 1)
            # Expect taking 2 frames (1-th and 4-th among 7 frames).
            dataset = SSv2(
                label_name_file,
                video_label_file,
                video_path_file,
                clip_sampler=clip_sampler,
                video_sampler=SequentialSampler,
                frames_per_clip=2,
            )
            expected = [(0, video_1), (1, video_2)]
            for sample, expected_sample in zip(dataset, expected):
                self.assertEqual(sample["label"], expected_sample[0])
                self.assertTrue(sample["video"].equal(expected_sample[1][:, (1, 4)]))
