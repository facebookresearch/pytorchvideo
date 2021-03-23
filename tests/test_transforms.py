# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import torch
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.transforms.functional import (
    repeat_temporal_frames_subsample,
    short_side_scale,
    uniform_temporal_subsample,
)
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)
from utils import create_dummy_video_frames


class TestTransforms(unittest.TestCase):
    def test_compose_with_video_transforms(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 30, 40)).to(
            dtype=torch.float32
        )
        test_clip = {"video": video, "label": 0}

        # Compose using torchvision and pytorchvideo transformst to ensure they interact
        # correctly.
        num_subsample = 10
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_subsample),
                            NormalizeVideo([video.mean()] * 3, [video.std()] * 3),
                            RandomShortSideScale(min_size=15, max_size=25),
                            RandomCropVideo(10),
                            RandomHorizontalFlipVideo(p=0.5),
                        ]
                    ),
                )
            ]
        )

        actual = transform(test_clip)
        c, t, h, w = actual["video"].shape
        self.assertEqual(c, 3)
        self.assertEqual(t, num_subsample)
        self.assertEqual(h, 10)
        self.assertEqual(w, 10)

    def test_uniform_temporal_subsample(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 30, 40)).to(
            dtype=torch.float32
        )
        actual = uniform_temporal_subsample(video, video.shape[1])
        self.assertTrue(actual.equal(video))

        video = thwc_to_cthw(create_dummy_video_frames(20, 30, 40)).to(
            dtype=torch.float32
        )
        actual = uniform_temporal_subsample(video, video.shape[1] // 2)
        self.assertTrue(actual.equal(video[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 19]]))

        video = thwc_to_cthw(create_dummy_video_frames(20, 30, 40)).to(
            dtype=torch.float32
        )
        actual = uniform_temporal_subsample(video, 1)
        self.assertTrue(actual.equal(video[:, 0:1]))

    def test_short_side_scale_width_shorter_pytorch(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 20, 10)).to(
            dtype=torch.float32
        )
        actual = short_side_scale(video, 5, backend="pytorch")
        self.assertEqual(actual.shape, (3, 20, 10, 5))

    def test_short_side_scale_height_shorter_pytorch(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 10, 20)).to(
            dtype=torch.float32
        )
        actual = short_side_scale(video, 5, backend="pytorch")
        self.assertEqual(actual.shape, (3, 20, 5, 10))

    def test_short_side_scale_equal_size_pytorch(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 10, 10)).to(
            dtype=torch.float32
        )
        actual = short_side_scale(video, 10, backend="pytorch")
        self.assertEqual(actual.shape, (3, 20, 10, 10))

    def test_short_side_scale_width_shorter_opencv(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 20, 10)).to(
            dtype=torch.float32
        )
        actual = short_side_scale(video, 5, backend="opencv")
        self.assertEqual(actual.shape, (3, 20, 10, 5))

    def test_short_side_scale_height_shorter_opencv(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 10, 20)).to(
            dtype=torch.float32
        )
        actual = short_side_scale(video, 5, backend="opencv")
        self.assertEqual(actual.shape, (3, 20, 5, 10))

    def test_short_side_scale_equal_size_opencv(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 10, 10)).to(
            dtype=torch.float32
        )
        actual = short_side_scale(video, 10, backend="opencv")
        self.assertEqual(actual.shape, (3, 20, 10, 10))

    def test_torchscriptable_input_output(self):
        video = thwc_to_cthw(create_dummy_video_frames(20, 30, 40)).to(
            dtype=torch.float32
        )

        # Test all the torchscriptable tensors.
        for transform in [UniformTemporalSubsample(10), RandomShortSideScale(10, 20)]:

            transform_script = torch.jit.script(transform)
            self.assertTrue(isinstance(transform_script, torch.jit.ScriptModule))

            # Seed before each transform to force determinism.
            torch.manual_seed(0)
            output = transform(video)
            torch.manual_seed(0)
            script_output = transform_script(video)
            self.assertTrue(output.equal(script_output))

    def test_repeat_temporal_frames_subsample(self):
        video = thwc_to_cthw(create_dummy_video_frames(32, 10, 10)).to(
            dtype=torch.float32
        )
        actual = repeat_temporal_frames_subsample(video, (1, 4))
        expected_shape = ((3, 32, 10, 10), (3, 8, 10, 10))
        for idx in range(len(actual)):
            self.assertEqual(actual[idx].shape, expected_shape[idx])

    # TODO: add a test case for short_side_scale in the next diff
    # (a sanity check to make sure the interp is not changed)
