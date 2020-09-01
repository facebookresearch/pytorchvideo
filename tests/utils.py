#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import os
import tempfile

import torch
import torchvision.io as io
from pytorchvideo.data.utils import thwc_to_cthw


def create_video_frames(num_frames: int, height: int, width: int):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())

    return torch.stack(data, 0)


@contextlib.contextmanager
def temp_encoded_video(num_frames: int, fps: int, height=10, width=10, prefix=None):
    """
    Creates a temporary lossless, mp4 video with synthetic content. Uses a context which
    deletes the video after exit.
    """
    # Lossless options.
    video_codec = "libx264rgb"
    options = {"crf": "0"}
    data = create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".mp4") as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, thwc_to_cthw(data)
    os.unlink(f.name)
