#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import os
import pathlib
import tempfile

import av
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms as transforms
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


@contextlib.contextmanager
def temp_encoded_video_with_audio(
    num_frames: int,
    fps: int,
    num_audio_samples: int,
    audio_rate: int = 48000,
    height=10,
    width=10,
    prefix=None,
):
    audio_data = torch.from_numpy(np.random.rand(1, num_audio_samples).astype("<i2"))
    video_data = create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".avi") as f:
        f.close()
        write_audio_video(
            f.name, video_data, audio_data, fps=fps, audio_rate=audio_rate
        )
        yield f.name, video_data, audio_data[0].to(torch.float64)


@contextlib.contextmanager
def temp_frame_video(frame_image_file_names, height=10, width=10):
    data = create_video_frames(len(frame_image_file_names), height, width)
    data = thwc_to_cthw(data)
    with tempfile.TemporaryDirectory() as root_dir:
        root_dir = pathlib.Path(root_dir)
        root_dir.mkdir(exist_ok=True)
        for i, file_name in enumerate(frame_image_file_names):
            im = transforms.ToPILImage()(data[:, i])
            im.save(root_dir / file_name, compress_level=0, optimize=False)
        yield root_dir, data


def write_audio_video(path, video, audio, fps=30, audio_rate=48000):
    video_array = torch.as_tensor(video, dtype=torch.uint8).numpy()
    audio_array = audio.numpy().astype("<i2")

    with av.open(path, "w") as container:

        # Add lossless h264 video stream.
        video_stream = container.add_stream("libx264rgb", fps)
        video_stream.width = video_array.shape[2]
        video_stream.height = video_array.shape[1]
        video_stream.pix_fmt = "rgb24"
        video_stream.options = {"crf": "0"}  # Lossless video option

        # Add lossless flac, stereo audio stream.
        audio_stream = container.add_stream("flac", rate=audio_rate, layout="stereo")
        audio_stream.codec_context.skip_frame = "NONE"
        audio_stream.time_base = f"1/{audio_rate}"
        audio_stream.options = {}

        num_audio_samples = audio_array.shape[-1]
        num_video_frames = len(video_array)
        num_audio_samples_per_frame = audio_rate // fps
        encoded_audio_index = 0
        encoded_video_index = 0

        # Video and audio stream packets are encoded interleaved by their time.
        # To do this we start both streams from pts 0 and then encode 1 video frame
        # and num_audio_samples_per_frame audio samples continuously until both
        # streams have finished encoding.
        while (
            encoded_audio_index < num_audio_samples
            or encoded_video_index < num_video_frames
        ):

            # Video frame encodings.
            if encoded_video_index < num_video_frames:
                frame = video_array[encoded_video_index]
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pict_type = "NONE"
                encoded_video_index += 1
                for packet in video_stream.encode(video_frame):
                    container.mux(packet)

            # Audio frame encodings.
            if encoded_audio_index < num_audio_samples:
                encode_packet_end = encoded_audio_index + num_audio_samples_per_frame
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_array[:, encoded_audio_index:encode_packet_end],
                    format="s16",
                    layout="stereo",
                )
                encoded_audio_index = encode_packet_end
                audio_frame.rate = audio_rate
                audio_frame.time_base = f"1/{audio_rate}"
                encoded_packets = audio_stream.encode(audio_frame)
                for packet in encoded_packets:
                    container.mux(packet)

        # Flush streams.
        for packet in audio_stream.encode(None):
            container.mux(packet)

        for packet in video_stream.encode(None):
            container.mux(packet)
