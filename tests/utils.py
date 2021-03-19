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
from pytorchvideo.data.dataset_manifest_utils import (
    EncodedVideoInfo,
    VideoFrameInfo,
    VideoInfo,
)
from pytorchvideo.data.utils import thwc_to_cthw


def create_dummy_video_frames(num_frames: int, height: int, width: int):
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
    data = create_dummy_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".mp4") as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, thwc_to_cthw(data).to(torch.float32)
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
    video_data = create_dummy_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".avi") as f:
        f.close()
        write_audio_video(
            f.name, video_data, audio_data, fps=fps, audio_rate=audio_rate
        )
        cthw_video_data = thwc_to_cthw(video_data).to(torch.float32)
        yield f.name, cthw_video_data, audio_data[0].to(torch.float32)


@contextlib.contextmanager
def temp_frame_video(frame_image_file_names, height=10, width=10):
    data = create_dummy_video_frames(len(frame_image_file_names), height, width)
    data = thwc_to_cthw(data)
    with tempfile.TemporaryDirectory() as root_dir:
        root_dir = pathlib.Path(root_dir)
        root_dir.mkdir(exist_ok=True)
        for i, file_name in enumerate(frame_image_file_names):
            im = transforms.ToPILImage()(data[:, i])
            im.save(root_dir / file_name, compress_level=0, optimize=False)
        yield root_dir, data.to(torch.float32)


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


MOCK_VIDEO_IDS = ["vid1", "vid2", "vid3", "vid4"]
MOCK_VIDEO_INFOS = {
    MOCK_VIDEO_IDS[0]: VideoInfo(
        video_id=MOCK_VIDEO_IDS[0], resolution="1080x1920", duration=100, fps=30
    ),
    MOCK_VIDEO_IDS[1]: VideoInfo(
        video_id=MOCK_VIDEO_IDS[1], resolution="1080x1920", duration=50, fps=60
    ),
    MOCK_VIDEO_IDS[2]: VideoInfo(
        video_id=MOCK_VIDEO_IDS[2], resolution="720x1280", duration=1000.09, fps=30
    ),
    MOCK_VIDEO_IDS[3]: VideoInfo(
        video_id=MOCK_VIDEO_IDS[3], resolution="720x1280", duration=17.001, fps=90
    ),
}


def get_flat_video_frames(directory, file_extension):
    frame_file_stem = "frame_"
    return {
        MOCK_VIDEO_IDS[0]: VideoFrameInfo(
            video_id=MOCK_VIDEO_IDS[0],
            location=f"{directory}/{MOCK_VIDEO_IDS[0]}",
            frame_file_stem=frame_file_stem,
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=3000,
            file_extension=file_extension,
        ),
        MOCK_VIDEO_IDS[1]: VideoFrameInfo(
            video_id=MOCK_VIDEO_IDS[1],
            location=f"{directory}/{MOCK_VIDEO_IDS[1]}",
            frame_file_stem=frame_file_stem,
            frame_string_length=16,
            min_frame_number=2,
            max_frame_number=3001,
            file_extension=file_extension,
        ),
        MOCK_VIDEO_IDS[2]: VideoFrameInfo(
            video_id=MOCK_VIDEO_IDS[2],
            location=f"{directory}/{MOCK_VIDEO_IDS[2]}",
            frame_file_stem=frame_file_stem,
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=30003,
            file_extension=file_extension,
        ),
        MOCK_VIDEO_IDS[3]: VideoFrameInfo(
            video_id=MOCK_VIDEO_IDS[3],
            location=f"{directory}/{MOCK_VIDEO_IDS[3]}",
            frame_file_stem=frame_file_stem,
            frame_string_length=16,
            min_frame_number=1,
            max_frame_number=1530,
            file_extension=file_extension,
        ),
    }


def get_encoded_video_infos(directory, exit_stack=None):
    encoded_video_infos = {}
    for video_id in MOCK_VIDEO_IDS:
        file_path, _ = (
            exit_stack.enter_context(temp_encoded_video(10, 10))
            if exit_stack
            else (f"{directory}/{video_id}.mp4", None)
        )
        encoded_video_infos[video_id] = EncodedVideoInfo(video_id, file_path)
    return encoded_video_infos
