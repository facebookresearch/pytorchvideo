import unittest
import os
import torch
import tempfile
import contextlib
import av
import pytest
import torchvision.io as io

from pytorchvideo.data.encoded_video import EncodedVideo


def _create_video_frames(num_frames: int, height: int, width: int):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())

    return torch.stack(data, 0)


@contextlib.contextmanager
def temp_video(
        num_frames: int,
        fps: int,
        height=300,
        width=300,
        lossless=False,
):
    video_codec = 'libx264'
    options = {}
    if lossless:
        video_codec = 'libx264rgb'
        options = {'crf': '0'}

    data = _create_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, data
    os.unlink(f.name)


class TestEncodedVideo(unittest.TestCase):
    def test_video_with_header_info_works(self):
        num_frames = 10
        fps = 5
        with temp_video(num_frames=num_frames, fps=fps, lossless=True) as (
            f_name,
            data,
        ):
            test_video = EncodedVideo(f_name)
            self.assertEqual(test_video.start_pts, 0)

            expected_end = test_video.seconds_to_video_pts(num_frames / fps)
            self.assertEqual(test_video.end_pts, expected_end)

            # All frames (0 - 2 seconds)
            frames = test_video.get_clip(0, test_video.seconds_to_video_pts(2))
            self.assertTrue(frames.equal(data))

            # Half frames (0 - 0.9 seconds)
            frames = test_video.get_clip(0, test_video.seconds_to_video_pts(0.9))
            self.assertTrue(frames.equal(data[:5]))

            # No frames (3 - 5 seconds)
            frames = test_video.get_clip(
                test_video.seconds_to_video_pts(3), test_video.seconds_to_video_pts(5)
            )
            self.assertEqual(len(frames.size()), 0)

            test_video.close()

    def test_open_video_failure(self):
        with pytest.raises(av.AVError):
            test_video = EncodedVideo("non_existent_file.txt")
            test_video.close()

    def test_decode_video_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"This is not an mp4 file")
            with pytest.raises(av.AVError):
                test_video = EncodedVideo(f.name)
                test_video.close()
