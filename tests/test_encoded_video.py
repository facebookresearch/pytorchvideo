import tempfile
import unittest

import av
import pytest
from pytorchvideo.data.encoded_video import EncodedVideo
from utils import temp_video


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
