# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import pathlib
import random
import tempfile
import unittest

import torch
from pytorchvideo.data import Ava
from pytorchvideo.data.clip_sampling import make_clip_sampler
from utils import temp_frame_video


AVA_FPS = 30


@contextlib.contextmanager
def temp_ava_dataset_2_videos():
    frame_names = [f"{str(i)}.png" for i in range(90)]
    # Create csv containing 2 test frame videos.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as frames_file:
        frames_file.write("original_vido_id video_id frame_id path labels\n".encode())
        # Frame video 1
        with temp_frame_video(frame_names) as (frame_1_video_dir, data_1):
            for i, frame_name in enumerate(frame_names):
                original_video_id_1 = str(frame_1_video_dir)
                video_id = "1"
                frame_id = str(i)
                path = pathlib.Path(frame_1_video_dir) / frame_name
                label = "0"
                frames_file.write(
                    f"{original_video_id_1} {video_id} {frame_id} {path} {label}\n".encode()
                )

            # Frame video 2
            with temp_frame_video(frame_names, height=5, width=5) as (
                frame_2_video_dir,
                data_2,
            ):
                for i, frame_name in enumerate(frame_names):
                    original_video_id_2 = str(frame_2_video_dir)
                    video_id = "2"
                    frame_id = str(i)
                    path = pathlib.Path(frame_2_video_dir) / frame_name
                    label = "1"
                    frames_file.write(
                        f"{original_video_id_2} {video_id} {frame_id} {path} {label}\n".encode()
                    )

                frames_file.close()
                yield frames_file.name, data_1, data_2, original_video_id_1, original_video_id_2


def get_random_bbox():
    bb_list = [round(random.random(), 3) for x in range(4)]
    converted_list = [str(element) for element in bb_list]
    return bb_list, ",".join(converted_list)


class TestAvaDataset(unittest.TestCase):
    def test_multiple_videos(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as data_file:
            with temp_ava_dataset_2_videos() as (
                frame_paths_file,
                video_1,
                video_2,
                video_1_name,
                video_2_name,
            ):
                # add bounding boxes
                # video 1
                bb_1_a, bb_1_a_string = get_random_bbox()
                action_1_a, iou_1_a = 1, 0.85
                bb_1_b, bb_1_b_string = get_random_bbox()
                action_1_b, iou_1_b = 2, 0.4

                data_file.write(
                    (
                        f"{video_1_name},902,{bb_1_a_string},"
                        + f"{str(action_1_a)},{str(iou_1_a)}\n"
                    ).encode()
                )
                data_file.write(
                    (
                        f"{video_1_name},902,{bb_1_b_string},"
                        + f"{str(action_1_b)},{str(iou_1_b)}\n"
                    ).encode()
                )
                # video 2
                bb_2_a, bb_2_a_string = get_random_bbox()
                action_2_a, iou_2_a = 3, 0.95
                bb_2_b, bb_2_b_string = get_random_bbox()
                action_2_b, iou_2_b = 4, 0.9

                data_file.write(
                    (
                        f"{video_2_name},902,{bb_2_a_string},"
                        + f"{str(action_2_a)},{str(iou_2_a)}\n"
                    ).encode()
                )
                data_file.write(
                    (
                        f"{video_2_name},902,{bb_2_b_string},"
                        + f"{str(action_2_b)},{str(iou_2_b)}\n"
                    ).encode()
                )

                data_file.close()

                dataset = Ava(
                    frame_paths_file=frame_paths_file,
                    frame_labels_file=data_file.name,
                    clip_sampler=make_clip_sampler("random", 1.0),
                )

                # All videos are of the form cthw and fps is 30
                # Clip is samples at time step = 2 secs in video
                sample_1 = next(dataset)
                self.assertTrue(sample_1["video"].equal(video_1[:, 45:75, :, :]))
                self.assertTrue(
                    torch.tensor(sample_1["boxes"]).equal(
                        torch.tensor([bb_1_a, bb_1_b])
                    )
                )
                self.assertTrue(
                    torch.tensor(sample_1["labels"]).equal(
                        torch.tensor([[action_1_a], [action_1_b]])
                    )
                )
                sample_2 = next(dataset)
                self.assertTrue(sample_2["video"].equal(video_2[:, 45:75, :, :]))
                self.assertTrue(
                    torch.tensor(sample_2["boxes"]).equal(
                        torch.tensor([bb_2_a, bb_2_b])
                    )
                )
                self.assertTrue(
                    torch.tensor(sample_2["labels"]).equal(
                        torch.tensor([[action_2_a], [action_2_b]])
                    )
                )

    def test_multiple_videos_with_label_map(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as label_map_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as data_file:
                with temp_ava_dataset_2_videos() as (
                    frame_paths_file,
                    video_1,
                    video_2,
                    video_1_name,
                    video_2_name,
                ):
                    # Create labelmap file
                    label_map = """item {
  name: "bend/bow (at the waist)"
  id: 1
}
item {
  name: "crouch/kneel"
  id: 3
}
item {
  name: "dance"
  id: 4
}"""
                    label_map_file.write(label_map.encode())
                    label_map_file.close()

                    # add bounding boxes
                    # video 1
                    bb_1_a, bb_1_a_string = get_random_bbox()
                    action_1_a, iou_1_a = 1, 0.85
                    bb_1_b, bb_1_b_string = get_random_bbox()
                    action_1_b, iou_1_b = 2, 0.4

                    data_file.write(
                        (
                            f"{video_1_name},902,{bb_1_a_string},"
                            + f"{str(action_1_a)},{str(iou_1_a)}\n"
                        ).encode()
                    )
                    data_file.write(
                        (
                            f"{video_1_name},902,{bb_1_b_string},"
                            + f"{str(action_1_b)},{str(iou_1_b)}\n"
                        ).encode()
                    )
                    # video 2
                    bb_2_a, bb_2_a_string = get_random_bbox()
                    action_2_a, iou_2_a = 3, 0.95
                    bb_2_b, bb_2_b_string = get_random_bbox()
                    action_2_b, iou_2_b = 4, 0.9

                    data_file.write(
                        (
                            f"{video_2_name},902,{bb_2_a_string},"
                            + f"{str(action_2_a)},{str(iou_2_a)}\n"
                        ).encode()
                    )
                    data_file.write(
                        (
                            f"{video_2_name},902,{bb_2_b_string},"
                            + f"{str(action_2_b)},{str(iou_2_b)}\n"
                        ).encode()
                    )

                    data_file.close()

                    dataset = Ava(
                        frame_paths_file=frame_paths_file,
                        frame_labels_file=data_file.name,
                        clip_sampler=make_clip_sampler("random", 1.0),
                        label_map_file=label_map_file.name,
                    )

                    # All videos are of the form cthw and fps is 30
                    # Clip is samples at time step = 2 secs in video
                    sample_1 = next(dataset)
                    self.assertTrue(sample_1["video"].equal(video_1[:, 45:75, :, :]))
                    self.assertTrue(
                        torch.tensor(sample_1["boxes"]).equal(torch.tensor([bb_1_a]))
                    )
                    self.assertTrue(
                        torch.tensor(sample_1["labels"]).equal(
                            torch.tensor([[action_1_a]])
                        )
                    )
                    sample_2 = next(dataset)
                    self.assertTrue(sample_2["video"].equal(video_2[:, 45:75, :, :]))
                    self.assertTrue(
                        torch.tensor(sample_2["boxes"]).equal(
                            torch.tensor([bb_2_a, bb_2_b])
                        )
                    )
                    self.assertTrue(
                        torch.tensor(sample_2["labels"]).equal(
                            torch.tensor([[action_2_a], [action_2_b]])
                        )
                    )
