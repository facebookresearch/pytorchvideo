# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
from typing import Tuple

import numpy as np
import torch
import tqdm

# Imports from PyTorchVideo and PyTorch3D
from pytorch3d.renderer import PerspectiveCameras
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset

from .dataset_utils import (
    generate_splits,
    get_geometry_data,
    objectron_to_pytorch3d,
    resize_images,
)
from .nerf_dataset import ListDataset


DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data", "objectron"
)


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch


def get_nerf_datasets(
    dataset_name: str,
    image_size: Tuple[int, int],
    data_root: str = DEFAULT_DATA_ROOT,
    **kwargs,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """
    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    if dataset_name != "objectron":
        raise ValueError("This data loader is only for the objectron dataset")

    # Use the bundle adjusted camera parameters
    sequence_geometry = get_geometry_data(os.path.join(data_root, "sfm_arframe.pbdata"))
    num_frames = len(sequence_geometry)

    # Check if splits are present else generate them on the first instance:
    splits_path = os.path.join(data_root, "splits.pt")
    if os.path.exists(splits_path):
        print("Loading splits...")
        splits = torch.load(splits_path)
        train_idx, val_idx, test_idx = splits
    else:
        print("Generating splits...")
        index_options = np.arange(num_frames)
        train_idx, val_idx, test_idx = generate_splits(index_options)
        torch.save([train_idx, val_idx, test_idx], splits_path)

    print("Loading video...")
    video_path = os.path.join(data_root, "video.MOV")
    # Load the video using the PyTorchVideo video class
    video = EncodedVideo.from_path(video_path)
    FPS = 30

    print("Loading all images and cameras...")
    # Load all the video frames
    frame_data = video.get_clip(start_sec=0, end_sec=(num_frames - 1) * 1.0 / FPS)
    frame_data = frame_data["video"].permute(1, 2, 3, 0)
    images = resize_images(frame_data, image_size)
    cameras = []

    for frame_id in tqdm.tqdm(range(num_frames)):
        I, P = sequence_geometry[frame_id]
        R = P[0:3, 0:3]
        T = P[0:3, 3]

        # Convert conventions
        R = R.transpose(0, 1)
        R, T = objectron_to_pytorch3d(R, T)

        # Get intrinsic parameters
        fx = I[0, 0]
        fy = I[1, 1]
        px = I[0, 2]
        py = I[1, 2]

        # Initialize the Perspective Camera
        scene_cam = PerspectiveCameras(
            R=R[None, ...],
            T=T[None, ...],
            focal_length=((fx, fy),),
            principal_point=((px, py),),
        ).to("cpu")

        cameras.append(scene_cam)

    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]

    return train_dataset, val_dataset, test_dataset
