# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import struct
import sys
from typing import List, Tuple

import numpy as np
import torch

# The AR Metadata captured with each frame in the video
from objectron.schema import (  # noqa: E402
    a_r_capture_metadata_pb2 as ar_metadata_protocol,
)
from PIL import Image
from pytorch3d.transforms import Rotate, RotateAxisAngle, Translate


# Imports from Objectron
module_path = os.path.abspath(os.path.join("..."))
if module_path not in sys.path:
    sys.path.append("../Objectron")


def objectron_to_pytorch3d(
    R: torch.Tensor, T: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms the R and T matrices from the Objectron world coordinate
    system to the PyTorch3d world system.
    Objectron cameras live in +X right, +Y Up, +Z from screen to us.
    Pytorch3d world is +X left, +Y up, +Z from us to screen.
    """
    rotation = Rotate(R=R)
    conversion = RotateAxisAngle(axis="y", angle=180)
    composed_transform = rotation.compose(conversion).get_matrix()
    composed_R = composed_transform[0, 0:3, 0:3]

    translation = Translate(x=T[None, ...])
    t_matrix = translation.compose(conversion).get_matrix()
    flipped_T = t_matrix[0, 3, :3]
    return composed_R, flipped_T


def generate_splits(
    index_options: List[int], train_fraction: float = 0.8
) -> List[List[int]]:
    """
    Get indices for train, val, test splits.
    """
    num_images = len(index_options)
    np.random.shuffle(index_options)
    train_index = int(train_fraction * num_images)
    val_index = train_index + ((num_images - train_index) // 2)
    train_indices = index_options[:train_index]
    val_indices = index_options[train_index:val_index]
    test_indices = index_options[val_index:]
    split_indices = [train_indices, val_indices, test_indices]
    return split_indices


def get_geometry_data(geometry_filename: str) -> List[List[torch.Tensor]]:
    """
    Utils function for parsing metadata files from the Objectron GitHub repo:
    https://github.com/google-research-datasets/Objectron/blob/master/notebooks/objectron-geometry-tutorial.ipynb  # noqa: B950
    """
    sequence_geometry = []
    with open(geometry_filename, "rb") as pb:
        proto_buf = pb.read()

        i = 0
        while i < len(proto_buf):
            # Read the first four Bytes in little endian '<' integers 'I' format
            # indicating the length of the current message.
            msg_len = struct.unpack("<I", proto_buf[i : i + 4])[0]
            i += 4
            message_buf = proto_buf[i : i + msg_len]
            i += msg_len
            frame_data = ar_metadata_protocol.ARFrame()
            frame_data.ParseFromString(message_buf)

            projection = np.reshape(frame_data.camera.projection_matrix, (4, 4))
            view = np.reshape(frame_data.camera.view_matrix, (4, 4))

            projection = torch.tensor(projection, dtype=torch.float32)
            view = torch.tensor(view, dtype=torch.float32)

            sequence_geometry.append((projection, view))
    return sequence_geometry


def resize_images(frames: List[torch.Tensor], image_size: List[int]) -> torch.Tensor:
    """
    Utils function to resize images
    """
    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # The dataset image is very large ...
    images = torch.FloatTensor(frames) / 255.0
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

    scale_factors = [s_new / s for s, s_new in zip(images.shape[1:3], image_size)]

    if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
        raise ValueError(
            "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
        )
    scale_factor = sum(scale_factors) * 0.5

    if scale_factor != 1.0:
        print(f"Rescaling dataset (factor={scale_factor})")
        images = torch.nn.functional.interpolate(
            images.permute(0, 3, 1, 2),
            size=tuple(image_size),
            mode="bilinear",
        ).permute(0, 2, 3, 1)

    return images
