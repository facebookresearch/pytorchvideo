# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .encoded_video_dataset import labeled_encoded_video_dataset


"""
    Action recognition video dataset for UCF101 stored as an encoded video.
    <https://www.crcv.ucf.edu/data/UCF101.php>
"""
Ucf101 = labeled_encoded_video_dataset
