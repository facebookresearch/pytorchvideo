# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .encoded_video_dataset import labeled_encoded_video_dataset


"""
    Action recognition video dataset for Kinetics-{400,600,700} stored as encoded videos.
    <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>
"""
Kinetics = labeled_encoded_video_dataset
