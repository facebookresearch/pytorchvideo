# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .labeled_video_dataset import labeled_video_dataset


"""
    Action recognition video dataset for Kinetics-{400,600,700}
    <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>
"""
Kinetics = labeled_video_dataset
