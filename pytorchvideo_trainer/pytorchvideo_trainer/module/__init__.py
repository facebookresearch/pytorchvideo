# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .byol import BYOLModule  # noqa
from .moco_v2 import MOCOV2Module  # noqa
from .simclr import SimCLRModule  # noqa
from .video_classification import VideoClassificationModule  # noqa


__all__ = [
    "VideoClassificationModule",
    "SimCLRModule",
    "BYOLModule",
    "MOCOV2Module",
]
