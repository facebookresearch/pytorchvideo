# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava import Ava  # noqa
from .charades import Charades  # noqa
from .clip_sampling import (  # noqa
    ClipSampler,
    RandomClipSampler,
    UniformClipSampler,
    make_clip_sampler,
)  # noqa
from .domsev import DomsevFrameDataset, DomsevVideoDataset  # noqa
from .epic_kitchen_forecasting import EpicKitchenForecasting  # noqa
from .epic_kitchen_recognition import EpicKitchenRecognition  # noqa
from .hmdb51 import Hmdb51  # noqa
from .kinetics import Kinetics  # noqa
from .labeled_video_dataset import (
    LabeledVideoDataset,
    labeled_video_dataset,
)  # noqa
from .ssv2 import SSv2
from .ucf101 import Ucf101  # noqa
