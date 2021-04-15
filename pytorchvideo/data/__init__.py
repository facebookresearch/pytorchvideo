# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .charades import Charades  # noqa
from .clip_sampling import (
    ClipSampler,
    RandomClipSampler,
    UniformClipSampler,
    make_clip_sampler,
)  # noqa
from .domsev import DomsevDataset  # noqa
from .encoded_video_dataset import (
    EncodedVideoDataset,
    labeled_encoded_video_dataset,
)  # noqa
from .epic_kitchen_forecasting import EpicKitchenForecasting  # noqa
from .epic_kitchen_recognition import EpicKitchenRecognition  # noqa
from .hmdb51 import Hmdb51  # noqa
from .kinetics import Kinetics  # noqa
from .ssv2 import SSv2
from .ucf101 import Ucf101  # noqa
