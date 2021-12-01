# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from collections import OrderedDict
from typing import Callable, List

import attr
import torch
from pytorchvideo.data.decoder import DecoderType
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo


FAIL_STRATEGY = ("RANDOM_FILL", "ZERO_FILL", "RETURN_NONE", "RAISE_ERROR")
HOOK_STATUS = ("PENDING", "SCHEDULED", "EXECUTING", "EXECUTED", "FAILED", "EARLY_EXIT")


@attr.s(repr=True)
class HookBase:
    """
    HookBase contains the basic attributes of a hook.
    """

    executor: Callable = attr.ib()
    conditional_execution_func: Callable = attr.ib()
    exit_func: Callable = attr.ib()
    inputs: List[str] = attr.ib(default=())
    outputs: List[str] = attr.ib(default=())
    fail_strategy: str = attr.ib(
        default="RAISE_ERROR",
        validator=lambda self_, attr_, val_: (val_) in FAIL_STRATEGY,
    )
    priority: int = attr.ib(
        default=1,
        validator=lambda self_, attr_, val_: val_ >= 1,
    )
    status: str = "PENDING"

    def run(
        self,
        status: OrderedDict,
    ):
        if self.conditional_execution_func():
            self._run(status)
        self.exit_func()

    def _run(
        self,
        status: OrderedDict,
    ):
        pass

    def get_inputs(
        self,
    ):
        return self.inputs

    def get_outputs(
        self,
    ):
        return self.outputs


def full_decode(status: OrderedDict, **args):
    decoder = args.get("decoder", DecoderType.PYAV)
    decode_audio = args.get("decode_audio", True)
    video = EncodedVideo.from_path(status["path"], decode_audio, decoder)
    frames = video.get_clip(0, video.duration)
    return frames


class DecodeHook(HookBase):
    def __init__(
        self,
        executor: Callable = full_decode,
        decode_audio: bool = True,
        decoder: str = DecoderType.PYAV,
        fail_strategy="RAISE_ERROR",
        priority=0,
    ):
        # Decoding params
        self.decode_audio = decode_audio
        self.decoder = decoder
        # Hook params
        self.executor = executor
        self.inputs = ["path"]
        self.outputs = ["video", "audio"] if decode_audio else ["video"]
        self.fail_strategy = fail_strategy
        self.priority = priority

    def _run(
        self,
        status: OrderedDict,
    ):
        frames = self.executor(
            status, decode_audio=self.decode_audio, decoder=self.decoder
        )
        return frames


class X3DClsHook(HookBase):
    def __init__(
        self,
        executor: Callable = full_decode,
        fail_strategy="RAISE_ERROR",
        priority=0,
    ):
        # Hook params
        self.executor = executor
        self.inputs = ["video"]
        self.outputs = ["action_class"]
        self.fail_strategy = fail_strategy
        self.priority = priority

        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        model = "x3d_s"

        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                ]
            ),
        )
        # Init network
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", model=model, pretrained=True
        )
        self.model = self.model.eval()

    def _run(
        self,
        status: OrderedDict,
    ):
        status = self.transform(status)
        inputs = status["video"]
        inputs = inputs[None, ...]
        output = self.model(inputs)
        return {"action_class": output}
