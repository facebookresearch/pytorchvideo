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


def center_keypoints_in_bbox(bboxes_per_frame, keypoints_per_frame):
    # calculate bbox center (x1, y1, x2, y2)
    bboxes_per_frame_center_x = (
        bboxes_per_frame[:, 0] + bboxes_per_frame[:, 2]
    ) / 2  # (x1+x2)/2
    bboxes_per_frame_center_y = (
        bboxes_per_frame[:, 1] + bboxes_per_frame[:, 3]
    ) / 2  # (y1+y2)/2

    # change origin of the keypoints to center of each bbox
    keypoints_per_frame[:, :, 0] = keypoints_per_frame[
        :, :, 0
    ] - bboxes_per_frame_center_x.unsqueeze(1)
    keypoints_per_frame[:, :, 1] = keypoints_per_frame[
        :, :, 1
    ] - bboxes_per_frame_center_y.unsqueeze(1)

    return keypoints_per_frame


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

        # frame and bounding-box tracker
        self.frame_tracker = []

    def _populate_frame_tracker(self, model, frames):
        """
        Generates a data structure to track bounding boxes and
        keypoint coordinates. Useful for extracting the frame-id given
        the bounding number from a video for action-recognition.
        """

        for frame_id, frame in enumerate(frames):
            model_outputs = model.predict(frame)

            # get bounding-box coordinates (x1, y1, x2, y2)
            bboxes_per_frame = (
                model_outputs["instances"][model_outputs["instances"].pred_classes == 0]
                .pred_boxes.tensor.to("cpu")
                .squeeze()
            )

            # get keypoints (slice to select only the x,y coordinates)
            keypoints_per_frame = (
                model_outputs["instances"][model_outputs["instances"].pred_classes == 0]
                .pred_keypoints[:, :, :2]
                .to("cpu")
            )

            # center keypoints wrt to the respective bounding box centers
            keypoints_per_frame = center_keypoints_in_bbox(
                bboxes_per_frame=bboxes_per_frame,
                keypoints_per_frame=keypoints_per_frame,
            )

            # sanity check
            if bboxes_per_frame.shape[0] != keypoints_per_frame.shape[0]:
                raise ValueError(
                    "bboxes_per_frame and keypoints_per_frame should have same 0th dim."
                )

            # append bbox_info to frame_tracker
            for i in range(bboxes_per_frame.shape[0]):
                bbox_coord = bboxes_per_frame[i, :]
                keypoint_per_bbox = keypoints_per_frame[i, :, :]

                bbox_info = {
                    "frame_id": frame_id,
                    "bbox_id": i,
                    "person_id": None,
                    "bbox_coord": bbox_coord,
                    "keypoint_coord": keypoint_per_bbox,
                }

                self.frame_tracker.append(bbox_info)

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
