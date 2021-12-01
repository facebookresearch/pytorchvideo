# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import OrderedDict
from typing import Callable

import cv2
import torch
from hook import HookBase


try:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
except Exception as _:
    raise ImportError(
        "Install detectron2: https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
    )

model_config = {
    "backend": "detectron2",
    "model": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    "threshold": 0.7,
}


def generate_predictor(model_config, *args):
    if model_config["backend"] == "detectron2":
        cfg = get_cfg()
        cfg.MODEL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.merge_from_file(model_zoo.get_config_file(model_config["model"]))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_config["threshold"]
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config["model"])

        predictor = DefaultPredictor(
            cfg,
        )
    else:
        raise ValueError("Incorrect backend.")

    return predictor


def people_keypoints_executor(image, predictor):
    return predictor(image)


class PeopleKeypointDetectionHook(HookBase):
    """
    Performs keypoint detection for humans.

    Args:
        model_config (dict): configuration for the model. The dict-keys are
            "backend", "model", and "threshold".
        executor: function that generates predictions.
    """

    def __init__(
        self,
        model_config: dict = model_config,
        executor: Callable = people_keypoints_executor,
    ):
        self.executor = executor
        self.model_config = model_config
        self.inputs = ["loaded_image", "bbox_coordinates"]
        self.outputs = ["keypoint_coordinates"]

        # generate different predictors for different backends.
        self.predictor = generate_predictor(model_config=self.model_config)

    def _run(self, status: OrderedDict):
        inputs = status["loaded_image"]
        outputs = self.executor(image=inputs, predictor=self.predictor)

        if model_config["backend"] == "detectron2":
            # keypoints is a tensor of shape (num_people, num_keypoint, (x, y, score))
            keypoints = outputs["instances"][
                outputs["instances"].pred_classes == 0
            ].pred_keypoints

        return {"keypoint_coordinates": keypoints}


def image_load_executor(image_path):
    # Returns an numpy array of shape (H,W,C) and dtype (uint8)
    return cv2.imread(image_path)


class ImageLoadHook(HookBase):
    def __init__(self, executor: Callable = image_load_executor):
        self.executor = executor
        self.inputs = ["image_path"]
        self.outputs = ["loaded_image"]

    def _run(self, status: OrderedDict):
        inputs = status["image_path"]
        image_arr = self.executor(image_path=inputs)

        return {"loaded_image": image_arr}


def people_detection_executor(loaded_image, predictor):
    # Returns a detectron2.structures.Boxes object
    # that stores a list of boxes as a Nx4 torch.Tensor.
    outputs = predictor(loaded_image)

    people_bbox = outputs["instances"][
        outputs["instances"].pred_classes == 0
    ].pred_boxes

    return people_bbox


det_models = {
    "faster_rcnn_R_50_C4": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    "faster_rcnn_R_50_FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
}


class Detectron2PeopleDetectionHook(HookBase):
    def __init__(
        self,
        executor: Callable = people_detection_executor,
        model_name: str = "faster_rcnn_R_50_C4",
        threshold=0.7,
    ):
        self.inputs = ["loaded_image"]
        self.outputs = ["bbox_coordinates"]
        self.executor = executor

        # Configure detectron2
        self.cfg = get_cfg()
        self.model_config = det_models[model_name]
        self.cfg.merge_from_file(model_zoo.get_config_file(self.model_config))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_config)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        if not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def _run(
        self,
        status,
    ):
        inputs = status["loaded_image"]
        bbox_coordinates = self.executor(
            loaded_image=inputs,
            predictor=self.predictor,
        )
        return {"bbox_coordinates": bbox_coordinates}
