# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from collections import OrderedDict
from typing import Callable

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
