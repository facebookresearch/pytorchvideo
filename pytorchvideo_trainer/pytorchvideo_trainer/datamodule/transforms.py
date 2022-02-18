# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import hydra
import torch
import torchvision
from PIL import Image, ImageFilter
from pytorchvideo.transforms import MixVideo
from torchvision.transforms import Compose


def build_transforms(transforms_config: Iterable[Mapping[str, Any]]) -> Compose:
    """
    A utility function to build data transforsm from a list of Hydra/Omega Conf
    objects. This utility method is called by
    `pytorchvideo_trainer.datamodule.PyTorchVideoDataModule` class to build a
    sequence of transforms applied during each phase(train, val and test).

    Uses torchvision.transforms.Compose to build a seuquence of transforms.

    Examples of config objects used by this method can be found in,
    `pytorchvide_trainer/conf/datamodule/transforms/`

    Args:
        transforms_config: A list of hydra config objects wherein, each element
        represents config associated with a single transforms.

        An example of this would be,
        ```
        - _target_: pytorchvideo.transforms.ApplyTransformToKey
            transform:
            - _target_: pytorchvideo.transforms.UniformTemporalSubsample
                num_samples: 16
            - _target_: pytorchvideo.transforms.Div255
            - _target_: pytorchvideo.transforms.Normalize
                mean: [0.45, 0.45, 0.45]
                std: [0.225, 0.225, 0.225]
            - _target_: pytorchvideo.transforms.ShortSideScale
                size: 224
            key: video
        - _target_: pytorchvideo.transforms.UniformCropVideo
            size: 224
        - _target_: pytorchvideo.transforms.RemoveKey
            key: audio
        ```
    """
    transform_list = [build_single_transform(config) for config in transforms_config]
    transform = Compose(transform_list)
    return transform


def build_single_transform(config: Mapping[str, Any]) -> Callable[..., object]:
    """
    A utility method to build a single transform from hydra / omega conf objects.

    If the key "transform" is present in the give config, it recursively builds
    and composes transforms using  the `torchvision.transforms.Compose` method.
    """
    config = dict(config)
    if "transform" in config:
        assert isinstance(config["transform"], Sequence)
        transform_list = [
            build_single_transform(transform) for transform in config["transform"]
        ]
        transform = Compose(transform_list)
        config.pop("transform")
        return hydra.utils.instantiate(config, transform=transform)
    return hydra.utils.instantiate(config)


class ApplyTransformToKeyOnList:
    """
    Applies transform to key of dictionary input wherein input is a list

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
         >>>  transforms.ApplyTransformToKeyOnList(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable) -> None:  # pyre-ignore[24]
        self._key = key
        self._transform = transform

    def __call__(
        self, x: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, List[torch.Tensor]]:
        x[self._key] = [self._transform(a) for a in x[self._key]]
        return x


class SlowFastPackPathway:
    """
    Transform for converting a video clip into a list of 2 clips with
    different temporal granualirity as needed by the SlowFast video
    model.

    For more details, refere to the paper,
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Args:
        alpha (int): Number of frames to sub-sample from the given clip
        to create the second clip.
    """

    def __init__(self, alpha: int) -> None:
        super().__init__()
        self.alpha = alpha

    def __call__(self, frames: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
        """
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class RepeatandConverttoList:
    """
    An utility transform that repeats each value in a
    key, value-style minibatch and replaces it with a list of values.

    Useful for performing multiple augmentations.
    An example such usecase can be found in
    `pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_mvit_16x4.yaml`

    Args:
        repead_num (int): Number of times to repeat each value.
    """

    def __init__(self, repeat_num: int) -> None:
        super().__init__()
        self.repeat_num = repeat_num

    # pyre-ignore[3]
    def __call__(self, sample_dict: Dict[str, Any]) -> Dict[str, List[Any]]:
        for k, v in sample_dict.items():
            sample_dict[k] = self.repeat_num * [v]
        return sample_dict


class MixVideoBatchWrapper:
    def __init__(
        self,
        mixup_alpha: float,
        cutmix_prob: float,
        cutmix_alpha: float,
        label_smoothing: float,
    ) -> None:
        """
        A wrapper for MixVideo (CutMix or Mixup) tranform in pytorchvideo.transforms.
        Extends the MixVideo transform to work on a batch dictionary objects.

        The dictionary object should consist of keys "video" and "label" representing
        video clips and their associated labels.
        """

        self.mix_video_transform = MixVideo(
            mixup_alpha=mixup_alpha,
            cutmix_prob=cutmix_prob,
            cutmix_alpha=cutmix_alpha,
            label_smoothing=label_smoothing,
        )

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        batch["video"], batch["label"] = self.mix_video_transform(
            batch["video"], batch["label"]
        )
        return batch


class ColorJitterVideoSSl:
    """
    A custom sequence of transforms that randomly performs Color jitter,
    Gaussian Blur and Grayscaling on the given clip.

    Particularly useful for the SSL tasks like SimCLR, MoCoV2, BYOL, etc.

    Args:
        bri_con_sat (list[float]): A list of 3 floats reprsenting brightness,
        constrast and staturation coefficients to use for the
        `torchvision.transforms.ColorJitter` transform.
        hue (float): Heu value to use in the `torchvision.transforms.ColorJitter`
        transform.
        p_color_jitter (float): The probability with which the Color jitter transform
        is randomly applied on the given clip.
        p_convert_gray (float): The probability with which the given clip is randomly
        coverted into grayscale.
        p_gaussian_blur (float): The probability with which the Gaussian transform
        is randomly applied on the given clip.
        gaussian_blur_sigma (list[float]): A list of 2 floats with in which
        the blur radius is randomly sampled for Gaussian blur transform.
    """

    def __init__(
        self,
        bri_con_sat: List[float],
        hue: float,
        p_color_jitter: float,
        p_convert_gray: float,
        p_gaussian_blur: float = 0.5,
        gaussian_blur_sigma: List[float] = (0.1, 2.0),
    ) -> None:

        self.color_jitter = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.ColorJitter(
                            bri_con_sat[0], bri_con_sat[1], bri_con_sat[2], hue
                        )
                    ],
                    p=p_color_jitter,
                ),
                torchvision.transforms.RandomGrayscale(p=p_convert_gray),
                torchvision.transforms.RandomApply(
                    [GaussianBlur(gaussian_blur_sigma)], p=p_gaussian_blur
                ),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        """
        c, t, h, w = frames.shape
        frames = frames.view(c, t * h, w)
        frames = self.color_jitter(frames)  # pyre-ignore[6,9]
        frames = frames.view(c, t, h, w)

        return frames


class GaussianBlur(object):
    """
    A PIL image version of Gaussian blur augmentation as
    in SimCLR https://arxiv.org/abs/2002.05709

    Args:
        sigma (list[float]): A list of 2 floats with in which
        the blur radius is randomly sampled during each step.
    """

    def __init__(self, sigma: List[float] = (0.1, 2.0)) -> None:
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        img (Image): A PIL image with single or 3 color channels.
        """
        sigma = self.sigma[0]
        if len(self.sigma) == 2:
            sigma = random.uniform(self.sigma[0], self.sigma[1])

        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img
