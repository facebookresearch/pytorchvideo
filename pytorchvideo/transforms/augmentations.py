# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video transforms that are used for advanced augmentation methods."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torchvision
import torchvision.transforms.functional_tensor as F_t
from torchvision.transforms.functional import InterpolationMode

# Maximum global magnitude used for video augmentation.
_AUGMENTATION_MAX_LEVEL = 10


def _check_fill_arg(kwargs):
    """
    Check if kwargs contains key ``fill``.
    """
    assert "fill" in kwargs, "Need to have fill in kwargs."


def _autocontrast(video: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Maximize contrast of a video by remapping its pixels per channel so that the lowest
    becomes black and the lightest becomes white.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
    """
    return torchvision.transforms.functional.autocontrast(video)


def _equalize(video: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Equalize the histogram of a video by applying a non-linear mapping to the input in
    order to create a uniform distribution of grayscale values in the output.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
    """
    if video.dtype != torch.uint8:
        video_type = video.dtype
        video = (video * 255).to(torch.uint8)
        return (torchvision.transforms.functional.equalize(video) / 255).to(video_type)
    return torchvision.transforms.functional.equalize(video)


def _invert(video: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Invert the colors of a video.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
    """
    return torchvision.transforms.functional.invert(video)


def _rotate(video: torch.Tensor, factor: float, **kwargs) -> torch.Tensor:
    """
    Rotate the image by angle.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): The rotation angle value in degrees, counter-clockwise.
    """
    _check_fill_arg(kwargs)
    return torchvision.transforms.functional.rotate(
        video, factor, fill=kwargs["fill"], interpolation=InterpolationMode.BILINEAR
    )


def _solarize(video: torch.Tensor, factor: float, **kwargs) -> torch.Tensor:
    """
    Solarize an video by inverting all pixel values above a threshold.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
    """
    if video.dtype == torch.uint8:
        return torchvision.transforms.functional.solarize(video, int(factor * 255.0))
    else:
        return torchvision.transforms.functional.solarize(video, factor)


def _adjust_contrast(video: torch.Tensor, factor: float, **kwargs) -> torch.Tensor:
    """
    Adjust contrast of an a video.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much to adjust the contrast. Can be any non-negative
            number. 0 gives a solid gray video, 1 gives the original video while 2
            increases the contrast by a factor of 2.
    """
    return torchvision.transforms.functional.adjust_contrast(video, factor)


def _adjust_saturation(video: torch.Tensor, factor: float, **kwargs) -> torch.Tensor:
    """
    Adjust the saturation of a video.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much to adjust the saturation. 0 will give a black and
            white video, 1 will give the original video while 2 will enhance the
            saturation by a factor of 2.
    """
    return torchvision.transforms.functional.adjust_saturation(video, factor)


def _adjust_brightness(video: torch.Tensor, factor: float, **kwargs) -> torch.Tensor:
    """
    Adjust brightness of a video.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        sharpness_factor (float): How much to adjust the sharpness. Can be any
            non-negative number. 0 gives a blurred video, 1 gives the original video
            while 2 increases the sharpness by a factor of 2.
    """
    return torchvision.transforms.functional.adjust_brightness(video, factor)


def _adjust_sharpness(video: torch.Tensor, factor: float, **kwargs) -> torch.Tensor:
    """
    Adjust the sharpness of a video.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much to adjust the sharpness. Can be any non-negative
            number. 0 gives a blurred video, 1 gives the original video while 2
            increases the sharpness by a factor of 2.
    """
    return torchvision.transforms.functional.adjust_sharpness(video, factor)


def _posterize(video: torch.Tensor, factor: float, **kwargs):
    """
    Posterize an image by reducing the number of bits for each color channel.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): The number of bits to keep for each channel (0-8).
    """
    if factor >= 8:
        return video
    if video.dtype != torch.uint8:
        video_type = video.dtype
        video = (video * 255).to(torch.uint8)
        return (torchvision.transforms.functional.posterize(video, factor) / 255).to(
            video_type
        )
    return torchvision.transforms.functional.posterize(video, factor)


def _shear_x(video: torch.Tensor, factor: float, **kwargs):
    """
    Shear the video along the horizontal axis.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much to shear along the horizontal axis using the affine
            matrix.
    """
    _check_fill_arg(kwargs)
    translation_offset = video.size(-2) * factor / 2
    return F_t.affine(
        video,
        [1, factor, translation_offset, 0, 1, 0],
        fill=kwargs["fill"],
        interpolation="bilinear",
    )


def _shear_y(video: torch.Tensor, factor: float, **kwargs):
    """
    Shear the video along the vertical axis.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much to shear along the vertical axis using the affine
            matrix.
    """
    _check_fill_arg(kwargs)
    translation_offset = video.size(-1) * factor / 2
    return F_t.affine(
        video,
        [1, 0, 0, factor, 1, translation_offset],
        fill=kwargs["fill"],
        interpolation="bilinear",
    )


def _translate_x(video: torch.Tensor, factor: float, **kwargs):
    """
    Translate the video along the vertical axis.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much (relative to the image size) to translate along the
            vertical axis.
    """
    _check_fill_arg(kwargs)
    translation_offset = factor * video.size(-1)
    return F_t.affine(
        video,
        [1, 0, translation_offset, 0, 1, 0],
        fill=kwargs["fill"],
        interpolation="bilinear",
    )


def _translate_y(video: torch.Tensor, factor: float, **kwargs):
    """
    Translate the video along the vertical axis.

    Args:
        video (torch.Tensor): Video tensor with shape (T, C, H, W).
        factor (float): How much (relative to the image size) to translate along the
            horizontal axis.
    """
    _check_fill_arg(kwargs)
    translation_offset = factor * video.size(-2)
    return F_t.affine(
        video,
        [1, 0, 0, 0, 1, translation_offset],
        fill=kwargs["fill"],
        interpolation="bilinear",
    )


def _randomly_negate(magnitude: float) -> float:
    """
    Negate input value with 50% chance.

    Args:
        magnitude (float): Input value.
    """
    return magnitude if torch.rand(1).item() > 0.5 else -magnitude


def _increasing_magnitude_to_arg(level: int, params: Tuple[float, float]) -> float:
    """
    Convert level to transform magnitude. This assumes transform magnitude increases
    linearly with level.

    Args:
        level (int): Level value.
        params (Tuple[float, float]): Params contains two values: 1) Base transform
            magnitude when level is 0; 2) Maxmimum increasing in transform magnitude
            when level is at Maxmimum.
    """
    magnitude = (level / _AUGMENTATION_MAX_LEVEL) * params[1]
    return (params[0] + magnitude,)


def _increasing_randomly_negate_to_arg(
    level: int, params: Tuple[float, float]
) -> Tuple[float]:
    """
    Convert level to transform magnitude. This assumes transform magnitude increases
    (or decreases with 50% chance) linearly with level.

    Args:
        level (int): Level value.
        params (Tuple[float, float]): Params contains two values: 1) Base transform
            magnitude when level is 0; 2) Maxmimum increasing in transform magnitude
            when level is at maxmimum.
    """
    magnitude = (level / _AUGMENTATION_MAX_LEVEL) * params[1]
    return (params[0] + _randomly_negate(magnitude),)


def _decreasing_int_to_arg(level: int, params: Tuple[int, int]) -> Tuple[int]:
    """
    Convert level to transform magnitude. This assumes transform magnitude decreases
    linearly with level. The return value is converted to int.

    Args:
        level (int): Level value.
        params (Tuple[float, float]): Params contains two values: 1) Base transform
            magnitude when level is 0; 2) Maxmimum decreasing in transform magnitude
            when level is at maxmimum.
    """
    magnitude = (level / _AUGMENTATION_MAX_LEVEL) * params[1]
    return (params[0] - int(magnitude),)


def _decreasing_to_arg(level: int, params: Tuple[float, float]) -> Tuple[float]:
    """
    Convert level to transform magnitude. This assumes transform magnitude decreases
    linearly with level.

    Args:
        level (int): Level value.
        params (Tuple[float, float]): Params contains two values: 1) Base transform
            magnitude when level is 0; 2) Maxmimum decreasing in transform magnitude
            when level is at maxmimum.
    """
    magnitude = (level / _AUGMENTATION_MAX_LEVEL) * params[1]
    return (params[0] - magnitude,)


# A dictionary that contains transform names (key) and their corresponding transform
# functions (value).
_NAME_TO_TRANSFORM_FUNC = {
    "AdjustBrightness": _adjust_brightness,
    "AdjustContrast": _adjust_contrast,
    "AdjustSaturation": _adjust_saturation,
    "AdjustSharpness": _adjust_sharpness,
    "AutoContrast": _autocontrast,
    "Equalize": _equalize,
    "Invert": _invert,
    "Rotate": _rotate,
    "Posterize": _posterize,
    "Solarize": _solarize,
    "ShearX": _shear_x,
    "ShearY": _shear_y,
    "TranslateX": _translate_x,
    "TranslateY": _translate_y,
}

# A dictionary that contains transform names (key) and their corresponding level
# functions (value), which converts the magnitude to the transform function arguments.
_LEVEL_TO_ARG = {
    "AdjustBrightness": _increasing_randomly_negate_to_arg,
    "AdjustContrast": _increasing_randomly_negate_to_arg,
    "AdjustSaturation": _increasing_randomly_negate_to_arg,
    "AdjustSharpness": _increasing_randomly_negate_to_arg,
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _increasing_randomly_negate_to_arg,
    "Posterize": _decreasing_int_to_arg,
    "Solarize": _decreasing_to_arg,
    "ShearX": _increasing_randomly_negate_to_arg,
    "ShearY": _increasing_randomly_negate_to_arg,
    "TranslateX": _increasing_randomly_negate_to_arg,
    "TranslateY": _increasing_randomly_negate_to_arg,
}

# A dictionary that contains transform names (key) and their corresponding maximum
# transform (value).
_TRANSFORM_MAX_PARAMS = {
    "AdjustBrightness": (1, 0.9),
    "AdjustContrast": (1, 0.9),
    "AdjustSaturation": (1, 0.9),
    "AdjustSharpness": (1, 0.9),
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": (0, 30),
    "Posterize": (4, 4),
    "Solarize": (1, 1),
    "ShearX": (0, 0.3),
    "ShearY": (0, 0.3),
    "TranslateX": (0, 0.45),
    "TranslateY": (0, 0.45),
}

# Hyperparameters for sampling magnitude.
SAMPLING_DEFAULT_HPARAS = {"sampling_std": 0.5}

# Hyperparameters for transform functions.
TRANSFORM_DEFAULT_HPARAS = {"fill": (0.5, 0.5, 0.5)}


class AugmentTransform:
    def __init__(
        self,
        transform_name: str,
        magnitude: int = 10,
        prob: float = 0.5,
        name_to_transform_func: Optional[Dict[str, Callable]] = None,
        level_to_arg: Optional[Dict[str, Callable]] = None,
        transform_max_paras: Optional[Dict[str, Tuple]] = None,
        transform_hparas: Optional[Dict[str, Any]] = None,
        sampling_type: str = "gaussian",
        sampling_hparas: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        The AugmentTransform composes a video transform that performs augmentation
        based on a maximum magnitude. AugmentTransform also offers flexible ways to
        generate augmentation magnitude based on different sampling strategies.

        Args:
            transform_name (str): The name of the video transform function.
            magnitude (int): Magnitude used for transform function.
            prob (float): The probablity of applying each transform function.
            name_to_transform_func (Optional[Dict[str, Callable]]): A Dictionary that
                contains mapping of the transform name to the transform function.
            level_to_arg (Optional[Dict[str, Callable]]): A Dictionary that contains
                mapping of the transform name to its level function, which converts
                the the magnitude to the transform function arguments.
            transform_max_paras (Optional[Dict[str, Tuple]]): A Dictionary that
                contains mapping of the transform name to its maximum transform
                magnitude.
            transform_hparas (Optional[Dict[Any]]): Transform hyper parameters.
                Needs to have key fill. By default, it uses transform_default_hparas.
            sampling_type (str): Sampling method for magnitude of transform. It should
                be either gaussian or uniform.
            sampling_hparas (Optional[Dict[Any]]): Hyper parameters for sampling. If
                gaussian sampling is used, it needs to have key sampling_std. By
                default, it uses transform_default_hparas.
        """

        assert sampling_type in ["gaussian", "uniform"]
        name_to_transform_func = name_to_transform_func or _NAME_TO_TRANSFORM_FUNC
        level_to_arg = level_to_arg or _LEVEL_TO_ARG
        transform_max_paras = transform_max_paras or _TRANSFORM_MAX_PARAMS
        self.transform_hparas = transform_hparas or TRANSFORM_DEFAULT_HPARAS
        self.sampling_type = sampling_type
        self.sampling_hparas = sampling_hparas or SAMPLING_DEFAULT_HPARAS
        assert "fill" in self.transform_hparas
        if self.sampling_type == "gaussian":
            assert "sampling_std" in self.sampling_hparas
        if self.sampling_type == "uniform":
            assert "sampling_data_type" in self.sampling_hparas
            assert "sampling_min" in self.sampling_hparas
            if self.sampling_hparas["sampling_data_type"] == "int":
                assert isinstance(self.sampling_hparas["sampling_min"], int)
            elif self.sampling_hparas["sampling_data_type"] == "float":
                assert isinstance(self.sampling_hparas["sampling_min"], (int, float))
        assert transform_name in name_to_transform_func

        self.max_level = _AUGMENTATION_MAX_LEVEL
        self.transform_name = transform_name
        self.magnitude = magnitude
        self.transform_fn = name_to_transform_func[transform_name]
        self.level_fn = level_to_arg[transform_name]
        self.level_paras = transform_max_paras[transform_name]
        self.prob = prob
        self.sampling_type = sampling_type

    def _get_magnitude(self) -> float:
        """
        Get magnitude based on sampling type.
        """
        if self.sampling_type == "gaussian":
            return max(
                0,
                min(
                    self.max_level,
                    torch.normal(
                        self.magnitude, self.sampling_hparas["sampling_std"], size=(1,)
                    ).item(),
                ),
            )
        elif self.sampling_type == "uniform":
            if self.sampling_hparas["sampling_data_type"] == "int":
                return torch.randint(
                    self.sampling_hparas["sampling_min"], self.magnitude + 1, size=(1,)
                ).item()
            elif self.sampling_hparas["sampling_data_type"] == "float":
                return (
                    torch.rand(size=(1,)).item()
                    * (self.magnitude - self.sampling_hparas["sampling_min"])
                    + self.sampling_hparas["sampling_min"]
                )
            else:
                raise ValueError("sampling_data_type must be either 'int' or 'float'")
        else:
            raise NotImplementedError

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        The input is a video tensor.

        Args:
            video (torch.Tensor): Input video tensor with shape (T, C, H, W).
        """
        if torch.rand(1).item() > self.prob:
            return video
        magnitude = self._get_magnitude()
        level_args = (
            self.level_fn(magnitude, self.level_paras)
            if self.level_fn is not None
            else ()
        )
        return self.transform_fn(video, *level_args, **self.transform_hparas)
