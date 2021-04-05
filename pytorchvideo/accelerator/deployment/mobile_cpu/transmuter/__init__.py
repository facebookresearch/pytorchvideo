from pytorchvideo.accelerator.deployment.common.model_transmuter import (
    EFFICIENT_BLOCK_TRANSMUTER_REGISTRY,
)

from .transmuter_mobile_cpu import EFFICIENT_BLOCK_TRANSMUTER_MOBILE_CPU


EFFICIENT_BLOCK_TRANSMUTER_REGISTRY[
    "mobile_cpu"
] = EFFICIENT_BLOCK_TRANSMUTER_MOBILE_CPU
