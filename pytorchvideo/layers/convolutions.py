from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.common_types import _size_3_t


class ConvReduce3D(nn.Module):
    """
    Builds a list of convolutional operators and performs summation on the outputs.

                            Conv3d, Conv3d, ...,  Conv3d
                                           ↓
                                          Sum
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[_size_3_t],
        stride: Optional[Tuple[_size_3_t]] = None,
        padding: Optional[Tuple[_size_3_t]] = None,
        padding_mode: Optional[Tuple[str]] = None,
        dilation: Optional[Tuple[_size_3_t]] = None,
        groups: Optional[Tuple[int]] = None,
        bias: Optional[Tuple[bool]] = None,
        reduction_method: str = "sum",
    ) -> None:
        """
        Args:
            in_channels int: number of input channels.
            out_channels int: number of output channels produced by the convolution(s).
            kernel_size tuple(_size_3_t): Tuple of sizes of the convolutionaling kernels.
            stride tuple(_size_3_t): Tuple of strides of the convolutions.
            padding tuple(_size_3_t): Tuple of paddings added to all three sides of the
                input.
            padding_mode tuple(string): Tuple of padding modes for each convs.
                Options include `zeros`, `reflect`, `replicate` or `circular`.
            dilation tuple(_size_3_t): Tuple of spacings between kernel elements.
            groups tuple(_size_3_t): Tuple of numbers of blocked connections from input
                channels to output channels.
            bias tuple(bool): If `True`, adds a learnable bias to the output.
            reduction_method str: Options include `sum` and `cat`.
        """
        super().__init__()
        assert reduction_method in ("sum", "cat")
        self.reduction_method = reduction_method
        conv_list = []
        for ind in range(len(kernel_size)):
            conv_param = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size[ind],
            }
            if stride is not None and stride[ind] is not None:
                conv_param["stride"] = stride[ind]
            if padding is not None and padding[ind] is not None:
                conv_param["padding"] = padding[ind]
            if dilation is not None and dilation[ind] is not None:
                conv_param["dilation"] = dilation[ind]
            if groups is not None and groups[ind] is not None:
                conv_param["groups"] = groups[ind]
            if bias is not None and bias[ind] is not None:
                conv_param["bias"] = bias[ind]
            if padding_mode is not None and padding_mode[ind] is not None:
                conv_param["padding_mode"] = padding_mode[ind]
            conv_list.append(nn.Conv3d(**conv_param))
        self.convs = nn.ModuleList(conv_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = []
        for ind in range(len(self.convs)):
            output.append(self.convs[ind](x))
        if self.reduction_method == "sum":
            output = torch.stack(output, dim=0).sum(dim=0, keepdim=False)
        elif self.reduction_method == "cat":
            output = torch.cat(output, dim=1)
        return output
