"""
nnblocks.py
=========

Individual NN blocks

Created by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Union, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Creates a block of layers each consisting of convolution operation,
    nonlinear activation and (optionally) batch normalization

    Args:
        ndim:
            Data dimensionality (1, 2, or 3)
        nb_layers:
            Number of layers in the block
        input_channels:
            Number of input channels for the block
        output_channels:
            Number of the output channels for the block
        kernel_size:
            Size of convolutional filter in pixels (Default: 3)
        stride:
            Stride of convolutional filter (Default: 1)
        padding:
            Value for edge padding (Default: 1)
        batch_norm:
            Add batch normalization to each layer in the block (Default: False)
        activation:
            non-linear activation ("relu", "lrelu", "tanh", "softplus", or None)

        Example:

        >>> # Get convolutional block with three 1D convolutions and batch normalization
        >>> convblock1d = ConvBlock(ndim=1, nlayers=3,
        >>>                         input_channels=1, output_channels=32,
        >>>                         batch_norm: bool = False)
    """
    def __init__(self,
                 ndim: int,
                 nlayers: int,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[Tuple[int], int] = 3,
                 stride: Union[Tuple[int], int] = 1,
                 padding: Union[Tuple[int], int] = 1,
                 batchnorm: bool = False,
                 activation: str = "lrelu",
                 **kwargs: float,
                 ) -> None:
        """
        Initializes module parameters
        """
        super(ConvBlock, self).__init__()
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        activation = get_activation(activation)
        block = []
        for idx in range(nlayers):
            input_channels = output_channels if idx > 0 else input_channels
            block.append(get_conv(ndim)(input_channels, output_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding))
            if activation is not None:
                block.append(activation())
            if batchnorm:
                block.append(get_bnorm(ndim)(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        output = self.block(x)
        return output


class UpsampleBlock(nn.Module):
    """
    Defines upsampling block performed using bilinear
    or nearest-neigbor interpolation followed by 1-by-1 convolution
    (the latter can be used to reduce a number of feature channels)

    Args:
        ndim:
            Data dimensionality (1, 2, or 3)
        input_channels:
            Number of input channels for the block
        output_channels:
            Number of the output channels for the block
        scale_factor:
            Scale factor for upsampling
        mode:
            Upsampling mode. Select between "bilinear" and "nearest"

        Example:

        >>> # Get a 2-dimenisonal upsampling module with 2x channel reduction
        >>> upsample2d = UpsampleBlock(ndim=1, input_channels=64, output_channels=32,
        >>>                            scale_factor=2, mode="bilinear")
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int,
                 output_channels: int,
                 scale_factor: int = 2,
                 mode: str = "bilinear") -> None:
        """
        Initializes module parameters
        """
        super(UpsampleBlock, self).__init__()
        if mode not in ("bilinear", "nearest"):
            raise NotImplementedError(
                "use 'bilinear' or 'nearest' for upsampling mode")
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        self.scale_factor = scale_factor
        self.mode = mode if ndim > 1 else "nearest"
        self.conv = get_conv(ndim)(
            input_channels, output_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


def get_bnorm(dim: int) -> Type[nn.Module]:
    bn_dict = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
    return bn_dict[dim]


def get_conv(dim: int) -> Type[nn.Module]:
    conv_dict = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    return conv_dict[dim]


def get_activation(activation: int) -> Type[nn.Module]:
    if activation is None:
        return
    activations = {"lrelu": nn.LeakyReLU, "tanh": nn.Tanh,
                   "softplus": nn.Softplus, "relu": nn.ReLU}
    return activations[activation]
