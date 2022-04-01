"""
nnblocks.py
=========

Individual NN blocks

by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""
from typing import Union, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
tt = torch.tensor

from warnings import warn, filterwarnings

filterwarnings("ignore", module="torch.nn.functional")


class ConvBlock(nn.Module):
    """
    Creates a block of layers each consisting of convolution operation,
    nonlinear activation and (optionally) batch normalization

    Parameters
    ----------
    ndim
        Data dimensionality (1, 2, or 3)
    nlayers
        Number of layers in the block
    input_channels
        Number of input feature channels for the block
    output_channels
        Number of output feature channels for the block
    kernel_size
        Size of convolutional filter in pixels (Default: 3)
    stride
        Stride of convolutional filter (Default: 1)
    padding
        Value for edge padding (Default: 1)
    batchnorm
        Add batch normalization to each layer in the block (Default: False)
    activation
        Non-linear activation: "relu", "lrelu", "tanh", "softplus", or None.
        (Default: "lrelu").
    pool
        Applies max-pooling operation at the end of the block (Default: True).

    Examples
    --------

    Get convolutional block with three 1D convolutions and batch normalization

    >>> convblock1d = ConvBlock(
    >>>     ndim=1, nlayers=3, input_channels=1,
    >>>     output_channels=32, batch_norm=True)

    Get convolutional block with two 2D convolutions and max-pooling at the end

    >>> convblock1d = ConvBlock(
    >>>     ndim=2, nlayers=2, input_channels=1,
    >>>     output_channels=32, pool=True)
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
                 pool: bool = False,
                 ) -> None:
        """
        Initializes module parameters
        """
        super(ConvBlock, self).__init__()
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        activation = get_activation(activation)
        block = []
        for i in range(nlayers):
            input_channels = output_channels if i > 0 else input_channels
            block.append(get_conv(ndim)(input_channels, output_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding))
            if activation is not None:
                block.append(activation())
            if batchnorm:
                block.append(get_bnorm(ndim)(output_channels))
        if pool:
            block.append(get_maxpool(ndim)(2, 2))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass
        """
        output = self.block(x)
        return output


class UpsampleBlock(nn.Module):
    """
    Upsampling performed using bilinear or nearest-neigbor interpolation
    followed by 1-by-1 convolution, which an be used to reduce a number of
    feature channels

    Parameters
    ----------
    ndim
        Data dimensionality (1, 2, or 3).
    input_channels
        Number of input channels for the block.
    output_channels
        Number of the output channels for the block.
    scale_factor
        Scale factor for upsampling.
    mode
        Upsampling mode. Select between "bilinear" and "nearest"
        (Default: bilinear for 2D, nearest for 1D and 3D).

    Examples
    --------

    2D upsampling with a 2x reduction in the number of feature channels

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
        warn_msg = ("'bilinear' mode is not supported for 1D and 3D;" +
                    " switching to 'nearest' mode")
        if mode not in ("bilinear", "nearest"):
            raise NotImplementedError(
                "Use 'bilinear' or 'nearest' for upsampling mode")
        if not 0 < ndim < 4:
            raise AssertionError("ndim must be equal to 1, 2 or 3")
        if mode == "bilinear" and ndim in (3, 1):
            warn(warn_msg, category=UserWarning)
            mode = "nearest"
        self.mode = mode
        self.scale_factor = scale_factor
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


class features_to_latent(nn.Module):
    """
    Maps features (usually, from a convolutional net/layer) to latent space
    """
    def __init__(self, input_dim: Tuple[int], latent_dim: int = 2) -> None:
        super(features_to_latent, self).__init__()
        self.reshape_ = torch.prod(tt(input_dim))
        self.fc_latent = nn.Linear(self.reshape_, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.reshape_)
        return self.fc_latent(x)


class latent_to_features(nn.Module):
    """
    Maps latent vector to feature space
    """
    def __init__(self, latent_dim: int, out_dim: Tuple[int]) -> None:
        super(latent_to_features, self).__init__()
        self.reshape_ = out_dim
        self.fc = nn.Linear(latent_dim, torch.prod(tt(out_dim)).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x.view(-1, *self.reshape_)


def get_bnorm(dim: int) -> Type[nn.Module]:
    bn_dict = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
    return bn_dict[dim]


def get_conv(dim: int) -> Type[nn.Module]:
    conv_dict = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    return conv_dict[dim]


def get_maxpool(dim: int) -> Type[nn.Module]:
    conv_dict = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
    return conv_dict[dim]


def get_activation(activation: int) -> Type[nn.Module]:
    if activation is None:
        return
    activations = {"lrelu": nn.LeakyReLU, "tanh": nn.Tanh,
                   "softplus": nn.Softplus, "relu": nn.ReLU}
    return activations[activation]
