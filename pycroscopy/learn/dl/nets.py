"""
nets.py
=========

Autoencoder and denoising autoencoder neural nets

by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.tensor as tt

from .nnblocks import (ConvBlock, UpsampleBlock, features_to_latent,
                       latent_to_features)


class FeatureExtractor(nn.Sequential):
    """
    Convolutional feature extractor
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int = 1,
                 layers_per_block: List[int] = None,
                 nfilters: int = 32,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 pool: bool = True,
                 ) -> None:
        """
        Initializes feature extractor module
        """
        super(FeatureExtractor, self).__init__()
        if layers_per_block is None:
            layers_per_block = [1, 2, 2]
        for i, layers in enumerate(layers_per_block):
            in_filters = input_channels if i == 0 else nfilters * i
            block = ConvBlock(ndim, layers, in_filters, nfilters * (i+1),
                              batchnorm=batchnorm, activation=activation,
                              pool=pool)
            self.add_module("c{}".format(i), block)


class Upsampler(nn.Sequential):
    """
    Convolutional upsampler (aka 'decoder')
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int = 96,
                 output_channels: int = 1,
                 layers_per_block: List[int] = None,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 activation_out: bool = False,
                 upsampling_mode: str = "bilinear",
                 ) -> None:
        """
        Initializes upsampler module
        """
        super(Upsampler, self).__init__()
        if layers_per_block is None:
            layers_per_block = [2, 2, 1]
        if activation_out:
            a_out = nn.Sigmoid() if output_channels == 1 else nn.Softmax(-1)

        nfilters = input_channels
        for i, layers in enumerate(layers_per_block):
            in_filters = nfilters if i == 0 else nfilters // i
            block = ConvBlock(ndim, layers, in_filters, nfilters // (i+1),
                              batchnorm=batchnorm, activation=activation,
                              pool=False)
            self.add_module("conv_block_{}".format(i), block)
            up = UpsampleBlock(ndim, nfilters // (i+1), nfilters // (i+1),
                               mode=upsampling_mode)
            self.add_module("up_{}".format(i), up)

        out = ConvBlock(ndim, 1, nfilters // (i+1), output_channels,
                        1, 1, 0, activation=None)
        self.add_module("output_layer", out)
        if activation_out:
            self.add_module("output_activation", a_out)


class AutoEncoder(nn.Module):
    """
    Convolutional autoencoder with latent space
    """
    def __init__(self,
                 ndim: int,
                 input_dim: Tuple[int],
                 latent_dim: int = 2,
                 layers_per_block: List[int] = [1, 2, 2],
                 nfilters: int = 32,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 activation_out: bool = True,
                 upsampling_mode: str = "bilinear"
                 ) -> None:
        """
        Initializes encoder, decoder, and latent parts of the model
        """
        super(AutoEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layers_per_block_e = layers_per_block
        layers_per_block_d = layers_per_block[::-1]
        encoder_channels_out = nfilters * len(layers_per_block)
        encoder_size_out = (tt(input_dim[1:]) // 2**len(layers_per_block)).tolist()

        self.encoder = FeatureExtractor(
            ndim, input_dim[0], layers_per_block_e,
            nfilters, batchnorm, activation, pool=True)
        self.features2latent = features_to_latent(
            [encoder_channels_out, *encoder_size_out], latent_dim)
        self.latent2features = latent_to_features(
            latent_dim, [encoder_channels_out, *encoder_size_out])
        self.decoder = Upsampler(
            ndim, encoder_channels_out, input_dim[0], layers_per_block_d,
            batchnorm, activation, activation_out, upsampling_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.encoder(x)
        x = self.features2latent(x)
        x = self.latent2features(x)
        x = self.decoder(x)
        return x

    def encode(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Encodes new data
        """
        x = self._2torch(x)
        with torch.no_grad():
            x = self.encoder(x)
            x = self.features2latent(x)
        return x

    def decode(self, x: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        Decodes latent coordinate(s) to data space
        """
        x = self._2torch(x)
        with torch.no_grad():
            x = self.latent2features(x)
            x = self.decoder(x)
        return x

    @classmethod
    def _2torch(cls, x: Union[np.ndarray, List]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = tt(x).float()
        x = x.view(1, -1) if x.ndim == 1 else x
        return x
