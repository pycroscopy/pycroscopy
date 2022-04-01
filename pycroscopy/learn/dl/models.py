"""
models.py
=========

Autoencoder and denoising autoencoder

by Maxim Ziatdinov (email: ziatdinovmax@gmail.com)
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
tt = torch.tensor

from .nnblocks import (ConvBlock, UpsampleBlock, features_to_latent,
                       latent_to_features)


class FeatureExtractor(nn.Sequential):
    """
    Convolutional feature extractor

    Parameters
    ----------
    ndim
        Data dimensionality (1, 2, or 3).
    input_channels
        Number of input feature channels
        (Defaults to a greyscale image with a single channel)
    layers_per_block
        Number of layers in each block (Default: [1, 2, 2]).
    nfilters
        Number of convolutional filters in the first convolutional block.
        The number of filters in each consecutive block is computed as
        :math:`block_i = nfilters * (i+1)` (Default: 32).
    batchnorm
        Add batch normalization to each layer in the block (Default: False).
    activation
        Non-linear activation: "relu", "lrelu", "tanh", "softplus", or None.
        (Default: "lrelu").
    pool
        Applies max-pooling operation at the end of the block (Default: True).

    Examples
    --------

    Create a feature extractor for RGB-like 2D data

    >>> layers_per_block = [2, 2, 3]
    >>> feature_extractor = FeatureExtractor(
    >>>     ndim=2, input_channels=3, layers_per_block=layers_per_block)

    Create a feature extractor for greyscale 2D data

    >>> layers_per_block = [2, 2, 3]
    >>> feature_extractor = FeatureExtractor(
    >>>     ndim=2, input_channels=1, layers_per_block=layers_per_block)
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

    Parameters
    ----------
    ndim
        Data dimensionality (1, 2, or 3).
    input_channels
        Number of input channels (convolutional filters) for the input layer.
        The number of filters in each consecutive block is computed as
        :math:`block_i = nfilters // (i+1)` (Default: 96).
    layers_per_block
        Number of layers in each block (Default: [2, 2, 1]).
    output_channels
        Number of the output channels (Deafult: 1)
    batchnorm
        Add batch normalization to each layer in the block (Default: False).
    activation
        Non-linear activation: "relu", "lrelu", "tanh", "softplus", or None.
        (Default: "lrelu").
    activation_out:
        Applies sigmoid (output_channels=1) or softmax (output_channels>1)
        activation to the final convolutional layer (Default: True)
    upsampling_mode
        Upsampling mode. Select between "bilinear" and "nearest"
        (Default: bilinear for 2D, nearest for 1D and 3D).
    """
    def __init__(self,
                 ndim: int,
                 input_channels: int = 96,
                 layers_per_block: List[int] = None,
                 output_channels: int = 1,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 activation_out: bool = True,
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

    Parameters
    ----------
    input_dim
        Input dimensions: (channels, length), (channels, height, width)
        or (height, width, depth).
    latent_dim
        Latent sapce dimensionality (Default: 2).
    layers_per_block
        List with the number of layers for each block of the encoder.
        The number of layers for the decoder is computed by reversing
        this list (Default: [1, 2, 2]).
    nfilters
        Number of convolutional filters in the first convolutional block
        of the encoder. The number of filters in each consecutive block
        is computed as :math:`block_i = nfilters * (i+1)`. The number of
        filters in the first layer of the decoder is equal to the number of
        filters in the last layer of the encoder and the number of filters
        in each consecutive block is computed as :math:`block_i = nfilters // (i+1)`
        (Default: 32).
    batchnorm
        Add batch normalization to each layer (Default: True).
    activation
        Non-linear activation: "relu", "lrelu", "tanh", "softplus", or None.
        (Default: "lrelu").
    activation_out:
        Applies sigmoid (output_channels=1) or softmax (output_channels>1)
        activation to the final convolutional layer (Default: True).
    upsampling_mode
        Upsampling mode. Select between "bilinear" and "nearest"
        (Default: bilinear for 2D, nearest for 1D and 3D).

    Examples
    --------

    Create autoencoder for dataset with 64 x 64 greyscale images

    >>> input_dim = (1, 64, 64)  # (channles, height, width)
    >>> ae = AutoEncoder(input_dim, latent_dim=2)
    """
    def __init__(self,
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
        self.input_dim = input_dim
        ndim = len(input_dim) - 1
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
            ndim, encoder_channels_out, layers_per_block_d, input_dim[0],
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

    def encode(self,
               x: Union[torch.Tensor, np.ndarray],
               **kwargs: int) -> np.ndarray:
        """
        Encodes new data
        """
        def _encode(xi) -> torch.Tensor:
            xi = self._2torch(xi).to(self.device)
            with torch.no_grad():
                xi = self.encoder(xi)
                xi = self.features2latent(xi)
            return xi.cpu()

        batch_size = kwargs.get("batch_size", 100)
        num_batches = len(x) // batch_size
        if num_batches == 0:
            num_batches += 1
        z_all = []
        for i in range(num_batches):
            xi = x[i*batch_size:(i+1)*batch_size]
            z_i = _encode(xi)
            z_all.append(z_i)
        xi = x[(i+1)*batch_size:]
        if len(xi) > 0:
            z_i = _encode(xi)
            z_all.append(z_i)

        return torch.cat(z_all).numpy()

    def decode(self, x: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
        """
        Decodes latent coordinate(s) to data space
        """
        x = self._2torch(x).to(self.device)
        with torch.no_grad():
            x = self.latent2features(x)
            x = self.decoder(x)
        return x.cpu().numpy()

    def decode_grid(self, d: int = 12, z1: Tuple[int] = None,
                    z2: Tuple[int] = None) -> np.ndarray:
        """
        Decodes a grid of latent coordinates to data sapce
        """
        if z1 is None:
            z1 = [-1.65, 1.65]
        if z2 is None:
            z2 = [-1.65, 1.65]
        grid_x = torch.linspace(z1[1], z1[0], d)
        grid_y = torch.linspace(z2[0], z2[1], d)
        decoded_grid = []
        for xi in grid_x:
            for yi in grid_y:
                decoded_grid.append(self.decode([xi, yi]))
        decoded_grid = np.concatenate(decoded_grid)
        return decoded_grid.reshape(-1, *self.input_dim[1:])

    @classmethod
    def _2torch(cls, x: Union[np.ndarray, List]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = tt(x).float()
        x = x.view(1, -1) if x.ndim == 1 else x
        return x


class DenoisingAutoEncoder(nn.Module):
    """
    Fully convolutional convolutional denosing autoencoder

    Parameters
    ----------
    n_dim
        Dimensionality (1, 2 or 3)
    layers_per_block
        List with the number of layers for each block of the encoder.
        The number of layers for the decoder is computed by reversing
        this list (Default: [1, 2, 2]).
    nfilters
        Number of convolutional filters in the first convolutional block
        of the encoder. The number of filters in each consecutive block
        is computed as :math:`block_i = nfilters * (i+1)`. The number of
        filters in the first layer of the decoder is equal to the number of
        filters in the last layer of the encoder and the number of filters
        in each consecutive block is computed as :math:`block_i = nfilters // (i+1)`
        (Default: 32).
    batchnorm
        Add batch normalization to each layer (Default: True).
    activation
        Non-linear activation: "relu", "lrelu", "tanh", "softplus", or None.
        (Default: "lrelu").
    activation_out:
        Applies sigmoid (output_channels=1) or softmax (output_channels>1)
        activation to the final convolutional layer (Default: True).
    upsampling_mode
        Upsampling mode. Select between "bilinear" and "nearest"
        (Default: bilinear for 2D, nearest for 1D and 3D).
    """
    def __init__(self,
                 ndim: Tuple[int],
                 input_channels: int = 1,
                 layers_per_block: List[int] = [1, 2, 2],
                 nfilters: int = 32,
                 batchnorm: bool = True,
                 activation: str = "lrelu",
                 activation_out: bool = False,
                 upsampling_mode: str = "bilinear"
                 ) -> None:
        """
        Initializes encoder, decoder, and latent parts of the model
        """
        super(DenoisingAutoEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layers_per_block_e = layers_per_block
        layers_per_block_d = layers_per_block[::-1]
        encoder_channels_out = nfilters * len(layers_per_block)

        self.encoder = FeatureExtractor(
            ndim, input_channels, layers_per_block_e,
            nfilters, batchnorm, activation, pool=True)
        self.decoder = Upsampler(
            ndim, encoder_channels_out, layers_per_block_d, input_channels,
            batchnorm, activation, activation_out, upsampling_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self,
                x: Union[torch.Tensor, np.ndarray],
                **kwargs: int) -> np.ndarray:
        """
        Predicts on new data
        """
        def _predict(xi) -> torch.Tensor:
            xi = self._2torch(xi).to(self.device)
            with torch.no_grad():
                xi = self.encoder(xi)
                xi = self.decoder(xi)
            return xi.cpu()

        batch_size = kwargs.get("batch_size", 100)
        num_batches = len(x) // batch_size
        if num_batches == 0:
            num_batches += 1
        z_all = []
        for i in range(num_batches):
            xi = x[i*batch_size:(i+1)*batch_size]
            z_i = _predict(xi)
            z_all.append(z_i)
        xi = x[(i+1)*batch_size:]
        if len(xi) > 0:
            z_i = _predict(xi)
            z_all.append(z_i)

        return torch.cat(z_all).numpy()

    @classmethod
    def _2torch(cls, x: Union[np.ndarray, List]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = tt(x).float()
        x = x.view(1, -1) if x.ndim == 1 else x
        return x
