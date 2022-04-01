import sys
import pytest
from numpy.testing import assert_equal
import torch

sys.path.append("../../../")

from pycroscopy.learn import models


@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_autoencoder_output(dim, size):
    input_dim = (1, *size)
    x = torch.randn(2, *input_dim)
    ae = models.AutoEncoder(input_dim, 2, [1, 1])
    out = ae(x)
    assert_equal(input_dim, out.shape[1:])


@pytest.mark.parametrize("ch", [1, 2, 3])
@pytest.mark.parametrize("ndim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_denoising_autoencoder_output(ndim, size, ch):
    input_dim = (ch, *size)
    x = torch.randn(2, *input_dim)
    dae = models.DenoisingAutoEncoder(ndim, ch, [1, 1])
    out = dae(x)
    assert_equal(input_dim, out.shape[1:])


@pytest.mark.parametrize("zdim", [1, 2, 5])
@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_autoencoder_encoding(zdim, dim, size):
    input_dim = (1, *size)
    x = torch.randn(12, *input_dim)
    ae = models.AutoEncoder(input_dim, zdim, [1, 1])
    z = ae.encode(x)
    assert_equal(zdim, z.shape[-1])


@pytest.mark.parametrize("zdim", [1, 2, 5])
@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_autoencoder_decoding(zdim, dim, size):
    input_dim = (1, *size)
    x = torch.randn(2, *input_dim)
    ae = models.AutoEncoder(input_dim, zdim, [1, 1])
    z = torch.randn(zdim)
    x_ = ae.decode(z)
    assert_equal(x_.shape[1:], x.shape[1:])


@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_autoencoder_decode_grid(dim, size):
    input_dim = (1, *size)
    grid_spacing = 4
    x = torch.randn(2, *input_dim)
    ae = models.AutoEncoder(input_dim, 2, [1, 1])
    grid = ae.decode_grid(grid_spacing)
    assert_equal(grid.shape[0], grid_spacing**2)
    assert_equal(grid.shape[1:], x.shape[2:])


@pytest.mark.parametrize("bsize", [2, 100])
@pytest.mark.parametrize("ch", [1, 2, 3])
@pytest.mark.parametrize("ndim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_denoising_autoencoder_batches(bsize, ndim, size, ch):
    input_dim = (ch, *size)
    x = torch.randn(5, *input_dim)
    dae = models.DenoisingAutoEncoder(
        ndim, ch, [1, 1])
    out = dae.predict(x, batch_size=bsize)
    assert_equal(out.shape[0], 5)


@pytest.mark.parametrize("bsize", [2, 100])
@pytest.mark.parametrize("dim, size", [(1, [8]), (2, [8, 8]), (3, [8, 8, 8])])
def test_autoencoder_batches(bsize, dim, size):
    input_dim = (1, *size)
    x = torch.randn(5, *input_dim)
    ae = models.AutoEncoder(input_dim, 2, [1, 1])
    out = ae.encode(x, batch_size=bsize)
    assert_equal(out.shape[0], 5)
