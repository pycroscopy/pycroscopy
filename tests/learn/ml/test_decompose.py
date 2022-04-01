import sys
import pytest
from numpy.testing import assert_equal, assert_
import numpy as np
sys.path.append("../../../")

from pycroscopy.learn import tensor_decomposition


@pytest.mark.parametrize("decomp_type", ["cp", "tucker"])
@pytest.mark.parametrize("in_dim", [(8, 4, 4), (8, 4, 4, 3)])
def test_input_dim(in_dim, decomp_type):
    x = np.random.randn(*in_dim)
    _, f = tensor_decomposition(x, 3)
    assert_equal(len(f), len(in_dim))


@pytest.mark.parametrize("decomp_type", ["cp", "tucker"])
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_factors_dim(rank, decomp_type):
    x = np.random.randn(8, 4, 4, 3)
    weights, factors = tensor_decomposition(x, rank)
    assert_equal(len(weights), rank)
    f_shapes = [f.shape[-1] == rank for f in factors]
    assert_(all(f_shapes))


@pytest.mark.parametrize("decomp_type", ["cp", "tucker"])
@pytest.mark.parametrize("start_with", [2, 3])
def test_flat_dim(start_with, decomp_type):
    x = np.random.randn(8, 4, 4, 3)
    weights, factors = tensor_decomposition(x, 3, decomp_type, start_with)
    assert_equal(len(factors), start_with + 1)


