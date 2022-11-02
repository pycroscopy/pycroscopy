"""
More generic ML and DL tools

ML part could contain
- cluster
- decomposition, e.g. SVD, PCA, NMF
- other unmixing (e.g., nonlinear unmixing, tensor factorization, etc.)
- gaussian mixture modelling

Dl part could contain
- cleaning / denoising
- autoencoders

Submodules
----------
.. autosummary::
    :toctree: _autosummary

"""

from .dl import nnblocks, models, datautils
from .dl.trainer import Trainer
from .ml import TensorFactor, MatrixFactor

__all__ = ['nnblocks', 'models', 'Trainer', 'datautils', 'TensorFactor', 'MatrixFactor']
