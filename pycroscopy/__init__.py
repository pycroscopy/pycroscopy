"""
The Pycroscopy package.

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    analysis
    io
    processing
    viz

"""
from . import core
from .core import *

from .__version__ import version as __version__
from .__version__ import date as __date__

__all__ = core.__all__
