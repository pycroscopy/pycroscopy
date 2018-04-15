"""
The Pycroscopy package.

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    core

"""
from . import core
from .core import *
from warnings import warn

from .io import translators
from . import analysis
from . import processing

from .__version__ import version as __version__
from .__version__ import time as __time__

__all__ = core.__all__
warn('You are using the unity_dev branch, which is aimed at a 1.0 release for pycroscopy. '
     'Be advised - this branch changes very significantly and frequently. It is therefore not meant for usage. '
     'Use the master or dev branches for regular purposes.')
