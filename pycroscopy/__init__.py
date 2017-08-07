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
from . import analysis
from .analysis import *
from . import io
from .io import *
from . import processing
from .processing import *
from . import viz
from .viz import *

from .__version__ import version as __version__
from .__version__ import date as __date__

__all__ = ['processing', 'analysis', 'io', 'viz', '__date__', '__version__']
__all__ += io.__all__
__all__ += processing.__all__
__all__ += analysis.__all__
__all__ += viz.__all__
