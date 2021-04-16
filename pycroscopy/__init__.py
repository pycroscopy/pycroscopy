"""
The Pycroscopy package.

Submodules
----------

.. autosummary::
    :toctree: _autosummary
"""
from warnings import warn
from .io import translators
from .io.translators import ImageTranslator  # Use pycroscopy version of ImageTranslator rather than pyUSID's
from . import analysis
from . import processing
from . import viz

from .__version__ import version as __version__
from .__version__ import time as __time__

warn('The subsequent version of pycroscopy will not be backwards compatible '
     'since it will be substantially overhauled', FutureWarning)
