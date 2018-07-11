"""
The Pycroscopy package.

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    core

"""
import pyUSID as core
from pyUSID.viz import *
from pyUSID.processing import *
from pyUSID.io import *
# For legacy reasons:
from pyUSID import USIDataset as PycroDataset
from warnings import warn
from .io import translators
from .io.translators import ImageTranslator  # Use pycroscopy version of ImageTranslator rather than pyUSID's
from . import analysis
from . import processing
from . import viz

from .__version__ import version as __version__
from .__version__ import time as __time__

warn('Contents of pycroscopy.core such as hdf_utils, plot_utils have been moved to pyUSID but will continue to be '
     'available implicitly till the next release. Please update import statements to import such modules directly from'
     'pyUSID. See - https://pycroscopy.github.io/pycroscopy/whats_new.html under June 28 2018', FutureWarning)

__all__ = core.__all__
__all__ += ['PycroDataset']
