"""
Physical or chemical model-based analysis of data

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    be_loop_fitter
    be_sho_fitter
    fit_methods
    fitter
    giv_bayesian
    guess_methods
    optimize

"""

from . import utils
from .utils import *
from . import be_sho_fitter
from .be_sho_fitter import BESHOfitter
from . import be_loop_fitter
from .be_loop_fitter import BELoopFitter
from . import fitter
from .fitter import Fitter
from .giv_bayesian import GIVBayesian
from .be_relax_fit import BERelaxFit

__all__ = ['Fitter', 'BESHOfitter', 'BELoopFitter', 'utils', 'GIVBayesian',
           'BERelaxFit']
__all__ += utils.__all__
