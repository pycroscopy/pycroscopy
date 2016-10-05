# Not supporting for __all__
from . import analysis
from .analysis import *
from . import io
from .io import *
from . import processing
from .processing import *
from . import viz
from .viz import *

from __version__ import version as __version__
from __version__ import date as __date__

# TODO: need to figure out what to do with external libs: numpy_groupies and pyqtgraph.

__all__ = ['processing', 'analysis', 'io', 'viz', '__date__', '__version__']
__all__+= io.__all__
__all__+= processing.__all__
__all__+= analysis.__all__
__all__+= viz.__all__
