"""
Pycroscopy's processing module

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    atom_finding
    cluster
    decomposition
    feature_extraction
    fft
    giv_utils
    gmode_utils
    image_processing
    image_transformation
    proc_utils
    process
    svd_utils

"""
from pycroscopy.core.processing.process import Process, parallel_compute
from . import fft
from . import gmode_utils
from . import histogram
from . import svd_utils
from .svd_utils import SVD, rebuild_svd
from . import decomposition
from .decomposition import Decomposition
from . import cluster
from .cluster import Cluster
from . import image_processing
from .image_processing import ImageWindow
from pycroscopy.core.processing.process import Process
from .signal_filter import SignalFilter
from .tree import ClusterTree

__all__ = ['Cluster', 'Decomposition', 'ImageWindow', 'SVD', 'fft', 'gmode_utils', 'histogram', 'svd_utils',
           'rebuild_svd', 'Process', 'parallel_compute', 'Process', 'SignalFilter', 'ClusterTree']
