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
from .process import Process, parallel_compute
from . import fft
from . import gmode_utils
from . import proc_utils
from . import svd_utils
from .svd_utils import SVD, rebuild_svd
from . import decomposition
from .decomposition import Decomposition
from . import cluster
from .cluster import Cluster
from . import image_processing
from .image_processing import ImageWindow
from . import giv_utils
from .feature_extraction import FeatureExtractorParallel, FeatureExtractorSerial
from .image_transformation import geoTransformerParallel, geoTransformerSerial
from . import process
from .process import Process
from .giv_bayesian import GIVBayesian
from .signal_filter import SignalFilter


def no_impl(*args, **kwargs):
    raise NotImplementedError("You need to install Multiprocess package (pip,github) to do a parallel Computation.\n"
                              "Switching to the serial version. ")

FeatureExtractor = FeatureExtractorSerial
geoTransformer = geoTransformerSerial

try:
    import multiprocessing
except ImportError:
    FeatureExtractorParallel = no_impl
    geoTransformerParallel = no_impl
else:
    FeatureExtractor = FeatureExtractorParallel
    geoTransformer = geoTransformerParallel

__all__ = ['Cluster', 'Decomposition', 'ImageWindow', 'SVD', 'fft', 'gmode_utils', 'proc_utils', 'svd_utils',
           'giv_utils', 'rebuild_svd', 'Process', 'parallel_compute', 'Process', 'GIVBayesian', 'SignalFilter']
