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
from . import fft
from . import gmode_utils
from . import proc_utils
from . import svd_utils
from .svd_utils import doSVD, rebuild_svd
from . import decomposition
from .decomposition import Decomposition
from . import cluster
from .cluster import Cluster
from . import image_processing
from .image_processing import ImageWindow
from . import giv_utils

def no_impl(*args,**kwargs):
    raise NotImplementedError("You need to install Multiprocess package (pip,github) to do a parallel Computation.\n"
                              "Switching to the serial version. ")

from .feature_extraction import FeatureExtractorParallel, FeatureExtractorSerial
from .image_transformation import geoTransformerParallel, geoTransformerSerial

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

__all__ = ['Cluster', 'Decomposition', 'ImageWindow', 'doSVD', 'fft', 'gmode_utils', 'proc_utils', 'svd_utils',
           'giv_utils', 'rebuild_svd']
