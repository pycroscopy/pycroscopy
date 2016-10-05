
from . import fft
from . import gmode_utils
from . import proc_utils
from . import svd_utils
from .svd_utils import doSVD
from . import decomposition
from .decomposition import Decomposition
from . import cluster
from .cluster import Cluster
from . import image_processing
from .image_processing import ImageWindow

def no_impl(*args,**kwargs):
    raise NotImplementedError("You need to install Multiprocess package (pip,github) to do a parallel Computation.\n"
                              "Switching to the serial version. ")

from .feature_extraction import FeatureExtractorParallel, FeatureExtractorSerial
from .geometric_transformation import geoTransformerParallel, geoTransformerSerial

FeatureExtractor = FeatureExtractorSerial
geoTransformer = geoTransformerSerial

try:
    import multiprocess
except ImportError:
    FeatureExtractorParallel = no_impl
    geoTransformerParallel = no_impl
else:
    FeatureExtractor = FeatureExtractorParallel
    geoTransformer = geoTransformerParallel

__all__ = ['Cluster', 'Decomposition', 'ImageWindow', 'doSVD', 'fft', 'gmode_utils', 'proc_utils', 'svd_utils']
