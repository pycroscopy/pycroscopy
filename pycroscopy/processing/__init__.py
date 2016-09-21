import fft
import gmode_utils


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
