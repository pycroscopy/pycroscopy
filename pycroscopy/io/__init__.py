"""
Translators to extract data from custom & proprietary microscope formats and write them to HDF5 files.

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    translators
    write_utils

"""
from . import translators
from .translators import *
from .ingestor import ingest
from .hdf_writer import HDFwriter
from .virtual_data import VirtualDataset, VirtualGroup, VirtualData

__all__ = translators.__all__
