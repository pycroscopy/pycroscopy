# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath
"""

from __future__ import division

import abc
from ..io_utils import getAvailableMem


class Translator(object):
    """
    Abstract class that defines the most basic functionality of a data format translator.
    A translator converts experimental data from binary / proprietary
    data formats to a single standardized HDF5 data file
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_mem_mb=1024):
        """
        Parameters
        -----------
        max_mem_mb : unsigned integer (Optional. Default = 1024)
            Maximum system memory (in megabytes) that the translator can use
            
        Returns
        -------
        Translator object
        """
        self.max_ram = min(max_mem_mb*1024**2, 0.75*getAvailableMem())

    @abc.abstractmethod
    def translate(self, filepath):
        """
        Abstract method.
        To be implemented by extensions of this class. God I miss Java!
        """
        pass

    @abc.abstractmethod
    def _parse_file_path(self, input_path):
        """
        Abstract method
        Parses the `input_path` to determine the `basename` and find
        the appropriate data files

        """
        pass

    @abc.abstractmethod
    def _read_data(self):
        """
        Abstract method
        Reads the data into the hdf5 datasets.
        """
