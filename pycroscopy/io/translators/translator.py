# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath
"""

from __future__ import division; # int/int = float
import abc # Abstract base class https://pymotw.com/2/abc/ <---- This needs to be implemented in a cleaner way
from ..io_utils import getAvailableMem

class Translator(object):
    """
    Abstract class that defines the most basic functionality of a data format translator.
    A translator converts experimental data from binary / proprietary
    data formats to a single standardized HDF5 data file
    """
    
    def __init__(self, max_mem_mb=1024):
        """
        Parameters
        -----------
        max_ram_mb : unsigned integer
            Maximum system memory (in megabytes) that the translator can use
            
        Returns
        -------
        Translator object
        """
        self.max_ram = min(max_mem_mb*1024**2, 0.75*getAvailableMem())
    
    @abc.abstractmethod
    def translate(self,filepath):
        """
        Abstract method.\n
        To be implemented by extensions of this class. God I miss Java!
        """
        pass