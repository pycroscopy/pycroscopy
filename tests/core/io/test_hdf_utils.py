# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

import unittest
import os
import h5py
import numpy as np

import sys
sys.path.append("../../../pycroscopy/")
from pycroscopy import MicroDataGroup, MicroDataset
from pycroscopy import HDFwriter
from pycroscopy.core.io import hdf_utils


class TestHDFUtils(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    unittest.main()