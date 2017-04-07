# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import least_squares
import itertools as itt
import multiprocessing as mp
import time as tm
from _warnings import warn
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

sys.path.append('../../../')
import pycroscopy
from pycroscopy.io.io_utils import recommendCores, realToCompound
from pycroscopy.io.microdata import MicroDataset, MicroDataGroup
from pycroscopy.io.io_hdf5 import ioHDF5
from pycroscopy.analysis.model import Model

class Gauss_Fit(Model):#, fitting_parms=None, num_cores=None):
        """
        Computes the guess and fit coefficients for the provided atom guess positions and writes these results to the
        given h5 group

        Parameters
        ----------

        h5_grp : h5py.Group reference
            Group containing the atom guess positions, cropped clean image and some necessary parameters
        fitting_parms : dictionary
            Parameters used for atom position fitting
        num_cores : unsigned int (Optional. Default = available logical cores - 2)
            Number of cores to compute with

        Returns
        -------
        h5_grp : h5py.Group reference
            Same group as the parameter but now with the 'Guess' and 'Fit' datasets
        """

        def __init__(self, h5_main, variables=['Frequency'], parallel=True):
            super(Gauss_Fit, self).__init__(h5_main, variables, parallel)




            pass

if __name__=='__main__':
    h5_path = 'C:\Users\o2d\Desktop\MouseWithoutBorders\Atom position refinement testing\Simulated_image\image 04.h5'
    folder_path, file_path = os.path.split(h5_path)

    print(image_path)

    file_base_name, file_extension = file_name.rsplit('.')
    h5_file = h5py.File(image_path, mode='r+')


    # look at the data tree in the h5

    # define a small function called 'print_tree' to look at the folder tree stucture
    def print_tree(parent):
        print(parent.name)
        if isinstance(parent, h5py.Group):
            for child in parent:
                print_tree(parent[child])

                # when we run into the SVD children store a reference to the group
                if child == 'S':
                    svd_grp = h5py.Group


    print('Datasets and datagroups within the file:')
    file_handle = h5_file
    print_tree(file_handle)
