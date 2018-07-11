"""
Created by Christopher Smith on 7/11/2018
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os
import h5py
import numpy as np
from skimage.io import imread
from skimage.measure import block_reduce

from pyUSID.io.image import ImageTranslator
from .df_utils.dm_utils import read_dm3, read_dm4


class ImageTranslator(ImageTranslator):
    """
    Translates data from a set of image files to an HDF5 file.  This version has been extended to
    support dm3 and dm4 files
    """

    def translate(self, image_path, h5_path=None, bin_factor=None, bin_func=np.mean, normalize=False, **image_args):
        """
        Basic method that adds Ptychography data to existing hdf5 thisfile
        You must have already done the basic translation with BEodfTranslator

        Parameters
        ----------------
        image_path : str
            Absolute path to folder holding the image files
        h5_path : str, optional
            Absolute path to where the HDF5 file should be located.
            Default is None
        bin_factor : array_like of uint, optional
            Downsampling factor for each dimension.  Default is None.
        bin_func : callable, optional
            Function which will be called to calculate the return value
            of each block.  Function must implement an axis parameter,
            i.e. numpy.mean.  Ignored if bin_factor is None.  Default is
            numpy.mean.
        normalize : boolean, optional
            Should the raw image be normalized when read in
            Default False
        image_args : dict
            Arguments to be passed to read_image.  Arguments depend on the type of image.

        Returns
        ----------
        h5_main : h5py.Dataset
            HDF5 Dataset object that contains the flattened images

        """
        image_path, h5_path = self._parse_file_path(image_path)

        image, image_parms = read_image(image_path, **image_args)
        usize, vsize = image.shape[:2]

        self.image_path = image_path
        self.h5_path = h5_path

        '''
        Check if a bin_factor is given.  Set up binning objects if it is.
        '''
        if bin_factor is not None:
            self.rebin = True
            if isinstance(bin_factor, int):
                self.bin_factor = (bin_factor, bin_factor)
            elif len(bin_factor) == 2:
                self.bin_factor = tuple(bin_factor)
            else:
                raise ValueError('Input parameter `bin_factor` must be a length 2 array_like or an integer.\n' +
                                 '{} was given.'.format(bin_factor))
            usize = int(usize / self.bin_factor[0])
            vsize = int(vsize / self.bin_factor[1])
            self.binning_func = block_reduce
            self.bin_func = bin_func

        image = self.binning_func(image, self.bin_factor, self.bin_func)

        image_parms['normalized'] = normalize
        image_parms['image_min'] = np.min(image)
        image_parms['image_max'] = np.max(image)
        '''
        Normalize Raw Image
        '''
        if normalize:
            image -= np.min(image)
            image = image / np.float32(np.max(image))

        h5_main = self._setup_h5(usize, vsize, image.dtype.type, image_parms)

        h5_main = self._read_data(image, h5_main)

        return h5_main


def read_image(image_path, *args, **kwargs):
    """
    Read the image file at `image_path` into a numpy array
    Parameters
    ----------
    image_path : str
        Path to the image file
    Returns
    -------
    image : numpy.ndarray
        Array containing the image from the file `image_path`
    image_parms : dict
        Dictionary containing image parameters.  If image type does not have
        parameters then an empty dictionary is returned.
    """
    ext = os.path.splitext(image_path)[1]
    if ext == '.dm3':
        kwargs.pop('as_grey', None)
        return read_dm3(image_path, *args, **kwargs)
    elif ext == '.dm4':
        kwargs.pop('as_grey', None)
        return read_dm4(image_path, *args, **kwargs)
    elif ext == '.txt':
        return np.loadtxt(image_path, *args, **kwargs), dict()
    else:
        # Set the as_grey argument to True is not already provided.
        kwargs['as_grey'] = (kwargs.pop('as_grey', True))
        return imread(image_path, *args, **kwargs), dict()