"""
Created on Feb 9, 2016

@author: Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os
import h5py
import numpy as np
from skimage.measure import block_reduce

from .io_image import read_image
from .translator import Translator, generate_dummy_main_parms
from .write_utils import Dimension, calc_chunks
from .hdf_utils import write_main_dataset, write_simple_attrs


class ImageTranslator(Translator):
    """
    Translate Pytchography data from a set of images to an HDF5 file
    """

    def __init__(self, *args, **kwargs):
        super(ImageTranslator, self).__init__(*args, **kwargs)

        self.rebin = False
        self.bin_factor = 1
        self.hdf = None
        self.binning_func = self.__no_bin
        self.bin_func = None
        self.image_path = None
        self.h5_path = None

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

    def _setup_h5(self, usize, vsize, data_type, image_parms):
        """
        Setup the HDF5 file in which to store the data including creating
        the Position and Spectroscopic datasets

        Parameters
        ----------
        usize : int
            Number of pixel columns in the images
        vsize : int
            Number of pixel rows in the images
        data_type : type
            Data type to save image as
        image_parms : dict
            Image parameters to be stored as attributes of the measurement
            group

        Returns
        -------
        h5_main : h5py.Dataset
            HDF5 Dataset that the images will be written into
        """
        num_pixels = usize * vsize

        root_parms = generate_dummy_main_parms()
        root_parms['data_type'] = 'ImageData'

        root_parms.update(image_parms)

        main_parms = {'image_size_u': usize,
                      'image_size_v': vsize,
                      'num_pixels': num_pixels,
                      'translator': 'Image'}

        self.h5_file = h5py.File(self.h5_path)

        # Root attributes first:
        write_simple_attrs(self.h5_file, root_parms)

        # Create the hdf5 data Group
        meas_grp = self.h5_file.create_group('Measurement_000')
        write_simple_attrs(meas_grp, main_parms)
        chan_grp = meas_grp.create_group('Channel_000')
        # Get the Position and Spectroscopic Datasets

        pos_dims = [Dimension('X', 'a.u.', np.arange(usize)), Dimension('Y', 'a.u.', np.arange(vsize))]

        chunking = calc_chunks([num_pixels, 1],
                                  data_type(0).itemsize,
                                  unit_chunks=[1, 1])

        h5_main = write_main_dataset(chan_grp, (usize*vsize, 1), 'Raw_Data', 'Intensity', 'a.u.',
                                     pos_dims, Dimension('None', 'a.u.', [1]), dtype=data_type, chunks=chunking)

        self.h5_file.flush()

        return h5_main

    @staticmethod
    def _parse_file_path(image_path):
        """
        Returns a list of all files in the directory given by path

        Parameters
        ---------------
        image_path : string / unicode
            absolute path to directory containing files

        Returns
        ----------
        image_path : str
            Absolute file path to the image
        image_ext : str
            File extension of image
        """
        if not os.path.exists(os.path.abspath(image_path)):
            raise ValueError('Specified image does not exist.')
        else:
            image_path = os.path.abspath(image_path)

        base_name, _ = os.path.splitext(image_path)

        h5_name = base_name + '.h5'
        h5_path = os.path.join(image_path, h5_name)

        return image_path, h5_path

    @staticmethod
    def __no_bin(image, *args, **kwargs):
        """
        Does absolutely nothing to the image.  Exists so that we can have
        a bin function to call whether we actually rebin the image or not.

        Parameters
        ----------
        image : ndarray
            Image
        args:
            Argument list
        kwargs:
            Keyword argument list

        Returns
        -------
        image : ndarray
            The input image
        """
        return image

    @staticmethod
    def _read_data(image, h5_main):
        """
        Read the image into the dataset

        image : numpy.array
            Numpy array containing the image
        h5_main : h5py.Dataset
            HDF5 Dataset that will hold the image
        """

        h5_main[:] = image.reshape(h5_main.shape)

        h5_main.file.flush()

        return h5_main
