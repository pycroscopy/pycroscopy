"""
Created on Feb 9, 2016

@author: Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os

import numpy as np
from skimage.measure import block_reduce
import h5py

from .df_utils.dm_utils import read_dm3
from pyUSID.io.image import read_image
from pyUSID.io.translator import Translator, generate_dummy_main_parms
from pyUSID.io.write_utils import Dimension, calc_chunks
from pyUSID.io.hdf_utils import get_h5_obj_refs, link_as_main, write_main_dataset, \
    write_simple_attrs, create_indexed_group


class MovieTranslator(Translator):
    """
    Translate Pytchography data from a set of images to an HDF5 file
    """

    def __init__(self, *args, **kwargs):
        super(MovieTranslator, self).__init__(*args, **kwargs)

        self.rebin = False
        self.bin_factor = 1
        self.h5_file = None
        self.binning_func = self.__no_bin
        self.bin_func = None
        self.image_ext = None

    def translate(self, h5_path, image_path, bin_factor=None, bin_func=np.mean, start_image=0, image_type='.tif'):
        """
        Basic method that adds Movie data to existing hdf5 file

        Parameters
        ----------------
        h5_path : str
            Absolute path to where the HDF5 file should be located
        image_path : str
            Absolute path to folder holding the image files
        bin_factor : array_like of uint, optional
            Downsampling factor for each dimension.  Default is None.
        bin_func : callable, optional
            Function which will be called to calculate the return value
            of each block.  Function must implement an axis parameter,
            i.e. numpy.mean.  Ignored if bin_factor is None.  Default is
            numpy.mean.
        start_image : int, optional
            Integer denoting which image in the file path should be considered the starting
            point.  Default is 0, start with the first image on the list.
        image_type : str, optional
            File extension of images to load.  Used to filter out other files in the same
            directory.  Default .tif

        Returns
        ----------
        h5_main : h5py.Dataset
            HDF5 Dataset object that contains the flattened images

        """
        self.image_ext = image_type

        image_path = os.path.abspath(image_path)
        h5_path = os.path.abspath(h5_path)
        
        if os.path.exists(h5_path):
            os.remove(h5_path)

        self.h5_file = h5py.File(h5_path, 'w')
        
        '''
        Get the list of all files with the provided extension and the number of files in the list
        '''
        if os.path.isfile(image_path):
            file_list, image_parms = read_dm3(image_path)
            usize = image_parms['SuperScan-Height']
            vsize = image_parms['SuperScan-Width']
            data_type = file_list.dtype.type
            num_images = file_list.shape[0] - start_image

        else:
            file_list = self._parse_file_path(image_path, image_type)

            # Set up the basic parameters associated with this set of images
            (usize, vsize), data_type, image_parms = self._getimagesize(os.path.join(image_path, file_list[0]))

            num_images = len(file_list) - start_image

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
            data_type = np.float32

        h5_main, h5_mean_spec, h5_ronch = self._setupH5(usize, vsize, np.float32, num_images, image_parms)

        self._read_data(file_list[start_image:],
                        h5_main, h5_mean_spec, h5_ronch, image_path)

        return h5_main

    def _read_data(self, image_stack, h5_main, h5_mean_spec, h5_ronch, image_path):
        """
        Iterates over the images in `file_list`, reading each image and downsampling if
        reqeusted, and writes the flattened image to file.  Also builds the Mean_Ronchigram
        and the Spectroscopic_Mean datasets at the same time.

        Parameters
        ----------
        image_stack : list of str
            List of all files in `image_path` that will be read
        h5_main : h5py.Dataset
            Dataset which will hold the Ronchigrams
        h5_mean_spec : h5py.Dataset
            Dataset which will hold the Spectroscopic Mean
        h5_ronch : h5py.Dataset
            Dataset which will hold the Mean Ronchigram
        image_path : str
            Absolute file path to the directory which hold the images

        Returns
        -------
        None
        """

        mean_ronch = np.zeros(h5_ronch.shape, dtype=np.float32)

        num_files = len(image_stack)

        if os.path.isfile(image_path):
            self.__save_dm3_frames(image_stack, h5_main, h5_mean_spec, h5_ronch, mean_ronch, num_files)
        else:
            self.__read_image_files(image_stack, h5_main, h5_mean_spec, h5_ronch, image_path, mean_ronch, num_files)

    def __save_dm3_frames(self, image_stack, h5_main, h5_mean_spec, h5_ronch, mean_ronch, num_frames):
        """

        :param image_stack:
        :param h5_main:
        :param h5_mean_spec:
        :param h5_ronch:
        :param mean_ronch:
        :param num_frames:
        :return:
        """
        for iframe, thisframe in enumerate(image_stack):
            selected = (iframe + 1) % round(num_frames / 16) == 0
            if selected:
                print('Processing file...{}% - reading: {}'.format(round(100 * iframe / num_frames), iframe))
            image = self.binning_func(thisframe, self.bin_factor, self.bin_func).flatten()
            h5_main[:, iframe] = image

            h5_mean_spec[iframe] = np.mean(image)

            mean_ronch += image

            self.h5_file.flush()

        h5_ronch[:] = mean_ronch / num_frames
        self.h5_file.flush()

    def __read_image_files(self, image_stack, h5_main, h5_mean_spec, h5_ronch, image_path, mean_ronch, num_files):
        """
        Read each image from `file_list` and save it in `h5_main`.

        Parameters
        ----------
        image_stack:
        :param h5_main:
        :param h5_mean_spec:
        :param h5_ronch:
        :param image_path:
        :param mean_ronch:
        :param num_files:
        :return:
        """
        for ifile, thisfile in enumerate(image_stack):

            selected = (ifile + 1) % round(num_files / 16) == 0
            if selected:
                print('Processing file...{}% - reading: {}'.format(round(100 * ifile / num_files), thisfile))

            image = read_image(os.path.join(image_path, thisfile), greyscale=True)
            image = self.binning_func(image, self.bin_factor, self.bin_func)
            image = image.flatten()
            h5_main[:, ifile] = image

            h5_mean_spec[ifile] = np.mean(image)

            mean_ronch += image

            self.h5_file.flush()
        h5_ronch[:] = mean_ronch / num_files
        self.h5_file.flush()

    @staticmethod
    def downSampRoncVec(ronch_vec, binning_factor):
        """
        Downsample the image by taking the mean over nearby values

        Parameters
        ----------
        ronch_vec : ndarray
            Image data
        binning_factor : int
            factor to reduce the size of the image by

        Returns
        -------
        ronc_mat3_mean : ndarray
            Flattened downsampled image
        """
        ccd_pix = int(np.sqrt(ronch_vec.size))
        ronc_mat = ronch_vec.reshape(ccd_pix, ccd_pix)
        ronc_mat2 = ronc_mat.reshape(ccd_pix, ccd_pix / binning_factor, binning_factor)
        ronc_mat2_mean = ronc_mat2.mean(2)  # take the mean along the 3rd dimension
        ronc_mat3 = ronc_mat2_mean.reshape(ccd_pix / binning_factor, binning_factor, -1)
        ronc_mat3_mean = ronc_mat3.mean(1)

        return ronc_mat3_mean.reshape(-1)

    @staticmethod
    def _parse_file_path(path, ftype='all'):
        """
        Returns a list of all files in the directory given by path
        
        Parameters
        ---------------
        path : string / unicode
            absolute path to directory containing files
        ftype : this file types to return in file_list. (optional. Default is all) 
        
        Returns
        ----------
        file_list : list of strings
            names of all files in directory located at path
        numfiles : unsigned int
            number of files in file_list
        """

        # Get all files in directory
        file_list = os.listdir(path)

        # If no file type specified, return full list
        if ftype == 'all':
            return file_list

        # Remove files of type other than the request ftype from the list
        new_file_list = []
        for this_thing in file_list:
            # Make sure it's really a file
            if not os.path.isfile(os.path.join(path, this_thing)):
                continue

            split = os.path.splitext(this_thing)
            ext = split[1]
            if ext == ftype:
                new_file_list.append(os.path.join(path, this_thing))

        return new_file_list

    @staticmethod
    def _getimagesize(image):
        """
        Returns the x and y size of the image in pixels
        
        Parameters
        ------------
        image : string / unicode
            absolute path to the image file
        
        Returns
        -----------
        (size, tmp.dtype) : Tuple 
        
        size : unsigned integer
            x and y dimenstions of image
        dtype : data type
            Datatype of the image
        """
        tmp, parms = read_image(image, get_parms=True)
        size = tmp.shape

        return size, tmp.dtype.type, parms

    def _setupH5(self, usize, vsize, data_type, num_images, main_parms):
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
        num_images : int
            Number of images in the movie
        main_parms : dict


        Returns
        -------
        h5_main : h5py.Dataset
            HDF5 Dataset that the images will be written into
        h5_mean_spec : h5py.Dataset
            HDF5 Dataset that the mean over all positions will be written
            into
        h5_ronch : h5py.Dataset
            HDF5 Dateset that the mean over all Spectroscopic steps will be
            written into
        """
        num_pixels = usize * vsize

        root_parms = generate_dummy_main_parms()
        root_parms['data_type'] = 'PtychographyData'

        main_parms['num_images'] = num_images
        main_parms['image_size_u'] = usize
        main_parms['image_size_v'] = vsize
        main_parms['num_pixels'] = num_pixels
        main_parms['translator'] = 'Movie'

        # Create the hdf5 data Group
        write_simple_attrs(self.h5_file, root_parms)
        meas_grp = create_indexed_group(self.h5_file, 'Measurement')
        write_simple_attrs(meas_grp, main_parms)
        chan_grp = create_indexed_group(meas_grp, 'Channel')

        # Build the Position and Spectroscopic Datasets
        spec_dim = Dimension('Time', 's', np.arange(num_images))
        pos_dims = [Dimension('X', 'a.u.', np.arange(usize)), Dimension('Y', 'a.u.', np.arange(vsize))]

        ds_chunking = calc_chunks([num_pixels, num_images],
                                  data_type(0).itemsize,
                                  unit_chunks=(num_pixels, 1))

        # Allocate space for Main_Data and Pixel averaged Data
        h5_main = write_main_dataset(chan_grp, (num_pixels, num_images), 'Raw_Data',
                                     'Intensity', 'a.u.',
                                     pos_dims, spec_dim,
                                     chunks=ds_chunking, dtype=data_type)
        h5_ronch = meas_grp.create_dataset('Mean_Ronchigram',
                                           data=np.zeros(num_pixels, dtype=np.float32),
                                           dtype=np.float32)
        h5_mean_spec = meas_grp.create_dataset('Spectroscopic_Mean',
                                               data=np.zeros(num_images, dtype=np.float32),
                                               dtype=np.float32)

        self.h5_file.flush()

        return h5_main, h5_mean_spec, h5_ronch

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
