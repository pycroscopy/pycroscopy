"""
Created on Feb 9, 2016

@author: Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os

import numpy as np
from skimage.data import imread
from skimage.measure import block_reduce
import h5py

from .df_utils.image_utils import no_bin
from .df_utils.dm_utils import read_dm3
from pyUSID.io.image import read_image
from pyUSID.io.translator import Translator, generate_dummy_main_parms
from pyUSID.io.write_utils import Dimension, calc_chunks
from pyUSID.io.hdf_utils import get_h5_obj_refs, link_as_main, write_main_dataset, \
    write_simple_attrs, create_indexed_group


class PtychographyTranslator(Translator):
    """
    Translate Pytchography data from a set of images to an HDF5 file
    """
    def __init__(self, *args, **kwargs):
        super(PtychographyTranslator, self).__init__(*args, **kwargs)

        self.rebin = False
        self.bin_factor = 1
        self.h5_file = None
        self.binning_func = no_bin
        self.bin_func = None
        self.image_ext = None

    def translate(self, h5_path, image_path, bin_factor=None, bin_func=np.mean, start_image=0, scan_size_x=None,
                  scan_size_y=None, image_type='.tif'):
        """
        Basic method that adds Ptychography data to existing hdf5 thisfile
        You must have already done the basic translation with BEodfTranslator
        
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
        scan_size_x : int, optional
            Number of Ronchigrams in the x direction.  Default is None, value will be determined
            from the number of images and `scan_size_y` if it is given.
        scan_size_y : int, optional
            Number of Ronchigrams in the y direction.  Default is None, value will be determined
            from the number of images and `scan_size_x` if it is given.
        image_type : str
            File extension of images to be read.  Default '.tif'

        Returns
        ----------
        h5_main : h5py.Dataset
            HDF5 Dataset object that contains the flattened images

        """
        self.image_ext = image_type
        
        image_path = os.path.abspath(image_path)
        h5_path = os.path.abspath(h5_path)

        # Open the hdf5 file and delete any contents
        if os.path.exists(h5_path):
            os.remove(h5_path)
        h5_file = h5py.File(h5_path, 'w')

        self.h5_file = h5_file

        # Get the list of all files with the .tif extension and the number of files in the list
        if image_type == '.dm3':
            file_list = [image_path]
            # image_path, _ = os.path.split(image_path)
            images, image_parms = read_dm3(image_path)
            usize = image_parms['SuperScan_Height']
            vsize = image_parms['SuperScan_Width']
            data_type = images.dtype
        else:
            file_list = self._parse_file_path(image_path, image_type)

            # Set up the basic parameters associated with this set of images
            (usize, vsize), data_type = self._getimagesize(os.path.join(image_path, file_list[0]))

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

        if scan_size_x is None:
            scan_size_x = int(np.floor(np.sqrt(len(file_list))))
            scan_size_y = scan_size_x
        if scan_size_y is None:
            scan_size_y = int(np.floor(len(file_list) / scan_size_x))

        num_files = scan_size_x*scan_size_y

        h5_main, h5_mean_spec, h5_ronch = self._setupH5(usize, vsize, np.float32, scan_size_x, scan_size_y)

        self._read_data(file_list[start_image:start_image+num_files],
                        h5_main, h5_mean_spec, h5_ronch, image_path)

        return h5_main

    def _read_data(self, file_list, h5_main, h5_mean_spec, h5_ronch, image_path):
        """
        Iterates over the images in `file_list`, reading each image and downsampling if
        reqeusted, and writes the flattened image to file.  Also builds the Mean_Ronchigram
        and the Spectroscopic_Mean datasets at the same time.

        Parameters
        ----------
        file_list : list of str
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

        num_files = len(file_list)

        for ifile, thisfile in enumerate(file_list):

            selected = (ifile + 1) % round(num_files / 16) == 0
            if selected:
                print('Processing file...{}% - reading: {}'.format(round(100 * ifile / num_files), thisfile))

            image, _ = read_image(os.path.join(image_path, thisfile), as_grey=True)
            image = self.binning_func(image, self.bin_factor, self.bin_func)
            image = image.flatten()
            h5_main[ifile, :] = image

            h5_mean_spec[ifile] = np.mean(image)

            mean_ronch += image

            self.h5_file.flush()

        h5_ronch[:] = mean_ronch / num_files
        self.h5_file.flush()

    # def downSampRoncVec(self, ronch_vec, binning_factor):
    #     """
    #     Downsample the image by taking the mean over nearby values
    #
    #     Parameters
    #     ----------
    #     ronch_vec : ndarray
    #         Image data
    #     binning_factor : int
    #         factor to reduce the size of the image by
    #
    #     Returns
    #     -------
    #     ronc_mat3_mean : ndarray
    #         Flattened downsampled image
    #     """
    #     ccd_pix = int(np.sqrt(ronch_vec.size))
    #     ronc_mat = ronch_vec.reshape(ccd_pix, ccd_pix)
    #     ronc_mat2 = ronc_mat.reshape(ccd_pix, ccd_pix / binning_factor, binning_factor)
    #     ronc_mat2_mean = ronc_mat2.mean(2)  # take the mean along the 3rd dimension
    #     ronc_mat3 = ronc_mat2_mean.reshape(ccd_pix / binning_factor, binning_factor, -1)
    #     ronc_mat3_mean = ronc_mat3.mean(1)
    #
    #     return ronc_mat3_mean.reshape(-1)

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
        tmp = imread(image)
        size = tmp.shape
        
        return size, tmp.dtype

    def _setupH5(self, usize, vsize, data_type, scan_size_x, scan_size_y):
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
        scan_size_x : int
            Number of images in the x dimension
        scan_size_y : int
            Number of images in the y dimension

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
        num_pixels = usize*vsize
        num_files = scan_size_x*scan_size_y

        root_parms = generate_dummy_main_parms()
        root_parms['data_type'] = 'PtychographyData'

        main_parms = {'num_images': num_files,
                      'image_size_u': usize,
                      'image_size_v': vsize,
                      'num_pixels': num_pixels,
                      'translator': 'Ptychography',
                      'scan_size_x': scan_size_x,
                      'scan_size_y': scan_size_y}

        # Create the hdf5 data Group
        write_simple_attrs(self.h5_file, root_parms)
        meas_grp = create_indexed_group(self.h5_file, 'Measurement')
        write_simple_attrs(meas_grp, main_parms)
        chan_grp = create_indexed_group(meas_grp, 'Channel')

        # Build the Position and Spectroscopic Datasets
        spec_desc = [Dimension('U', 'pixel', np.arange(usize)), Dimension('V', 'pixel', np.arange(vsize))]
        pos_desc = [Dimension('X', 'pixel', np.arange(scan_size_x)),
                    Dimension('Y', 'pixel', np.arange(scan_size_y))]

        ds_chunking = calc_chunks([num_files, num_pixels],
                                  data_type(0).itemsize,
                                  unit_chunks=(1, num_pixels))

    # Allocate space for Main_Data and Pixel averaged Data
        h5_main = write_main_dataset(chan_grp, (num_files, num_pixels), 'Raw_Data',
                                     'Intensity', 'a.u.',
                                     pos_desc, spec_desc,
                                     chunks=ds_chunking, dtype=data_type)
        h5_ronch = meas_grp.create_dataset('Mean_Ronchigram',
                                           data=np.zeros(num_pixels, dtype=np.float32),
                                           dtype=np.float32)
        h5_mean_spec = meas_grp.create_dataset('Spectroscopic_Mean',
                                               data=np.zeros(num_files, dtype=np.float32),
                                               dtype=np.float32)

        self.h5_file.flush()
        
        return h5_main, h5_mean_spec, h5_ronch

