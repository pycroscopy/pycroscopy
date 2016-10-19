"""
Created on Feb 9, 2016

@author: Chris Smith
"""

import os

import numpy as np
# from scipy.misc import imread

from skimage.data import imread
from skimage.measure import block_reduce

from .translator import Translator
from .utils import generateDummyMainParms
from ..hdf_utils import getH5DsetRefs, calc_chunks
from ..io_hdf5 import ioHDF5
from ..microdata import MicroDataGroup, MicroDataset


class PtychographyTranslator(Translator):
    """
    Translate Pytchography data from a set of images to an HDF5 file
    """
    def __init__(self, *args, **kwargs):
        super(PtychographyTranslator, self).__init__(*args, **kwargs)

        self.rebin = False
        self.bin_factor = 1
        self.hdf = None
        self.binning_func = self.__no_bin
        self.bin_func = None

    def translate(self, h5_path, image_path, bin_factor=None, bin_func=np.mean):
        """
        Basic method that adds Ptychography data to existing hdf5 thisfile
        You must have already done the basic translation with BEodfTranslator
        
        Parameters
        ----------------
        h5_path : str
            Absolute path to where the HDF5 file should be located
        image_path : str
            Absolute path to folder holding the image files
        bin_factor : array_like of uint
            Downsampling factor for each dimension
        bin_func : callable
            Function which will be called to calculate the return value
            of each block.  Function must implement an axis parameter,
            i.e. numpy.mean

        Returns
        ----------
        h5_main : h5py.Dataset
            HDF5 Dataset object that contains the flattened images
        """
                
        # Get the list of all files with the .tif extension and the number of files in the list

        file_list = self._parsefilepath(image_path, '.tif')

        num_files = len(file_list)

        # Open the hdf5 thisfile and delete any contents
        try:
            hdf = ioHDF5(h5_path)
            hdf.clear()
        except:
            raise

        self.hdf = hdf

        # Set up the basic parameters associated with this set of images
        
        (usize, vsize), data_type = self._getimagesize(os.path.join(image_path, file_list[0]))

        if bin_factor is not None:
            self.rebin = True
            if isinstance(bin_factor, int):
                self.bin_factor = (bin_factor, bin_factor)
            elif len(bin_factor) == 2:
                self.bin_factor = tuple(bin_factor)
            else:
                raise ValueError('Input parameter `bin_factor` must be a length 2 array_like or an integer.' +
                                 '{} was given.'.format(bin_factor))
            usize = int(usize / self.bin_factor[0])
            vsize = int(vsize / self.bin_factor[1])
            self.binning_func = block_reduce
            self.bin_func = bin_func

        num_pixels = usize*vsize
        
        scan_size = np.int(np.sqrt(num_files))
        num_files = scan_size**2
        
        h5_main, h5_mean_spec, h5_ronch = self._setupH5(num_files, usize,
                                                        vsize, np.float32,
                                                        scan_size)

        self._read_data(file_list[:num_files], h5_main, h5_mean_spec, h5_ronch, image_path, num_pixels)

        return h5_main

    def _read_data(self, file_list, h5_main, h5_mean_spec, h5_ronch, image_path, num_pixels):

        mean_ronch = np.zeros(num_pixels, dtype=np.float32)

        num_files = len(file_list)

        for ifile, thisfile in enumerate(file_list):

            selected = (ifile + 1) % round(num_files / 16) == 0
            if selected:
                print('Processing file...{}% - reading: {}'.format(round(100 * ifile / num_files), thisfile))

            image = imread(os.path.join(image_path, thisfile), as_grey=True)
            image = self.binning_func(image, self.bin_factor, self.bin_func)
            image = image.flatten()
            h5_main[ifile, :] = image

            h5_mean_spec[ifile] = np.mean(image)

            mean_ronch += image

            self.hdf.flush()

        h5_ronch[:] = mean_ronch / num_files
        self.hdf.flush()

    def downSampRoncVec(self, ronch_vec, binning_factor):
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
    def _parsefilepath(path, ftype='all'):
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
            numfiles = len(file_list)
            return file_list

        # Remove files of type other than the request ftype from the list
        new_file_list = []
        for this_thing in file_list:
            if not os.path.isfile(os.path.join(path, this_thing)):
                continue

            split = os.path.splitext(this_thing)
            ext = split[1]
            if ext == ftype:
                new_file_list.append(os.path.join(path,this_thing))

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

    
    def _setupH5(self, num_files, usize, vsize, data_type, scan_size):
        """
        Setup the HDF5 file in which to store the data including creating
        the Position and Spectroscopic datasets

        Parameters
        ----------
        num_files : int
            Number of images
        usize : int
            Number of pixel columns in the images
        vsize : int
            Number of pixel rows in the images
        data_type : type
            Data type to save image as
        scan_size : int
            Number of images in each dimension

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

        main_parms = generateDummyMainParms()
        main_parms['num_images'] = num_files
        main_parms['datatype'] = 'ptychography'
        main_parms['image_size_u'] = usize
        main_parms['image_size_v'] = vsize
        main_parms['num_pixels'] = num_pixels
        main_parms['translator'] = 'Ptychography'
        main_parms['scan_size_x'] = scan_size
        main_parms['scan_size_y'] = scan_size
    # Create the hdf5 data Group
        meas_grp = MicroDataGroup('Measurement_000', parent='/')
        meas_grp.attrs = main_parms
        chan_grp = MicroDataGroup('Channel_000')
    # Get the Position and Spectroscopic Datasets
    #     ds_spec_ind, ds_spec_vals = self._buildspectroscopicdatasets(usize, vsize, num_pixels)
        ds_spec_ind, ds_spec_vals = self._buildspectroscopicdatasets((usize, vsize),
                                                                     labels=['U', 'V'],
                                                                     units=['pixel', 'pixel'])
        ds_pos_ind, ds_pos_val = self._buildpositiondatasets([scan_size, scan_size], labels=['X', 'Y'], units=['pixel', 'pixel'])

        ds_chunking = calc_chunks([num_files, num_pixels], data_type(0).itemsize, unit_chunks=(1, num_pixels))

    # Allocate space for Main_Data and Pixel averaged Data
        ds_main_data = MicroDataset('Raw_Data', data=[], maxshape=(num_files, num_pixels),
                                    chunking=ds_chunking, dtype=data_type, compression='gzip')
        ds_mean_ronch_data = MicroDataset('Mean_Ronchigram',
                                          data=np.zeros(num_pixels, dtype=np.float32),
                                          dtype=np.float32)
        ds_mean_spec_data = MicroDataset('Spectroscopic_Mean',
                                         data=np.zeros(num_files, dtype=np.float32),
                                         dtype=np.float32)
    # Add datasets as children of Measurement_000 data group
        chan_grp.addChildren([ds_main_data, ds_spec_ind, ds_spec_vals, ds_pos_ind,
                              ds_pos_val, ds_mean_ronch_data, ds_mean_spec_data])
        meas_grp.addChildren([chan_grp])

        # print('Writing following tree to this file:')
        # meas_grp.showTree()

        h5_refs = self.hdf.writeData(meas_grp)
        h5_main = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
        h5_ronch = getH5DsetRefs(['Mean_Ronchigram'], h5_refs)[0]
        h5_mean_spec = getH5DsetRefs(['Spectroscopic_Mean'], h5_refs)[0]
        aux_ds_names = ['Position_Indices',
                        'Position_Values',
                        'Spectroscopic_Indices',
                        'Spectroscopic_Values']

        self._linkformain(h5_main, *getH5DsetRefs(aux_ds_names, h5_refs))

        self.hdf.flush()
        
        return h5_main, h5_mean_spec, h5_ronch

    def __no_bin(self, image, *args, **kwargs):
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
