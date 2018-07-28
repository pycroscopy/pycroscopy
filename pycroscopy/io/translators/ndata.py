"""
Created on Feb 9, 2016

@author: Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import json
import os
import zipfile
from warnings import warn

import h5py
import numpy as np
from skimage.measure import block_reduce
from skimage.util import crop

from .df_utils.image_utils import unnest_parm_dicts
from .df_utils.dm_utils import read_dm3
from pyUSID.io.translator import Translator, generate_dummy_main_parms
from pyUSID.io.write_utils import Dimension, calc_chunks
from pyUSID.io.hdf_utils import write_main_dataset, create_indexed_group, write_simple_attrs


class NDataTranslator(Translator):
    """
    Translate Pytchography data from a set of images to an HDF5 file
    """

    def __init__(self, *args, **kwargs):
        super(NDataTranslator, self).__init__(*args, **kwargs)

        self.rebin = False
        self.bin_factor = (1, 1, 1, 1)
        self.h5_f = None
        self.binning_func = self.__no_bin
        self.bin_func = None
        self.h5_main = None
        self.root_image_list = list()
        self.crop_method = 'percent'
        self.crop_ammount = None
        self.image_list_tag = None

    def translate(self, h5_path, image_path, bin_factor=None, bin_func=np.mean, start_image=0, scan_size_x=None,
                  scan_size_y=None, crop_ammount=None, crop_method='percent'):
        """
        Basic method that adds Ptychography data to existing hdf5 thisfile
        You must have already done the basic translation with BEodfTranslator

        Parameters
        ----------------
        h5_path : str
            Absolute path to where the HDF5 file should be located
        image_path : str
            Absolute path to folder holding the image files or the path to a specific file
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
        crop_ammount : uint or list of uints, optional
            How much should be cropped from the original image.  Can be a single unsigned
            integer or a list of unsigned integers.  A single integer will crop the same
            ammount from all edges.  A list of two integers will crop the x-dimension by
            the first integer and the y-dimension by the second integer.  A list of 4
            integers will crop the image as [left, right, top, bottom].
        crop_method : str, optional
            Which cropping method should be used.  How much of the image is removed is
            determined by the value of `crop_ammount`.
            'percent' - A percentage of the image is removed.
            'absolute' - The specific number of pixel is removed.
        Returns
        ----------
        h5_main : h5py.Dataset
            HDF5 Dataset object that contains the flattened images

        """
        # Open the hdf5 file and delete any contents
        if os.path.exists(h5_path):
            os.remove(h5_path)

        h5_f = h5py.File(h5_path, 'w')

        self.h5_f = h5_f
        self.crop_method = crop_method
        self.crop_ammount = crop_ammount

        '''
        Get the list of all files with the .tif extension and
        the number of files in the list
        '''
        image_path = os.path.abspath(image_path)
        file_list = self._parse_file_path(image_path)

        image_parm_list = self._getimageparms(file_list)

        '''
        Check if a bin_factor is given.  Set up binning objects if it is.
        '''
        if bin_factor is not None:
            self.rebin = True
            if isinstance(bin_factor, int):
                self.bin_factor = (1, 1, bin_factor, bin_factor)
            elif len(bin_factor) == 2:
                self.bin_factor = (1, 1) + bin_factor
            else:
                raise ValueError('Input parameter `bin_factor` must be a length 2 array_like or an integer.\n' +
                                 '{} was given.'.format(bin_factor))

            self.binning_func = block_reduce
            self.bin_func = bin_func

        h5_channels = self._setupH5(image_parm_list)

        self._read_data(file_list, h5_channels)

        self.h5_f.close()

        return

    # def _create_root_image(self, image_path):
    #     """
    #     Create the Groups and Datasets for a single root image
    #
    #     Parameters
    #     ----------
    #     image_path : str
    #         Path to the image file
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     image, image_parms = read_dm3(image_path)
    #     if image.ndim == 3:
    #         image = np.sum(image, axis=0)
    #
    #     '''
    #     Create the Measurement and Channel Groups to hold the
    #     image Datasets
    #     '''
    #     root_grp = VirtualGroup('/')
    #
    #     meas_grp = VirtualGroup('Measurement_')
    #
    #     chan_grp = VirtualGroup('Channel_')
    #     root_grp.add_children([meas_grp])
    #     meas_grp.add_children([chan_grp])
    #
    #     '''
    #     Set the Measurement Group attributes
    #     '''
    #     meas_grp.attrs.update(image_parms)
    #     usize, vsize = image.shape
    #     meas_grp.attrs['image_size_u'] = usize
    #     meas_grp.attrs['image_size_v'] = vsize
    #     meas_grp.attrs['translator'] = 'OneView'
    #     meas_grp.attrs['num_pixels'] = image.size
    #
    #     ds_raw_image = VirtualDataset('Raw_Data', np.reshape(image, (-1, 1)))
    #
    #     '''
    #     Build Spectroscopic and Position datasets for the image
    #     '''
    #     spec_desc = Dimension('Intensity', 'a.u.', [1])
    #     ds_spec_inds, ds_spec_vals = build_ind_val_dsets(spec_desc, is_spectral=True)
    #
    #     pos_dims = [Dimension('X', 'a.u.', np.arange(image.shape[0])),
    #                 Dimension('Y', 'a.u.', np.arange(image.shape[1]))]
    #     ds_pos_inds, ds_pos_vals = build_ind_val_dsets(pos_dims, is_spectral=False)
    #
    #     chan_grp.add_children([ds_raw_image, ds_spec_inds, ds_spec_vals,
    #                            ds_pos_inds, ds_pos_vals])
    #
    #     '''
    #     Write the data to file and get the handle for the image dataset
    #     '''
    #     image_refs = self.h5_f.write(root_grp)
    #
    #     h5_image = get_h5_obj_refs(['Raw_Data'], image_refs)[0]
    #
    #     '''
    #     Link references to raw
    #     '''
    #     aux_ds_names = ['Position_Indices', 'Position_Values', 'Spectroscopic_Indices', 'Spectroscopic_Values']
    #     link_as_main(h5_image, *get_h5_obj_refs(aux_ds_names, image_refs))
    #
    #     self.root_image_list.append(h5_image)

    def _read_data(self, file_list, h5_channels):
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
        h5_main_list = list()
        '''
        For each file, we must read the data then create the neccessary datasets, add them to the channel, and
        write it all to file
        '''

        '''
        Get zipfile handles for all the ndata1 files that were found in the image_path
        '''

        for ifile, (this_file, this_channel) in enumerate(zip(file_list, h5_channels)):
            _, ext = os.path.splitext(this_file)
            if ext in ['.ndata1', '.ndata']:
                '''
                Extract the data file from the zip archive and read it into an array
                '''
                this_zip = zipfile.ZipFile(this_file, 'r')
                tmp_path = this_zip.extract('data.npy')
                this_data = np.load(tmp_path)
                os.remove(tmp_path)
            elif ext == '.npy':
                # Read data directly from npy file
                this_data = np.load(this_file)

            '''
            Find the shape of the data, then calculate the final dimensions based on the crop and
            downsampling parameters
            '''
            while this_data.ndim < 4:
                this_data = np.expand_dims(this_data, 0)

            this_data = self.crop_ronc(this_data)
            scan_size_x, scan_size_y, usize, vsize = this_data.shape

            usize = int(round(1.0 * usize / self.bin_factor[-2]))
            vsize = int(round(1.0 * vsize / self.bin_factor[-1]))

            num_images = scan_size_x * scan_size_y
            num_pixels = usize * vsize

            '''
            Write these attributes to the Measurement group
            '''
            new_attrs = {'image_size_u': usize,
                         'image_size_v': vsize,
                         'scan_size_x': scan_size_x,
                         'scan_size_y': scan_size_y}

            write_simple_attrs(this_channel.parent, new_attrs)


            # Get the Position and Spectroscopic Datasets
            spec_desc = [Dimension('U', 'pixel', np.arange(usize)), Dimension('V', 'pixel', np.arange(vsize))]
            pos_desc = [Dimension('X', 'pixel', np.arange(scan_size_x)),
                        Dimension('Y', 'pixel', np.arange(scan_size_y))]

            ds_chunking = calc_chunks([num_images, num_pixels],
                                      np.float32(0).itemsize,
                                      unit_chunks=(1, num_pixels))

            # Allocate space for Main_Data and Pixel averaged DataX
            h5_main = write_main_dataset(this_channel, (num_images, num_pixels), 'Raw_Data',
                                         'Intensity', 'a.u.',
                                         pos_desc, spec_desc,
                                         chunks=ds_chunking, dtype=np.float32)

            h5_ronch = this_channel.create_dataset('Mean_Ronchigram',
                                                   data=np.zeros(num_pixels, dtype=np.float32))

            h5_mean_spec = this_channel.create_dataset('Mean_Spectrogram',
                                                       data=np.zeros(num_images, dtype=np.float32))

            this_data = self.binning_func(this_data, self.bin_factor, self.bin_func).reshape(h5_main.shape)

            h5_main[:, :] = this_data

            h5_mean_spec[:] = np.mean(this_data, axis=1)

            h5_ronch[:] = np.mean(this_data, axis=0)

            self.h5_f.flush()

            h5_main_list.append(h5_main)

        self.h5_f.flush()

    def crop_ronc(self, ronc):
        """
        Crop the input Ronchigram by the specified ammount using the specified method.

        Parameters
        ----------
        ronc : numpy.array
            Input image to be cropped.

        Returns
        -------
        cropped_ronc : numpy.array
            Cropped image
        """

        if self.crop_ammount is None:
            return ronc

        crop_ammount = self.crop_ammount
        crop_method = self.crop_method

        if crop_method == 'percent':
            crop_ammount = np.round(np.atleast_2d(crop_ammount) / 100.0 * ronc.shape)
            crop_ammount = tuple([tuple(row) for row in crop_ammount.astype(np.uint32)])
        elif crop_method == 'absolute':
            if isinstance(crop_ammount, int):
                crop_ammount = ((crop_ammount,), (crop_ammount,))
            elif len(crop_ammount) == 2:
                crop_ammount = ((crop_ammount[0],), (crop_ammount[1],))
            elif len(crop_ammount) == 4:
                crop_ammount = ((crop_ammount[0], crop_ammount[1]), (crop_ammount[2], crop_ammount[3]))
            else:
                raise ValueError('The crop_ammount should be an integer or list of 2 or 4 integers.')
        else:
            raise ValueError('Allowed values of crop_method are percent and absolute.')

        crop_ammount = ((0,), (0,)) + crop_ammount

        cropped_ronc = crop(ronc, crop_ammount)

        if any([dim == 0 for dim in cropped_ronc.shape]):
            warn("Requested crop ammount is greater than the image size.  No cropping will be done.")
            return ronc
        return cropped_ronc

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
    def _parse_file_path(image_path):
        """
        Returns a list of all files in the directory given by path

        Parameters
        ---------------
        path : string / unicode
            absolute path to directory containing files

        Returns
        ----------
        file_list : list of strings
            names of all files in directory located at path
        """
        allowed_image_types = ['.ndata1', '.npy', '.ndata']
        if os.path.isdir(image_path):
            # Image path is to a directory
            file_list = list()
            for root, dirs, files in os.walk(image_path):
                for thisfile in files:
                    _, ext = os.path.splitext(thisfile)
                    if ext not in allowed_image_types:
                        continue
                    else:
                        file_list.append(os.path.join(root, thisfile))
        else:
            # Image path is a file
            _, ext = os.path.splitext(image_path)
            if ext in allowed_image_types:
                file_list = [image_path]

        return file_list

    @staticmethod
    def _getimageparms(file_list):
        """
        Returns the image parameters for each file in the `file_list`

        Parameters
        ------------
        file_list : list of zipfile.ZipFile
            List of zipfile objects

        Returns
        -----------
        parm_list : list of dict
            List of image parameters from the files in `file_list`
        """
        parm_list = list()

        for fpath in file_list:
            base, ext = os.path.splitext(fpath)
            if ext in ['.ndata1', '.ndata']:
                zfile = zipfile.ZipFile(fpath, 'r')
                tmp_path = zfile.extract('metadata.json')
            elif ext == '.npy':
                folder, basename = os.path.split(base)
                same_name_path = base+'.json'
                metapath = os.path.join(folder, 'metadata.json')
                if os.path.exists(same_name_path):
                    tmp_path = same_name_path
                elif os.path.exists(metapath):
                    tmp_path = metapath
            metafile = open(tmp_path, 'r')
            metastring = metafile.read()
            parm_list.append(unnest_parm_dicts(json.loads(metastring)))
            metafile.close()

            if ext == '.ndata1':
                os.remove(tmp_path)

        return parm_list

    def _setupH5(self, image_parms):
        """
        Setup the HDF5 file in which to store the data
        Due to the structure of the ndata format, we can only create the Measurement and Channel groups here

        Parameters
        ----------
        image_parms : dict
            Dictionary of parameters

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
        root_parms = generate_dummy_main_parms()
        root_parms['data_type'] = 'PtychographyData'

        # Create the hdf5 data Group
        write_simple_attrs(self.h5_f, root_parms)

        h5_channels = list()
        for meas_parms in image_parms:
            # Create new measurement group for each set of parameters
            meas_grp = create_indexed_group(self.h5_f, 'Measurement')
            # Write the parameters as attributes of the group
            write_simple_attrs(meas_grp, meas_parms)

            chan_grp = create_indexed_group(meas_grp, 'Channel')

            h5_channels.append(chan_grp)

        self.h5_f.flush()

        return h5_channels

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
