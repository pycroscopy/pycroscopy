"""
Created on Jun 16, 2016

@author: Chris Smith -- csmith55@utk.edu
"""

from __future__ import division, print_function, absolute_import
import os
from multiprocessing import cpu_count
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.signal import blackman
from sklearn.utils import gen_batches

from pyUSID import USIDataset
from pyUSID.io.hdf_utils import get_h5_obj_refs, copy_attributes, link_h5_objects_as_attrs, find_results_groups, \
    link_as_main, check_for_old
from pyUSID.io.io_utils import get_available_memory
from pyUSID.io.write_utils import make_indices_matrix, get_aux_dset_slicing, INDICES_DTYPE, VALUES_DTYPE, calc_chunks
from ..io.hdf_writer import HDFwriter
from ..io.virtual_data import VirtualGroup, VirtualDataset
from .svd_utils import get_component_slice

windata32 = np.dtype({'names': ['Image Data'],
                      'formats': [np.float32]})
absfft32 = np.dtype({'names': ['FFT Magnitude'],
                     'formats': [np.float32]})
winabsfft32 = np.dtype({'names': ['Image Data', 'FFT Magnitude'],
                        'formats': [np.float32, np.float32]})
wincompfft32 = np.dtype({'names': ['Image Data', 'FFT Real', 'FFT Imag'],
                         'formats': [np.float32, np.float32, np.float32]})


class ImageWindow(object):
    """
    This class will handle the reading of a raw image file, creating windows from it, and writing those
    windows to an HDF5 file.
    """

    def __init__(self, h5_main, max_RAM_mb=1024, cores=None, reset=True, **image_args):
        """
        Setup the image windowing

        Parameters
        ----------
        h5_main : h5py.Dataset
            HDF5 Dataset containing the data to be windowed
        max_RAM_mb : int, optional
            integer maximum amount of ram, in Mb, to use in windowing
            Default 1024
        cores : int, optional
            integer number of [logical] CPU cores to use in windowing
            Defualt None, use number of available cores minus 2
        reset : Boolean, optional
            should all data in the hdf5 file be deleted

        """
        self.h5_file = h5_main.file
        self.hdf = HDFwriter(h5_main.file)

        # Ensuring that at least one core is available for use / 2 cores are available for other use
        max_cores = max(1, cpu_count() - 2)
        #         print 'max_cores',max_cores
        if cores is not None:
            cores = min(round(abs(cores)), max_cores)
        else:
            cores = max_cores
        self.cores = int(cores)

        self.max_memory = min(max_RAM_mb * 1024 ** 2, 0.75 * get_available_memory())
        if self.cores != 1:
            self.max_memory = int(self.max_memory / 2)

        '''
        Initialize class variables to None
        '''
        self.h5_raw = h5_main
        self.h5_norm = None
        self.h5_wins = None
        self.h5_clean = None
        self.h5_noise = None
        self.h5_fft_clean = None
        self.h5_fft_noise = None

    def do_windowing(self, win_x=None, win_y=None, win_step_x=1, win_step_y=1,
                     win_fft=None, *args, **kwargs):
        """
        Extract the windows from the normalized image and write them to the file

        Parameters
        ----------
        win_x : int, optional
            size of the window, in pixels, in the horizontal direction
            Default None, a guess will be made based on the FFT of the image
        win_y : int, optional
            size of the window, in pixels, in the vertical direction
            Default None, a guess will be made based on the FFT of the image
        win_step_x : int, optional
            step size, in pixels, to take between windows in the horizontal direction
            Default 1
        win_step_y : int, optional
            step size, in pixels, to take between windows in the vertical direction
            Default 1
        win_fft : str, optional
            What kind of fft should be stored with the windows.  Options are
            None - Only the window
            'abs' - Only the magnitude of the fft
            'data+abs' - The window and magnitude of the fft
            'data+complex' - The window and the complex fft
            Default None

        Returns
        -------
        h5_wins : HDF5 Dataset
            Dataset containing the flattened windows

        """
        h5_main = self.h5_raw

        if win_fft == 'data' or win_fft is None:
            win_fft = 'data'
            win_type = windata32
            win_func = self.win_data_func
        elif win_fft == 'abs':
            win_type = absfft32
            win_func = self.abs_fft_func
        elif win_fft == 'data+abs':
            win_type = winabsfft32
            win_func = self.win_abs_fft_func
        elif win_fft == 'data+complex':
            win_type = wincompfft32
            win_func = self.win_comp_fft_func
        else:
            warn('Invalid FFT option supplied.  Windowing will default to data only.')
            win_fft = 'data'
            win_type = windata32
            win_func = self.win_data_func

        '''
        If a window size has not been specified, obtain a guess value from 
        window_size_extract
        '''
        win_test, psf_width = self.window_size_extract(*args, **kwargs)
        if win_x is None:
            win_x = win_test
        if win_y is None:
            win_y = win_test

        image, h5_wins, win_pos_mat, have_old = self._setup_window_h5(h5_main, psf_width, win_fft, win_step_x,
                                                                      win_step_y, win_type, win_x, win_y)

        if have_old:
            self.h5_wins = h5_wins
            return h5_wins

        n_wins = win_pos_mat.shape[0]
        win_pix = win_x * win_y

        '''
        Create slice object from the positions
        '''
        win_slices = [[slice(x, x + win_x), slice(y, y + win_y)] for x, y in win_pos_mat]

        '''
        Calculate the size of a given batch that will fit in the available memory
        '''
        mem_per_win = win_x * win_y * h5_wins.dtype.itemsize
        if self.cores is None:
            free_mem = self.max_memory - image.size * image.itemsize
        else:
            free_mem = self.max_memory * 2 - image.size * image.itemsize
        batch_size = int(free_mem / mem_per_win)
        batch_slices = gen_batches(n_wins, batch_size)

        for ibatch, batch in enumerate(batch_slices):
            batch_wins = np.zeros([batch.stop - batch.start, win_pix], dtype=win_type)
            '''
            Read each slice and write it to the dataset
            '''
            for islice, this_slice in enumerate(win_slices[batch]):
                iwin = ibatch * batch_size + islice
                selected = iwin % np.rint(n_wins / 10) == 0

                if selected:
                    per_done = np.rint(100 * iwin / n_wins)
                    print('Windowing Image...{}% --pixels {}-{}, step # {}'.format(per_done,
                                                                                   (this_slice[0].start,
                                                                                    this_slice[1].start),
                                                                                   (this_slice[0].stop,
                                                                                    this_slice[1].stop),
                                                                                   islice))

                batch_wins[islice] = win_func(image[this_slice]).flatten()

            h5_wins[batch] = batch_wins
            self.hdf.flush()

        self.h5_wins = h5_wins

        return USIDataset(h5_wins)

    @staticmethod
    def _check_win_parameters(h5_main, win_step_x, win_step_y, win_x, win_y):
        """
        Check the window parameters
        
        Parameters
        ----------
        h5_main : h5py.Dataset
            Dataset containing the Raw Image
        win_step_x : uint
            Step size in the x-direction between windows.
        win_step_y : uint
            Step size in the y-direction between windows.
        win_x : uint
            Size of the window in the x-direction.
        win_y : uint
            Size of the window in the y-direction.

        Returns
        -------
        image : numpy.ndarray
            Array containing the original image reshaped by position.
        win_step_x : uint
            Step size in the x-direction between windows.
        win_step_y : uint
            Step size in the y-direction between windows.
        win_x : uint
            Size of the window in the x-direction.
        win_y : uint
            Size of the window in the y-direction.
        
        """
        '''
        Get the position indices of h5_main and reshape the flattened image back
        '''
        try:
            h5_pos = h5_main.file[h5_main.attrs['Position_Indices']][()]
            x_pix = len(np.unique(h5_pos[:, 0]))
            y_pix = len(np.unique(h5_pos[:, 1]))

        except KeyError:
            '''
            Position Indices dataset does not exist
            Assume square image
            '''
            x_pix = np.int(np.sqrt(h5_main.size))
            y_pix = x_pix

        except:
            raise
        image = h5_main[()].reshape(x_pix, y_pix)
        '''
            Step size must be less than 1/4th the image size
            '''
        win_step_x = min(x_pix / 4, win_step_x)
        win_step_y = min(y_pix / 4, win_step_y)
        '''
            Prevent windows from being less that twice the step size and more than half the image size
            '''
        win_x = max(2 * win_step_x, min(x_pix, win_x))
        win_y = max(2 * win_step_y, min(y_pix, win_y))
        print('Optimal window size determined to be {wx}x{wy} pixels.'.format(wx=win_x, wy=win_y))

        return image, win_step_x, win_step_y, win_x, win_y

    def _setup_window_h5(self, h5_main, psf_width, win_fft, win_step_x, win_step_y, win_type, win_x,
                         win_y):
        """
        Setup the hdf5 group for the windows
        
        Parameters
        ----------
        h5_main : h5py.Dataset
            Dataset containing the Raw Image
        psf_width : uint
            psf_width???  Someone who knows what this is should fill it in 
        win_fft : 
        win_step_x : uint
            Step size in the x-direction between windows.
        win_step_y : uint
            Step size in the y-direction between windows.
        win_type
        win_x : uint
            Size of the window in the x-direction.
        win_y : uint
            Size of the window in the y-direction.

        Returns
        -------
        

        """

        image, win_step_x, win_step_y, win_x, win_y = self._check_win_parameters(h5_main,
                                                                                 win_step_x, win_step_y,
                                                                                 win_x, win_y)

        '''
        Build the Spectroscopic and Position Datasets 
        '''
        ds_pix_inds, ds_pix_vals, ds_pos_inds, ds_pos_vals, win_pos_mat = self._get_window_pos_spec(
            image, win_step_x, win_step_y, win_x, win_y)
        im_x, im_y = image.shape
        n_wins = win_pos_mat.shape[0]
        win_pix = win_x * win_y

        '''
        Calculate the chunk size
        '''
        win_chunks = calc_chunks([n_wins, win_pix], win_type.itemsize, unit_chunks=[1, win_pix])

        parent = h5_main.parent

        check_parameters = {'fft_mode': win_fft,
                            'psf_width': psf_width,
                            'win_x': win_x,
                            'win_y': win_y,
                            'win_step_x': win_step_x,
                            'win_step_y': win_step_y,
                            'image_x': im_x,
                            'image_y': im_y}
        basename = h5_main.name.split('/')[-1]

        old_group = check_for_old(h5_main, '-Windowing', check_parameters)

        if len(old_group) > 0:
            old_group = old_group[-1]
            old = True
            h5_wins = old_group['Image_Windows']

        else:
            old = False
            '''
            Create the Windows Dataset and Datagroup
            '''
            ds_windows = VirtualDataset('Image_Windows',
                                        data=None,
                                        maxshape=[n_wins, win_pix],
                                        dtype=win_type,
                                        chunking=win_chunks,
                                        compression='gzip',
                                        attrs={'quantity': 'Intensity',
                                               'units': 'a.u.'})

            ds_group = VirtualGroup(basename + '-Windowing_', parent.name[1:])
            ds_group.add_children([ds_windows, ds_pos_inds, ds_pix_inds,
                                   ds_pos_vals, ds_pix_vals])
            ds_group.attrs['win_x'] = win_x
            ds_group.attrs['win_y'] = win_y
            ds_group.attrs['win_step_x'] = win_step_x
            ds_group.attrs['win_step_y'] = win_step_y
            ds_group.attrs['image_x'] = im_x
            ds_group.attrs['image_y'] = im_y
            ds_group.attrs['psf_width'] = psf_width
            ds_group.attrs['fft_mode'] = win_fft
            image_refs = self.hdf.write(ds_group)

            '''
            Get the hdf5 objects for the windows and ancillary datasets
            '''
            h5_wins = get_h5_obj_refs(['Image_Windows'], image_refs)[0]

            '''
            Link references to windowed dataset
            '''
            aux_ds_names = ['Position_Indices', 'Position_Values', 'Spectroscopic_Indices', 'Spectroscopic_Values']
            link_h5_objects_as_attrs(h5_wins, get_h5_obj_refs(aux_ds_names, image_refs))

            self.hdf.flush()

        return image, h5_wins, win_pos_mat, old

    @staticmethod
    def _get_window_pos_spec(image, win_step_x, win_step_y, win_x, win_y):
        """
        Create the position and spectroscopic datasets for the windows.
        
        Parameters
        ----------
        image : numpy.ndarray
            Raw Image
        win_step_x : uint
            Step size in the x-direction between windows.
        win_step_y : uint
            Step size in the y-direction between windows.
        win_x : uint
            Size of the window in the x-direction.
        win_y : uint
            Size of the window in the y-direction.

        Returns
        -------
        ds_pix_inds : VirtualDataset
            Spectroscopic Indices of the windows
        ds_pix_vals : VirtualDataset
            Spectroscopic Values of the windows
        ds_pos_inds : VirtualDataset
            Position Indices of the windows
        ds_pos_vals : VirtualDataset
            Position Values of the windows
        win_pos_mat : numpy.ndarray
            Array containing the positions of the window origins

        """
        im_x, im_y = image.shape
        x_steps = np.arange(0, im_x - win_x + 1, win_step_x, dtype=np.uint32)
        y_steps = np.arange(0, im_y - win_y + 1, win_step_y, dtype=np.uint32)
        nx = len(x_steps)
        ny = len(y_steps)
        win_pos_mat = np.array([np.repeat(x_steps, ny), np.tile(y_steps, nx)], dtype=np.uint32).T
        win_pix_mat = make_indices_matrix([win_x, win_y]).T

        '''
        Set up the HDF5 Group and Datasets for the windowed data
        '''
        ds_pos_inds = VirtualDataset('Position_Indices', data=win_pos_mat, dtype=INDICES_DTYPE)
        ds_pix_inds = VirtualDataset('Spectroscopic_Indices', data=win_pix_mat, dtype=INDICES_DTYPE)
        ds_pos_vals = VirtualDataset('Position_Values', data=win_pos_mat, dtype=VALUES_DTYPE)
        ds_pix_vals = VirtualDataset('Spectroscopic_Values', data=win_pix_mat, dtype=VALUES_DTYPE)
        pos_labels = get_aux_dset_slicing(['Window Origin X', 'Window Origin Y'], is_spectroscopic=False)
        ds_pos_inds.attrs['labels'] = pos_labels
        ds_pos_inds.attrs['units'] = ['pixel', 'pixel']
        ds_pos_vals.attrs['labels'] = pos_labels
        ds_pos_vals.attrs['units'] = ['pixel', 'pixel']
        pix_labels = get_aux_dset_slicing(['U', 'V'], is_spectroscopic=True)
        ds_pix_inds.attrs['labels'] = pix_labels
        ds_pix_inds.attrs['units'] = ['pixel', 'pixel']
        ds_pix_vals.attrs['labels'] = pix_labels
        ds_pix_vals.attrs['units'] = ['pixel', 'pixel']

        return ds_pix_inds, ds_pix_vals, ds_pos_inds, ds_pos_vals, win_pos_mat

    @staticmethod
    def win_data_func(image):
        """
        Returns the input image in the `windata32` format

        Parameters
        ----------
        image : numpy.ndarray
            Windowed image to take the FFT of

        Returns
        -------
        windows : numpy.ndarray
            Array the image in the windata32 format
        """
        windows = np.empty_like(image, dtype=windata32)
        windows['Image Data'] = image

        return windows

    @staticmethod
    def abs_fft_func(image):
        """
        Take the 2d FFT of each window in `windows` and return in the proper form.

        Parameters
        ----------
        image : numpy.ndarray
            Windowed image to take the FFT of

        Returns
        -------
        windows : numpy.ndarray
            Array of the Magnitude of the FFT of each window for the input
            `image`

        """
        windows = np.empty_like(image, dtype=absfft32)
        windows['FFT Magnitude'] = np.abs(np.fft.fftshift(np.fft.fft2(image)))

        return windows

    @staticmethod
    def win_abs_fft_func(image):
        """
        Take the 2d FFT of each window in `windows` and return in the proper form.

        Parameters
        ----------
        image : numpy.ndarray
            Windowed image to take the FFT of

        Returns
        -------
        windows : numpy.ndarray
            Array of windows and the Magnitude of the FFT of each window for the input
            `image`

        """
        windows = np.empty_like(image, dtype=winabsfft32)
        windows['Image Data'] = image
        windows['FFT Magnitude'] = np.abs(np.fft.fftshift(np.fft.fft2(image)))

        return windows

    @staticmethod
    def win_comp_fft_func(image):
        """
        Take the 2d FFT of each window in `windows` and return in the proper form.

        Parameters
        ----------
        image : numpy.ndarray
            Windowed image to take the FFT of

        Returns
        -------
        windows : numpy.ndarray
            Array of windows and the FFT of each window for the input `image`

        """
        windows = np.empty_like(image, dtype=wincompfft32)
        windows['Image Data'] = image
        win_fft = np.fft.fftshift(np.fft.fft2(image))
        windows['FFT Real'] = win_fft.real
        windows['FFT Imag'] = win_fft.imag

        return windows

    def build_clean_image(self, h5_win=None):
        """
        Reconstructs the cleaned image from the windowed dataset

        Parameters
        ----------
        h5_win : HDF5 dataset , optional
            The windowed image to be reconstructed.

        Returns
        -------
        h5_clean : HDF5 dataset
            The cleaned image

        """
        if h5_win is None:
            if self.clean_wins is None:
                warn('You must clean the image before rebuilding it.')
                return
            h5_win = self.clean_wins

        '''
        Get basic windowing information from attributes of 
        h5_win
        '''
        im_x = h5_win.parent.attrs['image_x']
        im_y = h5_win.parent.attrs['image_y']
        win_x = h5_win.parent.attrs['win_x']
        win_y = h5_win.parent.attrs['win_y']
        win_step_x = h5_win.parent.attrs['win_step_x']
        win_step_y = h5_win.parent.attrs['win_step_x']

        '''
        Calculate the steps taken to create original windows
        '''
        x_steps = np.arange(0, im_x - win_x + 1, win_step_x)
        y_steps = np.arange(0, im_y - win_y + 1, win_step_y)

        '''
        Initialize arrays to hold summed windows and counts for each position
        '''
        counts = np.zeros([im_x, im_y], np.uint8)
        accum = np.zeros([im_x, im_y], np.float32)

        nx = len(x_steps)
        ny = len(y_steps)
        n_wins = nx * ny

        '''
        Create slice object from the positions
        '''
        win_slices = [[slice(x, x + win_x), slice(y, y + win_y)] for x, y in np.array([np.tile(x_steps, nx),
                                                                                       np.repeat(y_steps, ny)]).T]

        '''
        Loop over all windows.  Increment counts for window positions and 
        add current window to total.
        '''
        ones = np.ones([win_x, win_y], dtype=counts.dtype)
        for islice, this_slice in enumerate(win_slices):
            selected = islice % np.rint(n_wins / 10) == 0
            if selected:
                per_done = np.rint(100 * islice / n_wins)
                print('Reconstructing Image...{}% -- step # {}'.format(per_done, islice))
            counts[this_slice] += ones

            accum[this_slice] += h5_win[islice].reshape(win_x, win_y)

        clean_image = accum / counts

        clean_image[np.isnan(clean_image)] = 0

        clean_grp = VirtualGroup('Cleaned_Image', h5_win.parent.name[1:])

        ds_clean = VirtualDataset('Cleaned_Image', clean_image)

        clean_grp.add_children([ds_clean])

        image_refs = self.hdf.write(clean_grp)
        self.hdf.flush()

        h5_clean = get_h5_obj_refs(['Cleaned_Image'], image_refs)[0]

        self.h5_clean = h5_clean

        return h5_clean

    # def clean_and_build(self, h5_win=None, components=None):
    #     """
    #     Rebuild the Image from the SVD results on the windows
    #     Optionally, only use components less than n_comp.
    #
    #     Parameters
    #     ----------
    #     h5_win : hdf5 Dataset, optional
    #         dataset containing the windowed image which SVD was performed on
    #     components: {int, iterable of int, slice} optional
    #         Defines which components to keep
    #
    #         Input Types
    #         integer : Components less than the input will be kept
    #         length 2 iterable of integers : Integers define start and stop of component slice to retain
    #         other iterable of integers or slice : Selection of component indices to retain
    #
    #     Returns
    #     -------
    #     clean_wins : HDF5 Dataset
    #         the cleaned windows
    #
    #     """
    #
    #     if h5_win is None:
    #         if self.h5_wins is None:
    #             warn('You must perform windowing on an image followed by SVD on the window before you can clean it.')
    #             return
    #         h5_win = self.h5_wins
    #     elif 'Image Data' not in h5_win.dtype.names:
    #         warn('The windows must have the real space image data in them to rebuild.')
    #         return
    #
    #     print('Cleaning the image by removing unwanted components.')
    #
    #     comp_slice = _get_component_slice(components)
    #
    #     '''
    #     Read the 1st n_comp components from the SVD results
    #     on h5_win
    #     '''
    #     win_name = h5_win.name.split('/')[-1]
    #
    #     try:
    #         win_svd = find_results_groups(h5_win, 'SVD')[-1]
    #
    #         h5_S = win_svd['S']
    #         h5_U = win_svd['U']
    #         h5_V = win_svd['V']
    #
    #     except KeyError:
    #         warnstring = 'SVD Results for {dset} were not found in {file}.'.format(dset=win_name, file=self.image_path)
    #         warn(warnstring)
    #         return
    #     except:
    #         raise
    #
    #     '''
    #     Get basic windowing information from attributes of
    #     h5_win
    #     '''
    #     im_x = h5_win.parent.attrs['image_x']
    #     im_y = h5_win.parent.attrs['image_y']
    #     win_x = h5_win.parent.attrs['win_x']
    #     win_y = h5_win.parent.attrs['win_y']
    #     win_step_x = h5_win.parent.attrs['win_step_x']
    #     win_step_y = h5_win.parent.attrs['win_step_x']
    #
    #     '''
    #     Calculate the steps taken to create original windows
    #     '''
    #     x_steps = np.arange(0, im_x - win_x, win_step_x)
    #     y_steps = np.arange(0, im_y - win_y, win_step_y)
    #
    #     '''
    #     Initialize arrays to hold summed windows and counts for each position
    #     '''
    #     counts = np.zeros([im_x, im_y], np.uint32)
    #     clean_image = np.zeros([im_x, im_y], np.float32)
    #
    #     nx = len(x_steps)
    #     ny = len(y_steps)
    #     n_wins = nx * ny
    #
    #     '''
    #     Create slice object from the positions
    #     '''
    #     h5_win_pos = h5_win.file[h5_win.attrs['Position_Indices']]
    #     win_slices = [[slice(x, x + win_x), slice(y, y + win_y)] for x, y in h5_win_pos]
    #
    #     '''
    #     Loop over all windows.  Increment counts for window positions and
    #     add current window to total.
    #     '''
    #     ones = np.ones([win_x, win_y], dtype=counts.dtype)
    #     ds_V = np.dot(np.diag(h5_S[comp_slice]), h5_V['Image_Data'][comp_slice, :])
    #
    #     for islice, this_slice in enumerate(win_slices):
    #         if islice % np.rint(n_wins / 10) == 0:
    #             per_done = np.rint(100 * islice / n_wins)
    #             print('Reconstructing Image...{}% -- step # {}'.format(per_done, islice))
    #
    #         counts[this_slice] += ones
    #
    #         this_win = np.dot(h5_U[islice, comp_slice], ds_V)
    #
    #         clean_image[this_slice] += this_win.reshape(win_x, win_y)
    #
    #     clean_image = np.divide(clean_image, counts)
    #
    #     clean_image[np.isnan(clean_image)] = 0
    #
    #     '''
    #     Calculate the removed noise and FFTs
    #     '''
    #     removed_noise = np.reshape(self.h5_raw, clean_image.shape) - clean_image
    #
    #     fft_clean = np.fft.fft2(clean_image)
    #     fft_noise = np.fft.fft2(removed_noise)
    #
    #     '''
    #     Create datasets for results, link them properly, and write them to file
    #     '''
    #     clean_grp = MicroDataGroup('Cleaned_Image_', win_svd.name[1:])
    #     ds_clean = MicroDataset('Cleaned_Image', clean_image.reshape(self.h5_raw.shape))
    #     ds_noise = MicroDataset('Removed_Noise', removed_noise.reshape(self.h5_raw.shape))
    #     ds_fft_clean = MicroDataset('FFT_Cleaned_Image', fft_clean.reshape(self.h5_raw.shape))
    #     ds_fft_noise = MicroDataset('FFT_Removed_Noise', fft_noise.reshape(self.h5_raw.shape))
    #
    #     clean_grp.addChildren([ds_clean, ds_noise, ds_fft_clean, ds_fft_noise])
    #
    #     if isinstance(comp_slice, slice):
    #         clean_grp.attrs['components_used'] = '{}-{}'.format(comp_slice.start, comp_slice.stop)
    #     else:
    #         clean_grp.attrs['components_used'] = comp_slice
    #
    #     image_refs = self.hdf.writeData(clean_grp)
    #     self.hdf.flush()
    #
    #     h5_clean = get_h5_obj_refs(['Cleaned_Image'], image_refs)[0]
    #     h5_noise = get_h5_obj_refs(['Removed_Noise'], image_refs)[0]
    #     h5_fft_clean = get_h5_obj_refs(['FFT_Cleaned_Image'], image_refs)[0]
    #     h5_fft_noise = get_h5_obj_refs(['FFT_Removed_Noise'], image_refs)[0]
    #
    #     copy_attributes(self.h5_raw, h5_clean, skip_refs=False)
    #     copy_attributes(self.h5_raw, h5_noise, skip_refs=False)
    #     copy_attributes(self.h5_raw, h5_fft_clean, skip_refs=False)
    #     copy_attributes(self.h5_raw, h5_fft_noise, skip_refs=False)
    #
    #     self.h5_clean = h5_clean
    #     self.h5_noise = h5_noise
    #
    #     return h5_clean

    def clean_and_build_batch(self, h5_win=None, components=None):
        """
        Rebuild the Image from the SVD results on the windows
        Optionally, only use components less than n_comp.

        Parameters
        ----------
        h5_win : hdf5 Dataset, optional
            dataset containing the windowed image which SVD was performed on
        components : {int, iterable of int, slice} optional
            Defines which components to keep
            Default - None, all components kept

            Input Types
            integer : Components less than the input will be kept
            length 2 iterable of integers : Integers define start and stop of component slice to retain
            other iterable of integers or slice : Selection of component indices to retain

        Returns
        -------
        clean_wins : HDF5 Dataset
            the cleaned windows

        """

        if h5_win is None:
            if self.h5_wins is None:
                warn('You must perform windowing on an image followed by SVD on the window before you can clean it.')
                return
            h5_win = self.h5_wins
        elif 'Image Data' not in h5_win.dtype.names:
            warn('The windows must have the real space image data in them to rebuild.')
            return

        print('Cleaning the image by removing unwanted components.')

        comp_slice = get_component_slice(components)

        '''
        Read the 1st n_comp components from the SVD results
        on h5_win
        '''
        win_name = h5_win.name.split('/')[-1]

        try:
            win_svd = find_results_groups(h5_win, 'SVD')[-1]

            h5_S = win_svd['S']
            h5_U = win_svd['U']
            h5_V = win_svd['V']

        except KeyError:
            warnstring = 'SVD Results for {dset} were not found in {file}.'.format(dset=win_name, file=self.image_path)
            warn(warnstring)
            return
        except:
            raise

        '''
        Get basic windowing information from attributes of
        h5_win
        '''
        im_x = h5_win.parent.attrs['image_x']
        im_y = h5_win.parent.attrs['image_y']
        win_x = h5_win.parent.attrs['win_x']
        win_y = h5_win.parent.attrs['win_y']

        '''
        Initialize arrays to hold summed windows and counts for each position
        '''
        counts = np.zeros([im_x, im_y], np.uint32)
        accum = np.zeros([im_x, im_y], np.float32)

        '''
        Create slice object from the positions
        '''
        ds_win_pos = h5_win.file[h5_win.attrs['Position_Indices']][()]
        win_slices = [[slice(x, x + win_x), slice(y, y + win_y)] for x, y in ds_win_pos]
        n_wins = ds_win_pos.shape[0]
        '''
        Create a matrix to add when counting.
        h5_V is usually small so go ahead and take S.V
        '''
        ones = np.ones([win_x, win_y], dtype=counts.dtype)
        ds_V = np.dot(np.diag(h5_S[comp_slice]), h5_V['Image Data'][comp_slice, :])

        '''
        Calculate the size of a given batch that will fit in the available memory
        '''
        mem_per_win = ds_V.itemsize * ds_V.shape[1]
        if self.cores is None:
            free_mem = self.max_memory - ds_V.size * ds_V.itemsize
        else:
            free_mem = self.max_memory * 2 - ds_V.size * ds_V.itemsize
        batch_size = int(free_mem / mem_per_win)
        batch_slices = gen_batches(n_wins, batch_size)

        print('Reconstructing in batches of {} windows.'.format(batch_size))

        '''
        Loop over all batches.  Increment counts for window positions and
        add current window to total.
        '''
        for ibatch, batch in enumerate(batch_slices):
            ds_U = h5_U[batch, comp_slice]
            batch_wins = np.dot(ds_U, ds_V).reshape([-1, win_x, win_y])
            del ds_U
            for islice, this_slice in enumerate(win_slices[batch]):
                iwin = ibatch * batch_size + islice
                if iwin % np.rint(n_wins / 10) == 0:
                    per_done = np.rint(100 * iwin / n_wins)
                    print('Reconstructing Image...{}% -- step # {}'.format(per_done, islice))

                counts[this_slice] += ones

                accum[this_slice] += batch_wins[islice]

        clean_image = np.divide(accum, counts)

        clean_image[np.isnan(clean_image)] = 0

        if h5_win.file.attrs['normalized']:
            '''
            Renormalize the cleaned image
            '''
            clean_image -= np.min(clean_image)
            clean_image = clean_image / np.max(clean_image)

        '''
        Calculate the removed noise and FFTs
        '''
        removed_noise = np.reshape(self.h5_raw, clean_image.shape) - clean_image
        blackman_window_rows = blackman(clean_image.shape[0])
        blackman_window_cols = blackman(clean_image.shape[1])
        fft_clean = np.fft.fft2(blackman_window_rows[:, np.newaxis] * clean_image * blackman_window_cols[np.newaxis, :])
        fft_noise = np.fft.fft2(
            blackman_window_rows[:, np.newaxis] * removed_noise * blackman_window_cols[np.newaxis, :])

        '''
        Create datasets for results, link them properly, and write them to file
        '''
        clean_grp = VirtualGroup('Cleaned_Image_', win_svd.name[1:])
        ds_clean = VirtualDataset('Cleaned_Image', clean_image.reshape(self.h5_raw.shape))
        ds_noise = VirtualDataset('Removed_Noise', removed_noise.reshape(self.h5_raw.shape))
        ds_fft_clean = VirtualDataset('FFT_Cleaned_Image', fft_clean.reshape(self.h5_raw.shape))
        ds_fft_noise = VirtualDataset('FFT_Removed_Noise', fft_noise.reshape(self.h5_raw.shape))

        clean_grp.add_children([ds_clean, ds_noise, ds_fft_clean, ds_fft_noise])

        if isinstance(comp_slice, slice):
            clean_grp.attrs['components_used'] = '{}-{}'.format(comp_slice.start, comp_slice.stop)
        else:
            clean_grp.attrs['components_used'] = comp_slice

        image_refs = self.hdf.write(clean_grp)
        self.hdf.flush()

        h5_clean = get_h5_obj_refs(['Cleaned_Image'], image_refs)[0]
        h5_noise = get_h5_obj_refs(['Removed_Noise'], image_refs)[0]
        h5_fft_clean = get_h5_obj_refs(['FFT_Cleaned_Image'], image_refs)[0]
        h5_fft_noise = get_h5_obj_refs(['FFT_Removed_Noise'], image_refs)[0]

        copy_attributes(self.h5_raw, h5_clean, skip_refs=False)
        copy_attributes(self.h5_raw, h5_noise, skip_refs=False)
        copy_attributes(self.h5_raw, h5_fft_clean, skip_refs=False)
        copy_attributes(self.h5_raw, h5_fft_noise, skip_refs=False)

        self.h5_clean = h5_clean
        self.h5_noise = h5_noise

        return h5_clean

    def clean_and_build_separate_components(self, h5_win=None, components=None):
        """
        Rebuild the Image from the SVD results on the windows
        Optionally, only use components less than n_comp.

        Parameters
        ----------
        h5_win : hdf5 Dataset, optional
            dataset containing the windowed image which SVD was performed on
        components : {int, iterable of int, slice} optional
            Defines which components to keep
            Default - None, all components kept
            \n
            Input Types
            integer : Components less than the input will be kept
            length 2 iterable of integers : Integers define start and stop of component slice to retain
            other iterable of integers or slice : Selection of component indices to retain

        Returns
        -------
        clean_wins : HDF5 Dataset
            the cleaned windows

        """

        if h5_win is None:
            if self.h5_wins is None:
                warn('You must perform windowing on an image followed by SVD on the window before you can clean it.')
                return
            h5_win = self.h5_wins
        elif 'Image Data' not in h5_win.dtype.names:
            warn('The windows must have the real space image data in them to rebuild.')
            return

        print('Cleaning the image by removing unwanted components.')
        comp_slice, num_comps = get_component_slice(components)

        '''
        Read the 1st n_comp components from the SVD results
        on h5_win
        '''
        win_name = h5_win.name.split('/')[-1]

        try:
            win_svd = find_results_groups(h5_win, 'SVD')[-1]

            h5_S = win_svd['S']
            h5_U = win_svd['U']
            h5_V = win_svd['V']

        except KeyError:
            warnstring = 'SVD Results for {dset} were not found in {file}.'.format(dset=win_name, file=self.image_path)
            warn(warnstring)
            return
        except:
            raise

        '''
        Get basic windowing information from attributes of
        h5_win
        '''
        im_x = h5_win.parent.attrs['image_x']
        im_y = h5_win.parent.attrs['image_y']
        win_x = h5_win.parent.attrs['win_x']
        win_y = h5_win.parent.attrs['win_y']

        '''
        Create slice object from the positions
        '''
        ds_win_pos = h5_win.file[h5_win.attrs['Position_Indices']][()]
        win_slices = [[slice(x, x + win_x), slice(y, y + win_y), slice(None)] for x, y in ds_win_pos]
        n_wins = len(ds_win_pos)

        '''
        Go ahead and take the dot product of S and V.  Get the number of components
        from the length of S
        '''
        ds_V = np.dot(np.diag(h5_S[comp_slice]), h5_V['Image Data'][comp_slice, :]).T
        num_comps = ds_V.shape[1]

        '''
        Initialize arrays to hold summed windows and counts for each position
        '''
        ones = np.ones([win_x, win_y, num_comps], dtype=np.uint32)
        counts = np.zeros([im_x, im_y, num_comps], dtype=np.uint32)
        clean_image = np.zeros([im_x, im_y, num_comps], dtype=np.float32)

        '''
        Calculate the size of a given batch that will fit in the available memory
        '''
        mem_per_win = ds_V.itemsize * (num_comps + ds_V.size)
        if self.cores is None:
            free_mem = self.max_memory - ds_V.size * ds_V.itemsize
        else:
            free_mem = self.max_memory / 2 - ds_V.size * ds_V.itemsize
        batch_size = int(free_mem / mem_per_win)
        if batch_size < 1:
            raise MemoryError('Not enough memory to perform Image Cleaning.')
        batch_slices = gen_batches(n_wins, batch_size)

        print('Reconstructing in batches of {} windows.'.format(batch_size))
        '''
        Loop over all batches.  Increment counts for window positions and
        add current window to total.
        '''
        for ibatch, batch in enumerate(batch_slices):
            ds_U = h5_U[batch, comp_slice]
            batch_wins = ds_U[:, None, :] * ds_V[None, :, :]
            for islice, this_slice in enumerate(win_slices[batch]):
                iwin = ibatch * batch_size + islice
                if iwin % np.rint(n_wins / 10) == 0:
                    per_done = np.rint(100 * iwin / n_wins)
                    print('Reconstructing Image...{}% -- step # {}'.format(per_done, iwin))

                counts[this_slice] += ones

                clean_image[this_slice] += batch_wins[islice].reshape(win_x, win_y, num_comps)

        del ds_U, ds_V

        clean_image /= counts
        del counts
        clean_image[np.isnan(clean_image)] = 0

        '''
        Create datasets for results, link them properly, and write them to file
        '''
        clean_grp = VirtualGroup('Cleaned_Image_', win_svd.name[1:])

        clean_chunking = calc_chunks([im_x * im_y, num_comps],
                                     clean_image.dtype.itemsize)
        ds_clean = VirtualDataset('Cleaned_Image',
                                  data=clean_image.reshape(im_x * im_y, num_comps),
                                  chunking=clean_chunking,
                                  compression='gzip')

        clean_grp.add_children([ds_clean])

        if isinstance(comp_slice, slice):
            clean_grp.attrs['components_used'] = '{}-{}'.format(comp_slice.start,
                                                                comp_slice.stop)
        else:
            clean_grp.attrs['components_used'] = comp_slice

        image_refs = self.hdf.write(clean_grp)
        self.hdf.flush()

        h5_clean = get_h5_obj_refs(['Cleaned_Image'], image_refs)[0]
        h5_comp_inds = h5_clean.file[h5_U.attrs['Spectroscopic_Indices']]
        h5_comp_vals = h5_clean.file[h5_U.attrs['Spectroscopic_Values']]
        h5_pos_inds = self.h5_file[self.h5_raw.attrs['Position_Indices']]
        h5_pos_vals = self.h5_file[self.h5_raw.attrs['Position_Values']]

        link_as_main(h5_clean, h5_pos_inds, h5_pos_vals, h5_comp_inds, h5_comp_vals)

        self.h5_clean = h5_clean

        return h5_clean

    def plot_clean_image(self, h5_clean=None, image_path=None, image_type='png',
                         save_plots=True, show_plots=False, cmap='gray'):
        """
        Plot the cleaned image stored in the HDF5 dataset h5_clean

        Parameters
        ----------
        h5_clean : HDF5 dataset, optional
            cleaned image to be plotted
        image_path : str, optional
            path to save cleaned image file
            Default None, '_clean' will be appened to the name of the input image
        image_type : str, optional
            image format to save the cleaned image as
            Default 'png', all formats recognized by matplotlib.pyplot.imsave
            are allowed
        save_plots : Boolean, pptional
            If true, the image will be saved to image_path
            with the extention specified by image_type
            Default True
        show_plots : Boolean, optional
            If true, the image will be displayed on the screen
            Default False
        cmap : str, optional
            matplotlib colormap string designation

        Returns
        -------
        clean_image : Axis_Image
            object holding the plot of the cleaned image

        """
        if h5_clean is None:
            if self.h5_clean is None:
                warn('You must clean an image before it can be plotted.')
                return
            h5_clean = self.h5_clean

        '''
        Get the position indices of h5_clean and reshape the flattened image back
        '''
        try:
            h5_pos = h5_clean.file[h5_clean.attrs['Position_Indices']][()]
            x_pix = len(np.unique(h5_pos[:, 0]))
            y_pix = len(np.unique(h5_pos[:, 1]))

        except KeyError:
            '''
        Position Indices dataset does not exist
        Assume square image
            '''
            x_pix = np.int(np.sqrt(h5_clean.size))
            y_pix = x_pix

        except:
            raise

        image = h5_clean[()].reshape(x_pix, y_pix)

        if save_plots:
            if image_path is None:
                image_dir, basename = os.path.split(self.h5_file.filename)
                basename, _ = os.path.splitext(basename)
                basename = basename + '_clean.' + image_type
                image_path = os.path.join(image_dir, basename)

            plt.imsave(image_path, image, format=image_type, cmap=cmap)

        clean_image = plt.imshow(image, cmap=cmap)
        if show_plots:
            plt.show()

        return clean_image

    def window_size_extract(self, num_peaks=2, save_plots=True, show_plots=False):
        """
        Take the normalized image and extract from it an optimal window size

        Parameters
        ----------
        num_peaks : int, optional
            number of peaks to use during least squares fit
            Default 2
        save_plots : Boolean, optional
            If True then a plot showing the quality of the fit will be
            generated and saved to disk.  Ignored if do_fit is false.
            Default True
        show_plots : Boolean, optional
            If True then a plot showing the quality of the fit will be
            generated and shown on screen.  Ignored if do_fit is false.
            Default False

        Returns
        -------
        window_size : int
            Optimal window size in pixels
        psf_width : int
            Estimate atom spacing in pixels

        """

        def __gauss_fit(p, x):
            """
            simple gaussian fitting function
            """
            a = p[0]
            s = p[1]

            g = a * np.exp(-(x / s) ** 2)

            return g

        def __gauss_chi(p, x, y):
            """
            Simple chi-squared fit
            """
            gauss = __gauss_fit(p, x)

            chi2 = ((y - gauss) / y) ** 2

            return chi2

        h5_main = self.h5_raw

        print('Determining appropriate window size from image.')
        '''
        Normalize the image
        '''
        immin = np.min(h5_main)
        immax = np.max(h5_main)
        image = np.float32(h5_main - immin) / (immax - immin)

        '''
        Reshape the image based on the position indices
        '''
        try:
            h5_pos = h5_main.file[h5_main.attrs['Position_Indices']][()]
            x_pix = len(np.unique(h5_pos[:, 0]))
            y_pix = len(np.unique(h5_pos[:, 1]))

        except KeyError:
            '''
            Position Indices dataset does not exist
            Assume square image
            '''
            x_pix = np.int(np.sqrt(h5_main.size))
            y_pix = x_pix

        except:
            raise
        image = image.reshape([x_pix, y_pix])

        '''
        Perform an fft on the normalize image 
        '''
        im_shape = np.min(image.shape)

        def __hamming(data):
            """
            Simple hamming filter
            """
            u, v = np.shape(data)
            u_vec = np.linspace(0, 1, u)
            v_vec = np.linspace(0, 1, v)
            u_mat, v_mat = np.meshgrid(u_vec, v_vec, indexing='ij')
            h_filter = np.multiply((1 - np.cos(2 * np.pi * u_mat)), (1 - np.cos(2 * np.pi * v_mat))) / 4.0

            return np.multiply(data, h_filter)

        im2 = image - np.mean(image)
        fim = np.fft.fftshift(np.fft.fft2(__hamming(im2)))

        imrange = np.arange(-im_shape / 2, im_shape / 2)
        uu, vv = np.meshgrid(imrange, imrange)

        '''
        Find max at each radial distance from the center
        '''
        r_n = int(im_shape / 4)
        r_min = 0
        r_max = im_shape / 2
        r_vec = np.linspace(r_min, r_max, r_n, dtype=np.float32).transpose()

        r_mat = np.abs(uu + 1j * vv)

        fimabs = np.abs(fim)
        fimabs_max = np.zeros(r_n - 1)

        for k in range(r_n - 1):
            r1 = r_vec[k]
            r2 = r_vec[k + 1]
            r_ind = np.where(np.logical_and(r_mat >= r1, r_mat <= r2))
            fimabs_max[k] = np.max(fimabs[r_ind])

        r_vec = r_vec[:-1] + (r_max - r_min) / (r_n - 1.0) / 2.0

        '''
        Find local maxima
        '''
        count = 0
        local_max = []
        for k in range(1, fimabs_max.size - 1):
            if fimabs_max[k - 1] < fimabs_max[k] and fimabs_max[k] > fimabs_max[k + 1]:
                count += 1
                local_max.append(k)

        '''
        Get points corresponding to local maxima
        '''
        r_loc_max_vec = r_vec[local_max]
        fimabs_loc_max_vec = fimabs_max[local_max]

        '''
        Remove points below the radius of the tallest peak
        '''
        fimabs_loc_max_ind = np.argmax(fimabs_loc_max_vec)
        fimabs_loc_max_vec = fimabs_loc_max_vec[fimabs_loc_max_ind:]
        r_loc_max_vec = r_loc_max_vec[fimabs_loc_max_ind:]

        '''
        Sort the peaks from largest to smallest
        '''
        sort_ind = np.argsort(fimabs_loc_max_vec)[::-1]
        fimabs_sort = fimabs_loc_max_vec[sort_ind]
        r_sort = r_loc_max_vec[sort_ind]

        '''
        Check to ensure there are at least 2 peaks left.
        '''
        if fimabs_sort.size == 1:
            warn('Only one peak in sorted vector.  Simple estimate being used.')
            window_size = im_shape / (r_sort[0] + 0.5)

            window_size = np.clip(window_size, 1, int(im_shape / 6.0))

            window_size = np.int(np.round(window_size * 2))

            gauss_guess = (2 * fimabs_sort[0], r_sort[0])

            psf_width = im_shape / gauss_guess[1] / np.pi

            return window_size, psf_width

        '''
        Only use specified number of peaks
        '''
        fimabs_sort = fimabs_sort[:num_peaks]
        r_sort = r_sort[:num_peaks]

        '''
        Fit to a gaussian
        '''
        gauss_guess = (2 * np.max(fimabs_sort), r_sort[0])

        fit_vec, pcov, info, errmsg, success = leastsq(__gauss_chi,
                                                       gauss_guess,
                                                       args=(r_sort, fimabs_sort),
                                                       full_output=1,
                                                       maxfev=250)

        psf_width = im_shape / fit_vec[1] / np.pi

        if save_plots or show_plots:
            guess_vec = __gauss_fit(gauss_guess, r_vec)
            fit_vec = __gauss_fit(fit_vec, r_vec)
            self.__plot_window_fit(r_vec, r_sort, fimabs_max, fimabs_sort,
                                   guess_vec, fit_vec, save_plots, show_plots)

        window_size = im_shape / (r_sort[0] + 0.5)

        window_size = np.clip(window_size, 1, int(im_shape / 6.0))

        window_size = np.int(np.round(window_size * 2))

        return window_size, psf_width

    def __plot_window_fit(self, r_vec, r_sort, fft_absimage, fft_abssort, guess, fit,
                          save_plots=True, show_plots=False):
        """
        Generate a plot showing the quality of the least-squares fit to the peaks of the FFT of the image

        Parameters
        ----------
        r_vec : numpy array
            1D array of unsorted radii in pixels
        r_sort : numpy array
            1D array of the sorted radii
        fft_absimage : numpy array
            1D array of the absolute value of the FFT of the normalized image
        fft_abssort : numpy array
            1D array of FFT_absimage after being sorted to match r_sort
        guess :  numpy array
            1D array of the gaussian guess
        fit : numpy array
            1D array of the fitted gaussian
        save_plots : Boolean, optional
            If True then a plot showing the quality of the fit will be
            generated and saved to disk.
            Default True
        show_plots : Boolean, optional
            If True then a plot showing the quality of the fit will be
            generated and shown on screen.
            Default False

        Returns
        -------
        None

        """

        fig = plt.figure(figsize=[8, 8], tight_layout=True)
        plt1, = plt.semilogy(r_vec, fft_absimage, label='magnitude')
        plt2, = plt.semilogy(r_sort, fft_abssort, 'ro', label='chosen peaks')
        plt3, = plt.semilogy(r_vec, guess, 'g', label='guess')
        plt4, = plt.semilogy(r_vec, fit, 'r', label='fit')
        ax = fig.gca()
        ax.autoscale(tight=True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=1)
        ax.set_xlabel('radius [pixels]')
        ax.set_ylabel('max magnitude')
        plt.legend(handles=[plt1, plt2, plt3, plt4])

        if save_plots:
            folder, filename = os.path.split(self.hdf.path)
            basename, junk = os.path.splitext(filename)

            plotname = '_'.join([basename, 'window_fit'])
            plotpath = os.path.join(folder, plotname + '.png')

            fig.savefig(plotpath, format='png')

        if show_plots:
            plt.show()

        # plt.close(fig)


def radially_average_correlation(data_mat, num_r_bin):
    """
    Calculates the radially average correlation functions for a given 2D image

    Parameters
    ----------
    data_mat : 2D real numpy array
        Image to analyze
    num_r_bin : unsigned int
        Number of spatial bins to analyze

    Returns
    --------
    a_mat : 2D real numpy array
        Noise spectrum of the image
    a_rad_avg_vec : 1D real numpy array
        Average value of the correlation as a function of feature size
    a_rad_max_vec : 1D real numpy array
        Maximum value of the correlation as a function of feature size
    a_rad_min_vec : 1D real numpy array
        Minimum value of the correlation as a function of feature size
    a_rad_std_vec : 1D real numpy array
        Standard deviation of the correlation as a function of feature size

    """
    x_size = data_mat.shape[0]
    y_size = data_mat.shape[1]

    x_mesh, y_mesh = np.meshgrid(np.linspace(-1, 1, x_size),
                                 np.linspace(-1, 1, y_size))
    r_vec = np.sqrt(x_mesh ** 2 + y_mesh ** 2).flatten()

    s_mat = (np.abs(np.fft.fftshift(np.fft.fft2(data_mat)))) ** 2
    a_mat = np.abs(np.fft.fftshift((np.fft.ifft2(s_mat))))

    min_a = np.min(a_mat)
    a_mat = a_mat - min_a
    max_a = np.max(a_mat)
    a_mat = a_mat / max_a

    a_vec = a_mat.flatten()

    # bin results based on r
    a_rad_avg_vec = np.zeros(num_r_bin)
    a_rad_max_vec = np.zeros(a_rad_avg_vec.shape)
    a_rad_min_vec = np.zeros(a_rad_avg_vec.shape)
    a_rad_std_vec = np.zeros(a_rad_avg_vec.shape)
    r_bin_vec = np.zeros(a_rad_avg_vec.shape)

    step = 1 / (num_r_bin * 1.0 - 1)
    for k, r_bin in enumerate(np.linspace(0, 1, num_r_bin)):
        b = np.where((r_vec < r_bin + step) and (r_vec > r_bin))[0]

        if b.size == 0:
            a_rad_avg_vec[k] = np.nan
            a_rad_min_vec[k] = np.nan
            a_rad_max_vec[k] = np.nan
            a_rad_std_vec[k] = np.nan
        else:
            a_bin = a_vec[b]
            a_rad_avg_vec[k] = np.mean(a_bin)
            a_rad_min_vec[k] = np.min(a_bin)
            a_rad_max_vec[k] = np.max(a_bin)
            a_rad_std_vec[k] = np.std(a_bin)
        r_bin_vec[k] = r_bin + 0.5 * step

    return a_mat, a_rad_avg_vec, a_rad_max_vec, a_rad_min_vec, a_rad_std_vec
