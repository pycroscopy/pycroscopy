"""
Utility functions for the Fake BEPS generator

"""
import os
import numpy as np
from sklearn.utils import gen_batches
from sidpy.proc.comp_utils import get_available_memory
from skimage.measure import block_reduce
# Pycroscopy imports
from sidpy.hdf.hdf_utils import get_attr, write_simple_attrs
from sidpy.hdf.dtype_utils import stack_real_to_compound
from sidpy.sid import Translator
from pyUSID.io.hdf_utils import link_as_main, copy_dataset, \
    write_main_dataset, create_indexed_group, create_results_group, \
    write_reduced_anc_dsets
from pyUSID.io.reg_ref import copy_all_region_refs , write_region_references
from pyUSID.io.write_utils import Dimension, calc_chunks
from pyUSID.io.image import read_image
from ...analysis.utils.be_loop import loop_fit_function
from ...analysis.utils.be_sho import SHOfunc
from ...analysis.be_sho_fitter import sho32
from ...analysis.be_loop_fitter import loop_fit32
from .df_utils.beps_gen_utils import get_noise_vec, beps_image_folder
from .df_utils.image_utils import no_bin
import h5py

# Deprecated imports:
from ..hdf_writer import HDFwriter
from ..write_utils import build_reduced_spec_dsets, build_ind_val_dsets
from ..virtual_data import VirtualGroup, VirtualDataset


class FakeBEPSGenerator(Translator):
    """

    """

    # TODO: Add other cycle fractions
    # TODO: Add support for other VS_modes
    # TODO: Add support for other field modes
    def __init__(self, *args, **kwargs):
        """

        """
        super(FakeBEPSGenerator, self).__init__(*args, **kwargs)
        self.N_x = None
        self.N_y = None
        self.n_steps = None
        self.n_bins = None
        self.start_freq = None
        self.end_freq = None
        self.n_cycles = None
        self.forc_cycles = None
        self.forc_repeats = None
        self.loop_a = None
        self.loop_b = None
        self.data_type = None
        self.mode = None
        self.field_mode = None
        self.n_pixels = None
        self.n_loops = None
        self.n_sho_bins = None
        self.n_spec_bins = None
        self.n_fields = None
        self.binning_func = no_bin
        self.cycle_fraction = None
        self.h5_path = None
        self.image_ext = None
        self.rebin = None
        self.bin_factor = None
        self.bin_func = None
        #self.max_ram = 1024**8


    def _read_data(self, folder):
        """

        Returns
        -------

        """
        print('In folder {}'.format(folder))
        file_list = self._parse_file_path(folder, self.image_ext)

        images = list()

        for image_file in file_list:
            image_path = os.path.join(folder, image_file)
            image = read_image(image_path, as_grayscale=True)
            image = self.binning_func(image, self.bin_factor, self.bin_func)
            images.append(image)

        self.N_x, self.N_y = image.shape
        self.n_pixels = self.N_x * self.N_y

        return images

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

        # Make sure we have a proper path to the images to use
        if path is None:
            path = os.path.join(os.getcwd(), 'df_utils/beps_data_gen_images')
        else:
            path = os.path.abspath(path)

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

    def translate(self, h5_path, n_steps=32, n_bins=37, start_freq=300E+3, end_freq=350E+3,
                  data_type='BEPSData', mode='DC modulation mode', field_mode='in and out-of-field',
                  n_cycles=1, FORC_cycles=1, FORC_repeats=1, loop_a=3, loop_b=4,
                  cycle_frac='full', image_folder=beps_image_folder, bin_factor=None,
                  bin_func=np.mean, image_type='.tif', simple_coefs=False):
        """

        Parameters
        ----------
        h5_path : str
            Desired path to write the new HDF5 file
        n_steps : uint, optional
            Number of voltage steps
            Default - 32
        n_bins : uint, optional
            Number of frequency bins
            Default - 37
        start_freq : float, optional
            Starting frequency in Hz
            Default - 300E+3
        end_freq : float, optional
            Final freqency in Hz
            Default - 350E+3
        data_type : str, optional
            Type of data to generate
            Options -  'BEPSData', 'BELineData'
            Default - 'BEPSData'
        mode  : str, optional
            Modulation mode to use when generating the data.
            Options - 'DC modulation mode', 'AC modulation mode'
            Default - 'DC modulation mode'
        field_mode : str, optional
            Field mode
            Options - 'in-field', 'out-of-field', 'in and out-of-field'
            Default - 'in and out-of-field'
        n_cycles : uint, optional
            Number of cycles
            Default - 1
        FORC_cycles : uint, optional
            Number of FORC cycles
            Default - 1
        FORC_repeats : uint, optional
            Number of FORC repeats
            Default - 1
        loop_a : float, optional
            Loop coefficient a
            Default - 1
        loop_b : float, optional
            Loop coefficient b
        cycle_frac : str
            Cycle fraction parameter.
            Default - 'full'
        image_folder : str
            Path to the images that will be used to generate the loop coefficients.  There must be 11 images named
            '1.tif', '2.tif', ..., '11.tif'
            Default - pycroscopy.io.translators.df_utils.beps_gen_utils.beps_image_folder
        bin_factor : array_like of uint, optional
            Downsampling factor for each dimension.  Default is None.
        bin_func : callable, optional
            Function which will be called to calculate the return value
            of each block.  Function must implement an axis parameter,
            i.e. numpy.mean.  Ignored if bin_factor is None.  Default is
            numpy.mean.
        image_type : str
            File extension of images to be read.  Default '.tif'
        simple_coefs : bool
            Should a simpler coefficient generation be used.  Ensures loops, but all loops are identical.
            Default False

        Returns
        -------

        """

        # Setup shared parameters
        self.n_steps = n_steps
        self.n_bins = n_bins
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.n_cycles = n_cycles
        self.forc_cycles = FORC_cycles
        self.forc_repeats = FORC_repeats
        self.loop_a = loop_a
        self.loop_b = loop_b
        self.data_type = data_type
        self.mode = mode
        self.field_mode = field_mode
        self.cycle_fraction = cycle_frac
        self.bin_factor = bin_factor
        self.bin_func = bin_func
        if field_mode == 'in and out-of-field':
            self.n_fields = 2
        else:
            self.n_fields = 1
        self.n_loops = FORC_cycles * FORC_repeats * n_cycles * self.n_fields
        self.n_sho_bins = n_steps * self.n_loops
        self.n_spec_bins = n_bins * self.n_sho_bins
        self.h5_path = h5_path
        self.image_ext = image_type
        self.simple_coefs = simple_coefs

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
            self.binning_func = block_reduce
            self.bin_func = bin_func
        print('Image folder is {}'.format(image_folder))
        images = self._read_data(image_folder)
        self.images = images
        data_gen_parms = {'N_x': self.N_x, 'N_y': self.N_y, 'n_steps;:': n_steps,
                          'n_bins': n_bins, 'start_freq': start_freq,
                          'end_freq': end_freq, 'n_cycles': n_cycles,
                          'forc_cycles': FORC_cycles, 'forc_repeats': FORC_repeats,
                          'loop_a': loop_a, 'loop_b': loop_b, 'data_type': data_type,
                          'VS_mode': mode, 'field_mode': field_mode, 'num_udvs_steps': self.n_spec_bins,
                          'VS_cycle_fraction': cycle_frac}

        # Build the hdf5 file and get the datasets to write the data to
        self._setup_h5(data_gen_parms)

        # Calculate the loop parameters
        coef_mat = self.calc_loop_coef_mat(images)

        # In-and-out of field coefficients
        if field_mode != 'in-field':
            coef_OF_mat = np.copy(coef_mat)
        if field_mode != 'out-of-field':
            coef_IF_mat = np.copy(coef_mat)
            coef_IF_mat[:, 4] -= 0.05

        self.coef_mat = coef_mat
        # Calculate the SHO fit and guess from the loop coefficients
        self._calc_sho(coef_OF_mat, coef_IF_mat)

        # Save the loop guess and fit to file
        coef_OF_mat = np.hstack((coef_OF_mat[:, :9], np.ones([coef_OF_mat.shape[0], 1])))
        coef_IF_mat = np.hstack((coef_IF_mat[:, :9], np.ones([coef_IF_mat.shape[0], 1])))

        coef_mat = np.hstack([coef_IF_mat[:, np.newaxis, :], coef_OF_mat[:, np.newaxis, :]])
        coef_mat = np.rollaxis(coef_mat, 1, coef_mat.ndim).reshape([coef_mat.shape[0], -1])

        self.h5_loop_fit[:] = np.tile(stack_real_to_compound(coef_mat, loop_fit32),
                                      [1, int(self.n_loops / self.n_fields)])

        self.h5_loop_guess[:] = np.tile(stack_real_to_compound(coef_mat * get_noise_vec(coef_mat.shape, 0.1),
                                                               loop_fit32),
                                        [1, int(self.n_loops / self.n_fields)])

        self.h5_file.flush()

        self._calc_raw()

        self.h5_file.flush()

        return self.h5_path

    def _build_ancillary_datasets(self):
        """

        Parameters
        ----------
        None

        Returns
        -------
        ds_pos_inds : VirtualDataset
            Position Indices
        ds_pos_vals : VirtualDataset
            Position Values
        ds_spec_inds : VirtualDataset
            Spectrosocpic Indices
        ds_spec_vals : VirtualDataset
            Spectroscopic Values

        """
        # create spectrogram at each pixel from the coefficients
        spec_step = np.arange(0, 1, 1 / self.n_steps)
        V_vec = 10 * np.arcsin(np.sin(self.n_fields * np.pi * spec_step)) * 2 / np.pi

        # build DC vector for typical BEPS
        Vdc_mat = np.vstack((V_vec, np.full(np.shape(V_vec), np.nan)))  # Add out-of-field values
        IF_vec = Vdc_mat.T.flatten()  # Base DC vector
        IF_vec = np.tile(IF_vec, self.n_cycles)  # Now with Cycles
        IF_vec = np.dot(1 + np.arange(self.forc_cycles)[:, None], IF_vec[None, :])  # Do a single FORC
        IF_vec = np.tile(IF_vec.flatten(), self.forc_repeats)  # Repeat the FORC

        IF_inds = np.logical_not(np.isnan(IF_vec))

        Vdc_vec = np.where(IF_inds, IF_vec, 0)

        # build AC vector
        Vac_vec = np.ones(np.shape(Vdc_vec))

        # Build the Spectroscopic Values matrix
        spec_dims = [self.n_fields, self.n_steps, self.n_cycles, self.forc_cycles, self.forc_repeats, self.n_bins]
        spec_labs = ['Field', 'DC_Offset', 'Cycle', 'FORC', 'FORC_repeat', 'Frequency']
        spec_units = ['', 'V', '', '', '', 'Hz']
        spec_start = [0, 0, 0, 0, 0, self.start_freq]
        spec_steps = [1, 1, 1, 1, 1, (self.end_freq - self.start_freq) / self.n_bins]

        # Remove dimensions with single values
        real_dims = np.argwhere(np.array(spec_dims) != 1).squeeze()
        spec_dims = [spec_dims[idim] for idim in real_dims]
        spec_labs = [spec_labs[idim] for idim in real_dims]
        spec_units = [spec_units[idim] for idim in real_dims]
        spec_start = [spec_start[idim] for idim in real_dims]
        spec_steps = [spec_steps[idim] for idim in real_dims]

        # Correct the DC Offset dimension
        spec_dims_corrected = list()
        for dim_size, dim_name, dim_units, step_size, init_val in zip(spec_dims,
                                                                      spec_labs, spec_units, spec_steps, spec_start):

            if dim_name == 'DC_Offset':
                value = Vdc_vec[::2]
            else:
                value = np.arange(dim_size) * step_size + init_val
            spec_dims_corrected.append(Dimension(dim_name, dim_units, value))

        pos_dims = list()
        for dim_size, dim_name, dim_units, step_size, init_val in zip([self.N_y, self.N_x], ['Y', 'X'], ['um', 'um'],
                                                                      [10 / self.N_y, 10 / self.N_x], [-5, -5]):
            pos_dims.append(Dimension(dim_name, dim_units, np.arange(dim_size) * step_size + init_val))

        return pos_dims, spec_dims_corrected

    def _setup_h5(self, data_gen_parms):
        """
        Setups up the hdf5 file structure before doing the actual generation

        Parameters
        ----------
        data_gen_parms : dict
            Dictionary containing the parameters to write to the Measurement Group as attributes

        Returns
        -------

        """

        '''
        Build the group structure down to the channel group
        '''
        # Set up the basic group structure
        root_parms = dict()
        root_parms['translator'] = 'FAKEBEPS'
        root_parms['data_type'] = data_gen_parms['data_type']

        # Write the file
        self.h5_f = h5py.File(self.h5_path, 'w')
        write_simple_attrs(self.h5_f, root_parms)

        meas_grp = create_indexed_group(self.h5_f, 'Measurement')
        chan_grp = create_indexed_group(meas_grp, 'Channel')

        write_simple_attrs(meas_grp, data_gen_parms)

        # Create the Position and Spectroscopic datasets for the Raw Data
        h5_pos_dims, h5_spec_dims = self._build_ancillary_datasets()

        h5_raw_data = write_main_dataset(chan_grp, (self.n_pixels, self.n_spec_bins),
                                                        'Raw_Data',
                                                        'Deflection',
                                                        'Volts',
                                                        h5_pos_dims, h5_spec_dims,
                                                        slow_to_fast=True,
                                                        dtype=np.complex64, verbose=True)

        '''
        Build the SHO Group
        '''
        sho_grp = create_results_group(h5_raw_data, 'SHO_Fit')


        # Build the Spectroscopic datasets for the SHO Guess and Fit
        h5_sho_spec_inds, h5_sho_spec_vals = write_reduced_anc_dsets(
            sho_grp, h5_raw_data.h5_spec_inds, h5_raw_data.h5_spec_vals, 'Frequency', is_spec=True)

        h5_sho_fit = write_main_dataset(sho_grp,
                                                       (self.n_pixels, int(self.n_spec_bins // self.n_bins)),
                                                       'Fit',
                                                       'SHO Parameters',
                                                       'a.u.',
                                                       None, None,
                                                       h5_pos_inds=h5_raw_data.h5_pos_inds,
                                                       h5_pos_vals=h5_raw_data.h5_pos_vals,
                                                       h5_spec_inds=h5_sho_spec_inds,
                                                       h5_spec_vals=h5_sho_spec_vals,
                                                       slow_to_fast=True, dtype=sho32)

        h5_sho_guess = copy_dataset(h5_sho_fit, sho_grp, alias='Guess')

        '''
        Build the loop group
        '''

        loop_grp = create_results_group(h5_sho_fit, 'Loop_Fit')

        # Build the Spectroscopic datasets for the loops

        h5_loop_spec_inds, h5_loop_spec_vals = write_reduced_anc_dsets(
            loop_grp, h5_sho_fit.h5_spec_inds, h5_sho_fit.h5_spec_vals,
            'DC_Offset', is_spec=True)

        h5_loop_fit = write_main_dataset(loop_grp,
                                                        (self.n_pixels, self.n_loops),
                                                        'Fit',
                                                        'Loop Fitting Parameters',
                                                        'a.u.',
                                                        None, None,
                                                        h5_pos_inds=h5_raw_data.h5_pos_inds,
                                                        h5_pos_vals=h5_raw_data.h5_pos_vals,
                                                        h5_spec_inds=h5_loop_spec_inds,
                                                        h5_spec_vals=h5_loop_spec_vals,
                                                        slow_to_fast=True, dtype=loop_fit32)

        h5_loop_guess = copy_dataset(h5_loop_fit, loop_grp, alias='Guess')
        copy_all_region_refs(h5_loop_guess, h5_loop_fit)

        self.h5_raw = h5_raw_data
        self.h5_sho_guess = h5_sho_guess
        self.h5_sho_fit = h5_sho_fit
        self.h5_loop_guess = h5_loop_guess
        self.h5_loop_fit = h5_loop_fit
        self.h5_spec_vals = h5_raw_data.h5_spec_vals
        self.h5_spec_inds = h5_raw_data.h5_spec_inds
        self.h5_sho_spec_inds = h5_sho_fit.h5_spec_inds
        self.h5_sho_spec_vals = h5_sho_fit.h5_spec_vals
        self.h5_loop_spec_inds = h5_loop_fit.h5_spec_inds
        self.h5_loop_spec_vals = h5_loop_fit.h5_spec_vals
        self.h5_file = h5_raw_data.file

        return

    def calc_loop_coef_mat(self, image_list):
        """
        Build the loop coefficient matrix

        Parameters
        ----------
        image_list : list of numpy.ndarray
            Images that will be used to generate the coefficients

        Returns
        -------
        coef_mat : numpy.ndarray
            Array of loop coefficients

        """

        # Setup the limits on the coefficients
        # Redoing coefficient limits

        coef_limits = [[-6.0, -3.0],  # 0 - loop bottom edge
                       [15.0, 25.0],  # 1 - loop height
                       [-5.0, -2.5],  # 2 - loop crossing 1
                       [1.0, 3.5],  # 3 - loop crossing 2
                       [-0.5, 0.5],  # 4 - loop slope
                       [0.1, 3],  # 5 - loop corner sharpness 1
                       [0.1, 3],  # 6 - loop corner shaprness 2
                       [0.1, 3],  # 7 - loop corner sharpness 3
                       [0.1, 8],  # 8 - loop corner sharpness 4
                       [315E3, 325E3],  # 9 - resonant frequency
                       [80.0, 180.0]]  # 10 - Q factor'''

        # build loop coef matrix
        coef_mat = np.zeros([self.n_pixels, 11])
        for coef_ind in range(11):
            if self.simple_coefs:
                coef_img = np.mean(coef_limits[coef_ind])
            else:
                coef_img = image_list[coef_ind]
                coef_img.astype('float32')
                coef_img = (coef_img - coef_img.min()) / (coef_img.max() - coef_img.min())
                coef_min = coef_limits[coef_ind][0]
                coef_max = coef_limits[coef_ind][1]
                coef_img = coef_img * (coef_max - coef_min) + coef_min

            coef_mat[:, coef_ind] = coef_img.flatten()

        return coef_mat

    def _calc_sho(self, coef_OF_mat, coef_IF_mat, amp_noise=0.1, phase_noise=0.1, q_noise=0.2, resp_noise=0.01):
        """
        Build the SHO dataset from the coefficient matrices

        Parameters
        ----------
        coef_OF_mat : numpy.ndarray
            Out-of-field coefficients
        coef_IF_mat : numpy.ndarray
            In-field coefficients
        amp_noise : float
            Noise factor for amplitude parameter
        phase_noise : float
            Noise factor for phase parameter
        q_noise : float
            Noise factor for Q-value parameter
        resp_noise : float
            Noide factor for w0 parameter

        Returns
        -------
        None

        """
        print(list(self.h5_sho_spec_vals.attrs))
        vdc_vec = self.h5_sho_spec_vals[self.h5_sho_fit.spec_dim_labels.index('DC_Offset')].squeeze()
        sho_field = self.h5_sho_spec_vals[self.h5_sho_fit.spec_dim_labels.index('Field')].squeeze()
        sho_of_inds = sho_field == 0
        sho_if_inds = sho_field == 1

        # determine how many pixels can be read at once
        mem_per_pix = vdc_vec.size * np.float32(0).itemsize
        #free_mem = self.max_ram - vdc_vec.size * vdc_vec.dtype.itemsize * 6
        free_mem = 1024
        batch_size = int(free_mem / mem_per_pix)
        batches = gen_batches(self.n_pixels, batch_size)

        one_cycle_length = vdc_vec[sho_of_inds].shape[-1]

        for pix_batch in batches:
            roll_len = one_cycle_length // 4
            vdc_OF_rolled = np.roll(vdc_vec[sho_of_inds], -1 * roll_len)
            vdc_IF_rolled = np.roll(vdc_vec[sho_if_inds], -1 * roll_len)

            R_OF = np.array([np.roll(loop_fit_function(vdc_OF_rolled, coef), roll_len)
                             for coef in coef_OF_mat[pix_batch]])

            R_OF = R_OF - R_OF.mean()

            R_IF = np.array([np.roll(loop_fit_function(vdc_IF_rolled, coef), roll_len)
                             for coef in coef_IF_mat[pix_batch]])

            R_IF = R_IF - R_IF.mean()

            R_mat = np.hstack([R_IF[:, np.newaxis, :], R_OF[:, np.newaxis, :]])
            R_mat = np.rollaxis(R_mat, 1, R_mat.ndim).reshape(R_mat.shape[0], -1)

            del R_OF, R_IF

            amp = np.abs(R_mat)
            resp = coef_OF_mat[pix_batch, 9, None] * np.ones_like(R_mat)
            q_val = coef_OF_mat[pix_batch, 10, None] * np.ones_like(R_mat) * 10
            phase = np.sign(R_mat) * np.pi / 2 + np.pi / 2

            self.h5_sho_fit[pix_batch, :] = stack_real_to_compound(np.hstack([amp,
                                                                              resp,
                                                                              q_val,
                                                                              phase,
                                                                              np.ones_like(R_mat)]),
                                                                   sho32)

            self.h5_sho_guess[pix_batch, :] = stack_real_to_compound(np.hstack([amp * get_noise_vec(self.n_sho_bins,
                                                                                                    amp_noise),
                                                                                resp * get_noise_vec(self.n_sho_bins,
                                                                                                     resp_noise),
                                                                                q_val * get_noise_vec(self.n_sho_bins,
                                                                                                      q_noise),
                                                                                phase * get_noise_vec(self.n_sho_bins,
                                                                                                      phase_noise),
                                                                                np.ones_like(R_mat)]),
                                                                     sho32)

            self.h5_file.flush()

        return

    def _calc_raw(self):
        """

        Returns
        -------

        """
        mem_per_pix = self.n_sho_bins * self.h5_sho_fit.dtype.itemsize + self.n_spec_bins * self.h5_raw.dtype.itemsize

        free_mem = get_available_memory()
        batch_size = int(free_mem / mem_per_pix)
        batches = gen_batches(self.n_pixels, batch_size)

        w_vec = self.h5_spec_vals[self.h5_raw.spec_dim_labels.index('Frequency')].squeeze()
        w_vec = w_vec[:self.n_bins]

        for pix_batch in batches:
            sho_chunk = self.h5_sho_fit[pix_batch, :].flatten()

            raw_data = np.zeros([sho_chunk.shape[0], self.n_bins], dtype=np.complex64)
            for iparm, sho_parms in enumerate(sho_chunk):
                raw_data[iparm, :] = SHOfunc(sho_parms, w_vec)

            self.h5_raw[pix_batch, :] = raw_data.reshape([-1, self.n_spec_bins])

            self.h5_file.flush()

        return