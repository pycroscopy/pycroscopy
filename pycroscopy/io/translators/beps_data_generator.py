"""
Utility functions for the Fake BEPS generator

"""
import os
import numpy as np
from PIL import Image
from sklearn.utils import gen_batches
# Pycroscopy imports
from ..io_hdf5 import ioHDF5
from ..hdf_utils import calc_chunks, getH5DsetRefs, link_as_main, get_attr, buildReducedSpec, reshape_to_Ndims, \
    get_dimensionality
from ..io_utils import realToCompound
from .utils import build_ind_val_dsets, generate_dummy_main_parms
from .translator import Translator
from ..microdata import MicroDataGroup, MicroDataset
from ...analysis.utils.be_loop import loop_fit_function
from ...analysis.be_sho_model import sho32
from ...analysis.be_loop_model import loop_fit32
from .df_utils.beps_gen_utils import build_loop_from_mat, get_noise_vec, beps_image_folder


class FakeDataGenerator(Translator):
    """

    """
    def __init__(self, *args, **kwargs):
        """

        """
        super(FakeDataGenerator, self).__init__(*args, **kwargs)
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

    def _read_data(self):
        """

        Returns
        -------

        """
        pass

    def _parse_file_path(self, input_path):
        """

        Parameters
        ----------
        input_path

        Returns
        -------

        """
        pass

    def translate(self, h5_path, N_x, N_y, n_steps, n_bins, start_freq, end_freq,
                  data_type='BEPSData', mode='DC modulation mode', field_mode='in and out-of-field',
                  n_cycles=1, FORC_cycles=1, FORC_repeats=1, loop_a=1, loop_b=4, image_folder=beps_image_folder):
        """

        Parameters
        ----------
        h5_path : str
            Desired path to write the new HDF5 file
        N_x : uint
            Number of pixels in the x-dimension
        N_y : uint
            Number of pixels in the y-dimension
        n_steps : uint
            Number of voltage steps
        n_bins : n_bins
            Number of frequency bins
        start_freq : float
            Starting frequency in Hz
        end_freq : float
            Final freqency in Hz
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
        image_folder : str
            Path to the images that will be used to generate the loop coefficients.  There must be 11 images named
            '1.tif', '2.tif', ..., '11.tif'
            Default - pycroscopy.io.translators.df_utils.beps_gen_utils.beps_image_folder

        Returns
        -------

        """

        # Setup shared parameters
        self.N_x = N_x
        self.N_y = N_y
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
        self.n_pixels = N_x*N_y
        if field_mode == 'in and out-of-field':
            self.n_fields = 2
        else:
            self.n_fields = 1
        self.n_loops = FORC_cycles*FORC_repeats*n_cycles*self.n_fields
        self.n_sho_bins = n_steps*self.n_loops
        self.n_spec_bins = n_bins*self.n_sho_bins
        self.h5_path = h5_path

        data_gen_parms = {'N_x': N_x, 'N_y': N_y, 'n_steps;:': n_steps,
                          'n_bins': n_bins, 'start_freq': start_freq,
                          'end_freq': end_freq, 'n_cycles': n_cycles,
                          'forc_cycles': FORC_cycles, 'forc_repeats': FORC_repeats,
                          'loop_a': loop_a, 'loop_b': loop_b, 'data_type': data_type,
                          'VS_mode': mode, 'field_mode': field_mode, 'num_udvs_steps': self.n_spec_bins}

        # Make sure we have a proper path to the images to use
        if image_folder is None:
            image_folder = os.path.join(os.getcwd(), 'df_utils/beps_data_gen_images')
        else:
            image_folder = os.path.abspath(image_folder)

        # Build the hdf5 file and get the datasets to write the data to
        self._setup_h5(data_gen_parms)

        # Calculate the loop parameters
        coef_mat = self.calc_loop_coef_mat(image_folder)

        # In-and-out of field coefficients
        if field_mode != 'in-field':
            coef_OF_mat = np.copy(coef_mat)
        if field_mode != 'out-of-field':
            coef_IF_mat = np.copy(coef_mat)
            coef_IF_mat[:, 4] -= 0.05

        # Calculate the SHO fit and guess from the loop coefficients
        self._calc_sho(coef_OF_mat, coef_IF_mat)

        # Save the loop guess and fit to file
        coef_OF_mat = np.hstack((coef_OF_mat[:, :9], np.ones([coef_OF_mat.shape[0], 1])))
        coef_IF_mat = np.hstack((coef_IF_mat[:, :9], np.ones([coef_IF_mat.shape[0], 1])))

        coef_mat = np.hstack([coef_IF_mat, coef_OF_mat])

        self.h5_loop_fit[:] = np.tile(realToCompound(coef_mat, loop_fit32),
                                      [1, int(self.n_loops / self.n_fields)])

        self.h5_loop_guess[:] = np.tile(realToCompound(coef_mat * get_noise_vec(coef_mat.shape, 0.1),
                                                       loop_fit32),
                                        [1, int(self.n_loops / self.n_fields)])

        self.h5_file.flush()

        self.h5_file.close()

        return self.h5_path

    def _build_ancillary_datasets(self):
        """

        Parameters
        ----------
        None

        Returns
        -------
        ds_pos_inds : MicroDataset
            Position Indices
        ds_pos_vals : MicroDataset
            Position Values
        ds_spec_inds : MicroDataset
            Spectrosocpic Indices
        ds_spec_vals : MicroDataset
            Spectroscopic Values

        """
        # create spectrogram at each pixel from the coefficients
        spec_step = np.arange(0, 1, 1 / self.n_steps)
        V_vec = 10 * np.arcsin(np.sin(self.n_fields * np.pi * spec_step)) * 2 / np.pi
        # V1 = V_vec[np.gradient(V_vec) > 0]
        # V2 = V_vec[np.gradient(V_vec) <= 0]
        # # V_mat = np.vstack([V1, V2])
        # bin_frequencies = np.linspace(self.start_freq, self.end_freq, self.n_bins)
        # w_mat = np.array([bin_frequencies.T] * (self.n_steps * self.n_fields)).T
        # bin_indices = np.arange(2000, 2000 + self.n_bins)
        # bin_step = np.arange(self.n_bins)
        # bin_fft_a = np.ones(self.n_bins)
        # bin_fft_p = np.arange(self.n_bins) ** 2
        # bin_fft = bin_fft_a * np.exp(1j * bin_fft_p)

        # build DC vector for typical BEPS
        Vdc_mat = np.vstack((V_vec, np.full(np.shape(V_vec), np.nan)))  # Add out-of-field values
        IF_vec = Vdc_mat.T.flatten()  # Base DC vector
        IF_vec = np.tile(IF_vec, self.n_cycles)  # Now with Cycles
        IF_vec = np.dot(1 + np.arange(self.forc_cycles)[:, None], IF_vec[None, :])  # Do a single FORC
        IF_vec = np.tile(IF_vec.flatten(), self.forc_repeats)  # Repeat the FORC

        IF_inds = np.logical_not(np.isnan(IF_vec))
        # OF_inds = np.isnan(IF_vec)

        Vdc_vec = np.where(IF_inds, IF_vec, 0)
        # OF_vec = np.where(OF_inds, Vdc_vec, np.nan)
        #
        # wave_type_vec = np.ones(np.shape(Vdc_vec))
        # wave_mod_vec = np.ones(np.shape(Vdc_vec))

        # build AC vector
        Vac_vec = np.ones(np.shape(Vdc_vec))

        # udvs_steps = self.n_steps * self.n_fields * self.n_cycles * self.forc_cycles* self.forc_repeats
        # # build the UDVS matrix
        # UDVS_mat = np.zeros((udvs_steps, 7))
        # UDVS_mat[:, 0] = np.arange(udvs_steps)  # step numbers
        # UDVS_mat[:, 1] = np.squeeze(Vdc_vec)  # DC
        # UDVS_mat[:, 2] = np.squeeze(Vac_vec)  # AC
        # UDVS_mat[:, 3] = np.squeeze(wave_type_vec)  # type
        # UDVS_mat[:, 4] = np.squeeze(wave_mod_vec)  # mod
        # UDVS_mat[:, 5] = np.squeeze(IF_vec)  # mod
        # UDVS_mat[:, 6] = np.squeeze(OF_vec)  # mod

        # Build the Spectroscopic Values matrix
        spec_dims = [self.n_bins, self.n_fields, self.n_steps, self.n_cycles, self.forc_cycles, self.forc_repeats]
        spec_labs = ['Frequency', 'Field', 'DC_Offset', 'Cycle', 'FORC', 'FORC_repeat']
        spec_units = ['Hz', '', 'V', '', '', '']
        spec_start = [self.start_freq, 0, 0, 0, 0, 0]
        spec_steps = [(self.end_freq - self.start_freq) / self.n_bins, 1, 1, 1, 1, 1]
        spec_inds, spec_vals = build_ind_val_dsets(spec_dims,
                                                   labels=spec_labs,
                                                   units=spec_units,
                                                   initial_values=spec_start,
                                                   steps=spec_steps)

        # Replace the dummy DC values with the correct ones
        spec_vals.data[spec_labs.index('DC_Offset'), :] = np.repeat(Vdc_vec, self.n_bins)

        position_ind_mat, position_val_mat = build_ind_val_dsets([self.N_x, self.N_y], False,
                                                                 steps=[10 / self.N_x, 10 / self.N_y],
                                                                 initial_values=[-5, -5],
                                                                 labels=['X', 'Y'],
                                                                 units=['um', 'um'])

        return position_ind_mat, position_val_mat, spec_inds, spec_vals

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
        root_grp = MicroDataGroup('')
        root_parms = generate_dummy_main_parms()
        root_parms['translator'] = 'FAKEBEPS'
        root_parms['data_type'] = data_gen_parms['data_type']
        root_grp.attrs = root_parms

        meas_grp = MicroDataGroup('Measurement_')
        chan_grp = MicroDataGroup('Channel_')

        meas_grp.attrs.update(data_gen_parms)

        # Create the Position and Spectroscopic datasets for the Raw Data
        ds_pos_inds, ds_pos_vals, ds_spec_inds, ds_spec_vals = self._build_ancillary_datasets()

        raw_chunking = calc_chunks([self.n_pixels,
                                    self.n_spec_bins],
                                   np.complex64(0).itemsize,
                                   unit_chunks=[1, self.n_bins])

        ds_raw_data = MicroDataset('Raw_Data', data=[],
                                   maxshape=[self.n_pixels, self.n_spec_bins],
                                   dtype=np.complex64,
                                   compression='gzip',
                                   chunking=raw_chunking,
                                   parent=meas_grp)

        chan_grp.addChildren([ds_pos_inds, ds_pos_vals, ds_spec_inds, ds_spec_vals,
                              ds_raw_data])
        meas_grp.addChildren([chan_grp])
        root_grp.addChildren([meas_grp])

        hdf = ioHDF5(self.h5_path)
        hdf.delete()
        h5_refs = hdf.writeData(root_grp)

        # Delete the MicroDatasets to save memory
        del ds_raw_data, ds_spec_inds, ds_spec_vals, ds_pos_inds, ds_pos_vals

        # Get the file and Raw_Data objects
        h5_raw = getH5DsetRefs(['Raw_Data'], h5_refs)[0]
        h5_chan_grp = h5_raw.parent

        # Get the Position and Spectroscopic dataset objects
        h5_pos_inds = getH5DsetRefs(['Position_Indices'], h5_refs)[0]
        h5_pos_vals = getH5DsetRefs(['Position_Values'], h5_refs)[0]
        h5_spec_inds = getH5DsetRefs(['Spectroscopic_Indices'], h5_refs)[0]
        h5_spec_vals = getH5DsetRefs(['Spectroscopic_Values'], h5_refs)[0]

        # Link the Position and Spectroscopic datasets as attributes of Raw_Data
        link_as_main(h5_raw, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

        '''
        Build the SHO Group
        '''
        sho_grp = MicroDataGroup('Raw_Data-SHO_Fit_', parent=h5_chan_grp.name)

        # Build the Spectroscopic datasets for the SHO Guess and Fit
        sho_spec_starts = np.where(h5_spec_inds[0] == 0)[0]
        sho_spec_labs = get_attr(h5_spec_inds, 'labels')
        ds_sho_spec_inds, ds_sho_spec_vals = buildReducedSpec(h5_spec_inds,
                                                              h5_spec_vals,
                                                              keep_dim=sho_spec_labs != 'Frequency',
                                                              step_starts=sho_spec_starts)

        sho_chunking = calc_chunks([self.n_pixels,
                                    self.n_sho_bins],
                                   sho32.itemsize,
                                   unit_chunks=[1, 1])
        ds_sho_fit = MicroDataset('Fit', data=[],
                                  maxshape=[self.n_pixels, self.n_sho_bins],
                                  dtype=sho32,
                                  compression='gzip',
                                  chunking=sho_chunking,
                                  parent=sho_grp)
        ds_sho_guess = MicroDataset('Guess', data=[],
                                    maxshape=[self.n_pixels, self.n_sho_bins],
                                    dtype=sho32,
                                    compression='gzip',
                                    chunking=sho_chunking,
                                    parent=sho_grp)

        sho_grp.addChildren([ds_sho_fit, ds_sho_guess, ds_sho_spec_inds, ds_sho_spec_vals])

        # Write the SHO group and datasets to the file and delete the MicroDataset objects
        h5_sho_refs = hdf.writeData(sho_grp)
        del ds_sho_fit, ds_sho_guess, ds_sho_spec_inds, ds_sho_spec_vals

        # Get the dataset handles for the fit and guess
        h5_sho_fit = getH5DsetRefs(['Fit'], h5_sho_refs)[0]
        h5_sho_guess = getH5DsetRefs(['Guess'], h5_sho_refs)[0]

        # Get the dataset handles for the SHO Spectroscopic datasets
        h5_sho_spec_inds = getH5DsetRefs(['Spectroscopic_Indices'], h5_sho_refs)[0]
        h5_sho_spec_vals = getH5DsetRefs(['Spectroscopic_Values'], h5_sho_refs)[0]

        # Link the Position and Spectroscopic datasets as attributes of the SHO Fit and Guess
        link_as_main(h5_sho_fit, h5_pos_inds, h5_pos_vals, h5_sho_spec_inds, h5_sho_spec_vals)
        link_as_main(h5_sho_guess, h5_pos_inds, h5_pos_vals, h5_sho_spec_inds, h5_sho_spec_vals)

        '''
        Build the loop group
        '''
        loop_grp = MicroDataGroup('Fit-Loop_Fit_', parent=h5_sho_fit.parent.name)

        # Build the Spectroscopic datasets for the loops
        loop_spec_starts = np.where(h5_sho_spec_inds[0] == 0)[0]
        loop_spec_labs = get_attr(h5_sho_spec_inds, 'labels')
        ds_loop_spec_inds, ds_loop_spec_vals = buildReducedSpec(h5_sho_spec_inds,
                                                                h5_sho_spec_vals,
                                                                keep_dim=loop_spec_labs != 'DC_Offset',
                                                                step_starts=loop_spec_starts)

        # Create the loop fit and guess MicroDatasets
        loop_chunking = calc_chunks([self.n_pixels, self.n_loops],
                                    loop_fit32.itemsize,
                                    unit_chunks=[1, 1])
        ds_loop_fit = MicroDataset('Fit', data=[],
                                   maxshape=[self.n_pixels, self.n_loops],
                                   dtype=loop_fit32,
                                   compression='gzip',
                                   chunking=loop_chunking,
                                   parent=loop_grp)

        ds_loop_guess = MicroDataset('Guess', data=[],
                                     maxshape=[self.n_pixels, self.n_loops],
                                     dtype=loop_fit32,
                                     compression='gzip',
                                     chunking=loop_chunking,
                                     parent=loop_grp)

        # Add the datasets to the loop group then write it to the file
        loop_grp.addChildren([ds_loop_fit, ds_loop_guess, ds_loop_spec_inds, ds_loop_spec_vals])
        h5_loop_refs = hdf.writeData(loop_grp)

        # Delete the MicroDatasets
        del ds_loop_spec_vals, ds_loop_spec_inds, ds_loop_guess, ds_loop_fit

        # Get the handles to the datasets
        h5_loop_fit = getH5DsetRefs(['Fit'], h5_loop_refs)[0]
        h5_loop_guess = getH5DsetRefs(['Guess'], h5_loop_refs)[0]
        h5_loop_spec_inds = getH5DsetRefs(['Spectroscopic_Indices'], h5_loop_refs)[0]
        h5_loop_spec_vals = getH5DsetRefs(['Spectroscopic_Values'], h5_loop_refs)[0]

        # Link the Position and Spectroscopic datasets to the Loop Guess and Fit
        link_as_main(h5_loop_fit, h5_pos_inds, h5_pos_vals, h5_loop_spec_inds, h5_loop_spec_vals)
        link_as_main(h5_loop_guess, h5_pos_inds, h5_pos_vals, h5_loop_spec_inds, h5_loop_spec_vals)

        self.h5_raw = h5_raw
        self.h5_sho_guess = h5_sho_guess
        self.h5_sho_fit = h5_sho_fit
        self.h5_loop_guess = h5_loop_guess
        self.h5_loop_fit = h5_loop_fit
        self.h5_spec_vals = h5_spec_vals
        self.h5_spec_inds = h5_spec_inds
        self.h5_sho_spec_inds = h5_sho_spec_inds
        self.h5_sho_spec_vals = h5_sho_spec_vals
        self.h5_loop_spec_inds = h5_loop_spec_inds
        self.h5_loop_spec_vals = h5_loop_spec_vals
        self.h5_file = h5_raw.file

        return

    def calc_loop_coef_mat(self, folder):
        """
        Build the loop coefficient matrix

        Parameters
        ----------
        folder : str
            Path to the folder holding the images

        Returns
        -------
        coef_mat : numpy.ndarray
            Array of loop coefficients

        """

        # Setup the limits on the coefficients
        coef_limits = [[-1.0, -0.4]]  # 0 - loop bottom edge
        coef_limits.append([0.5, 2.0])  # 1 - loop height
        coef_limits.append([3.0, 5.0])  # 2 - loop crossing 1
        coef_limits.append([-5.0, -3.0])  # 3 - loop crossing 2
        coef_limits.append([-0.001, 0.0])  # 4 - loop slope
        coef_limits.append([self.loop_a, self.loop_b])  # 5 - loop corner sharpness 1
        coef_limits.append([self.loop_a / 4, self.loop_b / 4])  # 6 - loop corner shaprness 2
        coef_limits.append([self.loop_a / 4, self.loop_b / 4])  # 7 - loop corner sharpness 3
        coef_limits.append([self.loop_a, self.loop_b])  # 8 - loop corner sharpness 4
        coef_limits.append([275E3, 325E3])  # 9 - resonant frequency
        coef_limits.append([100.0, 150.0])  # 10 - Q factor

        # build loop coef matrix
        coef_mat = np.zeros([self.n_pixels, 11])
        for coef_ind in range(11):
            image_name = str(coef_ind + 1) + '.tif'
            coef_img = np.array(Image.open(os.path.join(folder, image_name))) / 256
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
        vdc_vec = self.h5_sho_spec_vals[self.h5_sho_spec_vals.attrs['DC_Offset']].squeeze()
        field = self.h5_sho_spec_vals[self.h5_sho_spec_vals.attrs['Field']].squeeze()
        of_inds = field == 0
        if_inds = field == 1
        # determine how many pixels can be read at once
        mem_per_pix = vdc_vec.size*np.float32(0).itemsize
        free_mem = self.max_ram-vdc_vec.size*vdc_vec.dtype.itemsize*6
        batch_size = int(free_mem/mem_per_pix)
        batches = gen_batches(self.n_pixels, batch_size)

        for pix_batch in batches:
            R_OF = np.array([loop_fit_function(vdc_vec[of_inds], coef) for coef in coef_OF_mat[pix_batch]])
            R_IF = np.array([loop_fit_function(vdc_vec[if_inds], coef) for coef in coef_IF_mat[pix_batch]])
            R_mat = np.stack([R_IF, R_OF], axis=2).reshape(-1, self.n_sho_bins)

            del R_OF, R_IF

            amp = np.abs(R_mat)
            resp = coef_OF_mat[pix_batch, 9, None] * np.ones_like(R_mat)
            q_val = coef_OF_mat[pix_batch, 10, None] * np.ones_like(R_mat)
            phase = np.sign(R_mat) * np.pi / 2

            self.h5_sho_fit[pix_batch, :] = realToCompound(np.hstack([amp,
                                                                      resp,
                                                                      q_val,
                                                                      phase,
                                                                      np.ones_like(R_mat)]),
                                                           sho32)

            self.h5_sho_guess[pix_batch, :] = realToCompound(np.hstack([amp*get_noise_vec(self.n_sho_bins, amp_noise),
                                                                        resp*get_noise_vec(self.n_sho_bins, resp_noise),
                                                                        q_val*get_noise_vec(self.n_sho_bins, q_noise),
                                                                        phase*get_noise_vec(self.n_sho_bins, phase_noise),
                                                                        np.ones_like(R_mat)]),
                                                             sho32)

            self.h5_file.flush()

        return
