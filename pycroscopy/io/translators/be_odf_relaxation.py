# -*- coding: utf-8 -*-
"""
Created on Thursday May 26 11:23:00 2016

@author:  Rama Vasudevan, Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, remove  # File Path formatting
from warnings import warn

import numpy as np  # For array operations
from scipy.io.matlab import loadmat  # To load parameters stored in Matlab .mat file
import h5py

from .df_utils.be_utils import trimUDVS, getSpectroscopicParmLabel, generatePlotGroups, createSpecVals, maxReadPixels, \
    nf32
from pyUSID.io.translator import Translator, generate_dummy_main_parms
from pyUSID.io.write_utils import INDICES_DTYPE, Dimension
from pyUSID.io.hdf_utils import create_indexed_group, write_main_dataset, write_simple_attrs


class BEodfRelaxationTranslator(Translator):
    """
    Translates old Relaxation data into the new H5 format. This is for the files generated from
    the old BEPSDAQ program utilizing two cards simultaneously.
    At present, this version of the translator only works for Out of field measurements
    It will not work for in-field. This should be fixed at a later date.
    
    """
    def __init__(self, max_mem_mb=1024):
        super(BEodfRelaxationTranslator, self).__init__(max_mem_mb)
        self.FFT_BE_wave = None
        self.h5_file = None
        self.ds_main = None
        self.mean_resp = None
        self.max_resp = None
        self.min_resp = None

    def translate(self, file_path, show_plots=True, save_plots=True, do_histogram=False):
        """
        Basic method that translates .dat data file(s) to a single .h5 file
        
        Inputs:
            file_path -- Absolute file path for one of the data files. 
            It is assumed that this file is of the OLD data format. 
            
        Outputs:
            Nothing
        """
        file_path = path.abspath(file_path)
        (folder_path, basename) = path.split(file_path)
        (basename, path_dict) = self._parse_file_path(file_path)

        h5_path = path.join(folder_path, basename + '.h5')
        if path.exists(h5_path):
            remove(h5_path)
        self.h5_file = h5py.File(h5_path, 'w')

        isBEPS = True
        parm_dict = self.__getParmsFromOldMat(path_dict['old_mat_parms'])

        ignored_plt_grps = ['in-field']  # Here we assume that there is no in-field.
        # If in-field data is captured then the translator would have to be modified.

        # Technically, we could do away with this if statement, as isBEPS is always true for this translation
        if isBEPS:
            parm_dict['data_type'] = 'BEPSData'

            std_expt = parm_dict['VS_mode'] != 'load user defined VS Wave from file'

            if not std_expt:
                warn('This translator does not handle user defined voltage spectroscopy')
                return

            spec_label = getSpectroscopicParmLabel(parm_dict['VS_mode'])

            # Check file sizes:
        if 'read_real' in path_dict.keys():
            real_size = path.getsize(path_dict['read_real'])
            imag_size = path.getsize(path_dict['read_imag'])
        else:
            real_size = path.getsize(path_dict['write_real'])
            imag_size = path.getsize(path_dict['write_imag'])

        if real_size != imag_size:
            raise ValueError("Real and imaginary file sizes DON'T match!. Ending")

        num_rows = int(parm_dict['grid_num_rows'])
        num_cols = int(parm_dict['grid_num_cols'])
        num_pix = num_rows * num_cols
        tot_bins = real_size / (num_pix * 4)  # Finding bins by simple division of entire datasize

        # Check for case where only a single pixel is missing.
        check_bins = real_size / ((num_pix - 1) * 4)

        if tot_bins % 1 and check_bins % 1:
            warn('Aborting! Some parameter appears to have changed in-between')
            return
        elif not tot_bins % 1:
            #             Everything's ok
            pass
        elif not check_bins % 1:
            tot_bins = check_bins
            warn('Warning:  A pixel seems to be missing from the data.  File will be padded with zeros.')

        tot_bins = int(tot_bins)
        (bin_inds, bin_freqs, bin_FFT, ex_wfm, dc_amp_vec) = self.__readOldMatBEvecs(path_dict['old_mat_parms'])
        """
        Because this is the old data format and there is a discrepancy in the number of bins (they seem to be 2 less 
        than the actual number), we need to re-calculate it based on the available data. This is done below.
        """

        band_width = parm_dict['BE_band_width_[Hz]'] * (0.5 - parm_dict['BE_band_edge_trim'])
        st_f = parm_dict['BE_center_frequency_[Hz]'] - band_width
        en_f = parm_dict['BE_center_frequency_[Hz]'] + band_width
        bin_freqs = np.linspace(st_f, en_f, len(bin_inds), dtype=np.float32)

        # Forcing standardized datatypes:
        bin_inds = np.int32(bin_inds)
        bin_freqs = np.float32(bin_freqs)
        bin_FFT = np.complex64(bin_FFT)
        ex_wfm = np.float32(ex_wfm)

        self.FFT_BE_wave = bin_FFT

        (UDVS_labs, UDVS_units, UDVS_mat) = self.__buildUDVSTable(parm_dict)

        # Remove the unused plot group columns before proceeding:
        (UDVS_mat, UDVS_labs, UDVS_units) = trimUDVS(UDVS_mat, UDVS_labs, UDVS_units, ignored_plt_grps)

        spec_inds = np.zeros(shape=(2, tot_bins), dtype=INDICES_DTYPE)

        # Will assume that all excitation waveforms have same number of bins
        # Here, the denominator is 2 because only out of field measruements. For IF + OF, should be 1
        num_actual_udvs_steps = UDVS_mat.shape[0] / 2
        bins_per_step = tot_bins / num_actual_udvs_steps

        # Some more checks
        if bins_per_step % 1:
            warn('Non integer number of bins per step!')
            return
        else:
            bins_per_step = int(bins_per_step)

        num_actual_udvs_steps = int(num_actual_udvs_steps)

        stind = 0
        for step_index in range(UDVS_mat.shape[0]):
            if UDVS_mat[step_index, 2] < 1E-3:  # invalid AC amplitude
                continue  # skip
            spec_inds[0, stind:stind + bins_per_step] = np.arange(bins_per_step, dtype=INDICES_DTYPE)  # Bin step
            spec_inds[1, stind:stind + bins_per_step] = step_index * np.ones(bins_per_step,
                                                                             dtype=INDICES_DTYPE)  # UDVS step
            stind += bins_per_step
        del stind, step_index

        # Some very basic information that can help the processing / analysis crew
        parm_dict['num_bins'] = tot_bins
        parm_dict['num_pix'] = num_pix
        parm_dict['num_udvs_steps'] = num_actual_udvs_steps

        global_parms = generate_dummy_main_parms()
        global_parms['grid_size_x'] = parm_dict['grid_num_cols']
        global_parms['grid_size_y'] = parm_dict['grid_num_rows']
        global_parms['experiment_date'] = parm_dict['File_date_and_time']

        # assuming that the experiment was completed:
        global_parms['current_position_x'] = parm_dict['grid_num_cols'] - 1
        global_parms['current_position_y'] = parm_dict['grid_num_rows'] - 1
        global_parms['data_type'] = parm_dict['data_type']  # self.__class__.__name__
        global_parms['translator'] = 'ODF'
        write_simple_attrs(self.h5_file, global_parms)

        # Create Measurement and Channel groups
        meas_grp = create_indexed_group(self.h5_file, 'Measurement')
        write_simple_attrs(meas_grp, parm_dict)

        chan_grp = create_indexed_group(meas_grp, 'Channel')
        chan_grp.attrs['Channel_Input'] = parm_dict['IO_Analog_Input_1']

        # Create Auxilliary Datasets
        h5_ex_wfm = chan_grp.create_dataset('Excitation_Waveform', data=ex_wfm)

        udvs_slices = dict()
        for col_ind, col_name in enumerate(UDVS_labs):
            udvs_slices[col_name] = (slice(None), slice(col_ind, col_ind + 1))
        h5_UDVS = chan_grp.create_dataset('UDVS',
                                          data=UDVS_mat,
                                          dtype=np.float32)
        write_simple_attrs(h5_UDVS, {'labels': UDVS_labs, 'units': UDVS_units})

        h5_bin_steps = chan_grp.create_dataset('Bin_Steps',
                                               data=np.arange(bins_per_step, dtype=np.uint32),
                                               dtype=np.uint32)

        # Need to add the Bin Waveform type - infer from UDVS
        exec_bin_vec = self.signal_type * np.ones(len(bin_inds), dtype=np.int32)
        h5_wfm_typ = chan_grp.create_dataset('Bin_Wfm_Type',
                                             data=exec_bin_vec,
                                             dtype=np.int32)

        h5_bin_inds = chan_grp.create_dataset('Bin_Indices',
                                              data=bin_inds,
                                              dtype=np.uint32)
        h5_bin_freq = chan_grp.create_dataset('Bin_Frequencies',
                                              data=bin_freqs,
                                              dtype=np.float32)
        h5_bin_FFT = chan_grp.create_dataset('Bin_FFT',
                                             data=bin_FFT,
                                             dtype=np.complex64)
        # Noise floor should be of shape: (udvs_steps x 3 x positions)
        h5_noise_floor = chan_grp.create_dataset('Noise_Floor',
                                                 shape=(num_pix, num_actual_udvs_steps),
                                                 dtype=nf32,
                                                 chunks=(1, num_actual_udvs_steps))

        """ 
        ONLY ALLOCATING SPACE FOR MAIN DATA HERE!
        Chunk by each UDVS step - this makes it easy / quick to:
            1. read data for a single UDVS step from all pixels
            2. read an entire / multiple pixels at a time
        The only problem is that a typical UDVS step containing 50 steps occupies only 400 bytes.
        This is smaller than the recommended chunk sizes of 10,000 - 999,999 bytes
        meaning that the metadata would be very substantial.
        This assumption is fine since we almost do not handle any user defined cases
        """

        """
        New Method for chunking the Main_Data dataset.  Chunking is now done in N-by-N squares of UDVS steps by pixels.
        N is determined dinamically based on the dimensions of the dataset.  Currently it is set such that individual
        chunks are less than 10kB in size.
        
        Chris Smith -- csmith55@utk.edu
        """
        pos_dims = [Dimension('X', 'nm', num_cols), Dimension('Y', 'nm', num_rows)]

        # Create Spectroscopic Values and Spectroscopic Values Labels datasets
        spec_vals, spec_inds, spec_vals_labs, spec_vals_units, spec_vals_names = createSpecVals(UDVS_mat, spec_inds,
                                                                                                bin_freqs,
                                                                                                exec_bin_vec,
                                                                                                parm_dict, UDVS_labs,
                                                                                                UDVS_units)

        spec_dims = list()
        for row_ind, row_name in enumerate(spec_vals_labs):
            spec_dims.append(Dimension(row_name,
                                            spec_vals_units[row_ind],
                                            spec_vals[row_ind]))

        pixel_chunking = maxReadPixels(10240, num_pix * num_actual_udvs_steps,
                                       bins_per_step, np.dtype('complex64').itemsize)
        chunking = np.floor(np.sqrt(pixel_chunking))
        chunking = max(1, chunking)
        chunking = min(num_actual_udvs_steps, num_pix, chunking)
        self.h5_main = write_main_dataset(chan_grp, (num_pix, tot_bins), 'Raw_Data',
                                          'Piezoresponse', 'V',
                                          pos_dims, spec_dims,
                                          dtype=np.complex64,
                                          chunks=(chunking, chunking * bins_per_step),
                                          compression='gzip')

        self.mean_resp = np.zeros(shape=(self.ds_main.shape[1]), dtype=np.complex64)
        self.max_resp = np.zeros(shape=(self.ds_main.shape[0]), dtype=np.float32)
        self.min_resp = np.zeros(shape=(self.ds_main.shape[0]), dtype=np.float32)

        # Now read the raw data files:
        self._read_data(path_dict['read_real'], path_dict['read_imag'], parm_dict)
        self.h5_file.flush()

        generatePlotGroups(self.ds_main, self.mean_resp, folder_path, basename, self.max_resp,
                           self.min_resp, max_mem_mb=self.max_ram, spec_label=spec_label, show_plots=show_plots,
                           save_plots=save_plots, do_histogram=do_histogram)

        self.h5_file.close()

        return h5_path

    def _read_data(self, real_path, imag_path, parm_dict):
        """
        Reads the imaginary and real data files one pixel at a time andwrites to the H5 dataset.
        
        Inputs:
            real_path -- file path of the .dat file containing the real component of the data
            imag_path -- file path of the .dat file containing the imaginary component of the data
            parm_dict--dictionary of parameters for the experiment            
            
        Outputs: None
        """
        print('---- reading data one pixel at a time----------')

        num_pix = int(parm_dict['grid_num_rows']) * int(parm_dict['grid_num_cols'])
        # print 'Number of rows is: ', parm_dict['grid_num_rows']
        # print 'Number of cols is: ', parm_dict['grid_num_cols']
        # print 'Rows * cols is:', int(parm_dict['grid_num_rows'])*int(parm_dict['grid_num_cols'])

        if path.getsize(real_path) != path.getsize(imag_path):
            print('Sizes of real and imaginary files NOT matching!!!!')
        if 1.0 * path.getsize(real_path) % num_pix != 0:
            print('Incomplete dataset!!!')

        bytes_per_pix = path.getsize(real_path) / num_pix
        f_real = open(real_path, "rb")
        f_imag = open(imag_path, "rb")

        for pix_ind in range(num_pix):
            print('Reading pixel #{}, file position {}'.format(pix_ind, hex(pix_ind * bytes_per_pix)))
            pix_vec = np.fromstring(f_real.read(int(bytes_per_pix)), dtype='f') + \
                1j * np.fromstring(f_imag.read(int(bytes_per_pix)), dtype='f')

            # Make chronologically correct
            pix_mat = np.reshape(pix_vec, (parm_dict['BE_bins_per_read'],
                                           parm_dict['VS_steps_per_full_cycle'], parm_dict['BE_repeats']))
            pix_mat_temp = np.transpose(pix_mat, (1, 2, 0))
            pix_vec2 = np.reshape(pix_mat_temp, -1)

            # Calculate the mean, min, max
            self.max_resp[pix_ind] = np.max(np.abs(pix_vec2))
            self.min_resp[pix_ind] = np.min(np.abs(pix_vec2))
            self.mean_resp = (1 / (pix_ind + 1)) * (pix_vec2 + pix_ind * self.mean_resp)

            # Write to file now
            self.ds_main[pix_ind, :] = np.complex64(pix_vec2)
            self.h5_file.flush()

        f_real.close()
        f_imag.close()

        print('Finished writing data to .h5')

    @staticmethod
    def __readOldMatBEvecs(file_path):
        """
    Returns information about the excitation BE waveform present in the .mat file
    
    Inputs:
        filepath -- Absolute filepath of the .mat parameter file
    
    Outputs:
        Tuple -- (bin_inds, bin_w, bin_FFT, BE_wave, dc_amp_vec_full)\n
        bin_inds -- Bin indices\n
        bin_w -- Excitation bin Frequencies\n
        bin_FFT -- FFT of the BE waveform for the excited bins\n
        BE_wave -- Band Excitation waveform\n
        dc_amp_vec_full -- spectroscopic waveform. 
        This information will be necessary for fixing the UDVS for AC modulation for example
        """

        matread = loadmat(file_path, squeeze_me=True)
        BE_wave = matread['BE_wave_1']
        bin_inds = matread['bin_ind_s'] - 1  # Python base 0. note also _s, for this case
        bin_w = matread['bin_w']
        dc_amp_vec_full = matread['dc_amp_vec_full']
        FFT_full = np.fft.fftshift(np.fft.fft(BE_wave))
        bin_FFT = np.conjugate(FFT_full[bin_inds])

        return bin_inds, bin_w, bin_FFT, BE_wave, dc_amp_vec_full

    def _parse_file_path(self, data_filepath):
        """
        Returns the basename and a dictionary containing the absolute file paths for the
        real and imaginary data files, text and mat parameter files in a dictionary
        
        Parameters
        ----------
        data_filepath : str
            Absolute path of the real / imaginary data file (.dat)

        Returns
        -------
        basename : str
        path_dict : dict

        """
        (folder_path, basename) = path.split(data_filepath)
        (super_folder, basename) = path.split(folder_path)

        if basename.endswith('_c'):
            # Old old data format where the folder ended with a _d for some reason
            base_name = basename[:-2]

        """
        A single pair of real and imaginary files are / were generated for:
            BE-Line and BEPS (compiled version only generated out-of-field or 'read')
        Two pairs of real and imaginary files were generated for later BEPS datasets
            These have 'read' and 'write' prefixes to denote out or in field respectively
        """
        path_dict = dict()

        real_path = path.join(folder_path, base_name + '_sub_real.dat')
        imag_path = path.join(folder_path, base_name + '_sub_imag.dat')

        path_dict['read_real'] = real_path
        path_dict['read_imag'] = imag_path
        path_dict['old_mat_parms'] = data_filepath

        return basename, path_dict

    @staticmethod
    def __getParmsFromOldMat(file_path):
        """
        Formats parameters found in the old parameters .mat file into a dictionary
        as though the dataset had a parms.txt describing it
        
        Inputs:
            file_path -- absolute filepath of the .mat file containing the parameters
            
        Outputs -- dictionary containing parameters
        """
        parm_dict = dict()
        matread = loadmat(file_path, squeeze_me=True)

        parm_dict['IO_rate'] = str(int(matread['AO_rate'] / 1E+6)) + ' MHz'

        position_vec = matread['position_vec']
        parm_dict['grid_current_row'] = position_vec[0]
        parm_dict['grid_current_col'] = position_vec[1]
        parm_dict['grid_num_rows'] = int(position_vec[2])
        parm_dict['grid_num_cols'] = int(position_vec[3])

        if position_vec[0] != position_vec[1] or position_vec[2] != position_vec[3]:
            warn('WARNING: Incomplete dataset. Translation not guaranteed!')
            parm_dict['grid_num_rows'] = int(position_vec[0])  # set to number of present cols and rows
            parm_dict['grid_num_cols'] = int(position_vec[1])

        BE_parm_vec_1 = matread['BE_parm_vec_1']
        # Not required for translation but necessary to have
        if BE_parm_vec_1[0] == 3:
            parm_dict['BE_phase_content'] = 'chirp-sinc hybrid'
        else:
            parm_dict['BE_phase_content'] = 'Unknown'
        parm_dict['BE_center_frequency_[Hz]'] = BE_parm_vec_1[1]
        parm_dict['BE_band_width_[Hz]'] = BE_parm_vec_1[2]
        parm_dict['BE_amplitude_[V]'] = BE_parm_vec_1[3]
        parm_dict['BE_band_edge_trim'] = -1 * BE_parm_vec_1[6]  # 150 most likely
        parm_dict['BE_phase_variation'] = BE_parm_vec_1[5]  # 0.01 most likely
        parm_dict['BE_repeats'] = 2 ** int(BE_parm_vec_1[8])
        parm_dict['File_date_and_time'] = 0  # For now ignoring.
        parm_dict['BE_bins_per_read'] = matread['bins_per_band_s']
        assembly_parm_vec = matread['assembly_parm_vec']

        if assembly_parm_vec[2] == 0:
            parm_dict['VS_measure_in_field_loops'] = 'out-of-field'
        elif assembly_parm_vec[2] == 1:
            parm_dict['VS_measure_in_field_loops'] = 'in and out-of-field'
        else:
            parm_dict['VS_measure_in_field_loops'] = 'in-field'

        parm_dict['IO_Analog_Input_1'] = '+/- 10V, FFT'
        if assembly_parm_vec[3] == 0:
            parm_dict['IO_Analog_Input_2'] = 'off'
        else:
            parm_dict['IO_Analog_Input_2'] = '+/- 10V, FFT'

        # num_driving_bands = assembly_parm_vec[0] # 0 = 1, 1 = 2 bands
        # band_combination_order = assembly_parm_vec[1] # 0 parallel 1 series

        VS_parms = matread['SS_parm_vec']
        dc_amp_vec_full = matread['dc_amp_vec_full']

        VS_start_V = VS_parms[4]
        VS_start_loop_amp = VS_parms[5]
        VS_final_loop_amp = VS_parms[6]
        # VS_read_write_ratio = VS_parms[8] #1 <- SS_read_write_ratio

        parm_dict['VS_set_pulse_amplitude_[V]'] = VS_parms[9]  # 0 <- SS_set_pulse_amp
        parm_dict['VS_read_voltage_[V]'] = VS_parms[3]
        parm_dict['VS_steps_per_full_cycle'] = len(dc_amp_vec_full)  # VS_parms[7]
        parm_dict['VS_cycle_fraction'] = 'full'
        parm_dict['VS_cycle_phase_shift'] = 0
        parm_dict['VS_number_of_cycles'] = VS_parms[2]
        parm_dict['FORC_num_of_FORC_cycles'] = 1
        parm_dict['FORC_V_high1_[V]'] = 0
        parm_dict['FORC_V_high2_[V]'] = 0
        parm_dict['FORC_V_low1_[V]'] = 0
        parm_dict['FORC_V_low2_[V]'] = 0

        if VS_parms[0] == 0:
            parm_dict['VS_mode'] = 'DC modulation mode'
            parm_dict['VS_amplitude_[V]'] = 0.5 * (max(dc_amp_vec_full) -
                                                   min(dc_amp_vec_full))  # SS_max_offset_amplitude
            parm_dict['VS_offset_[V]'] = max(dc_amp_vec_full) + min(dc_amp_vec_full)
        elif VS_parms[0] == 1:
            # FORC
            parm_dict['VS_mode'] = 'DC modulation mode'
            parm_dict['VS_amplitude_[V]'] = 1  # VS_parms[1] # SS_max_offset_amplitude
            parm_dict['VS_offset_[V]'] = 0
            parm_dict['VS_number_of_cycles'] = 1
            parm_dict['FORC_num_of_FORC_cycles'] = VS_parms[2]
            parm_dict['FORC_V_high1_[V]'] = VS_start_V
            parm_dict['FORC_V_high2_[V]'] = VS_start_V
            parm_dict['FORC_V_low1_[V]'] = VS_start_V - VS_start_loop_amp
            parm_dict['FORC_V_low2_[V]'] = VS_start_V - VS_final_loop_amp
        elif VS_parms[0] == 2:
            # AC mode 
            parm_dict['VS_mode'] = 'AC modulation mode with time reversal'
            parm_dict['VS_amplitude_[V]'] = 0.5 * VS_final_loop_amp
            parm_dict[
                'VS_offset_[V]'] = 0  # this is not correct. Fix manually when it comes to UDVS generation?
        else:
            parm_dict['VS_mode'] = 'Custom'

        return parm_dict

    def __buildUDVSTable(self, parm_dict):
        """
        Generates the UDVS table using the parameters
        
        Inputs:
            parm_dict -- Dictionary of parameters present in the text files
        
        Outputs:
            tuple (labels, table)
            labels -- List of strings - Labels for columns in the UDVS table
            table -- UDVS data table
        """

        def translateVal(target, strvals, numvals):
            """
            Internal function - Interprets the provided value using the provided lookup table
            """

            if len(strvals) is not len(numvals):
                return None
            for strval, fltval in zip(strvals, numvals):
                if target == strval:
                    return fltval
            return None  # not found in list

        # Extract values from parm text file
        BE_signal_type = 1
        # This is necessary when normalzing the AI by the AO
        self.harmonic = BE_signal_type
        self.signal_type = BE_signal_type

        BE_amp = parm_dict['BE_amplitude_[V]']

        VS_amp = parm_dict['VS_amplitude_[V]']
        VS_offset = parm_dict['VS_offset_[V]']
        VS_steps = parm_dict['VS_steps_per_full_cycle']
        VS_cycles = parm_dict['VS_number_of_cycles']
        VS_fraction = translateVal(parm_dict['VS_cycle_fraction'], ['full', '1/2', '1/4', '3/4'], [1., 0.5, 0.25, 0.75])
        VS_shift = parm_dict['VS_cycle_phase_shift']
        if VS_shift is not 0:
            VS_shift = translateVal(VS_shift, ['1/4', '1/2', '3/4'], [0.25, 0.5, 0.75])
        VS_in_out_cond = translateVal(parm_dict['VS_measure_in_field_loops'],
                                      ['out-of-field', 'in-field', 'in and out-of-field'], [0, 1, 2])
        VS_ACDC_cond = translateVal(parm_dict['VS_mode'],
                                    ['DC modulation mode', 'AC modulation mode with time reversal',
                                     'load user defined VS Wave from file', 'current mode'],
                                    [0, 2, 3, 4])
        self.expt_type = VS_ACDC_cond
        FORC_cycles = parm_dict['FORC_num_of_FORC_cycles']
        FORC_A1 = parm_dict['FORC_V_high1_[V]']
        FORC_A2 = parm_dict['FORC_V_high2_[V]']
        FORC_B1 = parm_dict['FORC_V_low1_[V]']
        FORC_B2 = parm_dict['FORC_V_low2_[V]']

        # % build vector of voltage spectroscopy values

        if VS_ACDC_cond == 0 or VS_ACDC_cond == 4:  # DC voltage spectroscopy or current mode
            VS_amp_vec_1 = np.arange(0, 1 + 1 / (VS_steps / 4), 1 / (VS_steps / 4))
            VS_amp_vec_2 = np.flipud(VS_amp_vec_1[:-1])
            VS_amp_vec_3 = -VS_amp_vec_1[1:]
            VS_amp_vec_4 = VS_amp_vec_1[1:-1] - 1
            VS_amp_vec = VS_amp * (np.hstack((VS_amp_vec_1, VS_amp_vec_2, VS_amp_vec_3, VS_amp_vec_4)))
            VS_amp_vec = np.roll(VS_amp_vec,
                                 int(np.floor(VS_steps / VS_fraction * VS_shift)))  # apply phase shift to VS wave
            VS_amp_vec = VS_amp_vec[:int(np.floor(VS_steps * VS_fraction))]  # cut VS waveform
            VS_amp_vec = np.tile(VS_amp_vec, VS_cycles)  # repeat VS waveform
            VS_amp_vec = VS_amp_vec + VS_offset

        if FORC_cycles > 1:
            VS_amp_vec = VS_amp_vec / np.max(np.abs(VS_amp_vec))
            FORC_cycle_vec = np.arange(0, FORC_cycles + 1, FORC_cycles / (FORC_cycles - 1))
            FORC_A_vec = FORC_cycle_vec * (FORC_A2 - FORC_A1) / FORC_cycles + FORC_A1
            FORC_B_vec = FORC_cycle_vec * (FORC_B2 - FORC_B1) / FORC_cycles + FORC_B1
            FORC_amp_vec = (FORC_A_vec - FORC_B_vec) / 2
            FORC_off_vec = (FORC_A_vec + FORC_B_vec) / 2

            VS_amp_mat = np.tile(VS_amp_vec, [FORC_cycles, 1])
            FORC_amp_mat = np.tile(FORC_amp_vec, [len(VS_amp_vec), 1]).transpose()
            FORC_off_mat = np.tile(FORC_off_vec, [len(VS_amp_vec), 1]).transpose()
            VS_amp_mat = VS_amp_mat * FORC_amp_mat + FORC_off_mat
            VS_amp_vec = VS_amp_mat.reshape(int(FORC_cycles * VS_cycles * VS_fraction * VS_steps))

        BE_repeats = parm_dict['BE_repeats']
        total_steps = len(VS_amp_vec) * BE_repeats  # Needed for relaxation datasets

        # % Build UDVS table:
        if VS_ACDC_cond is 0 or VS_ACDC_cond is 4:  # relaxation measurements

            num_VS_steps = total_steps * 2  # To account for IF and OOF

            UD_VS_table_label = ['step_num', 'dc_offset', 'ac_amp', 'wave_type', 'wave_mod', 'in-field', 'out-of-field']
            UD_VS_table = np.zeros(shape=(num_VS_steps, 7), dtype=np.float32)
            UD_VS_table_unit = ['', 'V', 'A', '', '', 'V', 'V']

            UD_VS_table[:, 0] = np.arange(0, num_VS_steps)  # Python base 0
            for step_num in np.arange(0, VS_steps):
                step_values = (np.arange(int(step_num) * int(BE_repeats) * 2,
                                         (int(step_num) + 1) * int(BE_repeats) * 2))
                UD_VS_table[step_values, 1] = VS_amp_vec[step_num]

            BE_IF_switch = np.abs(np.imag(np.exp(1j * np.pi / 2 * np.arange(1, num_VS_steps + 1))))
            BE_OF_switch = np.abs(np.real(np.exp(1j * np.pi / 2 * np.arange(1, num_VS_steps + 1))))

            if VS_in_out_cond is 0:  # out of field only
                UD_VS_table[:, 2] = BE_amp * BE_OF_switch
            elif VS_in_out_cond is 1:  # in field only
                UD_VS_table[:, 2] = BE_amp * BE_IF_switch
            elif VS_in_out_cond is 2:  # both in and out of field
                UD_VS_table[:, 2] = BE_amp * np.ones(num_VS_steps)

            UD_VS_table[:, 3] = np.ones(num_VS_steps)  # wave type
            UD_VS_table[:, 4] = np.ones(num_VS_steps) * BE_signal_type  # wave mod

            UD_VS_table[:, 5] = float('NaN') * np.ones(num_VS_steps)
            UD_VS_table[:, 6] = float('NaN') * np.ones(num_VS_steps)

            UD_VS_table[BE_IF_switch == 1, 5] = UD_VS_table[BE_IF_switch == 1, 1]
            UD_VS_table[BE_OF_switch == 1, 6] = UD_VS_table[BE_IF_switch == 1, 1]

        return UD_VS_table_label, UD_VS_table_unit, UD_VS_table
