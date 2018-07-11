# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:24:12 2015

@author: Suhas Somnath, Stephen Jesse
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, listdir, remove
from warnings import warn
import h5py
import numpy as np
from scipy.io.matlab import loadmat  # To load parameters stored in Matlab .mat file

from .df_utils.be_utils import trimUDVS, getSpectroscopicParmLabel, parmsToDict, generatePlotGroups, \
    createSpecVals, requires_conjugate, nf32
from pyUSID.io.translator import Translator, generate_dummy_main_parms
from pyUSID.io.write_utils import INDICES_DTYPE, VALUES_DTYPE, Dimension, calc_chunks
from pyUSID.io.hdf_utils import write_ind_val_dsets, write_main_dataset, write_region_references, \
    create_indexed_group, write_simple_attrs, write_book_keeping_attrs, copy_attributes,\
    write_reduced_spec_dsets
from pyUSID.io.usi_data import USIDataset
from pyUSID.io.io_utils import get_available_memory

class BEodfTranslator(Translator):
    """
    Translates either the Band Excitation (BE) scan or Band Excitation 
    Polarization Switching (BEPS) data format from the old data format(s) to .h5
    """

    def __init__(self, *args, **kwargs):
        super(BEodfTranslator, self).__init__(*args, **kwargs)
        self.h5_raw = None
        self.num_rand_spectra = kwargs.pop('num_rand_spectra', 1000)
        self.FFT_BE_wave = None
        self.signal_type = None
        self.expt_type = None

    def translate(self, file_path, show_plots=True, save_plots=True, do_histogram=False, verbose=False):
        """
        Translates .dat data file(s) to a single .h5 file
        
        Parameters
        -------------
        file_path : String / Unicode
            Absolute file path for one of the data files. 
            It is assumed that this file is of the OLD data format.
        show_plots : (optional) Boolean
            Whether or not to show intermediate plots
        save_plots : (optional) Boolean
            Whether or not to save plots to disk
        do_histogram : (optional) Boolean
            Whether or not to construct histograms to visualize data quality. Note - this takes a fair amount of time
        verbose : (optional) Boolean
            Whether or not to print statements
            
        Returns
        ----------
        h5_path : String / Unicode
            Absolute path of the resultant .h5 file
        """
        file_path = path.abspath(file_path)
        (folder_path, basename) = path.split(file_path)
        (basename, path_dict) = self._parse_file_path(file_path)

        h5_path = path.join(folder_path, basename + '.h5')
        tot_bins_multiplier = 1
        udvs_denom = 2

        if 'parm_txt' in path_dict.keys():
            (isBEPS, parm_dict) = parmsToDict(path_dict['parm_txt'])
        elif 'old_mat_parms' in path_dict.keys():
            parm_dict = self.__get_parms_from_old_mat(path_dict['old_mat_parms'])
            if parm_dict['VS_steps_per_full_cycle']==0: isBEPS=False
            else: isBEPS=True
        else:
            raise IOError('No parameters file found! Cannot translate this dataset!')

        ignored_plt_grps = []
        if isBEPS:
            parm_dict['data_type'] = 'BEPSData'

            field_mode = parm_dict['VS_measure_in_field_loops']
            std_expt = parm_dict['VS_mode'] != 'load user defined VS Wave from file'

            if not std_expt:
                raise ValueError('This translator does not handle user defined voltage spectroscopy')

            spec_label = getSpectroscopicParmLabel(parm_dict['VS_mode'])

            if parm_dict['VS_mode'] in ['DC modulation mode', 'current mode']:
                if field_mode == 'in and out-of-field':
                    tot_bins_multiplier = 2
                    udvs_denom = 1
                else:
                    if field_mode == 'out-of-field':
                        ignored_plt_grps = ['in-field']
                    else:
                        ignored_plt_grps = ['out-of-field']
            else:
                tot_bins_multiplier = 1
                udvs_denom = 1

        else:
            spec_label = 'None'
            parm_dict['data_type'] = 'BELineData'

        # Check file sizes:
        if 'read_real' in path_dict.keys():
            real_size = path.getsize(path_dict['read_real'])
            imag_size = path.getsize(path_dict['read_imag'])
        else:
            real_size = path.getsize(path_dict['write_real'])
            imag_size = path.getsize(path_dict['write_imag'])

        if real_size != imag_size:
            raise ValueError("Real and imaginary file sizes DON'T match!. Ending")

        #Check here if a second channel for current is present
        # Look for the file containing the current data

        file_names = listdir(folder_path)
        aux_files = []
        current_data_exists = False
        for fname in file_names:
            if 'AI2' in fname:
                if 'write' in fname:
                    current_file = path.join(folder_path, fname)
                    current_data_exists=True
                aux_files.append(path.join(folder_path, fname))

        add_pix = False
        num_rows = int(parm_dict['grid_num_rows'])
        num_cols = int(parm_dict['grid_num_cols'])
        num_pix = num_rows * num_cols
        tot_bins = real_size / (num_pix * 4)
        # Check for case where only a single pixel is missing.
        check_bins = real_size / ((num_pix - 1) * 4)

        if tot_bins % 1 and check_bins % 1:
            raise ValueError('Aborting! Some parameter appears to have changed in-between')
        elif not tot_bins % 1:
            # Everything's ok
            pass
        elif not check_bins % 1:
            tot_bins = check_bins
            warn('Warning:  A pixel seems to be missing from the data.  File will be padded with zeros.')
            add_pix = True

        tot_bins = int(tot_bins) * tot_bins_multiplier

        if 'parm_mat' in path_dict.keys():
            (bin_inds, bin_freqs, bin_FFT, ex_wfm) = self.__read_parms_mat(path_dict['parm_mat'], isBEPS)
        elif 'old_mat_parms' in path_dict.keys():
            (bin_inds, bin_freqs, bin_FFT, ex_wfm, dc_amp_vec) = self.__read_old_mat_be_vecs(path_dict['old_mat_parms'])
        else:
            band_width = parm_dict['BE_band_width_[Hz]'] * (0.5 - parm_dict['BE_band_edge_trim'])
            st_f = parm_dict['BE_center_frequency_[Hz]'] - band_width
            en_f = parm_dict['BE_center_frequency_[Hz]'] + band_width
            bin_freqs = np.linspace(st_f, en_f, tot_bins, dtype=np.float32)

            warn('No parms .mat file found.... Filling dummy values into ancillary datasets.')
            bin_inds = np.zeros(shape=tot_bins, dtype=np.int32)
            bin_FFT = np.zeros(shape=tot_bins, dtype=np.complex64)
            ex_wfm = np.zeros(shape=100, dtype=np.float32)

        # Forcing standardized datatypes:
        bin_inds = np.int32(bin_inds)
        bin_freqs = np.float32(bin_freqs)
        bin_FFT = np.complex64(bin_FFT)
        ex_wfm = np.float32(ex_wfm)

        self.FFT_BE_wave = bin_FFT

        if isBEPS:
            (UDVS_labs, UDVS_units, UDVS_mat) = self.__build_udvs_table(parm_dict)

            #             Remove the unused plot group columns before proceeding:
            (UDVS_mat, UDVS_labs, UDVS_units) = trimUDVS(UDVS_mat, UDVS_labs, UDVS_units, ignored_plt_grps)

            old_spec_inds = np.zeros(shape=(2, tot_bins), dtype=INDICES_DTYPE)

            #             Will assume that all excitation waveforms have same number of bins
            num_actual_udvs_steps = UDVS_mat.shape[0] / udvs_denom
            bins_per_step = tot_bins / num_actual_udvs_steps

            if bins_per_step % 1:
                print('UDVS mat shape: {}, total bins: {}, bins per step: {}'.format(UDVS_mat.shape, tot_bins,
                                                                                     bins_per_step))
                raise ValueError('Non integer number of bins per step!')

            bins_per_step = int(bins_per_step)
            num_actual_udvs_steps = int(num_actual_udvs_steps)

            stind = 0
            for step_index in range(UDVS_mat.shape[0]):
                if UDVS_mat[step_index, 2] < 1E-3:  # invalid AC amplitude
                    continue
                # Bin step
                old_spec_inds[0, stind:stind + bins_per_step] = np.arange(bins_per_step, dtype=INDICES_DTYPE)
                # UDVS step
                old_spec_inds[1, stind:stind + bins_per_step] = step_index * np.ones(bins_per_step, dtype=INDICES_DTYPE)
                stind += bins_per_step
            del stind, step_index

        else:  # BE Line
            self.signal_type = 1
            self.expt_type = 1  # Stephen has not used this index for some reason
            num_actual_udvs_steps = 1
            bins_per_step = tot_bins
            UDVS_labs = ['step_num', 'dc_offset', 'ac_amp', 'wave_type', 'wave_mod', 'be-line']
            UDVS_units = ['', 'V', 'A', '', '', '']
            UDVS_mat = np.array([1, 0, parm_dict['BE_amplitude_[V]'], 1, 1, 1],
                                dtype=np.float32).reshape(1, len(UDVS_labs))

            old_spec_inds = np.vstack((np.arange(tot_bins, dtype=INDICES_DTYPE),
                                       np.zeros(tot_bins, dtype=INDICES_DTYPE)))

        # Some very basic information that can help the processing / analysis crew
        parm_dict['num_bins'] = tot_bins
        parm_dict['num_pix'] = num_pix
        parm_dict['num_udvs_steps'] = num_actual_udvs_steps
        parm_dict['num_steps'] = num_actual_udvs_steps

        udvs_slices = dict()
        for col_ind, col_name in enumerate(UDVS_labs):
            udvs_slices[col_name] = (slice(None), slice(col_ind, col_ind + 1))

        # Need to add the Bin Waveform type - infer from UDVS        
        exec_bin_vec = self.signal_type * np.ones(len(bin_inds), dtype=np.int32)

        if self.expt_type == 2:
            # Need to double the vectors:
            exec_bin_vec = np.hstack((exec_bin_vec, -1 * exec_bin_vec))
            bin_inds = np.hstack((bin_inds, bin_inds))
            bin_freqs = np.hstack((bin_freqs, bin_freqs))
            # This is wrong but I don't know what else to do
            bin_FFT = np.hstack((bin_FFT, bin_FFT))

        # Create Spectroscopic Values and Spectroscopic Values Labels datasets
        # This is an old and legacy way of doing things. Ideally, all we would need ot do is just get the unit values
        spec_vals, spec_inds, spec_vals_labs, spec_vals_units, spec_vals_labs_names = createSpecVals(UDVS_mat,
                                                                                                     old_spec_inds,
                                                                                                     bin_freqs,
                                                                                                     exec_bin_vec,
                                                                                                     parm_dict,
                                                                                                     UDVS_labs,
                                                                                                     UDVS_units)
        # Not sure what is happening here but this should work.
        spec_dim_dict = dict()
        for entry in spec_vals_labs_names:
            spec_dim_dict[entry[0] + '_parameters'] = entry[1]

        spec_vals_slices = dict()

        for row_ind, row_name in enumerate(spec_vals_labs):
            spec_vals_slices[row_name] = (slice(row_ind, row_ind + 1), slice(None))

        if path.exists(h5_path):
            remove(h5_path)

        # First create the file
        h5_f = h5py.File(h5_path)

        # Then write root level attributes
        global_parms = generate_dummy_main_parms()
        global_parms['grid_size_x'] = parm_dict['grid_num_cols']
        global_parms['grid_size_y'] = parm_dict['grid_num_rows']
        try:
            global_parms['experiment_date'] = parm_dict['File_date_and_time']
        except KeyError:
            global_parms['experiment_date'] = '1:1:1'

        # assuming that the experiment was completed:
        global_parms['current_position_x'] = parm_dict['grid_num_cols'] - 1
        global_parms['current_position_y'] = parm_dict['grid_num_rows'] - 1
        global_parms['data_type'] = parm_dict['data_type']
        global_parms['translator'] = 'ODF'
        write_simple_attrs(h5_f, global_parms)
        write_book_keeping_attrs(h5_f)

        # Then create the measurement group
        h5_meas_group = create_indexed_group(h5_f, 'Measurement')

        # Write attributes at the measurement group level
        write_simple_attrs(h5_meas_group, parm_dict)

        # Create the Channel group
        h5_chan_grp = create_indexed_group(h5_meas_group, 'Channel')

        # Write channel group attributes
        write_simple_attrs(h5_chan_grp, {'Channel_Input': 'IO_Analog_Input_1',
                                         'channel_type': 'BE'})

        # Now the datasets!
        h5_chan_grp.create_dataset('Excitation_Waveform', data=ex_wfm)

        h5_udvs = h5_chan_grp.create_dataset('UDVS', data=UDVS_mat)
        write_region_references(h5_udvs, udvs_slices, add_labels_attr=True, verbose=verbose)
        write_simple_attrs(h5_udvs, {'units': UDVS_units}, verbose=verbose)
        
        # ds_udvs_labs = MicroDataset('UDVS_Labels',np.array(UDVS_labs))
        h5_chan_grp.create_dataset('UDVS_Indices', data=old_spec_inds[1])

        # ds_spec_labs = MicroDataset('Spectroscopic_Labels',np.array(['Bin','UDVS_Step']))
        h5_chan_grp.create_dataset('Bin_Step', data=np.arange(bins_per_step, dtype=INDICES_DTYPE),
                                   dtype=INDICES_DTYPE)

        h5_chan_grp.create_dataset('Bin_Indices', data=bin_inds, dtype=INDICES_DTYPE)
        h5_chan_grp.create_dataset('Bin_Frequencies', data=bin_freqs)
        h5_chan_grp.create_dataset('Bin_FFT', data=bin_FFT)
        h5_chan_grp.create_dataset('Bin_Wfm_Type', data=exec_bin_vec)

        pos_dims = [Dimension('X', 'm', np.arange(num_cols)), Dimension('Y', 'm', np.arange(num_rows))]
        h5_pos_ind, h5_pos_val = write_ind_val_dsets(h5_chan_grp, pos_dims, is_spectral=False, verbose=verbose)

        h5_spec_inds = h5_chan_grp.create_dataset('Spectroscopic_Indices', data=spec_inds, dtype=INDICES_DTYPE)        
        h5_spec_vals = h5_chan_grp.create_dataset('Spectroscopic_Values', data=np.array(spec_vals), dtype=VALUES_DTYPE)
        for dset in [h5_spec_inds, h5_spec_vals]:
            write_region_references(dset, spec_vals_slices, add_labels_attr=True, verbose=verbose)
            write_simple_attrs(dset, {'units': spec_vals_units}, verbose=verbose)
            write_simple_attrs(dset, spec_dim_dict)

        # Noise floor should be of shape: (udvs_steps x 3 x positions)
        h5_chan_grp.create_dataset('Noise_Floor', (num_pix, num_actual_udvs_steps), dtype=nf32,
                                   chunks=(1, num_actual_udvs_steps))

        """
        New Method for chunking the Main_Data dataset.  Chunking is now done in N-by-N squares
        of UDVS steps by pixels.  N is determined dynamically based on the dimensions of the
        dataset.  Currently it is set such that individual chunks are less than 10kB in size.

        Chris Smith -- csmith55@utk.edu
        """
        BEPS_chunks = calc_chunks([num_pix, tot_bins],
                                  np.complex64(0).itemsize,
                                  unit_chunks=(1, bins_per_step))
        self.h5_raw = write_main_dataset(h5_chan_grp, (num_pix, tot_bins), 'Raw_Data', 'Piezoresponse', 'V', None, None,
                                         dtype=np.complex64, chunks=BEPS_chunks, compression='gzip',
                                         h5_pos_inds=h5_pos_ind, h5_pos_vals=h5_pos_val, h5_spec_inds=h5_spec_inds,
                                         h5_spec_vals=h5_spec_vals, verbose=verbose)

        self._read_data(UDVS_mat, parm_dict, path_dict, real_size, isBEPS, add_pix)

        generatePlotGroups(self.h5_raw, self.mean_resp, folder_path, basename,
                           self.max_resp, self.min_resp, max_mem_mb=self.max_ram,
                           spec_label=spec_label, show_plots=show_plots, save_plots=save_plots,
                           do_histogram=do_histogram, debug=verbose)

        self.h5_raw = USIDataset(self.h5_raw)

        # Go ahead and read the current data in the second (current) channel
        if current_data_exists:                     #If a .dat file matches
            self._read_secondary_channel(h5_meas_group, aux_files)

        h5_f.close()

        return h5_path


    def _read_data(self, UDVS_mat, parm_dict, path_dict, real_size, isBEPS, add_pix):
        """
        Checks if the data is BEPS or BELine and calls the correct function to read the data from
        file

        Parameters
        ----------
        UDVS_mat : numpy.ndarray of float
            UDVS table
        parm_dict : dict
            Experimental parameters
        path_dict : dict
            Dictionary of data files to be read
        real_size : dict
            Size of each data file
        isBEPS : boolean
            Is the data BEPS
        add_pix : boolean
            Does the reader need to add extra pixels to the end of the dataset

        Returns
        -------
        None
        """
        # Now read the raw data files:
        if not isBEPS:
            # Do this for all BE-Line (always small enough to read in one shot)
            self.__quick_read_data(path_dict['read_real'], path_dict['read_imag'], parm_dict['num_udvs_steps'])
        elif real_size < self.max_ram and parm_dict['VS_measure_in_field_loops'] == 'out-of-field':
            # Do this for out-of-field BEPS ONLY that is also small (256 MB)
            self.__quick_read_data(path_dict['read_real'], path_dict['read_imag'], parm_dict['num_udvs_steps'])
        elif real_size < self.max_ram and parm_dict['VS_measure_in_field_loops'] == 'in-field':
            # Do this for in-field only
            self.__quick_read_data(path_dict['write_real'], path_dict['write_imag'], parm_dict['num_udvs_steps'])
        else:
            # Large BEPS datasets OR those with in-and-out of field
            self.__read_beps_data(path_dict, UDVS_mat.shape[0], parm_dict['VS_measure_in_field_loops'], add_pix)
        self.h5_raw.file.flush()

    def __read_beps_data(self, path_dict, udvs_steps, mode, add_pixel=False):
        """
        Reads the imaginary and real data files pixelwise and writes to the H5 file 
        
        Parameters 
        --------------------
        path_dict : dictionary
            Dictionary containing the absolute paths of the real and imaginary data files
        udvs_steps : unsigned int
            Number of UDVS steps
        mode : String / Unicode
            'in-field', 'out-of-field', or 'in and out-of-field'
        add_pixel : boolean. (Optional; default is False)
            If an empty pixel worth of data should be written to the end             
        
        Returns 
        -------------------- 
        None
        """

        print('---- reading pixel-by-pixel ----------')

        bytes_per_pix = self.h5_raw.shape[1] * 4
        step_size = self.h5_raw.shape[1] / udvs_steps

        if mode == 'out-of-field':
            parsers = [BEodfParser(path_dict['read_real'], path_dict['read_imag'],
                                   self.h5_raw.shape[0], bytes_per_pix)]
        elif mode == 'in-field':
            parsers = [BEodfParser(path_dict['write_real'], path_dict['write_imag'],
                                   self.h5_raw.shape[0], bytes_per_pix)]
        elif mode == 'in and out-of-field':
            # each file will only have half the udvs steps:
            if 0.5 * udvs_steps % 1:
                raise ValueError('Odd number of UDVS')

            udvs_steps = int(0.5 * udvs_steps)
            # be careful - each pair contains only half the necessary bins - so read half
            parsers = [BEodfParser(path_dict['write_real'], path_dict['write_imag'],
                                   self.h5_raw.shape[0], int(bytes_per_pix / 2)),
                       BEodfParser(path_dict['read_real'], path_dict['read_imag'],
                                   self.h5_raw.shape[0], int(bytes_per_pix / 2))]

            if step_size % 1:
                raise ValueError('strange number of bins per UDVS step. Exiting')

            step_size = int(step_size)

        rand_spectra = self.__get_random_spectra(parsers, self.h5_raw.shape[0], udvs_steps, step_size,
                                                 num_spectra=self.num_rand_spectra)
        take_conjugate = requires_conjugate(rand_spectra)

        self.mean_resp = np.zeros(shape=(self.h5_raw.shape[1]), dtype=np.complex64)
        self.max_resp = np.zeros(shape=(self.h5_raw.shape[0]), dtype=np.float32)
        self.min_resp = np.zeros(shape=(self.h5_raw.shape[0]), dtype=np.float32)

        numpix = self.h5_raw.shape[0]
        """ 
        Don't try to do the last step if a pixel is missing.   
        This will be handled after the loop. 
        """
        if add_pixel:
            numpix -= 1

        for pix_indx in range(numpix):
            if self.h5_raw.shape[0] > 5:
                if pix_indx % int(round(self.h5_raw.shape[0] / 10)) == 0:
                    print('Reading... {} complete'.format(round(100 * pix_indx / self.h5_raw.shape[0])))

            # get the raw stream from each parser
            pxl_data = list()
            for prsr in parsers:
                pxl_data.append(prsr.read_pixel())

            # interleave if both in and out of field
            # we are ignoring user defined possibilities...
            if mode == 'in and out-of-field':
                in_fld = pxl_data[0]
                out_fld = pxl_data[1]

                in_fld_2 = in_fld.reshape(udvs_steps, step_size)
                out_fld_2 = out_fld.reshape(udvs_steps, step_size)
                raw_mat = np.empty((udvs_steps * 2, step_size), dtype=out_fld.dtype)
                raw_mat[0::2, :] = in_fld_2
                raw_mat[1::2, :] = out_fld_2
                raw_vec = raw_mat.reshape(in_fld.size + out_fld.size).transpose()
            else:
                raw_vec = pxl_data[0]  # only one parser
            self.max_resp[pix_indx] = np.max(np.abs(raw_vec))
            self.min_resp[pix_indx] = np.min(np.abs(raw_vec))
            self.mean_resp = (1 / (pix_indx + 1)) * (raw_vec + pix_indx * self.mean_resp)

            if take_conjugate:
                raw_vec = np.conjugate(raw_vec)
            self.h5_raw[pix_indx, :] = np.complex64(raw_vec[:])
            self.h5_raw.file.flush()

        # Add zeros to main_data for the missing pixel. 
        if add_pixel:
            self.h5_raw[-1, :] = 0 + 0j

        print('---- Finished reading files -----')

    def __quick_read_data(self, real_path, imag_path, udvs_steps):
        """
        Returns information about the excitation BE waveform present in the .mat file

        Parameters
        -----------
        real_path : String / Unicode
            Absolute file path of the real data file
        imag_path : String / Unicode
            Absolute file path of the real data file
        udvs_steps : unsigned int
            Number of UDVS steps
        """
        print('---- reading all data at once ----------')

        parser = BEodfParser(real_path, imag_path, self.h5_raw.shape[0], self.h5_raw.shape[1] * 4)

        step_size = self.h5_raw.shape[1] / udvs_steps
        rand_spectra = self.__get_random_spectra([parser], self.h5_raw.shape[0], udvs_steps, step_size,
                                                 num_spectra=self.num_rand_spectra)
        take_conjugate = requires_conjugate(rand_spectra)
        raw_vec = parser.read_all_data()
        if take_conjugate:
            print('Taking conjugate to ensure positive Quality factors')
            raw_vec = np.conjugate(raw_vec)

        raw_mat = raw_vec.reshape(self.h5_raw.shape[0], self.h5_raw.shape[1])

        # Write to the h5 dataset:
        self.mean_resp = np.mean(raw_mat, axis=0)
        self.max_resp = np.amax(np.abs(raw_mat), axis=0)
        self.min_resp = np.amin(np.abs(raw_mat), axis=0)
        self.h5_raw[:, :] = np.complex64(raw_mat)
        self.h5_raw.file.flush()

        print('---- Finished reading files -----')

    def _parse_file_path(self, data_filepath):
        """
        Returns the basename and a dictionary containing the absolute file paths for the
        real and imaginary data files, text and mat parameter files in a dictionary
        
        Parameters 
        --------------------
        data_filepath: String / Unicode
            Absolute path of any file in the same directory as the .dat files
        
        Returns 
        --------------------
        basename : String / Unicode
            Basename of the dataset      
        path_dict : Dictionary
            Dictionary containing absolute paths of all necessary data and parameter files
        """
        (folder_path, basename) = path.split(data_filepath)
        (super_folder, basename) = path.split(folder_path)

        if basename.endswith('_d') or basename.endswith('_c'):
            # Old old data format where the folder ended with a _d or _c to denote a completed spectroscopic run
            basename = basename[:-2]
        """
        A single pair of real and imaginary files are / were generated for:
            BE-Line and BEPS (compiled version only generated out-of-field or 'read')
        Two pairs of real and imaginary files were generated for later BEPS datasets
            These have 'read' and 'write' prefixes to denote out or in field respectively
        """
        path_dict = dict()

        for file_name in listdir(folder_path):
            abs_path = path.join(folder_path, file_name)
            if file_name.endswith('.txt') and file_name.find('parm') > 0:
                path_dict['parm_txt'] = abs_path
            elif file_name.find('.mat') > 0:
                if file_name.find('more_parms') > 0:
                    path_dict['parm_mat'] = abs_path
                elif file_name == (basename + '.mat'):
                    path_dict['old_mat_parms'] = abs_path
            elif file_name.endswith('.dat'):
                # Need to account for the second AI channel here
                file_tag = 'read'
                if file_name.find('write') > 0:
                    file_tag = 'write'
                if file_name.find('real') > 0:
                    file_tag += '_real'
                elif file_name.find('imag') > 0:
                    file_tag += '_imag'
                path_dict[file_tag] = abs_path

        return basename, path_dict

    def _read_secondary_channel(self, h5_meas_group, aux_file_path):
        """
        Reads secondary channel stored in AI .mat file
        Currently works for in-field measurements only, but should be updated to
        include both in and out of field measurements

        Parameters
        -----------
        h5_meas_group : h5 group
            Reference to the Measurement group
        aux_file_path : String / Unicode
            Absolute file path of the secondary channel file.
        """
        print('---- Reading Secondary Channel  ----------')
        if len(aux_file_path)>1:
            print('Detected multiple files, assuming in and out of field')
            aux_file_paths = aux_file_path
        else:
            aux_file_paths = list(aux_file_path)

        freq_index = self.h5_raw.spec_dim_labels.index('Frequency')
        num_pix = self.h5_raw.shape[0]
        spectral_len = 1

        for i in range(len(self.h5_raw.spec_dim_sizes)):
            if i == freq_index:
                continue
            spectral_len = spectral_len * self.h5_raw.spec_dim_sizes[i]

        #num_forc_cycles = self.h5_raw.spec_dim_sizes[self.h5_raw.spec_dim_labels.index("FORC")]
        #num_dc_steps =  self.h5_raw.spec_dim_sizes[self.h5_raw.spec_dim_labels.index("DC_Offset")]

        # create a new channel
        h5_current_channel_group = create_indexed_group(h5_meas_group, 'Channel')

        # Copy attributes from the main channel
        copy_attributes(self.h5_raw.parent, h5_current_channel_group)

        # Modify attributes that are different
        write_simple_attrs(h5_current_channel_group, {'Channel_Input': 'IO_Analog_Input_2',
                                                      'channel_type': 'Current'}, verbose=True)

        #Get the reduced dimensions
        h5_current_spec_inds, h5_current_spec_values = write_reduced_spec_dsets(h5_current_channel_group,
                                                        self.h5_raw.h5_spec_inds,
                                                        self.h5_raw.h5_spec_vals, 'Frequency')


        h5_current_main = write_main_dataset(h5_current_channel_group,  # parent HDF5 group
                                             (num_pix, spectral_len),  # shape of Main dataset
                                             'Raw_Data',  # Name of main dataset
                                             'Current',  # Physical quantity contained in Main dataset
                                             'nA',  # Units for the physical quantity
                                             None,  # Position dimensions
                                             None,  # Spectroscopic dimensions
                                             h5_pos_inds=self.h5_raw.h5_pos_inds,
                                             h5_pos_vals=self.h5_raw.h5_pos_vals,
                                             h5_spec_inds=h5_current_spec_inds,
                                             h5_spec_vals=h5_current_spec_values,
                                             dtype=np.float32,  # data type / precision
                                             main_dset_attrs={'IO_rate': 4E+6, 'Amplifier_Gain': 9})

        # Now calculate the number of positions that can be stored in memory in one go.
        b_per_position = np.float32(0).itemsize * spectral_len

        max_pos_per_read = int(np.floor((get_available_memory()) / b_per_position))

        # if self._verbose:
        print('Allowed to read {} pixels per chunk'.format(max_pos_per_read))

        #Open the read and write files and write them to the hdf5 file
        for aux_file in aux_file_paths:
            if 'write' in aux_file:
                infield = True
            else:
                infield=False

            cur_file = open(aux_file, "rb")

            start_pix = 0

            while start_pix < num_pix:
                end_pix = min(num_pix, start_pix + max_pos_per_read)

                # TODO: Fix for when it won't fit in memory.

                #if max_pos_per_read * b_per_position > num_pix * b_per_position:
                cur_data = np.frombuffer(cur_file.read(), dtype='f')
                #else:
                #cur_data = np.frombuffer(cur_file.read(max_pos_per_read * b_per_position), dtype='f')

                cur_data = cur_data.reshape(end_pix - start_pix, spectral_len//2)

                # Write to h5
                if infield:
                    h5_current_main[start_pix:end_pix, ::2] = cur_data
                else:
                    h5_current_main[start_pix:end_pix, 1::2] = cur_data
                start_pix = end_pix


    @staticmethod
    def __read_old_mat_be_vecs(file_path):
        """
        Returns information about the excitation BE waveform present in the 
        more parms.mat file
        
        Parameters 
        --------------------
        filepath : String or unicode
            Absolute filepath of the .mat parameter file
        
        Returns 
        --------------------
        bin_inds : 1D numpy unsigned int array
            Indices of the excited and measured frequency bins
        bin_w : 1D numpy float array
            Excitation bin Frequencies
        bin_FFT : 1D numpy complex array
            FFT of the BE waveform for the excited bins
        BE_wave : 1D numpy float array
            Band Excitation waveform
        dc_amp_vec_full : 1D numpy float array
            spectroscopic waveform. 
            This information will be necessary for fixing the UDVS for AC modulation for example
        """
        matread = loadmat(file_path, squeeze_me=True)
        BE_wave = matread['BE_wave']
        bin_inds = matread['bin_ind'] - 1  # Python base 0
        bin_w = matread['bin_w']
        dc_amp_vec_full = matread['dc_amp_vec_full']
        FFT_full = np.fft.fftshift(np.fft.fft(BE_wave))
        bin_FFT = np.conjugate(FFT_full[bin_inds])
        return bin_inds, bin_w, bin_FFT, BE_wave, dc_amp_vec_full

    @staticmethod
    def __get_parms_from_old_mat(file_path):
        """
        Formats parameters found in the old parameters .mat file into a dictionary
        as though the dataset had a parms.txt describing it
        
        Parameters 
        --------------------
        file_path : Unicode / String
            absolute filepath of the .mat file containing the parameters
            
        Returns 
        --------------------
        parm_dict : dictionary
            Parameters describing experiment
        """
        parm_dict = dict()
        matread = loadmat(file_path, squeeze_me=True)

        parm_dict['IO_rate'] = str(int(matread['AO_rate'] / 1E+6)) + ' MHz'

        position_vec = matread['position_vec']
        parm_dict['grid_current_row'] = position_vec[0]
        parm_dict['grid_current_col'] = position_vec[1]
        parm_dict['grid_num_rows'] = position_vec[2]
        parm_dict['grid_num_cols'] = position_vec[3]

        if position_vec[0] != position_vec[1] or position_vec[2] != position_vec[3]:
            warn('WARNING: Incomplete dataset. Translation not guaranteed!')
            parm_dict['grid_num_rows'] = position_vec[0]  # set to number of present cols and rows
            parm_dict['grid_num_cols'] = position_vec[1]

        BE_parm_vec_1 = matread['BE_parm_vec_1']
        # Not required for translation but necessary to have
        if BE_parm_vec_1[0] == 3:
            parm_dict['BE_phase_content'] = 'chirp-sinc hybrid'
        else:
            parm_dict['BE_phase_content'] = 'Unknown'
        parm_dict['BE_center_frequency_[Hz]'] = BE_parm_vec_1[1]
        parm_dict['BE_band_width_[Hz]'] = BE_parm_vec_1[2]
        parm_dict['BE_amplitude_[V]'] = BE_parm_vec_1[3]
        parm_dict['BE_band_edge_smoothing_[s]'] = BE_parm_vec_1[4]  # 150 most likely
        parm_dict['BE_phase_variation'] = BE_parm_vec_1[5]  # 0.01 most likely
        parm_dict['BE_window_adjustment'] = BE_parm_vec_1[6]
        parm_dict['BE_points_per_step'] = 2 ** int(BE_parm_vec_1[7])
        parm_dict['BE_repeats'] = 2 ** int(BE_parm_vec_1[8])
        try:
            parm_dict['BE_bins_per_read'] = matread['bins_per_band_s']
        except KeyError:
            parm_dict['BE_bins_per_read'] = len(matread['bin_w'])

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

        # num_driving_bands = assembly_parm_vec[0]  # 0 = 1, 1 = 2 bands
        # band_combination_order = assembly_parm_vec[1]  # 0 parallel 1 series

        VS_parms = matread['SS_parm_vec']
        dc_amp_vec_full = matread['dc_amp_vec_full']

        VS_start_V = VS_parms[4]
        VS_start_loop_amp = VS_parms[5]
        VS_final_loop_amp = VS_parms[6]
        # VS_read_write_ratio = VS_parms[8]  # 1 <- SS_read_write_ratio

        parm_dict['VS_set_pulse_amplitude_[V]'] = VS_parms[9]  # 0 <- SS_set_pulse_amp
        parm_dict['VS_read_voltage_[V]'] = VS_parms[3]
        parm_dict['VS_steps_per_full_cycle'] = VS_parms[7]
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
            parm_dict['VS_amplitude_[V]'] = 0.5 * (
                max(dc_amp_vec_full) - min(dc_amp_vec_full))  # SS_max_offset_amplitude
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
            parm_dict['VS_offset_[V]'] = 0  # this is not correct. Fix manually when it comes to UDVS generation?
        else:
            parm_dict['VS_mode'] = 'Custom'

        return parm_dict

    @staticmethod
    def __read_parms_mat(file_path, is_beps):
        """
        Returns information about the excitation BE waveform present in the more parms.mat file
        
        Parameters 
        --------------------
        file_path : String / Unicode
            Absolute filepath of the .mat parameter file
        is_beps : Boolean
            Whether or not this is BEPS or BE-Line
        
        Returns 
        --------------------
        BE_bin_ind : 1D numpy unsigned int array
            Indices of the excited and measured frequency bins
        BE_bin_w : 1D numpy float array
            Excitation bin Frequencies
        BE_bin_FFT : 1D numpy complex array
            FFT of the BE waveform for the excited bins
        ex_wfm : 1D numpy float array
            Band Excitation waveform
        """
        if not path.exists(file_path):
            raise IOError('NO "More parms" file found')
        if is_beps:
            fft_name = 'FFT_BE_wave'
        else:
            fft_name = 'FFT_BE_rev_wave'
        matread = loadmat(file_path, variable_names=['BE_bin_ind', 'BE_bin_w', fft_name])
        BE_bin_ind = np.squeeze(matread['BE_bin_ind']) - 1  # From Matlab (base 1) to Python (base 0)
        BE_bin_w = np.squeeze(matread['BE_bin_w'])
        FFT_full = np.complex64(np.squeeze(matread[fft_name]))
        # For whatever weird reason, the sign of the imaginary portion is flipped. Correct it:
        # BE_bin_FFT = np.conjugate(FFT_full[BE_bin_ind])
        BE_bin_FFT = np.zeros(len(BE_bin_ind), dtype=np.complex64)
        BE_bin_FFT.real = np.real(FFT_full[BE_bin_ind])
        BE_bin_FFT.imag = -1 * np.imag(FFT_full[BE_bin_ind])

        ex_wfm = np.real(np.fft.ifft(np.fft.ifftshift(FFT_full)))

        return BE_bin_ind, BE_bin_w, BE_bin_FFT, ex_wfm

    def __build_udvs_table(self, parm_dict):
        """
        Generates the UDVS table using the parameters
        
        Parameters 
        --------------------
        parm_dict : dictionary
            Parameters describing experiment
        
        Returns 
        --------------------      
        UD_VS_table_label : List of strings
            Labels for columns in the UDVS table
        UD_VS_table_unit : List of strings
            Units for the columns in the UDVS table
        UD_VS_table : 2D numpy float array
            UDVS data table
        """

        def translate_val(target, strvals, numvals):
            """
            Internal function - Interprets the provided value using the provided lookup table

            Parameters
            ----------
            target : String
                Item we are looking for in the strvals list
            strvals : list of strings
                List of source values
            numvals : list of numbers
                List of results
            """

            if len(strvals) is not len(numvals):
                return None
            for strval, fltval in zip(strvals, numvals):
                if target == strval:
                    return fltval
            return None  # not found in list

        # % Extract values from parm text file
        BE_signal_type = translate_val(parm_dict['BE_phase_content'],
                                       ['chirp-sinc hybrid', '1/2 harmonic excitation',
                                        '1/3 harmonic excitation', 'pure sine'],
                                       [1, 2, 3, 4])
        # This is necessary when normalzing the AI by the AO
        self.harmonic = BE_signal_type
        self.signal_type = BE_signal_type
        if BE_signal_type is 4:
            self.harmonic = 1
        BE_amp = parm_dict['BE_amplitude_[V]']

        VS_amp = parm_dict['VS_amplitude_[V]']
        VS_offset = parm_dict['VS_offset_[V]']
        # VS_read_voltage = parm_dict['VS_read_voltage_[V]']
        VS_steps = parm_dict['VS_steps_per_full_cycle']
        VS_cycles = parm_dict['VS_number_of_cycles']
        VS_fraction = translate_val(parm_dict['VS_cycle_fraction'],
                                    ['full', '1/2', '1/4', '3/4'],
                                    [1., 0.5, 0.25, 0.75])
        VS_shift = parm_dict['VS_cycle_phase_shift']
        if VS_shift is not 0:
            VS_shift = translate_val(VS_shift, ['1/4', '1/2', '3/4'], [0.25, 0.5, 0.75])
        VS_in_out_cond = translate_val(parm_dict['VS_measure_in_field_loops'],
                                       ['out-of-field', 'in-field', 'in and out-of-field'], [0, 1, 2])
        VS_ACDC_cond = translate_val(parm_dict['VS_mode'],
                                     ['DC modulation mode', 'AC modulation mode with time reversal',
                                      'load user defined VS Wave from file', 'current mode'],
                                     [0, 2, 3, 4])
        self.expt_type = VS_ACDC_cond
        FORC_cycles = parm_dict['FORC_num_of_FORC_cycles']
        FORC_A1 = parm_dict['FORC_V_high1_[V]']
        FORC_A2 = parm_dict['FORC_V_high2_[V]']
        # FORC_repeats = parm_dict['# of FORC repeats']
        FORC_B1 = parm_dict['FORC_V_low1_[V]']
        FORC_B2 = parm_dict['FORC_V_low2_[V]']

        # % build vector of voltage spectroscopy values

        if VS_ACDC_cond == 0 or VS_ACDC_cond == 4:  # DC voltage spectroscopy or current mode
            VS_amp_vec_1 = np.arange(0, 1 + 1 / (VS_steps / 4), 1 / (VS_steps / 4))
            VS_amp_vec_2 = np.flipud(VS_amp_vec_1[:-1])
            VS_amp_vec_3 = -VS_amp_vec_1[1:]
            VS_amp_vec_4 = VS_amp_vec_1[1:-1] - 1
            vs_amp_vec = VS_amp * (np.hstack((VS_amp_vec_1, VS_amp_vec_2, VS_amp_vec_3, VS_amp_vec_4)))
            # apply phase shift to VS wave
            vs_amp_vec = np.roll(vs_amp_vec, int(np.floor(VS_steps / VS_fraction * VS_shift)))
            # cut VS waveform
            vs_amp_vec = vs_amp_vec[:int(np.floor(VS_steps * VS_fraction))]
            # repeat VS waveform
            vs_amp_vec = np.tile(vs_amp_vec, int(VS_cycles))
            vs_amp_vec = vs_amp_vec + VS_offset

        elif VS_ACDC_cond == 2:  # AC voltage spectroscopy with time reversal
            vs_amp_vec = VS_amp * np.arange(1 / (VS_steps / 2 / VS_fraction), 1 + 1 / (VS_steps / 2 / VS_fraction),
                                            1 / (VS_steps / 2 / VS_fraction))
            vs_amp_vec = np.roll(vs_amp_vec,
                                 int(np.floor(VS_steps / VS_fraction * VS_shift)))  # apply phase shift to VS wave
            vs_amp_vec = vs_amp_vec[:int(np.floor(VS_steps * VS_fraction / 2))]  # cut VS waveform
            vs_amp_vec = np.tile(vs_amp_vec, VS_cycles * 2)  # repeat VS waveform

        if FORC_cycles > 1:
            vs_amp_vec = vs_amp_vec / np.max(np.abs(vs_amp_vec))
            FORC_cycle_vec = np.arange(0, FORC_cycles + 1, FORC_cycles / (FORC_cycles - 1))
            FORC_A_vec = FORC_cycle_vec * (FORC_A2 - FORC_A1) / FORC_cycles + FORC_A1
            FORC_B_vec = FORC_cycle_vec * (FORC_B2 - FORC_B1) / FORC_cycles + FORC_B1
            FORC_amp_vec = (FORC_A_vec - FORC_B_vec) / 2
            FORC_off_vec = (FORC_A_vec + FORC_B_vec) / 2

            VS_amp_mat = np.tile(vs_amp_vec, [int(FORC_cycles), 1])
            FORC_amp_mat = np.tile(FORC_amp_vec, [len(vs_amp_vec), 1]).transpose()
            FORC_off_mat = np.tile(FORC_off_vec, [len(vs_amp_vec), 1]).transpose()
            VS_amp_mat = VS_amp_mat * FORC_amp_mat + FORC_off_mat
            vs_amp_vec = VS_amp_mat.reshape(int(FORC_cycles * VS_cycles * VS_fraction * VS_steps))

        # Build UDVS table:
        if VS_ACDC_cond is 0 or VS_ACDC_cond is 4:  # DC voltage spectroscopy or current mode

            if VS_ACDC_cond is 0:
                UD_dc_vec = np.vstack((vs_amp_vec, np.zeros(len(vs_amp_vec))))
            if VS_ACDC_cond is 4:
                UD_dc_vec = np.vstack((vs_amp_vec, vs_amp_vec))

            UD_dc_vec = UD_dc_vec.transpose().reshape(UD_dc_vec.size)
            num_VS_steps = UD_dc_vec.size

            UD_VS_table_label = ['step_num', 'dc_offset', 'ac_amp', 'wave_type', 'wave_mod', 'in-field', 'out-of-field']
            UD_VS_table_unit = ['', 'V', 'A', '', '', 'V', 'V']
            udvs_table = np.zeros(shape=(num_VS_steps, 7), dtype=np.float32)

            udvs_table[:, 0] = np.arange(0, num_VS_steps)  # Python base 0
            udvs_table[:, 1] = UD_dc_vec

            BE_IF_switch = np.abs(np.imag(np.exp(1j * np.pi / 2 * np.arange(1, num_VS_steps + 1))))
            BE_OF_switch = np.abs(np.real(np.exp(1j * np.pi / 2 * np.arange(1, num_VS_steps + 1))))

            if VS_in_out_cond is 0:  # out of field only
                udvs_table[:, 2] = BE_amp * BE_OF_switch
            elif VS_in_out_cond is 1:  # in field only
                udvs_table[:, 2] = BE_amp * BE_IF_switch
            elif VS_in_out_cond is 2:  # both in and out of field
                udvs_table[:, 2] = BE_amp * np.ones(num_VS_steps)

            udvs_table[:, 3] = np.ones(num_VS_steps)  # wave type
            udvs_table[:, 4] = np.ones(num_VS_steps) * BE_signal_type  # wave mod

            udvs_table[:, 5] = float('NaN') * np.ones(num_VS_steps)
            udvs_table[:, 6] = float('NaN') * np.ones(num_VS_steps)

            udvs_table[BE_IF_switch == 1, 5] = udvs_table[BE_IF_switch == 1, 1]
            udvs_table[BE_OF_switch == 1, 6] = udvs_table[BE_IF_switch == 1, 1]

        elif VS_ACDC_cond is 2:  # AC voltage spectroscopy

            num_VS_steps = vs_amp_vec.size
            half = int(0.5 * num_VS_steps)

            if num_VS_steps is not half * 2:
                raise ValueError('Odd number of UDVS steps found. Exiting!')

            UD_dc_vec = VS_offset * np.ones(num_VS_steps)
            UD_VS_table_label = ['step_num', 'dc_offset', 'ac_amp', 'wave_type', 'wave_mod', 'forward', 'reverse']
            UD_VS_table_unit = ['', 'V', 'A', '', '', 'A', 'A']
            udvs_table = np.zeros(shape=(num_VS_steps, 7), dtype=np.float32)
            udvs_table[:, 0] = np.arange(1, num_VS_steps + 1)
            udvs_table[:, 1] = UD_dc_vec
            udvs_table[:, 2] = vs_amp_vec
            udvs_table[:, 3] = np.ones(num_VS_steps)
            udvs_table[:half, 4] = BE_signal_type * np.ones(half)
            udvs_table[half:, 4] = -1 * BE_signal_type * np.ones(half)
            udvs_table[:, 5] = float('NaN') * np.ones(num_VS_steps)
            udvs_table[:, 6] = float('NaN') * np.ones(num_VS_steps)
            udvs_table[:half, 5] = vs_amp_vec[:half]
            udvs_table[half:, 6] = vs_amp_vec[half:]

        return UD_VS_table_label, UD_VS_table_unit, udvs_table

    @staticmethod
    def __get_random_spectra(parsers, num_pixels, num_udvs_steps, num_bins, num_spectra=100, verbose=False):
        """
        Parameters
        ----------
        parsers : list of BEodfParser objects
            parsers to seek into files to grab spectra
        num_pixels : unsigned int
            Number of spatial positions in the image
        num_udvs_steps : unsigned int
            Number of UDVS steps
        num_bins : unsigned int
            Number of frequency bins in every UDVS step
        num_spectra : unsigned int
            Total number of spectra to be extracted
        verbose : Boolean, optional
            Whether or not to print debugging statements

        Returns
        -------
        chosen_spectra : 2D complex numpy array
            spectrogram or spectra arranged as [instance, spectrum]
        """
        num_pixels = int(num_pixels)
        num_udvs_steps = int(num_udvs_steps)
        num_bins = int(num_bins)

        num_spectra = min(num_spectra, len(parsers) * num_pixels * num_udvs_steps)
        selected_pixels = np.random.randint(0, num_pixels, size=num_spectra)
        selected_steps = np.random.randint(0, num_udvs_steps, size=num_spectra)
        selected_parsers = np.random.randint(0, len(parsers), size=num_spectra)

        if verbose:
            print('Selecting the following random pixels, UDVS steps, parsers')
            print(np.vstack((selected_pixels, selected_steps, selected_parsers)))

        chosen_spectra = np.zeros(shape=(num_spectra, num_bins), dtype=np.complex64)

        for spectra_index in range(num_spectra):
            prsr = parsers[selected_parsers[spectra_index]]
            prsr.seek_to_pixel(selected_pixels[spectra_index])
            raw_vec = prsr.read_pixel()
            spectrogram = raw_vec.reshape(num_udvs_steps, -1)
            chosen_spectra[spectra_index] = spectrogram[selected_steps[spectra_index]]

        for prsr in parsers:
            prsr.reset()

        return chosen_spectra


class BEodfParser(object):
    def __init__(self, real_path, imag_path, num_pix, bytes_per_pix):
        """
        This object reads the two binary data files (real and imaginary data).
        Use separate parser instances for in-field and out-field data sets.
        
        Parameters 
        --------------------
        real_path : String / Unicode
            absolute path of the binary file containing the real portion of the data
        imag_path : String / Unicode
            absolute path of the binary file containing the imaginary portion of the data
        num_pix : unsigned int
            Number of pixels in this image
        bytes_per_pix : unsigned int
            Number of bytes per pixel
        """
        self.f_real = open(real_path, "rb")
        self.f_imag = open(imag_path, "rb")

        self.__num_pix__ = num_pix
        self.__bytes_per_pix__ = bytes_per_pix
        self.__pix_indx__ = 0

    def read_pixel(self):
        """
        Returns the content of the next pixel

        Returns 
        -------
        raw_vec : 1D numpy complex64 array
            Content of one pixel's data
        """
        if self.__num_pix__ is not None:
            if self.__pix_indx__ is self.__num_pix__:
                warn('BEodfParser - No more pixels to read!')
                return None

        self.f_real.seek(self.__pix_indx__ * self.__bytes_per_pix__, 0)
        real_vec = np.fromstring(self.f_real.read(self.__bytes_per_pix__), dtype='f')

        self.f_imag.seek(self.__pix_indx__ * self.__bytes_per_pix__, 0)
        imag_vec = np.fromstring(self.f_imag.read(self.__bytes_per_pix__), dtype='f')

        raw_vec = np.zeros(len(real_vec), dtype=np.complex64)
        raw_vec.real = real_vec
        raw_vec.imag = imag_vec

        self.__pix_indx__ += 1

        if self.__pix_indx__ is self.__num_pix__:
            self.f_real.close()
            self.f_imag.close()

        return raw_vec

    def read_all_data(self):
        """
        Returns the complete contents of the file pair

        Returns 
        -------
        raw_vec : 1D numpy complex64 array
            Entire content of the file pair
        """
        self.f_real.seek(0, 0)
        self.f_imag.seek(0, 0)

        d_real = np.fromstring(self.f_real.read(), dtype='f')
        d_imag = np.fromstring(self.f_imag.read(), dtype='f')

        full_file = d_real + 1j * d_imag

        self.f_real.close()
        self.f_imag.close()

        return full_file

    def seek_to_pixel(self, pixel_ind):
        """

        Parameters
        ----------
        pixel_ind

        Returns
        -------

        """
        if self.__num_pix__ is not None:
            pixel_ind = min(pixel_ind, self.__num_pix__)
        self.__pix_indx__ = pixel_ind

    def reset(self):
        """

        """
        self.f_real.seek(0, 0)
        self.f_imag.seek(0, 0)
        self.__pix_indx__ = 0
