#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 3:19:46 2017

@author: anugrahsaxena, Suhas Somnath, Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, remove
from warnings import warn
import h5py
import numpy as np
from scipy.io.matlab import loadmat  # To load parameters stored in Matlab .mat file

from .df_utils.be_utils import parmsToDict
from .gmode_line import GLineTranslator
from pyUSID.io.translator import generate_dummy_main_parms
from pyUSID.io.write_utils import VALUES_DTYPE, Dimension
from pyUSID.io.hdf_utils import write_simple_attrs, create_indexed_group, write_ind_val_dsets, write_main_dataset


class GTuneTranslator(GLineTranslator):
    """
    Translates G-mode Tune (bigtimedata.dat) files from actual BE line experiments to HDF5
    """

    def __init__(self, *args, **kwargs):
        super(GLineTranslator, self).__init__(*args, **kwargs)

    def translate(self, file_path):
        """
        The main function that translates the provided file into a .h5 file
        
        Parameters
        ----------
        file_path : String / unicode
            Absolute path of any file in the directory

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the h5 file

        """
        file_path = path.abspath(file_path)
        # Figure out the basename of the data:
        (basename, parm_paths, data_paths) = super(GTuneTranslator, self)._parse_file_path(file_path)

        (folder_path, unused) = path.split(file_path)
        h5_path = path.join(folder_path, basename + '.h5')

        if path.exists(h5_path):
            remove(h5_path)

        # Load parameters from .mat file
        matread = loadmat(parm_paths['parm_mat'],
                          variable_names=['AI_wave', 'BE_wave_AO_0', 'BE_wave_AO_1', 'BE_wave_train',
                                          'BE_wave', 'total_cols', 'total_rows'])
        be_wave = np.float32(np.squeeze(matread['BE_wave']))
        be_wave_train = np.float32(np.squeeze(matread['BE_wave_train']))

        num_cols = int(matread['total_cols'][0][0])
        expected_rows = int(matread['total_rows'][0][0])

        self.points_per_pixel = len(be_wave)
        self.points_per_line = len(be_wave_train)

        # Load parameters from .txt file - 'BE_center_frequency_[Hz]', 'IO rate'
        is_beps, parm_dict = parmsToDict(parm_paths['parm_txt'])

        # Get file byte size:
        # For now, assume that bigtime_00 always exists and is the main file
        file_size = path.getsize(data_paths[0])

        # Calculate actual number of lines since the first few lines may not be saved
        self.num_rows = 1.0 * file_size / (4 * self.points_per_pixel * num_cols)
        if self.num_rows % 1:
            warn('Error - File has incomplete rows')
            return None
        else:
            self.num_rows = int(self.num_rows)

        samp_rate = parm_dict['IO_rate_[Hz]']
        ex_freq_nominal = parm_dict['BE_center_frequency_[Hz]']

        # method 1 for calculating the correct excitation frequency:
        pixel_duration = 1.0 * self.points_per_pixel / samp_rate
        num_periods = pixel_duration * ex_freq_nominal
        ex_freq_correct = 1 / (pixel_duration / np.floor(num_periods))

        # correcting the excitation frequency - will be VERY useful during analysis and filtering
        parm_dict['BE_center_frequency_[Hz]'] = ex_freq_correct

        # Some very basic information that can help the processing crew
        parm_dict['points_per_line'] = self.points_per_line
        parm_dict['num_bins'] = self.points_per_pixel
        parm_dict['grid_num_rows'] = self.num_rows
        parm_dict['data_type'] = 'G_mode_line'


        if self.num_rows != expected_rows:
            print('Note: {} of {} lines found in data file'.format(self.num_rows, expected_rows))

        # Calculate number of points to read per line:
        self.__bytes_per_row__ = int(file_size / self.num_rows)

        # First finish writing all global parameters, create the file too:
        h5_file = h5py.File(h5_path, 'w')
        global_parms = generate_dummy_main_parms()

        global_parms['data_type'] = 'G_mode_line'
        global_parms['translator'] = 'G_mode_line'
        write_simple_attrs(h5_file, global_parms)

        # Next create the Measurement and Channel groups and write the appropriate parameters to them
        meas_grp = create_indexed_group(h5_file, 'Measurement')
        write_simple_attrs(meas_grp, parm_dict)

        # Now that the file has been created, go over each raw data file:
        """ 
        We only allocate the space for the main data here.
        This does NOT change with each file. The data written to it does.
        The auxiliary datasets will not change with each raw data file since
        only one excitation waveform is used
        """
        pos_desc = Dimension('Y', 'm', np.arange(self.num_rows))
        spec_desc = Dimension('Excitation', 'V', np.tile(VALUES_DTYPE(be_wave), num_cols))

        h5_pos_ind, h5_pos_val = write_ind_val_dsets(meas_grp, pos_desc, is_spectral=False)
        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(meas_grp, spec_desc, is_spectral=True)


        for f_index in data_paths.keys():
            chan_grp = create_indexed_group(meas_grp, 'Channel')

            h5_main = write_main_dataset(chan_grp, (self.num_rows, self.points_per_pixel * num_cols), 'Raw_Data',
                                         'Deflection', 'V',
                                         None, None,
                                         h5_pos_inds=h5_pos_ind, h5_pos_vals=h5_pos_val,
                                         h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                                         chunks=(1, self.points_per_pixel), dtype=np.float16)

            # Now transfer scan data in the dat file to the h5 file:
            super(GTuneTranslator, self)._read_data(data_paths[f_index], h5_main)

        h5_file.close()
        print('G-Tune translation complete!')

        return h5_path
