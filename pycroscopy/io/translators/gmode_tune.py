#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 3:19:46 2017

@author: anugrahsaxena
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, listdir, remove
from warnings import warn

import numpy as np
from scipy.io.matlab import loadmat  # To load parameters stored in Matlab .mat file

from .df_utils.be_utils import parmsToDict
from .gmode_line import GLineTranslator
from .utils import generate_dummy_main_parms, build_ind_val_dsets
from ..hdf_utils import getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5
from ..microdata import MicroDataGroup, MicroDataset


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

        # Need to take the complex conjugate if reading from a .mat file
        # FFT_BE_wave = np.conjugate(np.complex64(np.squeeze(matread['FFT_BE_wave'])))

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

        # method 2 for calculating the exact excitation frequency:
        """
        fft_ex_wfm = np.abs(np.fft.fftshift(np.fft.fft(be_wave)))
        w_vec = np.linspace(-0.5 * samp_rate, 0.5 * samp_rate - 1.0*samp_rate / self.points_per_pixel,
                            self.points_per_pixel)
        hot_bins = np.squeeze(np.argwhere(fft_ex_wfm > 1E+3))
        ex_freq_correct = w_vec[hot_bins[-1]]
        """

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
        meas_grp = MicroDataGroup('Measurement_000')
        meas_grp.attrs = parm_dict

        spm_data = MicroDataGroup('')
        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = 'G_mode_line'
        global_parms['translator'] = 'G_mode_line'
        spm_data.attrs = global_parms
        spm_data.addChildren([meas_grp])

        hdf = ioHDF5(h5_path)
        # hdf.clear()
        hdf.writeData(spm_data)

        # Now that the file has been created, go over each raw data file:
        # 1. write all ancillary data. Link data. 2. Write main data sequentially

        """ 
        We only allocate the space for the main data here.
        This does NOT change with each file. The data written to it does.
        The auxiliary datasets will not change with each raw data file since
        only one excitation waveform is used
        """
        ds_main_data = MicroDataset('Raw_Data', data=[],
                                    maxshape=(self.num_rows, self.points_per_pixel * num_cols),
                                    chunking=(1, self.points_per_pixel), dtype=np.float16)
        ds_main_data.attrs['quantity'] = ['Deflection']
        ds_main_data.attrs['units'] = ['V']

        ds_pos_ind, ds_pos_val = build_ind_val_dsets([self.num_rows], is_spectral=False,
                                                     labels=['Y'], units=['m'])
        ds_spec_inds, ds_spec_vals = build_ind_val_dsets([self.points_per_pixel * num_cols], is_spectral=True,
                                                         labels=['Excitation'], units=['V'])
        ds_spec_vals.data = np.atleast_2d(np.tile(np.float32(be_wave), num_cols))  # Override the default waveform

        aux_ds_names = ['Position_Indices', 'Position_Values',
                        'Spectroscopic_Indices', 'Spectroscopic_Values']

        for f_index in data_paths.keys():
            chan_grp = MicroDataGroup('{:s}{:03d}'.format('Channel_', f_index), '/Measurement_000/')
            chan_grp.addChildren([ds_main_data, ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals])

            # print('Writing following tree to file:')
            # chan_grp.showTree()
            h5_refs = hdf.writeData(chan_grp)

            h5_main = getH5DsetRefs(['Raw_Data'], h5_refs)[0]  # We know there is exactly one main data

            # Reference linking can certainly take place even before the datasets have reached their final size         
            linkRefs(h5_main, getH5DsetRefs(aux_ds_names, h5_refs))

            # Now transfer scan data in the dat file to the h5 file:
            super(GTuneTranslator, self)._read_data(data_paths[f_index], h5_main)

        hdf.close()
        print('G-Tune translation complete!')

        return h5_path
