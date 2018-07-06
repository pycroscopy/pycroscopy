# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:58:35 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, remove  # File Path formatting
from warnings import warn

import h5py
import numpy as np  # For array operations

from pyUSID.io.translator import Translator, generate_dummy_main_parms
from pyUSID.io.write_utils import Dimension
from pyUSID.io.hdf_utils import write_main_dataset, create_indexed_group, write_simple_attrs


class GIVTranslator(Translator):
    """
    Translates G-mode Fast IV datasets from .mat files to .h5
    """
    def __init__(self, *args, **kwargs):
        super(GIVTranslator, self).__init__(*args, **kwargs)
        self.raw_datasets = None

    def _parse_file_path(self, input_path):
        pass

    def translate(self, parm_path):
        """      
        The main function that translates the provided file into a .h5 file
        
        Parameters
        ------------
        parm_path : string / unicode
            Absolute file path of the parameters .mat file. 
            
        Returns
        ----------
        h5_path : string / unicode
            Absolute path of the translated h5 file
        """
        parm_path = path.abspath(parm_path)
        parm_dict, excit_wfm = self._read_parms(parm_path)
        folder_path, base_name = path.split(parm_path)
        waste, base_name = path.split(folder_path)

        # Until a better method is provided....
        with h5py.File(path.join(folder_path, 'line_1.mat'), 'r') as h5_mat_line_1:
            num_ai_chans = h5_mat_line_1['data'].shape[1]
        
        h5_path = path.join(folder_path, base_name+'.h5')
        if path.exists(h5_path):
            remove(h5_path)

        with h5py.File(h5_path) as h5_f:

            h5_meas_grp = create_indexed_group(h5_f, 'Measurement')
            global_parms = generate_dummy_main_parms()
            global_parms.update({'data_type': 'gIV', 'translator': 'gIV'})
            write_simple_attrs(h5_meas_grp, global_parms)

            # Only prepare the instructions for the dimensions here
            spec_dims = Dimension('Bias', 'V', excit_wfm)
            pos_dims = Dimension('Y', 'm', np.linspace(0, parm_dict['grid_scan_height_[m]'],
                                                       parm_dict['grid_num_rows']))

            self.raw_datasets = list()

            for chan_index in range(num_ai_chans):

                h5_chan_grp = create_indexed_group(h5_meas_grp, 'Channel')
                write_simple_attrs(h5_chan_grp, parm_dict)
                """
                Minimize file size to the extent possible.
                DAQs are rated at 16 bit so float16 should be most appropriate.
                For some reason, compression is effective only on time series data
                """
                h5_raw = write_main_dataset(h5_chan_grp, (parm_dict['grid_num_rows'], excit_wfm.size), 'Raw_Data',
                                            'Current',
                                            '1E-{} A'.format(parm_dict['IO_amplifier_gain']), pos_dims, spec_dims,
                                            dtype=np.float16, chunks=(1, excit_wfm.size), compression='gzip')

                self.raw_datasets.append(h5_raw)

            # Now that the N channels have been made, populate them with the actual data....
            self._read_data(parm_dict, folder_path)

        return h5_path

    def _read_data(self, parm_dict, folder_path):
        """
        Reads raw data and populates the h5 datasets

        Parameters
        ----------
        parm_dict : Dictionary
            dictionary containing parameters for this data
        folder_path : string / unicode
            Absolute path of folder containing the data
        """
        if parm_dict['excitation_extra_pts'] == 0:
            main_data = slice(parm_dict['excitation_pulse_points'], None)
        else:
            main_data = slice(parm_dict['excitation_pulse_points'], -1 * parm_dict['excitation_extra_pts'])

        for line_ind in range(parm_dict['grid_num_rows']):
            if line_ind % np.round(parm_dict['grid_num_rows']/10) == 0:
                print('Reading data in line {} of {}'.format(line_ind+1, parm_dict['grid_num_rows']))
            file_path = path.join(folder_path, 'line_'+str(line_ind+1)+'.mat')
            if path.exists(file_path):
                with h5py.File(file_path, 'r') as h5_f:
                    h5_data = h5_f['data']
                    if h5_data.shape[0] >= parm_dict['excitation_length'] and \
                            h5_data.shape[1] == len(self.raw_datasets):
                        for chan, h5_chan in enumerate(self.raw_datasets):
                            h5_chan[line_ind, :] = np.float16(h5_data[main_data, chan])
                            h5_chan.file.flush()
                    else:
                        warn('No data found for Line '+str(line_ind))
            else:
                warn('File not found for: line '+str(line_ind))
        print('Finished reading all data!')

    @staticmethod
    def _read_parms(parm_path):
        """
        Copies experimental parameters from the .mat file to a dictionary
        
        Parameters
        ----------
        parm_path : string / unicode
            Absolute path of the parameters file
        
        Returns 
        -------
        parm_dict : dictionary
            Dictionary containing all relevant parameters
        excit_wfm : 1d numpy float array
            Excitation waveform
        """
        with h5py.File(parm_path, 'r') as h5_f:
            parm_dict = dict()

            parm_dict['IO_samp_rate_[Hz]'] = np.uint32(h5_f['samp_rate'][0][0])
            parm_dict['IO_amplifier_gain'] = np.uint32(h5_f['amp_gain'][0][0])

            parm_dict['excitation_frequency_[Hz]'] = np.float32(h5_f['frequency'][0][0])
            parm_dict['excitation_amplitude_[V]'] = np.float32(h5_f['amplitude'][0][0])
            parm_dict['excitation_offset_[V]'] = np.float32(h5_f['offset'][0][0])
            excit_wfm = np.float32(np.squeeze(h5_f['excit_wfm'].value))

            # Make sure to truncate the data to the point when the
            pts_per_cycle = int(np.round(1.0*parm_dict['IO_samp_rate_[Hz]']/parm_dict['excitation_frequency_[Hz]']))
            extra_pts = len(excit_wfm) % pts_per_cycle
            parm_dict['excitation_extra_pts'] = extra_pts

            """ New versions could have used a pulse at the beginning of the line to "wake up" the material.
            This pulse is always an integer number of cycles long
            For now, let's remove this section from the excitation waveform and data"""

            # This pulse may or may not have been used:
            try:
                pulse_duration = np.float32(h5_f['pulse_duration'][0][0])
                pulse_height = np.float32(h5_f['pulse_height'][0][0])
            except KeyError:
                pulse_duration = 0.0
                pulse_height = 0.0
            if pulse_duration == 0.0:
                pulse_height = 0.0

            parm_dict['excitation_pulse_height_[V]'] = np.float32(pulse_height)
            parm_dict['excitation_pulse_time_[sec]'] = np.float32(pulse_duration)
            pulse_points = int(np.round(pulse_duration * parm_dict['IO_samp_rate_[Hz]']))
            parm_dict['excitation_pulse_points'] = np.uint32(pulse_points)

            line_time = np.float32(h5_f['line_time'][0][0]) - pulse_duration
            excess_time = line_time - 1.0*extra_pts/parm_dict['IO_samp_rate_[Hz]']
            parm_dict['excitation_duration_[sec]'] = line_time - excess_time
            if extra_pts > 0:
                excit_wfm = excit_wfm[pulse_points:-extra_pts]
            else:
                excit_wfm = excit_wfm[pulse_points:]
            parm_dict['excitation_length'] = len(excit_wfm)

            parm_dict['grid_num_rows'] = np.uint32(h5_f['num_lines'][0][0])
            parm_dict['grid_num_cols'] = np.uint32(np.floor(len(excit_wfm) / pts_per_cycle))

            parm_dict['grid_scan_height_[m]'] = np.float32(h5_f['scan_height'][0][0])
            parm_dict['grid_scan_width_[m]'] = np.float32(h5_f['scan_width'][0][0])
            parm_dict['grid_scan_speed_[ms-1]'] = np.float32(h5_f['scan_speed'][0][0])

        return parm_dict, excit_wfm
