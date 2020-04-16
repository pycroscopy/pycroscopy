# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 15:21:46 2015

@author: Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from os import path, listdir, remove
from warnings import warn
import h5py
import numpy as np
from scipy.io.matlab import loadmat  # To load parameters stored in Matlab .mat file

from .df_utils.be_utils import parmsToDict
from pyUSID.io.translator import Translator
from pyUSID.io.write_utils import VALUES_DTYPE, Dimension
from pyUSID.io.hdf_utils import write_main_dataset, create_indexed_group, write_simple_attrs, write_ind_val_dsets

if sys.version_info.major == 3:
    unicode = str


class GLineTranslator(Translator):
    """
    Translated G-mode line (bigtimedata.dat) files from actual BE line experiments to HDF5
    """
    def __init__(self, *args, **kwargs):
        super(Translator, self).__init__(*args, **kwargs)
        self.points_per_pixel = 1
        self.num_rows = 1
        self.__bytes_per_row__ = 1

    @staticmethod
    def is_valid_file(data_path):
        """
        Checks whether the provided file can be read by this translator

        Parameters
        ----------
        data_path : str
            Path to raw data file

        Returns
        -------
        obj : str
            Path to file that will be accepted by the translate() function if
            this translator is indeed capable of translating the provided file.
            Otherwise, None will be returned
        """

        data_path = path.abspath(data_path)

        orig_path = None
        if path.isfile(data_path):
            ext = data_path.split('.')[-1]
            if ext.lower() not in ['jpg', 'png', 'jpeg', 'tiff', 'mat', 'txt',
                                   'dat', 'xls', 'xlsx']:
                return None

            # we only care about the folder names at this point...
            orig_path = data_path
            # Assume that the file is amongst all other data files
            folder_path, _ = path.split(data_path)
        else:
            folder_path = data_path

        data_path = path.join(folder_path, listdir(path=folder_path)[0])
        basename, parm_paths, data_paths = GLineTranslator._parse_file_path(data_path)

        # The provided file must either p
        if orig_path is not None:
            if not any([orig_path in sel_dict.values() for sel_dict in
                        [parm_paths, data_paths]]):
                return None

        if len(parm_paths) == 2 and len(data_paths) > 0:
            return parm_paths['parm_txt']

        return None

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
        (basename, parm_paths, data_paths) = self._parse_file_path(file_path)
        
        (folder_path, unused) = path.split(file_path)
        h5_path = path.join(folder_path, basename+'.h5')
        
        if path.exists(h5_path):
            remove(h5_path)
        
        # Load parameters from .mat file - 'BE_wave', 'FFT_BE_wave', 'total_cols', 'total_rows'
        matread = loadmat(parm_paths['parm_mat'], variable_names=['BE_wave', 'FFT_BE_wave', 'total_cols', 'total_rows'])
        be_wave = np.float32(np.squeeze(matread['BE_wave']))

        # Need to take the complex conjugate if reading from a .mat file
        # FFT_BE_wave = np.conjugate(np.complex64(np.squeeze(matread['FFT_BE_wave'])))
        
        num_cols = int(matread['total_cols'][0][0])
        expected_rows = int(matread['total_rows'][0][0])
        self.points_per_pixel = len(be_wave)
        
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
        parm_dict['num_bins'] = self.points_per_pixel
        parm_dict['grid_num_rows'] = self.num_rows
        parm_dict['data_type'] = 'G_mode_line'
            
        if self.num_rows != expected_rows:
            print('Note: {} of {} lines found in data file'.format(self.num_rows, expected_rows))
        
        # Calculate number of points to read per line:
        self.__bytes_per_row__ = int(file_size/self.num_rows)

        # First finish writing all global parameters, create the file too:
        h5_f = h5py.File(h5_path, 'w')
        global_parms = dict()
        global_parms['data_type'] = 'G_mode_line'
        global_parms['translator'] = 'G_mode_line'
        write_simple_attrs(h5_f, global_parms)

        meas_grp = create_indexed_group(h5_f, 'Measurement')
        write_simple_attrs(meas_grp, parm_dict)

        pos_desc = Dimension('Y', 'm', np.arange(self.num_rows))
        spec_desc = Dimension('Excitation', 'V', np.tile(VALUES_DTYPE(be_wave), num_cols))

        first_dat = True
        for key in data_paths.keys():
            # Now that the file has been created, go over each raw data file:
            # 1. write all ancillary data. Link data. 2. Write main data sequentially

            """ We only allocate the space for the main data here.
            This does NOT change with each file. The data written to it does.
            The auxiliary datasets will not change with each raw data file since
            only one excitation waveform is used"""
            chan_grp = create_indexed_group(meas_grp, 'Channel')

            if first_dat:
                if len(data_paths) > 1:
                    # All positions and spectra are shared between channels
                    h5_pos_inds, h5_pos_vals = write_ind_val_dsets(meas_grp, pos_desc, is_spectral=False)
                    h5_spec_inds, h5_spec_vals = write_ind_val_dsets(meas_grp, spec_desc, is_spectral=True)
                elif len(data_paths) == 1:
                    h5_pos_inds, h5_pos_vals = write_ind_val_dsets(chan_grp, pos_desc, is_spectral=False)
                    h5_spec_inds, h5_spec_vals = write_ind_val_dsets(chan_grp, spec_desc, is_spectral=True)

                first_dat = False
            else:
                pass

            h5_main = write_main_dataset(chan_grp, (self.num_rows, self.points_per_pixel * num_cols), 'Raw_Data',
                                         'Deflection', 'V',
                                         None, None,
                                         h5_pos_inds=h5_pos_inds, h5_pos_vals=h5_pos_vals,
                                         h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                                         chunks=(1, self.points_per_pixel), dtype=np.float16)

            # Now transfer scan data in the dat file to the h5 file:
            self._read_data(data_paths[key], h5_main)
            
        h5_f.close()
        print('G-Line translation complete!')

        return h5_path

    @staticmethod
    def _parse_file_path(data_filepath):
        """
        Goes through the file directory and figures out the basename and the 
        parameter (text and .mat), data file paths (for each analog input channel)
        
        Parameters
        -----------------
        data_filepath : string / unicode
            absolute path of any file in the data folder
        
        Returns
        ----------------
        basename : string / unicode
            base name of the experiment\n
        parm_paths : dictionary
            paths for the text and .mat parameter files\n
            parm_text : absolute file path of the parameter text file\n
            parm_mat : absolute file path of the parameter .mat file
        data_paths : dictionary of the paths for the big-time data files.
            key : index of the analog input that generated the data file\n
            value : absolute file path of the data file
        """
        # Return (basename, parameter text path)
        folder_path, _ = path.split(data_filepath)
        _, basename = path.split(folder_path)

        # There may be one or two bigdata files. May need both paths
        parm_paths = dict()
        data_paths = dict()
        targ_str = 'bigtime_0'
        for file_name in listdir(folder_path):

            if file_name.endswith('.txt') and file_name.find('parm') > 0:
                parm_paths['parm_txt'] = path.join(folder_path, file_name)

            elif file_name.endswith('_all.mat'):
                parm_paths['parm_mat'] = path.join(folder_path, file_name)

            elif file_name.endswith('.dat'):
                ind = file_name.find(targ_str)
                if ind > 0:
                    data_paths[int(file_name[ind + len(targ_str)])] = path.join(folder_path, file_name)

        return basename, parm_paths, data_paths

    def _read_data(self, filepath, h5_dset):
        """
        Reads the .dat file and populates the .h5 dataset

        Parameters
        ---------
        filepath : String / unicode
            absolute path of the data file for a particular analog input channel
        h5_dset : HDF5 dataset reference
            Reference to the target Raw_Data dataset

        Returns
        ---------
        None
        """
        # Create data matrix - Only need 16 bit floats (time) 
                
        # Read line by line and write to h5                 
        with open(filepath, 'rb') as file_handl:
            for row_indx in range(self.num_rows):
                
                if row_indx % 10 == 0:
                    print('Reading line {} of {}'.format(row_indx, self.num_rows))
                
                file_handl.seek(row_indx*self.__bytes_per_row__, 0)
                data_vec = np.fromstring(file_handl.read(self.__bytes_per_row__), dtype='f')
                h5_dset[row_indx] = np.float16(data_vec)
                h5_dset.file.flush()
        
        print('Finished reading file: {}!'.format(filepath))
