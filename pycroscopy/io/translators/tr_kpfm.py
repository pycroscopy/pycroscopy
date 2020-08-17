# -*- coding: utf-8 -*-
"""
Created on Thursday July 27 2017

@author: Rama Vasudevan, Suhas Somnath, Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from os import path, remove, listdir  # File Path formatting
import re
import numpy as np  # For array operations
import h5py
from scipy.io import loadmat

from sidpy.sid import Translator
from sidpy.hdf.hdf_utils import write_simple_attrs

from pyUSID.io.write_utils import Dimension
from pyUSID.io.hdf_utils import write_main_dataset, create_indexed_group

if sys.version_info.major == 3:
    unicode = str


class TRKPFMTranslator(Translator):
    """
    Translates  trKPFM datasets from .mat and .dat files to .h5
    """

    def __init__(self, *args, **kwargs):
        super(TRKPFMTranslator, self).__init__(*args, **kwargs)
        self.raw_datasets = None

    @staticmethod
    def is_valid_file(data_path):
        """
        Checks whether the provided file can be read by this translator

        Parameters
        ----------
        data_path : str
            Path to folder or any data / parameter file within the folder

        Returns
        -------
        obj : str
            Path to file that will be accepted by the translate() function if
            this translator is indeed capable of translating the provided file.
            Otherwise, None will be returned
        """

        def get_chan_ind(line):
            match_obj = re.match(r'(.*)_ch(..).dat', line, re.M | re.I)
            type_list = [str, int]
            if match_obj:
                return \
                    [type_caster(match_obj.group(ind)) for ind, type_caster in
                     zip(range(1, 1 + len(type_list)), type_list)][-1]
            else:
                return None

        if path.isfile(data_path):
            # Assume that the file is amongst all other data files
            folder_path, _ = path.split(data_path)
        else:
            folder_path = data_path

        #  Now looking at the folder with  all necessary files:
        file_list = listdir(path=folder_path)
        parm_file_name = None
        raw_data_paths = list()
        for item in file_list:
            if item.endswith('parm.mat'):
                parm_file_name = item
            elif isinstance(get_chan_ind(item), int):
                raw_data_paths.append(item)

        # Both the parameter and data files MUST be found:
        if parm_file_name is not None and len(raw_data_paths) > 0:
            # Returning the path to the parameter file since this is what the translate() expects:
            return path.join(folder_path, parm_file_name)

        return None

    def _parse_file_path(self, input_path):
        folder_path, base_name = path.split(input_path)
        base_name = base_name[:-8]

        h5_path = path.join(folder_path, base_name + '.h5')
        if path.exists(h5_path):
            remove(h5_path)

        self.h5_path = h5_path
        # Until a better method is provided....
        self.file_list = list()
        for file in listdir(folder_path):
            if '.dat' in file:
                self.file_list.append(path.join(folder_path, file))
        self.file_list = sorted(self.file_list)

    @staticmethod
    def _parse_spectrogram_size(file_handle):
        """
        
        Parameters
        ----------
        file_handle

        Returns
        -------
        data_length: int, size of the spectrogram
        count: int, number of pixels in dataset +1

        """"""
        """

        f = file_handle

        cont_cond = True
        count = 0

        data_lengths = []

        while cont_cond:
            #print(count, f.tell())
            count += 1

            data_length = np.fromfile(f, dtype=np.float32, count=1)

            if data_length > 0:
                data_lengths.append(int(data_length))
                f.seek(int(data_length - 1) * 4, 1)
            else:
                cont_cond = False

        if len(np.unique(np.array(data_lengths))) > 1:
            print("Unequal data lengths! Cannot continue")
        else:
            print("Equal data lengths")

        return data_lengths[0], count

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
        excit_wfm = excit_wfm[1::2]
        self._parse_file_path(parm_path)

        num_dat_files = len(self.file_list)

        f = open(self.file_list[0], 'rb')
        spectrogram_size, count_vals = self._parse_spectrogram_size(f)
        print("Excitation waveform shape: ", excit_wfm.shape)
        print("spectrogram size:", spectrogram_size)
        num_pixels = parm_dict['grid_num_rows'] * parm_dict['grid_num_cols']
        print('Number of pixels: ', num_pixels)
        print('Count Values: ', count_vals)
        #if (num_pixels + 1) != count_vals:
        #    print("Data size does not match number of pixels expected. Cannot continue")

        #Find how many channels we have to make
        num_ai_chans = num_dat_files // 2  # Division by 2 due to real/imaginary

        # Now start creating datasets and populating:
        #Start with getting an h5 file
        h5_file = h5py.File(self.h5_path)

        #First create a measurement group
        h5_meas_group = create_indexed_group(h5_file, 'Measurement')

        #Set up some parameters that will be written as attributes to this Measurement group
        global_parms = dict()
        global_parms['data_type'] = 'trKPFM'
        global_parms['translator'] = 'trKPFM'
        write_simple_attrs(h5_meas_group, global_parms)
        write_simple_attrs(h5_meas_group, parm_dict)

        #Now start building the position and spectroscopic dimension containers
        #There's only one spectroscpoic dimension and two position dimensions

        #The excit_wfm only has the DC values without any information on cycles, time, etc.
        #What we really need is to add the time component. For every DC step there are some time steps.

        num_time_steps = (spectrogram_size-5) //excit_wfm.size //2 #Need to divide by 2 because it considers on and off field

        #There should be three spectroscopic axes
        #In order of fastest to slowest varying, we have
        #time, voltage, field

        time_vec = np.linspace(0, parm_dict['IO_time'], num_time_steps)
        print('Num time steps: {}'.format(num_time_steps))
        print('DC Vec size: {}'.format(excit_wfm.shape))
        print('Spectrogram size: {}'.format(spectrogram_size))

        field_vec = np.array([0,1])

        spec_dims = [Dimension ('Time', 's', time_vec),Dimension('Field', 'Binary', field_vec),
                     Dimension('Bias', 'V', excit_wfm)]

        pos_dims = [Dimension('Cols', 'm', int(parm_dict['grid_num_cols'])),
                    Dimension('Rows', 'm', int(parm_dict['grid_num_rows']))]


        self.raw_datasets = list()

        for chan_index in range(num_ai_chans):
            chan_grp = create_indexed_group(h5_meas_group,'Channel')

            if chan_index == 0:
                write_simple_attrs(chan_grp,{'Harmonic': 1})
            else:
                write_simple_attrs(chan_grp,{'Harmonic': 2})

            h5_raw = write_main_dataset(chan_grp,  # parent HDF5 group
                                        (num_pixels, spectrogram_size - 5),
                                        # shape of Main dataset
                                        'Raw_Data',  # Name of main dataset
                                        'Deflection',  # Physical quantity contained in Main dataset
                                        'V',  # Units for the physical quantity
                                        pos_dims,  # Position dimensions
                                        spec_dims,  # Spectroscopic dimensions
                                        dtype=np.complex64,  # data type / precision
                                        compression='gzip',
                                        chunks=(1, spectrogram_size - 5),
                                        main_dset_attrs={'quantity': 'Complex'})

            #h5_refs = hdf.write(chan_grp, print_log=False)
            #h5_raw = get_h5_obj_refs(['Raw_Data'], h5_refs)[0]
            #link_h5_objects_as_attrs(h5_raw, get_h5_obj_refs(aux_ds_names, h5_refs))
            self.raw_datasets.append(h5_raw)
            self.raw_datasets.append(h5_raw)

        # Now that the N channels have been made, populate them with the actual data....
        self._read_data(parm_dict, parm_path, spectrogram_size)

        h5_file.file.close()

        #hdf.close()
        return self.h5_path

    def _read_data(self, parm_dict, parm_path, data_length):
        """
        Reads raw data and populates the h5 datasets

        Parameters
        ----------
        parm_dict : Dictionary
            dictionary containing parameters for this data
        folder_path : string / unicode
            Absolute path of folder containing the data
        """
        # Determine number of pixels
        num_pixels = parm_dict['grid_num_rows'] * parm_dict['grid_num_cols']

        # The four files in TRKPFM are for real and imaginary parts for 1st, 2nd harmonic
        # Create a list of [True,False,True,False] so files can be written to
        # the appropraite channel

        real_imag = np.zeros(shape=(len(self.file_list), 1))
        real_imag[::2] = 1
        real_cond = []
        for entry in real_imag:
            if entry > 0:
                real_cond.append(True)
            else:
                real_cond.append(False)

        # Scan through all the .dat files available
        for ifile, file_path in enumerate(self.file_list):
            f = open(file_path, 'rb')
            results_p = self.read_file(data_length, f)
            spectrogram_matrix = np.array(results_p[:])
            b_axis = spectrogram_matrix.shape[2]
            c_axis = spectrogram_matrix.shape[1]
            # dall = np.transpose(spectrogram_matrix, (0, 2, 1)).reshape(num_pixels * c_axis, b_axis)
            dall = np.transpose(spectrogram_matrix, (0, 2, 1)).reshape(-1, b_axis)

            _, ia, ic = np.unique(dall, axis=0, return_index=True, return_inverse=True)
            reprowind = np.setdiff1d(ic, ia)

            if len(reprowind > 0):
                dall[reprowind, :] = np.nan

            # Write to the datasets
            h5_main = self.raw_datasets[ifile]

            if real_cond[ifile]:
                print('Dall Size is: ', dall.shape)
                # Do some error catching. In case the last pixel is absent, then just ignore it.
                try:
                    h5_main[:, :] = dall.reshape(h5_main.shape) + 1j * 0
                except ValueError:
                    h5_main[:-1, :] = dall.reshape(h5_main.shape[0] - 1, h5_main.shape[1]) + 1j * 0
            else:
                # Error catching. In case the last pixel is absent, then just ignore it.
                try:
                    h5_main[:, :] += 0 + 1j * dall.reshape(h5_main.shape)
                except ValueError:
                    h5_main[:-1, :] += 0 + 1j * dall.reshape(h5_main.shape[0] - 1, h5_main.shape[1])
            h5_main.file.flush()

    @staticmethod
    def read_file(data_length, f):
        start_point = 0
        count = 0
        count_vals = []
        f.seek(start_point * 4, 0)
        cont_cond = True
        results_p = []
        while cont_cond:
            count_vals.append(count)
            count += 1

            data_vec = np.fromfile(f, dtype=np.float32, count=int(data_length))
            data_vec1 = data_vec[5:int(data_length)]

            if len(data_vec) > 1:

                s1 = data_vec[3]
                s2 = data_vec[4]
                # print('Data_mat and s1,s2:', data_vec1.shape, s1, s2)
                data_mat1 = data_vec1.reshape(int(s2), int(s1)).T
                results_p.append(data_mat1)

            else:
                cont_cond = False

        f.close()
        return results_p

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
            Excitation waveform containing the full DC amplitude vector
        """

        h5_f = loadmat(parm_path)
        parm_dict = dict()

        parm_dict['IO_samp_rate_[Hz]'] = np.uint32(h5_f['IO_rate'][0][0])
        parm_dict['IO_time'] = np.float32(h5_f['IO_time'][0][0])

        excit_wfm = np.float32(np.squeeze(h5_f['dc_amp_vec']))

        parm_dict['grid_num_rows'] = np.int(h5_f['num_rows'][0][0])
        parm_dict['grid_num_cols'] = np.int(h5_f['num_cols'][0][0])

        return parm_dict, excit_wfm
