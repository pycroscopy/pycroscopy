# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:50:47 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np  # For array operations
from os import path
from pyUSID.io.write_utils import Dimension
from pyUSID.io.numpy_translator import NumpyTranslator


class AscTranslator(NumpyTranslator):
    """
    Translates Scanning Tunnelling Spectroscopy (STS) data in .asc files obtained from Omicron STMs.
    """

    def translate(self, file_path, max_v=1):
        """
        Translates the provided .asc file to .h5

        Parameters
        ----------
        file_path : string / unicode
            Absolute path of the source .ASC STS file from Omicron STMs
        max_v : float


        Returns
        -------
        h5_path : string / unicode
            Absolute path of the translated file
        """

        max_v = abs(max_v)

        file_path = path.abspath(file_path)
        folder_path, file_name = path.split(file_path)
        file_name = file_name[:-4]

        # Extracting the raw data into memory
        file_handle = open(file_path, 'r')
        string_lines = file_handle.readlines()
        file_handle.close()

        # Extract parameters from the first few header lines
        parm_dict = self.__read_parms(string_lines)

        num_rows = int(parm_dict['y-pixels'])
        num_cols = int(parm_dict['x-pixels'])
        num_pos = num_rows * num_cols
        spectra_length = int(parm_dict['z-points'])

        # num_headers = len(string_lines) - num_pos
        num_headers = 403

        # Extract the STS data from subsequent lines
        raw_data_2d = self._read_data(string_lines, num_pos, spectra_length, num_headers)

        # Generate the x / voltage / spectroscopic axis:
        volt_vec = np.linspace(-1 * max_v, 1 * max_v, spectra_length)

        h5_path = path.join(folder_path, file_name + '.h5')

        # pass on the the necessary pieces of information onto the numpy translate that will handle the creation and
        # writing to the h5 file.

        pos_dims = [Dimension('X', 'nm', num_cols), Dimension('Y', 'nm', num_rows)]
        spec_dims = Dimension('Bias', 'V', volt_vec)

        h5_path = super(AscTranslator, self).translate(h5_path, 'STS', raw_data_2d, 'Current', 'nA', pos_dims,
                                                       spec_dims, translator_name='ASC', parm_dict=parm_dict)

        return h5_path

    def _read_data(self, string_lines, num_pos, spectra_length, num_headers):
        """
        Reads the data from lines of the data file

        Parameters
        ----------
        string_lines : list of strings
            Lines containing the data in string format, separated by tabs
        num_pos : unsigned int
            Number of pixels
        spectra_length : unsigned int
            Number of points in the spectral / voltage axis
        num_headers : unsigned int
            Number of header lines to ignore

        Returns
        -------
        raw_data_2d : 2D numpy array
            Data arranged as [position x voltage points]
        """
        raw_data_2d = np.zeros(shape=(num_pos, spectra_length), dtype=np.float32)
        for line_ind in range(num_pos):
            this_line = string_lines[num_headers + line_ind]
            string_spectrum = this_line.split('\t')[:-1]  # omitting the new line
            raw_data_2d[line_ind] = np.array(string_spectrum, dtype=np.float32)
        return raw_data_2d

    def _parse_file_path(self, input_path):
        pass

    @staticmethod
    def __read_parms(string_lines):
        """
        Returns the parameters regarding the experiment as dictionary

        Parameters
        ----------
        string_lines : list of strings
            Lines from the data file in string representation

        Returns
        -------
        parm_dict : dictionary
            Dictionary of parameters regarding the experiment
        """
        # Reading parameters stored in the first few rows of the file
        parm_dict = dict()
        for line in string_lines[3:17]:
            line = line.replace('# ', '')
            line = line.replace('\n', '')
            temp = line.split('=')
            test = temp[1].strip()
            try:
                test = float(test)
            except ValueError:
                pass
            parm_dict[temp[0].strip()] = test

        return parm_dict
