# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:58:35 2017

@author: Suhas Somnath
"""

from __future__ import division
import numpy as np  # For array operations
from os import path
from .numpy_translator import NumpyTranslator


class AscTranslator(NumpyTranslator):
    """
    Translates Scanning Tunnelling Spectroscopy (STS) data in .asc files obtained from Omicron STMs.
    """

    def translate(self, file_path, max_v=1):

        # Extracting the data into memory
        folder_path, file_name = path.split(file_path)
        file_name = file_name[:-4]

        file_handle = open(file_path, 'r')
        string_lines = file_handle.readlines()
        file_handle.close()

        parm_dict = self.__read_parms(string_lines)

        num_rows = int(parm_dict['y-pixels'])
        num_cols = int(parm_dict['x-pixels'])
        num_pos = num_rows * num_cols
        spectra_length = int(parm_dict['z-points'])

        # num_headers = len(string_lines) - num_pos
        num_headers = 403

        raw_data_2d = self._read_data(string_lines, num_pos, spectra_length, num_headers)

        # Generate the x axis:
        volt_vec = np.linspace(-1 * max_v, 1 * max_v, spectra_length)

        h5_path = path.join(folder_path, file_name + '.h5')

        h5_path = super(AscTranslator, self).translate(h5_path, raw_data_2d, num_rows, num_cols, qty_name='Current',
                                                       data_unit='nA', spec_name='Bias', spec_unit='V',
                                                       spec_val=volt_vec, scan_height=100, scan_width=200,
                                                       spatial_unit='nm')

        return h5_path

    def _read_data(self, string_lines, num_pos, spectra_length, num_headers):
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
