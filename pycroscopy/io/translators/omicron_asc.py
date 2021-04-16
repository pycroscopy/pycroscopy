# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:50:47 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from os import path
import re
import numpy as np  # For array operations
from pyUSID import Dimension
from pyUSID.io.array_translator import ArrayTranslator


class AscTranslator(ArrayTranslator):
    """
    Translates Scanning Tunnelling Spectroscopy (STS) data in .asc files obtained from Omicron STMs.
    """

    def translate(self, file_path):
        """
        Translates the provided .asc file to .h5

        Parameters
        ----------
        file_path : str
            Absolute path of the source .ASC STS file from Omicron STMs


        Returns
        -------
        h5_path : str
            Absolute path of the translated file
        """

        file_path = path.abspath(file_path)
        folder_path, file_name = path.split(file_path)
        file_name = file_name[:-4]

        # Extracting the raw data into memory
        file_handle = open(file_path, 'r')
        string_lines = file_handle.readlines()
        file_handle.close()

        # Extract parameters from the first few header lines
        parm_dict, num_headers = self._read_parms(string_lines)

        num_rows = int(parm_dict['Main-y_pixels'])
        num_cols = int(parm_dict['Main-x_pixels'])
        num_pos = num_rows * num_cols
        spectra_length = int(parm_dict['Main-z_points'])

        # Extract the STS data from subsequent lines
        raw_data_2d = self._read_data(string_lines, num_pos, spectra_length, num_headers)

        # Generate the x / voltage / spectroscopic axis:
        volt_vec = np.linspace(parm_dict['Spectroscopy-Device_1_Start [Volt]'],
                               parm_dict['Spectroscopy-Device_1_End [Volt]'],
                               spectra_length)

        h5_path = path.join(folder_path, file_name + '.h5')

        # pass on the the necessary pieces of information onto the numpy translate that will handle the creation and
        # writing to the h5 file.

        pos_dims = [Dimension('X', 'nm', np.linspace(parm_dict['Main-x_offset'], parm_dict['Main-x_length'],
                                                     parm_dict['Main-x_pixels'])),
                    Dimension('Y', 'nm', np.linspace(parm_dict['Main-y_offset'], parm_dict['Main-y_length'],
                                                     parm_dict['Main-y_pixels']))]
        spec_dims = Dimension('Bias', 'V', volt_vec)

        h5_path = super(AscTranslator, self).translate(h5_path, 'STS', raw_data_2d, 'Tunnelling current',
                                                       parm_dict['Main-value_unit'], pos_dims,
                                                       spec_dims, translator_name='ASC', parm_dict=parm_dict)

        return h5_path

    def _read_data(self, string_lines, num_pos, spectra_length, num_headers):
        """
        Reads the data from lines of the data file

        Parameters
        ----------
        string_lines : :class:`list` of str
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
    def _read_parms(string_lines):
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

        def parse_header(line):
            # ".  Aux2_V:"
            match_obj = re.match(r'\.  (.*):', line, re.M | re.I)
            type_list = [str]
            if match_obj:
                raw_vals = [type_caster(match_obj.group(ind)) for ind, type_caster in
                            zip(range(1, 1 + len(type_list)), type_list)]
                return raw_vals[0]
            else:
                return None

        def parse_parm(line):
            # ".  .  Auto_Flush_Period = 0.1 Second"
            match_obj = re.match(r'\.  \.  (.*) = (.*)', line, re.M | re.I)
            type_list = [str, str]
            if match_obj:
                raw_vals = [type_caster(match_obj.group(ind)) for ind, type_caster in
                            zip(range(1, 1 + len(type_list)), type_list)]

                # Some cleaning:
                raw_vals[0] = raw_vals[0].replace('-', '_').strip()  # We use '-' as a level separator
                raw_vals[1] = raw_vals[1].replace('--', '').strip()

                # often, units are on the values side, see if these can be transitioned over to the key:
                vals_split = raw_vals[1].split(' ')
                if len(vals_split) == 2:
                    raw_vals = [raw_vals[0] + ' [' + vals_split[1] + ']', vals_split[0]]
                try:
                    raw_vals[1] = float(raw_vals[1])
                    # convert those values that should be integers:
                    if raw_vals[1] % 1 == 0:
                        raw_vals[1] = int(raw_vals[1])
                except ValueError:
                    pass
                return {raw_vals[0]: raw_vals[1]}
            else:
                return None

        def flatten_dict(nested_dict, separator='-'):
            # From: https://codereview.stackexchange.com/questions/21033/flatten-dictionary-in-python-functional-style
            def expand(outer_key, outer_value):
                if isinstance(outer_value, dict):
                    return [(outer_key + separator + inner_key, inner_value) for inner_key, inner_value in flatten_dict(outer_value).items()]
                else:
                    return [(outer_key, outer_value)]

            items = [item for outer_key, outer_value in nested_dict.items() for item in expand(outer_key, outer_value)]

            return dict(items)

        # #############################################################################################################

        temp_dict = dict()

        line = string_lines[1]
        if line.startswith('# Created by SPIP'):
            line = line.replace('# Created by SPIP ', '').replace('\n', '')
            ind = line.index(' ')
            temp_dict['SPIP_version'] = line[:ind]
            temp_dict['creation_time'] = line[ind + 1:]

        # #################################################################################################

        line_offset = 3
        for line_ind, line in enumerate(string_lines[line_offset:]):
            if parse_header(line) is not None:
                line_offset += line_ind
                break
            line = line.replace('# ', '')
            line = line.replace('\n', '')
            temp = line.split('=')
            test = temp[1].strip()
            try:
                test = float(test)
                # convert those values that should be integers:
                if test % 1 == 0:
                    test = int(test)
            except ValueError:
                pass
            temp_dict[temp[0].strip().replace('-', '_')] = test

        main_dict = {'Main': temp_dict.copy()}

        # #################################################################################################

        curr_cat_name = None
        temp_dict = dict()

        for ind, line in enumerate(string_lines[line_offset:]):
            if line.strip().startswith('# Start of Data:'):
                line_offset += ind + 1
                break
            header_name = parse_header(line)
            if header_name:
                if curr_cat_name is not None:
                    main_dict[curr_cat_name] = temp_dict.copy()
                    temp_dict = dict()
                curr_cat_name = header_name
            else:
                this_parm = parse_parm(line)
                if this_parm is None:
                    continue
                temp_dict.update(this_parm)
        if len(temp_dict) > 0:
            main_dict[curr_cat_name] = temp_dict.copy()

        # #################################################################################################

        return flatten_dict(main_dict), line_offset
