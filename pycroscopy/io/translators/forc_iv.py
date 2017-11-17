# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:58:35 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, remove
import h5py
import numpy as np  # For array operations
from scipy.io import loadmat

from .translator import Translator
from .utils import build_ind_val_dsets
from ..microdata import MicroDataset  # building blocks for defining hierarchical storage in the H5 file


class ForcIVTranslator(Translator):
    """
    Translates FORC IV datasets from .mat files to .h5
    """
    def __init__(self, *args, **kwargs):
        super(ForcIVTranslator, self).__init__(*args, **kwargs)
        self.h5_read = None

    def _read_data(self):
        pass

    def _parse_file_path(self, input_path):
        pass

    def translate(self, raw_data_path):
        """
        The main function that translates the provided file into a .h5 file

        Parameters
        ------------
        raw_data_path : string / unicode
            Absolute file path of the data .mat file.

        Returns
        ----------
        h5_path : string / unicode
            Absolute path of the translated h5 file
        """
        folder_path, file_name = path.split(raw_data_path)

        h5_path = path.join(folder_path, file_name[:-4] + '.h5')
        if path.exists(h5_path):
            remove(h5_path)

        self.h5_read = True
        try:
            h5_f = h5py.File(raw_data_path, 'r')
        except ImportError:
            self.h5_read = False
            h5_f = loadmat(raw_data_path)

        excite_cell = h5_f['dc_amp_cell3']
        test = excite_cell[0][0]
        if self.h5_read:
            excitation_vec = h5_f[test]
        else:
            excitation_vec = np.float32(np.squeeze(test))

        current_cell = h5_f['current_cell3']

        num_rows = current_cell.shape[0]
        num_cols = current_cell.shape[1]
        num_iv_pts = excitation_vec.size

        current_data = np.zeros(shape=(num_rows * num_cols, num_iv_pts), dtype=np.float32)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                pix_ind = row_ind * num_cols + col_ind
                if self.h5_read:
                    curr_val = np.squeeze(h5_f[current_cell[row_ind][col_ind]].value)
                else:
                    curr_val = np.float32(np.squeeze(current_cell[row_ind][col_ind]))
                current_data[pix_ind, :] = 1E+9 * curr_val

        parm_dict = self._read_parms(h5_f)

        ds_main = MicroDataset('Raw_Data', data=current_data, dtype=np.float32, compression='gzip',
                               chunking=(min(num_cols * num_rows, 100), num_iv_pts))
        ds_main.attrs = {'quantity': 'Current', 'units': '1E-9 A'}

        ds_pos_ind, ds_pos_val = build_ind_val_dsets([num_rows, num_cols], is_spectral=False,
                                                     labels=['Y', 'X'], units=['m', 'm'], verbose=False)
        ds_spec_inds, ds_spec_vals = build_ind_val_dsets([num_iv_pts], is_spectral=True,
                                                         labels=['DC Bias'], units=['V'], verbose=False)
        ds_spec_vals.data = np.atleast_2d(excitation_vec)

        return super(ForcIVTranslator, self).simple_write(h5_path, 'FORC_IV', ds_main,
                                                          [ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals],
                                                          parm_dict)

    def _read_parms(self, raw_data_file_handle):
        """
        Copies experimental parameters from the .mat file to a dictionary

        Parameters
        ----------
        raw_data_file_handle : h5py.Group object or dictionary
            Handle to the file containing the raw data

        Returns
        -------
        parm_dict : dictionary
            Dictionary containing all relevant parameters
        """
        parm_dict = dict()
        exceptions = ['Z_cell3', 'current_cell3', 'dc_amp_cell3', 'loop_area_rel', '__version__', '__header__',
                      '__globals__']
        for att_name in raw_data_file_handle:
            if att_name not in exceptions:
                if not self.h5_read:
                    parm_dict[att_name] = raw_data_file_handle[att_name][0][0]
        return parm_dict
