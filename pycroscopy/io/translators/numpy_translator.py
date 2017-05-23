# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:58:35 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np  # For array operations

from .translator import Translator
from .utils import build_ind_val_dsets
from ..hdf_utils import calc_chunks
from ..microdata import MicroDataset  # building blocks for defining hierarchical storage in the H5 file


class NumpyTranslator(Translator):
    """
    Writes a numpy array to .h5
    """

    def _read_data(self):
        pass

    def _parse_file_path(self, input_path):
        pass

    def translate(self, h5_path, main_data, num_rows, num_cols, qty_name='Unknown', data_unit='a. u.',
                  spec_name='Spectroscopic_Variable', spec_val=None, spec_unit='a. u.', data_type='generic',
                  translator_name='numpy', scan_height=None, scan_width=None, spatial_unit='m', parms_dict={}):
        """
        The main function that translates the provided data into a .h5 file

        Parameters
        ----------
        h5_path
        main_data
        num_rows
        num_cols
        qty_name
        data_unit
        spec_name
        spec_val
        spec_unit
        data_type
        translator_name
        scan_height
        scan_width
        spatial_unit
        parms_dict

        Returns
        -------
        h5_path : string / unicode
            Absolute path of the translated h5 file
        """
        if main_data.ndim != 2:
            raise ValueError('Main dataset must be a 2-dimensional array arranged as [positions x spectra]')

        spectra_length = main_data.shape[1]

        ds_main = MicroDataset('Raw_Data', data=main_data, dtype=np.float32, compression='gzip',
                               chunking=calc_chunks(main_data.shape, np.float32(0).itemsize,
                                                    unit_chunks=(1, spectra_length)))
        ds_main.attrs = {'quantity': qty_name, 'units': data_unit}

        pos_steps = None
        if scan_width is not None and scan_height is not None:
            pos_steps = [1.0 * scan_height / num_rows, 1.0 * scan_width / num_cols]

        ds_pos_ind, ds_pos_val = build_ind_val_dsets([num_rows, num_cols], is_spectral=False, steps=pos_steps,
                                                     labels=['Y', 'X'], units=[spatial_unit, spatial_unit],
                                                     verbose=False)
        ds_spec_inds, ds_spec_vals = build_ind_val_dsets([spectra_length], is_spectral=True,
                                                         labels=[spec_name], units=[spec_unit], verbose=False)
        if spec_val is not None:
            if type(spec_val) in [list, np.ndarray]:
                ds_spec_vals.data = np.float32(np.atleast_2d(spec_val))

        parms_dict.update({'translator': 'NumpyTranslator'})

        return super(NumpyTranslator, self).simple_write(h5_path, data_type, translator_name, ds_main,
                                                         [ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals],
                                                         parm_dict=parms_dict)
