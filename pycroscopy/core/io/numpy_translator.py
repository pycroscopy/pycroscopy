# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:58:35 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from os import path, remove
import sys
import h5py
import numpy as np  # For array operations

from .write_utils import AuxillaryDescriptor
from .translator import Translator, generate_dummy_main_parms
from .hdf_utils import write_main_dataset, write_simple_attrs

if sys.version_info.major == 3:
    unicode = str


class NumpyTranslator(Translator):
    """
    Writes a numpy array to .h5
    """

    def translate(self, h5_path, data_name, raw_data, quantity, units, pos_dims, spec_dims,
                  translator_name='NumpyTranslator', parm_dict=None):
        """
        Writes the provided datasets and parameters to an h5 file

        Parameters
        ----------
        h5_path : String / Unicode
            Absolute path of the h5 file to be written
        data_name : String / Unicode
            Name of the data type. Example - 'SEM'
        raw_data : np.ndarray
            2D matrix formatted as [position, spectral]
        quantity : String / Unicode
            Name of the physical quantity stored in the dataset. Example - 'Current'
        units : String / Unicode
            Name of units for the quantity stored in the dataset. Example - 'A' for amperes
        pos_dims : AuxillaryDescriptor
            Object specifying the instructions necessary for building the Position indices and values datasets
        spec_dims : AuxillaryDescriptor
            Object specifying the instructions necessary for building the Spectroscopic indices and values datasets
        translator_name : String / unicode, Optional
            Name of the translator. Example - 'HitachiSEMTranslator'
        parm_dict : dictionary (Optional)
            Dictionary of parameters that will be written under the group 'Measurement_000'

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the written h5 file

        """
        for arg in [h5_path, data_name, translator_name, quantity, units]:
            assert isinstance(arg, (str, unicode))
            assert len(arg) > 0
        assert isinstance(raw_data, np.ndarray)
        assert raw_data.ndim == 2

        for ind, anc_dic in enumerate([pos_dims, spec_dims]):
            assert isinstance(anc_dic, AuxillaryDescriptor)
            # Check to make sure that the product of the position and spectroscopic dimension sizes match with
            # that of raw_data
            assert raw_data.shape[ind] == np.product(anc_dic.sizes)

        if path.exists(h5_path):
            remove(h5_path)

        if parm_dict is None:
            parm_dict = {}

        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = data_name
        global_parms['translator'] = translator_name

        # Begin writing to file:
        with h5py.File(h5_path) as h5_f:

            # Root attributes first:
            write_simple_attrs(h5_f, global_parms)

            # measurement group next
            meas_grp = h5_f.create_group('Measurement_000')
            write_simple_attrs(meas_grp, parm_dict)

            # channel group next
            chan_grp = meas_grp.create_group('Channel_000')

            _ = write_main_dataset(chan_grp, raw_data, 'Raw_Data', quantity, units, pos_dims, spec_dims)

        return h5_path
