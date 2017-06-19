# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import abc
from os import path, remove
from ..io_utils import getAvailableMem
from ..microdata import MicroDataGroup, MicroDataset
from .utils import generate_dummy_main_parms
from ..hdf_utils import getH5DsetRefs, linkRefs
from ..io_hdf5 import ioHDF5  # Now the translator is responsible for writing the data.


class Translator(object):
    """
    Abstract class that defines the most basic functionality of a data format translator.
    A translator converts experimental data from binary / proprietary
    data formats to a single standardized HDF5 data file
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_mem_mb=1024):
        """
        Parameters
        -----------
        max_mem_mb : unsigned integer (Optional. Default = 1024)
            Maximum system memory (in megabytes) that the translator can use
            
        Returns
        -------
        Translator object
        """
        self.max_ram = min(max_mem_mb*1024**2, 0.75*getAvailableMem())

    @abc.abstractmethod
    def translate(self, filepath):
        """
        Abstract method.
        To be implemented by extensions of this class. God I miss Java!
        """
        pass

    @abc.abstractmethod
    def _parse_file_path(self, input_path):
        """
        Abstract method
        Parses the `input_path` to determine the `basename` and find
        the appropriate data files

        """
        pass

    @abc.abstractmethod
    def _read_data(self):
        """
        Abstract method
        Reads the data into the hdf5 datasets.
        """

    @staticmethod
    def simple_write(h5_path, data_name, translator_name, ds_main, aux_dset_list, parm_dict=None):
        """
        Writes the provided datasets and parameters to an h5 file
        
        Parameters
        ----------
        h5_path : String / Unicode
            Absolute path of the h5 file to be written
        data_name : String / Unicode
            Name of the data type
        translator_name : String / unicode
            Name of the translator
        ds_main : MicroDataset object
            Main dataset
        aux_dset_list : list of MicroDataset objects
            auxillary datasets to be written to the file
        parm_dict : dictionary (Optional)
            Dictionary of parameters

        Returns
        -------
        h5_path : String / unicode
            Absolute path of the written h5 file

        """
        if parm_dict is None:
            parm_dict = {}
        chan_grp = MicroDataGroup('Channel_000')
        chan_grp.addChildren([ds_main])
        chan_grp.addChildren(aux_dset_list)
        meas_grp = MicroDataGroup('Measurement_000')
        meas_grp.attrs = parm_dict
        meas_grp.addChildren([chan_grp])
        spm_data = MicroDataGroup('')
        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = data_name
        global_parms['translator'] = translator_name
        spm_data.attrs = global_parms
        spm_data.addChildren([meas_grp])

        aux_dset_names = list()
        for dset in aux_dset_list:
            if isinstance(dset, MicroDataset):
                aux_dset_names.append(dset.name)

        if path.exists(h5_path):
            remove(h5_path)

        hdf = ioHDF5(h5_path)
        h5_refs = hdf.writeData(spm_data, print_log=False)
        h5_raw = getH5DsetRefs([ds_main.name], h5_refs)[0]
        linkRefs(h5_raw, getH5DsetRefs(aux_dset_names, h5_refs))
        hdf.close()
        return h5_path
