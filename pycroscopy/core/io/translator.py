# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import abc
import time as tm
from os import path, remove

from .io_utils import get_available_memory
from .virtual_data import VirtualGroup, VirtualDataset
from .hdf_utils import get_h5_obj_refs, link_h5_objects_as_attrs
from .hdf_writer import HDFwriter  # Now the translator is responsible for writing the data.


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
        self.max_ram = min(max_mem_mb * 1024 ** 2, 0.75 * get_available_memory())

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
        ds_main : VirtualDataset object
            Main dataset
        aux_dset_list : list of VirtualDataset objects
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
        chan_grp = VirtualGroup('Channel_000')
        chan_grp.add_children([ds_main])
        chan_grp.add_children(aux_dset_list)
        meas_grp = VirtualGroup('Measurement_000')
        meas_grp.attrs = parm_dict
        meas_grp.add_children([chan_grp])
        spm_data = VirtualGroup('')
        global_parms = generate_dummy_main_parms()
        global_parms['data_type'] = data_name
        global_parms['translator'] = translator_name
        spm_data.attrs = global_parms
        spm_data.add_children([meas_grp])

        aux_dset_names = list()
        for dset in aux_dset_list:
            if isinstance(dset, VirtualDataset):
                aux_dset_names.append(dset.name)

        if path.exists(h5_path):
            remove(h5_path)

        hdf = HDFwriter(h5_path)
        h5_refs = hdf.write(spm_data, print_log=False)
        h5_raw = get_h5_obj_refs([ds_main.name], h5_refs)[0]
        link_h5_objects_as_attrs(h5_raw, get_h5_obj_refs(aux_dset_names, h5_refs))
        hdf.close()
        return h5_path


def generate_dummy_main_parms():
    """
    Generates a (dummy) dictionary of parameters that will be used at the root level of the h5 file

    Returns
    ----------
    main_parms : dictionary
        Dictionary containing basic descriptors that describe a dataset
    """
    main_parms = dict()
    main_parms['translate_date'] = tm.strftime("%Y_%m_%d")
    main_parms['instrument'] = 'cypher_west'
    main_parms['xcams_id'] = 'abc'
    main_parms['user_name'] = 'John Doe'
    main_parms['sample_name'] = 'PZT'
    main_parms['sample_description'] = 'Thin Film'
    main_parms['project_name'] = 'Band Excitation'
    main_parms['project_id'] = 'CNMS_2015B_X0000'
    main_parms['comments'] = 'Band Excitation data'
    main_parms['data_tool'] = 'be_analyzer'
    # This parameter actually need not be a dummy and can be extracted from the parms file
    main_parms['experiment_date'] = '2015-10-05 14:55:05'
    main_parms['experiment_unix_time'] = tm.time()
    # Need to fill in the x and y grid size here
    main_parms['grid_size_x'] = 1
    main_parms['grid_size_y'] = 1
    # Need to fill in the current X, Y, Z, Laser position here
    main_parms['current_position_x'] = 1
    main_parms['current_position_y'] = 1

    return main_parms


