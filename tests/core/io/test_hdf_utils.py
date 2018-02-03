# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

import unittest
import os
import sys
import h5py
import numpy as np
sys.path.append("../../../pycroscopy/")
from pycroscopy import MicroDataGroup, MicroDataset
from pycroscopy import HDFwriter
from pycroscopy.core.io import hdf_utils


class TestHDFUtils(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def __create_test_h5_file():
        file_path = 'test_hdf_utils.h5'
        TestHDFUtils.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            num_rows = 3
            num_cols = 5
            source_dset_name = 'source_main'
            tool_name = 'Fitter'

            source_pos_data = np.vstack((np.tile(np.arange(num_rows), num_cols),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            dset_source_pos_inds = MicroDataset('Position_Indices', source_pos_data, dtype=np.uint16)
            dset_source_pos_vals = MicroDataset('Position_Values', source_pos_data, dtype=np.float16)

            num_cycles = 2
            num_cycle_pts = 7

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = MicroDataset(source_dset_name, source_main_data,
                                            attrs={'units': 'A', 'quantity': 'Current'})

            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            dset_source_spec_inds = MicroDataset('Spectroscopic_Indices', source_spec_data, dtype=np.uint16)
            dset_source_spec_vals = MicroDataset('Spectroscopic_Values', source_spec_data, dtype=np.float16)

            group_source = MicroDataGroup('Raw_Measurement',
                                          children=[dset_source_main, dset_source_spec_inds, dset_source_spec_vals,
                                                    dset_source_pos_vals, dset_source_pos_inds],
                                          attrs={'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4],
                                                 'att_4': ['str_1', 'str_2', 'str_3']})

            writer = HDFwriter(h5_f)
            h5_refs_list = writer.write(group_source)

            [h5_source_main] = hdf_utils.get_h5_obj_refs([dset_source_main.name], h5_refs_list)
            h5_source_group = h5_source_main.parent
            [h5_pos_inds] = hdf_utils.get_h5_obj_refs([dset_source_pos_inds.name], h5_refs_list)
            [h5_pos_vals] = hdf_utils.get_h5_obj_refs([dset_source_pos_vals.name], h5_refs_list)
            [h5_source_spec_inds] = hdf_utils.get_h5_obj_refs([dset_source_spec_inds.name], h5_refs_list)
            [h5_source_spec_vals] = hdf_utils.get_h5_obj_refs([dset_source_spec_vals.name], h5_refs_list)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            # Now add a few results:

            num_cycles = 1
            num_cycle_pts = 7

            results_spec_data = np.expand_dims(np.arange(num_cycle_pts), 0)
            dset_results_spec_inds = MicroDataset('Spectroscopic_Indices', results_spec_data, dtype=np.uint16)
            dset_results_spec_vals = MicroDataset('Spectroscopic_Values', results_spec_data, dtype=np.float16)

            results_1_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_results_1_main = MicroDataset('results_main', results_1_main_data,
                                               attrs={'units': 'pF', 'quantity': 'Capacitance'})

            group_results_1 = MicroDataGroup(source_dset_name + '_' + tool_name + '_',
                                             parent=h5_source_group.name,
                                             children=[dset_results_1_main, dset_results_spec_inds,
                                                       dset_results_spec_vals],
                                             attrs={'att_1': 'string_val', 'att_2': 1.2345,
                                                    'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']})

            h5_refs_list = writer.write(group_results_1)

            [h5_results_1_main] = hdf_utils.get_h5_obj_refs([dset_results_1_main.name], h5_refs_list)
            [h5_results_1_spec_inds] = hdf_utils.get_h5_obj_refs([dset_results_spec_inds.name], h5_refs_list)
            [h5_results_1_spec_vals] = hdf_utils.get_h5_obj_refs([dset_results_spec_vals.name], h5_refs_list)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_1_spec_inds, h5_results_1_spec_vals]:
                h5_results_1_main.attrs[dset.name.split('/')[-1]] = dset.ref

            # add another result with different parameters

            results_2_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_results_2_main = MicroDataset('results_main', results_2_main_data,
                                               attrs={'units': 'pF', 'quantity': 'Capacitance'})

            group_results_2 = MicroDataGroup(source_dset_name + '_' + tool_name + '_',
                                             parent=h5_source_group.name,
                                             children=[dset_results_2_main, dset_results_spec_inds,
                                                       dset_results_spec_vals],
                                             attrs={'att_1': 'other_string_val', 'att_2': 5.4321,
                                                    'att_3': [4, 1, 3], 'att_4': ['s', 'str_2', 'str_3']})

            h5_refs_list = writer.write(group_results_2)

            [h5_results_2_main] = hdf_utils.get_h5_obj_refs([dset_results_2_main.name], h5_refs_list)
            [h5_results_2_spec_inds] = hdf_utils.get_h5_obj_refs([dset_results_spec_inds.name], h5_refs_list)
            [h5_results_2_spec_vals] = hdf_utils.get_h5_obj_refs([dset_results_spec_vals.name], h5_refs_list)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_2_spec_inds, h5_results_2_spec_vals]:
                h5_results_2_main.attrs[dset.name.split('/')[-1]] = dset.ref

    def test_something(self):
        self.__create_test_h5_file()


if __name__ == '__main__':
    unittest.main()
