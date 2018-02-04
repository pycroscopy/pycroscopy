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
from pycroscopy.core.io.pycro_data import PycroDataset

test_h5_file_path = 'test_hdf_utils.h5'

class TestHDFUtils(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def __ensure_test_h5_file():
        if not os.path.exists(test_h5_file_path):
            TestHDFUtils.__create_test_h5_file()

    @staticmethod
    def __create_test_h5_file():
        if os.path.exists(test_h5_file_path):
            os.remove(test_h5_file_path)
        with h5py.File(test_h5_file_path) as h5_f:
            num_rows = 3
            num_cols = 5
            source_dset_name = 'source_main'
            tool_name = 'Fitter'

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            dset_source_pos_inds = MicroDataset('Position_Indices', source_pos_data, dtype=np.uint16,
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'nm']})
            dset_source_pos_vals = MicroDataset('Position_Values', source_pos_data, dtype=np.float16,
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'nm']})

            num_cycles = 2
            num_cycle_pts = 7

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = MicroDataset(source_dset_name, source_main_data,
                                            attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}
                                                   })

            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            dset_source_spec_inds = MicroDataset('Spectroscopic_Indices', source_spec_data, dtype=np.uint16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})
            dset_source_spec_vals = MicroDataset('Spectroscopic_Values', source_spec_data, dtype=np.float16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})

            dset_ancillary = MicroDataset('Ancillary', np.arange(5))

            group_source = MicroDataGroup('Raw_Measurement',
                                          children=[dset_source_main, dset_source_spec_inds, dset_source_spec_vals,
                                                    dset_source_pos_vals, dset_source_pos_inds, dset_ancillary],
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
            dset_results_spec_inds = MicroDataset('Spectroscopic_Indices', results_spec_data, dtype=np.uint16,
                                                  attrs={'labels': ['Bias'], 'units': ['V']})
            dset_results_spec_vals = MicroDataset('Spectroscopic_Values', results_spec_data, dtype=np.float16,
                                                  attrs={'labels': ['Bias'], 'units': ['V']})

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

    # start off with the most popular functions:
    def test_get_attr_illegal_01(self):
        self.__ensure_test_h5_file()
        with self.assertRaises(AssertionError):
            hdf_utils.get_attr(np.arange(3), 'units')

    def test_get_attr_illegal_02(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(AssertionError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 14)

    def test_get_attr_illegal_03(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(AssertionError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], ['quantity', 'units'])

    def test_get_attr_illegal_04(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_get_attr_legal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'units')
            self.assertEqual(returned, 'A')

    def test_get_attr_legal_02(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/Position_Indices'], 'labels')
            self.assertTrue(np.all(returned == ['X', 'Y']))

    def test_get_attr_legal_03(self):
        self.__ensure_test_h5_file()
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main_Fitter_000']
            for key, expected_value in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_value))

    def test_get_attributes_01(self):
        self.__ensure_test_h5_file()
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        sub_attrs = ['att_1', 'att_4', 'att_3']
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main_Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group, sub_attrs)
            self.assertIsInstance(returned_attrs, dict)
            for key in sub_attrs:
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_get_attributes_all(self):
        self.__ensure_test_h5_file()
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main_Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group)
            self.assertIsInstance(returned_attrs, dict)
            for key in attrs.keys():
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_get_attributes_illegal(self):
        self.__ensure_test_h5_file()
        sub_attrs = ['att_1', 'att_4', 'does_not_exist']
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main_Fitter_000']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_attributes(h5_group, sub_attrs)

    def test_get_auxillary_datasets_single(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxillary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_get_auxillary_datasets_single(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxillary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_get_auxillary_datasets_multiple(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_pos_vals = h5_f['/Raw_Measurement/Position_Values']
            ret_val = hdf_utils.get_auxillary_datasets(h5_main, aux_dset_name=['Position_Indices',
                                                                               'Position_Values'])
            self.assertEqual(set(ret_val), set([h5_pos_inds, h5_pos_vals]))

    def test_get_auxillary_datasets_all(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = [h5_f['/Raw_Measurement/Position_Indices'],
                        h5_f['/Raw_Measurement/Position_Values'],
                        h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Spectroscopic_Values']]
            ret_val = hdf_utils.get_auxillary_datasets(h5_main)
            self.assertEqual(set(expected), set(ret_val))

    def test_get_auxillary_datasets_illegal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_auxillary_datasets(h5_main, aux_dset_name='Does_Not_Exist')

    def test_get_data_descriptor_main(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            self.assertEqual(hdf_utils.get_data_descriptor(h5_main), 'Current (A)')

    def test_get_data_descriptor_main(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_pos = h5_f['/Raw_Measurement/Ancillary']
            self.assertEqual(hdf_utils.get_data_descriptor(h5_pos), 'unknown quantity (unknown units)')

    def test_get_dimensionality_legal_no_sort(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_shapes = [[7, 2],
                               [7],
                               [5, 3]]
            for h5_dset, exp_shape in zip(h5_dsets, expected_shapes):
                self.assertTrue(np.all(exp_shape == hdf_utils.get_dimensionality(h5_dset)))

    def test_get_dimensionality_legal_w_sort(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_shapes = [[2, 7],
                               [7],
                               [3, 5]]
            sort_orders = [[1, 0],
                           [0],
                           [1, 0]]
            for h5_dset, s_oder, exp_shape in zip(h5_dsets, sort_orders, expected_shapes):
                self.assertTrue(np.all(exp_shape == hdf_utils.get_dimensionality(h5_dset, index_sort=s_oder)))

    def test_get_formatted_labels_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_labels = [['Bias (V)', 'Cycle ()'], ['Bias (V)'], ['X (nm)', 'Y (nm)']]
            for h5_dset, exp_labs in zip(h5_dsets, expected_labels):
                self.assertTrue(np.all(exp_labs == hdf_utils.get_formatted_labels(h5_dset)))

    def test_get_formatted_labels_illegal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Ancillary'],
                        h5_f['/Raw_Measurement/source_main']]  # This will have labels and units but of different sizes
            for h5_dset, err_type in zip(h5_dsets, [KeyError, ValueError]):
                with self.assertRaises(err_type):
                    _ = hdf_utils.get_formatted_labels(h5_dset)

    def test_get_group_refs_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_refs = [h5_f['/Raw_Measurement/Ancillary'],
                       h5_f['/Raw_Measurement/source_main_Fitter_000'],
                       h5_f['/Raw_Measurement/source_main_Fitter_001'],
                       h5_f['/Raw_Measurement/source_main_Fitter_000/results_main']]
            group_prefix = 'source_main_Fitter'
            expected_objs = set([h5_f['/Raw_Measurement/source_main_Fitter_000'],
                                 h5_f['/Raw_Measurement/source_main_Fitter_001']])
            ret_vals = set(hdf_utils.get_group_refs(group_prefix, h5_refs))
            self.assertTrue(ret_vals == expected_objs)

    def test_get_group_refs_failure(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_refs = [h5_f['/Raw_Measurement/Ancillary'],
                       h5_f,
                       np.arange(15),
                       h5_f['/Raw_Measurement/source_main_Fitter_000/results_main']]
            group_prefix = 'source_main_Blah'
            self.assertTrue(hdf_utils.get_group_refs(group_prefix, h5_refs) == [])

    def test_get_h5_obj_refs_legal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f,
                              4.123,
                              np.arange(6),
                              h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/Spectroscopic_Values']]
            chosen_objs = [h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Position_Indices', 'source_main_Fitter_000', 'Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(chosen_objs), set(returned_h5_objs))

    def test_get_h5_obj_refs_same_name(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f['/Raw_Measurement/source_main_Fitter_001/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/Spectroscopic_Values']]
            expected_objs = [h5_f['/Raw_Measurement/source_main_Fitter_001/Spectroscopic_Indices'],
                             h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(expected_objs), set(returned_h5_objs))

    def test_check_is_main_legal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main_Fitter_001/results_main']]
            not_main_dsets = [h5_f,
                              4.123,
                              np.arange(6),
                              h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/Spectroscopic_Values']]
            for dset in expected_dsets:
                self.assertTrue(hdf_utils.check_if_main(dset))
            for dset in not_main_dsets:
                self.assertFalse(hdf_utils.check_if_main(dset))

    def test_get_sort_order_simple(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main_Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_order = [[0, 1], [0], [0, 1]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    """           
    def test_get_region_ref_indices(self):
        assert False

    def test_get_all_main_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main_Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main_Fitter_001/results_main']]
            main_dsets = hdf_utils.get_all_main(h5_f, verbose=False)
            for dset in main_dsets:
                self.assertTrue(dset in expected_dsets)
    """

if __name__ == '__main__':
    unittest.main()
