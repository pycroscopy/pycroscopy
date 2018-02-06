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
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'um']})
            # make the values more interesting:
            source_pos_data = np.vstack((source_pos_data[:, 0] * 50, source_pos_data[:, 1] * 1.25)).T
            dset_source_pos_vals = MicroDataset('Position_Values', source_pos_data, dtype=np.float16,
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'um']})

            num_cycles = 2
            num_cycle_pts = 7

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = MicroDataset(source_dset_name, source_main_data,
                                            attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}
                                                   })
            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            dset_source_spec_inds = MicroDataset('Spectroscopic_Indices', source_spec_data, dtype=np.uint16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})
            source_spec_data = np.vstack((np.tile(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)),
                                                  num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            dset_source_spec_vals = MicroDataset('Spectroscopic_Values', source_spec_data, dtype=np.float16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})

            dset_ancillary = MicroDataset('Ancillary', np.arange(5))

            group_source = MicroDataGroup('Raw_Measurement',
                                          children=[dset_source_main, dset_source_spec_inds, dset_source_spec_vals,
                                                    dset_source_pos_vals, dset_source_pos_inds, dset_ancillary,
                                                    MicroDataGroup('Misc')],
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

            results_spec_inds = np.expand_dims(np.arange(num_cycle_pts), 0)
            dset_results_spec_inds = MicroDataset('Spectroscopic_Indices', results_spec_inds, dtype=np.uint16,
                                                  attrs={'labels': ['Bias'], 'units': ['V']})
            results_spec_vals = np.expand_dims(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)), 0)
            dset_results_spec_vals = MicroDataset('Spectroscopic_Values', results_spec_vals, dtype=np.float16,
                                                  attrs={'labels': ['Bias'], 'units': ['V']})

            results_1_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_results_1_main = MicroDataset('results_main', results_1_main_data,
                                               attrs={'units': 'pF', 'quantity': 'Capacitance'})

            group_results_1 = MicroDataGroup(source_dset_name + '-' + tool_name + '_',
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

            group_results_2 = MicroDataGroup(source_dset_name + '-' + tool_name + '_',
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
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            for key, expected_value in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_value))

    def test_get_attributes_01(self):
        self.__ensure_test_h5_file()
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        sub_attrs = ['att_1', 'att_4', 'att_3']
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group, sub_attrs)
            self.assertIsInstance(returned_attrs, dict)
            for key in sub_attrs:
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_get_attributes_all(self):
        self.__ensure_test_h5_file()
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group)
            self.assertIsInstance(returned_attrs, dict)
            for key in attrs.keys():
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_get_attributes_illegal(self):
        self.__ensure_test_h5_file()
        sub_attrs = ['att_1', 'att_4', 'does_not_exist']
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
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
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
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
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
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
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_labels = [['Bias (V)', 'Cycle ()'], ['Bias (V)'], ['X (nm)', 'Y (um)']]
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
                       h5_f['/Raw_Measurement/source_main-Fitter_000'],
                       h5_f['/Raw_Measurement/source_main-Fitter_001'],
                       h5_f['/Raw_Measurement/source_main-Fitter_000/results_main']]
            group_prefix = 'source_main-Fitter'
            expected_objs = set([h5_f['/Raw_Measurement/source_main-Fitter_000'],
                                 h5_f['/Raw_Measurement/source_main-Fitter_001']])
            ret_vals = set(hdf_utils.get_group_refs(group_prefix, h5_refs))
            self.assertTrue(ret_vals == expected_objs)

    def test_get_group_refs_failure(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_refs = [h5_f['/Raw_Measurement/Ancillary'],
                       h5_f,
                       np.arange(15),
                       h5_f['/Raw_Measurement/source_main-Fitter_000/results_main']]
            group_prefix = 'source_main_Blah'
            self.assertTrue(hdf_utils.get_group_refs(group_prefix, h5_refs) == [])

    def test_get_h5_obj_refs_legal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f,
                              4.123,
                              np.arange(6),
                              h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/Spectroscopic_Values']]
            chosen_objs = [h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Position_Indices', 'source_main-Fitter_000', 'Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(chosen_objs), set(returned_h5_objs))

    def test_get_h5_obj_refs_same_name(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_obj_refs = [h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                           h5_f['/Raw_Measurement/Spectroscopic_Values']]
            expected_objs = [h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices'],
                             h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']]

            target_ref_names = ['Spectroscopic_Indices']

            returned_h5_objs = hdf_utils.get_h5_obj_refs(target_ref_names, h5_obj_refs)

            self.assertEqual(set(expected_objs), set(returned_h5_objs))

    def test_check_is_main_legal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            for dset in expected_dsets:
                self.assertTrue(hdf_utils.check_if_main(dset))

    def test_check_is_main_illegal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            not_main_dsets = [h5_f,
                              4.123,
                              np.arange(6),
                              h5_f['/Raw_Measurement/Position_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/Spectroscopic_Values']]
            for dset in not_main_dsets:
                self.assertFalse(hdf_utils.check_if_main(dset))

    def test_get_sort_order_simple(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_order = [[0, 1], [0], [0, 1]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    def test_get_sort_order_reversed(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [np.flipud(h5_f['/Raw_Measurement/Spectroscopic_Indices']),
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        np.fliplr(h5_f['/Raw_Measurement/Position_Indices'])]
            expected_order = [[1, 0], [0], [1, 0]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    def test_get_source_dataset_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                        h5_f['/Raw_Measurement/source_main-Fitter_001']]
            h5_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            for h5_grp in h5_groups:
                self.assertEqual(h5_main, hdf_utils.get_source_dataset(h5_grp))

    def test_get_source_dataset_illegal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_source_dataset(h5_f['/Raw_Measurement/Misc'])

    def test_get_unit_values_source_spec_all(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float16(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False))),
                        'Cycle': [0., 1.]}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, is_spec=True)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_spec_all_explicit(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float16(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False))),
                        'Cycle': [0., 1.]}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, is_spec=True, dim_names=['Cycle', 'Bias'])
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_illegal_key(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, is_spec=True, dim_names=['Cycle', 'Does not exist'])

    def test_get_unit_values_illegal_dset(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Ancillary']
            with self.assertRaises(AssertionError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, is_spec=True, dim_names=['Cycle', 'Bias'])

    def test_get_unit_values_source_spec_single(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float16(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)))}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, is_spec=True, dim_names='Bias')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_pos_all(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            num_rows = 3
            num_cols = 5
            expected = {'X': np.float16(np.arange(num_cols) * 50),
                        'Y': np.float16(np.arange(num_rows) * 1.25)}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, is_spec=False)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_pos_single(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            num_rows = 3
            expected = {'Y': np.float16(np.arange(num_rows) * 1.25)}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, is_spec=False, dim_names='Y')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_find_dataset_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            expected_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices']]
            ret_val = hdf_utils.find_dataset(h5_group, 'Spectroscopic_Indices')
            self.assertEqual(set(ret_val), set(expected_dsets))

    def test_find_dataset_illegal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            ret_val = hdf_utils.find_dataset(h5_group, 'Does_Not_Exist')
            self.assertEqual(len(ret_val), 0)

    def test_find_results_groups_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                               h5_f['/Raw_Measurement/source_main-Fitter_001']]
            ret_val = hdf_utils.find_results_groups(h5_main, 'Fitter')
            self.assertEqual(set(ret_val), set(expected_groups))

    def test_find_results_groups_illegal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ret_val = hdf_utils.find_results_groups(h5_main, 'Blah')
            self.assertEqual(len(ret_val), 0)

    def test_check_for_matching_attrs_dset_no_attrs(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=None))

    def test_check_for_matching_attrs_dset_matching_attrs(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'quantity':'Current'}
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_dset_one_mismatched_attrs(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'blah': 'meh'}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': 'string_val'}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_02(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': ['str_1', 'str_2', 'str_3']}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_03(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 'str_3']}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_04(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 45]}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_old_exact_match(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_check_for_old_subset_but_match(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_check_for_old_exact_match_02(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'other_string_val', 'att_2': 5.4321,
                     'att_3': [4, 1, 3], 'att_4': ['s', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_001'])

    def test_check_for_old_fail_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': [4, 1, 3], 'att_2': ['s', 'str_2', 'str_3'],
                     'att_3': 'other_string_val', 'att_4': 5.4321}
            ret_val = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertIsInstance(ret_val, list)
            self.assertEqual(len(ret_val), 0)

    def test_check_for_old_fail_02(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_x': [4, 1, 3], 'att_z': ['s', 'str_2', 'str_3'],
                     'att_y': 'other_string_val', 'att_4': 5.4321}
            ret_val = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertIsInstance(ret_val, list)
            self.assertEqual(len(ret_val), 0)

    def test_link_as_main(self):
        file_path = 'link_as_main.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            dset_source_pos_inds = MicroDataset('PosIndices', source_pos_data, dtype=np.uint16,
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'um']})
            dset_source_pos_vals = MicroDataset('PosValues', source_pos_data, dtype=np.float16,
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'um']})

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = MicroDataset('source_main', source_main_data,
                                            attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}
                                                   })
            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            dset_source_spec_inds = MicroDataset('SpecIndices', source_spec_data, dtype=np.uint16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})
            dset_source_spec_vals = MicroDataset('SpecValues', source_spec_data, dtype=np.float16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, dset_source_main)
            h5_pos_inds = writer._create_dataset(h5_f, dset_source_pos_inds)
            h5_pos_vals = writer._create_dataset(h5_f, dset_source_pos_vals)
            h5_spec_inds = writer._create_dataset(h5_f, dset_source_spec_inds)
            h5_spec_vals = writer._create_dataset(h5_f, dset_source_spec_vals)

            self.assertFalse(hdf_utils.check_if_main(h5_main))

            # Now need to link as main!
            hdf_utils.link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)

            # Finally:
            self.assertTrue(hdf_utils.check_if_main(h5_main))

        os.remove(file_path)

    def test_link_as_main_size_mismatch(self):
        file_path = 'link_as_main.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            dset_source_pos_inds = MicroDataset('PosIndices', source_pos_data, dtype=np.uint16,
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'um']})
            dset_source_pos_vals = MicroDataset('PosValues', source_pos_data, dtype=np.float16,
                                                attrs={'labels': ['X', 'Y'], 'units': ['nm', 'um']})

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = MicroDataset('source_main', source_main_data,
                                            attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}
                                                   })
            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            dset_source_spec_inds = MicroDataset('SpecIndices', source_spec_data, dtype=np.uint16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})
            dset_source_spec_vals = MicroDataset('SpecValues', source_spec_data, dtype=np.float16,
                                                 attrs={'labels': ['Bias', 'Cycle'], 'units': ['V', '']})

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, dset_source_main)
            h5_pos_inds = writer._create_dataset(h5_f, dset_source_pos_inds)
            h5_pos_vals = writer._create_dataset(h5_f, dset_source_pos_vals)
            h5_spec_inds = writer._create_dataset(h5_f, dset_source_spec_inds)
            h5_spec_vals = writer._create_dataset(h5_f, dset_source_spec_vals)

            self.assertFalse(hdf_utils.check_if_main(h5_main))

            # Now need to link as main!
            with self.assertRaises(AssertionError):
                hdf_utils.link_as_main(h5_main, h5_spec_inds, h5_pos_vals, h5_pos_inds, h5_spec_vals)

        os.remove(file_path)

    def test_link_h5_obj_as_alias(self):
        file_path = 'link_as_alias.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, MicroDataset('main', np.arange(5)))
            h5_anc = writer._create_dataset(h5_f, MicroDataset('Ancillary', np.arange(3)))
            h5_group = writer._create_group(h5_f, MicroDataGroup('Results'))

            # Linking to dataset:
            hdf_utils.link_h5_obj_as_alias(h5_main, h5_anc, 'Blah')
            hdf_utils.link_h5_obj_as_alias(h5_main, h5_group, 'Something')
            self.assertEqual(h5_f[h5_main.attrs['Blah']], h5_anc)
            self.assertEqual(h5_f[h5_main.attrs['Something']], h5_group)

            # Linking ot Group:
            hdf_utils.link_h5_obj_as_alias(h5_group, h5_main, 'Center')
            hdf_utils.link_h5_obj_as_alias(h5_group, h5_anc, 'South')
            self.assertEqual(h5_f[h5_group.attrs['Center']], h5_main)
            self.assertEqual(h5_f[h5_group.attrs['South']], h5_anc)

            # Linking to file:
            hdf_utils.link_h5_obj_as_alias(h5_f, h5_main, 'Paris')
            hdf_utils.link_h5_obj_as_alias(h5_f, h5_group, 'France')
            self.assertEqual(h5_f[h5_f.attrs['Paris']], h5_main)
            self.assertEqual(h5_f[h5_f.attrs['France']], h5_group)

            # Non h5 object
            with self.assertRaises(AssertionError):
                hdf_utils.link_h5_obj_as_alias(h5_group, np.arange(5), 'Center')

            # H5 reference but not the object
            with self.assertRaises(AssertionError):
                hdf_utils.link_h5_obj_as_alias(h5_group, h5_f.attrs['Paris'], 'Center')

        os.remove(file_path)

    def test_link_h5_objects_as_attrs(self):
        file_path = 'link_h5_objects_as_attrs.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, MicroDataset('main', np.arange(5)))
            h5_anc = writer._create_dataset(h5_f, MicroDataset('Ancillary', np.arange(3)))
            h5_group = writer._create_group(h5_f, MicroDataGroup('Results'))

            hdf_utils.link_h5_objects_as_attrs(h5_f, [h5_anc, h5_main, h5_group])
            for exp, name in zip([h5_main, h5_anc, h5_group], ['main', 'Ancillary', 'Results']):
                self.assertEqual(exp, h5_f[h5_f.attrs[name]])

            # Single object
            hdf_utils.link_h5_objects_as_attrs(h5_main, h5_anc)
            self.assertEqual(h5_f[h5_main.attrs['Ancillary']], h5_anc)

            # Linking to a group:
            hdf_utils.link_h5_objects_as_attrs(h5_group, [h5_anc, h5_main])
            for exp, name in zip([h5_main, h5_anc], ['main', 'Ancillary']):
                self.assertEqual(exp, h5_group[h5_group.attrs[name]])

            with self.assertRaises(AssertionError):
                hdf_utils.link_h5_objects_as_attrs(h5_main, np.arange(4))

        os.remove(file_path)

    def test_reshape_to_n_dims_h5_no_sort_reqd(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=False)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(h5_main[()], (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, (1, 0, 3, 2))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_main, get_labels=True, sort_dims=True)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(h5_main[()], (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, (1, 0, 3, 2))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

    def test_reshape_to_n_dims_h5_not_main_dset(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            h5_spec = h5_f['/Raw_Measurement/Spectroscopic_Indices']

            # Not main
            with self.assertRaises(AssertionError):
                _ = hdf_utils.reshape_to_n_dims(h5_main)

            # Not main and not helping that we are supplign incompatible ancillary datasets
            with self.assertRaises(AssertionError):
                _ = hdf_utils.reshape_to_n_dims(h5_main, h5_pos=h5_pos, h5_spec=h5_spec)

            # main but we are supplign incompatible ancillary datasets
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000/results_main']
            with self.assertRaises(AssertionError):
                _ = hdf_utils.reshape_to_n_dims(h5_main, h5_pos=h5_pos, h5_spec=h5_spec)

    def test_reshape_to_n_dim_numpy(self):
        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7
        # arrange as slow, fast instead of fast, slow
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T

        source_main_data = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                for cycle_ind in range(num_cycles):
                    for bias_ind in range(num_cycle_pts):
                        val = 1E+3*row_ind + 1E+2*col_ind + 1E+1*cycle_ind + bias_ind
                        source_main_data[row_ind*num_cols + col_ind, cycle_ind*num_cycle_pts + bias_ind] = val

        # make spectroscopic slow, fast instead of fast, slow
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))
        n_dim, success = hdf_utils.reshape_to_n_dims(source_main_data, h5_pos = source_pos_data,
                                                             h5_spec=source_spec_data, get_labels=False)
        expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
        self.assertTrue(np.allclose(expected_n_dim, n_dim))

    def test_reshape_to_n_dim_sort_required(self):
        file_path = 'reshape_to_n_dim_sort_required.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7
            # arrange as slow, fast instead of fast, slow
            source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                         np.tile(np.arange(num_cols), num_rows))).T
            dset_source_pos_inds = MicroDataset('Position_Indices', source_pos_data, dtype=np.uint16,
                                                attrs={'labels': ['Y', 'X'], 'units': ['nm', 'um']})
            dset_source_pos_vals = MicroDataset('Position_Values', source_pos_data, dtype=np.float16,
                                                attrs={'labels': ['Y', 'X'], 'units': ['nm', 'um']})

            source_main_data = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
            for row_ind in range(num_rows):
                for col_ind in range(num_cols):
                    for cycle_ind in range(num_cycles):
                        for bias_ind in range(num_cycle_pts):
                            val = 1E+3*row_ind + 1E+2*col_ind + 1E+1*cycle_ind + bias_ind
                            source_main_data[row_ind*num_cols + col_ind, cycle_ind*num_cycle_pts + bias_ind] = val

            # source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = MicroDataset('source_main', source_main_data,
                                            attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}})
            # make spectroscopic slow, fast instead of fast, slow
            source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                          np.tile(np.arange(num_cycle_pts), num_cycles)))
            dset_source_spec_inds = MicroDataset('Spectroscopic_Indices', source_spec_data, dtype=np.uint16,
                                                 attrs={'labels': ['Cycle', 'Bias'], 'units': ['', 'V', ]})
            dset_source_spec_vals = MicroDataset('Spectroscopic_Values', source_spec_data, dtype=np.float16,
                                                 attrs={'labels': ['Cycle', 'Bias'], 'units': ['', 'V', ]})
            group_source = MicroDataGroup('Raw_Measurement',
                                          children=[dset_source_main, dset_source_spec_inds, dset_source_spec_vals,
                                                    dset_source_pos_vals, dset_source_pos_inds])

            writer = HDFwriter(h5_f)
            h5_refs_list = writer.write(group_source)

            [h5_source_main] = hdf_utils.get_h5_obj_refs([dset_source_main.name], h5_refs_list)
            [h5_pos_inds] = hdf_utils.get_h5_obj_refs([dset_source_pos_inds.name], h5_refs_list)
            [h5_pos_vals] = hdf_utils.get_h5_obj_refs([dset_source_pos_vals.name], h5_refs_list)
            [h5_source_spec_inds] = hdf_utils.get_h5_obj_refs([dset_source_spec_inds.name], h5_refs_list)
            [h5_source_spec_vals] = hdf_utils.get_h5_obj_refs([dset_source_spec_vals.name], h5_refs_list)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_source_main, get_labels=True, sort_dims=True)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['Y', 'X', 'Cycle', 'Bias'])]))
            expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

        os.remove(file_path)




    """  
    def test_calc_chunks(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = None
        ret_val = hdf_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)
        print(ret_val)
        assert False  
               
    def test_get_indices_for_region_ref(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            reg_ref = hdf_utils.get_attr(h5_main, 'even_rows')
            ret_val = hdf_utils.get_indices_for_region_ref(h5_main, reg_ref)
            expected_data = h5_main[0:None:2]
            data_slices = []
            for item in ret_val:
                print(h5_main[item[0], item[1]].shape)
                data_slices.append(h5_main[item[0], item[1]])
            data_slices = np.vstack(data_slices)
            print(data_slices.shape, expected_data.shape)
            self.assertTrue(np.allclose(data_slices, expected_data))

    def test_get_all_main_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            main_dsets = hdf_utils.get_all_main(h5_f, verbose=False)
            for dset in main_dsets:
                self.assertTrue(dset in expected_dsets)
                
                h5_main.attrs[alias_name] = h5_ancillary.ref
    """


if __name__ == '__main__':
    unittest.main()
