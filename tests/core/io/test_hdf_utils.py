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
import shutil
sys.path.append("../../../pycroscopy/")
from pycroscopy import VirtualGroup, VirtualDataset
from pycroscopy import HDFwriter
from pycroscopy.core.io import hdf_utils, write_utils
from pycroscopy.core.io.pycro_data import PycroDataset

test_h5_file_path = 'test_hdf_utils.h5'

if sys.version_info.major == 3:
    unicode = str


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
            pos_attrs = {'units': ['nm', 'um'],
                         'labels': {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}}
            dset_source_pos_inds = VirtualDataset('Position_Indices', source_pos_data, dtype=np.uint16, attrs=pos_attrs)
            # make the values more interesting:
            source_pos_data = np.vstack((source_pos_data[:, 0] * 50, source_pos_data[:, 1] * 1.25)).T
            dset_source_pos_vals = VirtualDataset('Position_Values', source_pos_data, dtype=np.float16, attrs=pos_attrs)

            num_cycles = 2
            num_cycle_pts = 7

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = VirtualDataset(source_dset_name, source_main_data,
                                              attrs={'units': 'A', 'quantity': 'Current',
                                                     'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                                'odd_rows': (slice(1, None, 2), slice(None))}
                                                     })
            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''],
                                 'labels': {'Bias': (slice(0, 1), slice(None)), 'Cycle': (slice(1, 2), slice(None))}}
            dset_source_spec_inds = VirtualDataset('Spectroscopic_Indices', source_spec_data, dtype=np.uint16,
                                                   attrs=source_spec_attrs)
            source_spec_data = np.vstack((np.tile(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)),
                                                  num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            dset_source_spec_vals = VirtualDataset('Spectroscopic_Values', source_spec_data, dtype=np.float16,
                                                   attrs=source_spec_attrs)

            dset_ancillary = VirtualDataset('Ancillary', np.arange(5))

            group_source = VirtualGroup('Raw_Measurement',
                                        children=[dset_source_main, dset_source_spec_inds, dset_source_spec_vals,
                                                  dset_source_pos_vals, dset_source_pos_inds, dset_ancillary,
                                                  VirtualGroup('Misc')],
                                        attrs={'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4],
                                                 'att_4': ['str_1', 'str_2', 'str_3']})

            writer = HDFwriter(h5_f)
            h5_refs_list = writer.write(group_source, print_log=False)
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
            results_spec_attrs = {'units': ['V'], 'labels': {'Bias': (slice(0, 1), slice(None))}}
            dset_results_spec_inds = VirtualDataset('Spectroscopic_Indices', results_spec_inds, dtype=np.uint16,
                                                    attrs=results_spec_attrs)
            results_spec_vals = np.expand_dims(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)), 0)
            dset_results_spec_vals = VirtualDataset('Spectroscopic_Values', results_spec_vals, dtype=np.float16,
                                                    attrs=results_spec_attrs)

            results_1_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_results_1_main = VirtualDataset('results_main', results_1_main_data,
                                                 attrs={'units': 'pF', 'quantity': 'Capacitance'})

            group_results_1 = VirtualGroup(source_dset_name + '-' + tool_name + '_',
                                           parent=h5_source_group.name,
                                           children=[dset_results_1_main, dset_results_spec_inds,
                                                       dset_results_spec_vals],
                                           attrs={'att_1': 'string_val', 'att_2': 1.2345,
                                                    'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']})

            h5_refs_list = writer.write(group_results_1, print_log=False)

            [h5_results_1_main] = hdf_utils.get_h5_obj_refs([dset_results_1_main.name], h5_refs_list)
            [h5_results_1_spec_inds] = hdf_utils.get_h5_obj_refs([dset_results_spec_inds.name], h5_refs_list)
            [h5_results_1_spec_vals] = hdf_utils.get_h5_obj_refs([dset_results_spec_vals.name], h5_refs_list)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_1_spec_inds, h5_results_1_spec_vals]:
                h5_results_1_main.attrs[dset.name.split('/')[-1]] = dset.ref

            # add another result with different parameters

            results_2_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_results_2_main = VirtualDataset('results_main', results_2_main_data,
                                                 attrs={'units': 'pF', 'quantity': 'Capacitance'})

            group_results_2 = VirtualGroup(source_dset_name + '-' + tool_name + '_',
                                           parent=h5_source_group.name,
                                           children=[dset_results_2_main, dset_results_spec_inds,
                                                       dset_results_spec_vals],
                                           attrs={'att_1': 'other_string_val', 'att_2': 5.4321,
                                                    'att_3': [4, 1, 3], 'att_4': ['s', 'str_2', 'str_3']})

            h5_refs_list = writer.write(group_results_2, print_log=False)

            [h5_results_2_main] = hdf_utils.get_h5_obj_refs([dset_results_2_main.name], h5_refs_list)
            [h5_results_2_spec_inds] = hdf_utils.get_h5_obj_refs([dset_results_spec_inds.name], h5_refs_list)
            [h5_results_2_spec_vals] = hdf_utils.get_h5_obj_refs([dset_results_spec_vals.name], h5_refs_list)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_2_spec_inds, h5_results_2_spec_vals]:
                h5_results_2_main.attrs[dset.name.split('/')[-1]] = dset.ref

    # start off with the most popular functions:
    def test_get_attr_illegal_01(self):
        self.__ensure_test_h5_file()
        with self.assertRaises(TypeError):
            hdf_utils.get_attr(np.arange(3), 'units')

    def test_get_attr_illegal_02(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 14)

    def test_get_attr_illegal_03(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], ['quantity', 'units'])

    def test_get_region_illegal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                hdf_utils.get_region(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_get_region_legal_01(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_source = h5_f['/Raw_Measurement/source_main']
            returned = hdf_utils.get_region(h5_source, 'even_rows')
            self.assertTrue(np.all(returned == h5_source[range(0, h5_source.shape[0], 2)]))

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
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
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
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Bias'])
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_illegal_key(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Does not exist'])

    def test_get_unit_values_illegal_dset(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Ancillary']
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Bias'])

    def test_get_unit_values_source_spec_single(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float16(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)))}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Bias')
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
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
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
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Y')
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
            pos_attrs = {'units': ['nm', 'um'],
                         'labels': {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}}
            dset_source_pos_inds = VirtualDataset('PosIndices', source_pos_data, dtype=np.uint16, attrs=pos_attrs)
            dset_source_pos_vals = VirtualDataset('PosValues', source_pos_data, dtype=np.float16, attrs=pos_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = VirtualDataset('source_main', source_main_data,
                                              attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}
                                                   })
            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''],
                                 'labels': {'Bias': (slice(0, 1), slice(None)), 'Cycle': (slice(1, 2), slice(None))}}
            dset_source_spec_inds = VirtualDataset('SpecIndices', source_spec_data, dtype=np.uint16,
                                                   attrs=source_spec_attrs)
            dset_source_spec_vals = VirtualDataset('SpecValues', source_spec_data, dtype=np.float16,
                                                   attrs=source_spec_attrs)

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
            pos_attrs = {'units': ['nm', 'um'],
                         'labels': {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}}
            dset_source_pos_inds = VirtualDataset('PosIndices', source_pos_data, dtype=np.uint16, attrs=pos_attrs)
            dset_source_pos_vals = VirtualDataset('PosValues', source_pos_data, dtype=np.float16, attrs=pos_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = VirtualDataset('source_main', source_main_data,
                                              attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}
                                                   })
            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''],
                                 'labels': {'Bias': (slice(0, 1), slice(None)), 'Cycle': (slice(1, 2), slice(None))}}
            dset_source_spec_inds = VirtualDataset('SpecIndices', source_spec_data, dtype=np.uint16,
                                                   attrs=source_spec_attrs)
            dset_source_spec_vals = VirtualDataset('SpecValues', source_spec_data, dtype=np.float16,
                                                   attrs=source_spec_attrs)

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, dset_source_main)
            h5_pos_inds = writer._create_dataset(h5_f, dset_source_pos_inds)
            h5_pos_vals = writer._create_dataset(h5_f, dset_source_pos_vals)
            h5_spec_inds = writer._create_dataset(h5_f, dset_source_spec_inds)
            h5_spec_vals = writer._create_dataset(h5_f, dset_source_spec_vals)

            self.assertFalse(hdf_utils.check_if_main(h5_main))

            # Now need to link as main!
            with self.assertRaises(ValueError):
                hdf_utils.link_as_main(h5_main, h5_spec_inds, h5_pos_vals, h5_pos_inds, h5_spec_vals)

        os.remove(file_path)

    def test_link_h5_obj_as_alias(self):
        file_path = 'link_as_alias.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, VirtualDataset('main', np.arange(5)))
            h5_anc = writer._create_dataset(h5_f, VirtualDataset('Ancillary', np.arange(3)))
            h5_group = writer._create_group(h5_f, VirtualGroup('Results'))

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
            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias(h5_group, np.arange(5), 'Center')

            # H5 reference but not the object
            with self.assertRaises(TypeError):
                hdf_utils.link_h5_obj_as_alias(h5_group, h5_f.attrs['Paris'], 'Center')

        os.remove(file_path)

    def test_link_h5_objects_as_attrs(self):
        file_path = 'link_h5_objects_as_attrs.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            h5_main = writer._create_dataset(h5_f, VirtualDataset('main', np.arange(5)))
            h5_anc = writer._create_dataset(h5_f, VirtualDataset('Ancillary', np.arange(3)))
            h5_group = writer._create_group(h5_f, VirtualGroup('Results'))

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

            with self.assertRaises(TypeError):
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
            with self.assertRaises(ValueError):
                _ = hdf_utils.reshape_to_n_dims(h5_main)

            # Not main and not helping that we are supplign incompatible ancillary datasets
            with self.assertRaises(ValueError):
                _ = hdf_utils.reshape_to_n_dims(h5_main, h5_pos=h5_pos, h5_spec=h5_spec)

            # main but we are supplign incompatible ancillary datasets
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000/results_main']
            with self.assertRaises(ValueError):
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
            pos_attrs = {'units': ['nm', 'um'],
                         'labels': {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}}
            dset_source_pos_inds = VirtualDataset('Position_Indices', source_pos_data, dtype=np.uint16, attrs=pos_attrs)
            dset_source_pos_vals = VirtualDataset('Position_Values', source_pos_data, dtype=np.float16, attrs=pos_attrs)

            source_main_data = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
            for row_ind in range(num_rows):
                for col_ind in range(num_cols):
                    for cycle_ind in range(num_cycles):
                        for bias_ind in range(num_cycle_pts):
                            val = 1E+3*row_ind + 1E+2*col_ind + 1E+1*cycle_ind + bias_ind
                            source_main_data[row_ind*num_cols + col_ind, cycle_ind*num_cycle_pts + bias_ind] = val

            # source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            dset_source_main = VirtualDataset('source_main', source_main_data,
                                              attrs={'units': 'A', 'quantity': 'Current',
                                                   'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                                              'odd_rows': (slice(1, None, 2), slice(None))}})
            # make spectroscopic slow, fast instead of fast, slow
            source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                          np.tile(np.arange(num_cycle_pts), num_cycles)))
            source_spec_attrs = {'units': ['V', ''],
                                 'labels': {'Bias': (slice(0, 1), slice(None)), 'Cycle': (slice(1, 2), slice(None))}}
            dset_source_spec_inds = VirtualDataset('Spectroscopic_Indices', source_spec_data, dtype=np.uint16,
                                                   attrs=source_spec_attrs)
            dset_source_spec_vals = VirtualDataset('Spectroscopic_Values', source_spec_data, dtype=np.float16,
                                                   attrs=source_spec_attrs)
            group_source = VirtualGroup('Raw_Measurement',
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
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['X', 'Y', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
            self.assertTrue(np.allclose(expected_n_dim, n_dim))

        os.remove(file_path)

    def test_reshape_from_n_dims_pos_and_spec_provided(self):
        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7

        # the N dimensional dataset should be arranged in the following order:
        # [positions slowest to fastest, spectroscopic slowest to fastest]
        source_nd = np.zeros(shape=(num_rows, num_cols, num_cycles, num_cycle_pts), dtype=np.float16)
        expected_2d = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                for cycle_ind in range(num_cycles):
                    for bias_ind in range(num_cycle_pts):
                        val = 1E+3 * row_ind + 1E+2 * col_ind + 1E+1 * cycle_ind + bias_ind
                        expected_2d[row_ind * num_cols + col_ind, cycle_ind * num_cycle_pts + bias_ind] = val
                        source_nd[row_ind, col_ind, cycle_ind, bias_ind] = val

        # case 1: Pos and Spec both arranged as slow to fast:
        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                      np.repeat(np.arange(num_cycles), num_cycle_pts)))

        ret_2d, success = hdf_utils.reshape_from_n_dims(source_nd, h5_pos=source_pos_data, h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 2: Only Pos arranged as slow to fast:
        main_pos_sorted = np.transpose(source_nd, (0, 1, 3, 2))
        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles),))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_pos_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 3: only Spec both arranged as slow to fast:
        main_spec_sorted = np.transpose(source_nd, (1, 0, 2, 3))
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                      np.repeat(np.arange(num_cycles), num_cycle_pts)))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_spec_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 4: neither pos nor spec arranged as slow to fast:
        main_not_sorted = np.transpose(source_nd, (1, 0, 3, 2))
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles),))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_not_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

    def test_reshape_from_n_dims_pos_and_spec_may_may_not_be_provided(self):
        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7

        # the N dimensional dataset should be arranged in the following order:
        # [positions slowest to fastest, spectroscopic slowest to fastest]
        source_nd = np.zeros(shape=(num_rows, num_cols, num_cycles, num_cycle_pts), dtype=np.float16)
        expected_2d = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
        for row_ind in range(num_rows):
            for col_ind in range(num_cols):
                for cycle_ind in range(num_cycles):
                    for bias_ind in range(num_cycle_pts):
                        val = 1E+3 * row_ind + 1E+2 * col_ind + 1E+1 * cycle_ind + bias_ind
                        expected_2d[row_ind * num_cols + col_ind, cycle_ind * num_cycle_pts + bias_ind] = val
                        source_nd[row_ind, col_ind, cycle_ind, bias_ind] = val

        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                      np.repeat(np.arange(num_cycles), num_cycle_pts)))

        # case 1: only pos provided:
        ret_2d, success = hdf_utils.reshape_from_n_dims(source_nd, h5_pos=source_pos_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 2: only spec provided:
        ret_2d, success = hdf_utils.reshape_from_n_dims(source_nd, h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 3: neither pos nor spec provided:
        with self.assertRaises(ValueError):
            _ = hdf_utils.reshape_from_n_dims(source_nd)

    def test_simple_region_ref_copy(self):
        # based on test_hdf_writer.test_write_legal_reg_ref_multi_dim_data()
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_orig_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_orig_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}}

            writer._write_dset_attributes(h5_orig_dset, attrs.copy())
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_orig_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:

            self.assertTrue(np.all([x in list(attrs['labels'].keys()) for x in hdf_utils.get_attr(h5_orig_dset,
                                                                                                  'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_orig_dset[h5_orig_dset.attrs['even_rows']], h5_orig_dset[h5_orig_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

            # Now write a new dataset without the region reference:
            h5_new_dset = writer._create_simple_dset(h5_f, VirtualDataset('other', data))
            self.assertIsInstance(h5_orig_dset, h5py.Dataset)
            h5_f.flush()

            for key in attrs['labels'].keys():
                hdf_utils.simple_region_ref_copy(h5_orig_dset, h5_new_dset, key)

            # now check to make sure that this dataset also has the same region references:
            written_data = [h5_new_dset[h5_new_dset.attrs['even_rows']], h5_new_dset[h5_new_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_get_all_main_legal(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            main_dsets = hdf_utils.get_all_main(h5_f, verbose=False)
            # self.assertEqual(set(main_dsets), set(expected_dsets))
            self.assertEqual(len(main_dsets), len(expected_dsets))
            self.assertTrue(np.all([x.name == y.name for x, y in zip(main_dsets, expected_dsets)]))
    
    def __validate_aux_dset_pair(self, h5_group, h5_inds, h5_vals, dim_names, dim_units, inds_matrix,
                                 vals_matrix=None, base_name=None, h5_main=None, is_spectral=True):
        if vals_matrix is None:
            vals_matrix = inds_matrix
        if base_name is None:
            if is_spectral:
                base_name = 'Spectroscopic'
            else:
                base_name = 'Position'
        else:
            self.assertIsInstance(base_name, (str, unicode))

        for h5_dset, exp_dtype, exp_name, ref_data in zip([h5_inds, h5_vals],
                                                          [write_utils.INDICES_DTYPE, write_utils.VALUES_DTYPE],
                                                          [base_name + '_Indices', base_name + '_Values'],
                                                          [inds_matrix, vals_matrix]):
            if isinstance(h5_main, h5py.Dataset):
                self.assertEqual(h5_main.file[h5_main.attrs[exp_name]], h5_dset)
            self.assertIsInstance(h5_dset, h5py.Dataset)
            self.assertEqual(h5_dset.parent, h5_group)
            self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
            self.assertTrue(np.allclose(ref_data, h5_dset[()]))
            self.assertEqual(h5_dset.dtype, exp_dtype)
            self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
            self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
            self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
            # assert region references
            for dim_ind, curr_name in enumerate(dim_names):
                expected = np.squeeze(ref_data[:, dim_ind])
                if is_spectral:
                    expected = np.squeeze(ref_data[dim_ind])
                self.assertTrue(np.allclose(expected,
                                            np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

    def test_write_ind_val_dsets_legal_bare_minimum_pos(self):
        num_cols = 3
        num_rows = 2
        sizes = [num_cols, num_rows]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T
        file_path = 'test_write_ind_val_dsets.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_f, descriptor, is_spectral=False)

            self. __validate_aux_dset_pair(h5_f, h5_inds, h5_vals, dim_names, dim_units, pos_data,
                                           is_spectral=False)

        os.remove(file_path)

    def test_write_ind_val_dsets_legal_bare_minimum_spec(self):
        num_cols = 3
        num_rows = 2
        sizes = [num_cols, num_rows]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        spec_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        file_path = 'test_write_ind_val_dsets.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group("Blah")
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_group, descriptor, is_spectral=True)

            self.__validate_aux_dset_pair(h5_group, h5_inds, h5_vals, dim_names, dim_units, spec_data,
                                          is_spectral=True)
        os.remove(file_path)

    def test_write_ind_val_dsets_legal_override_steps_offsets_base_name(self):
        num_cols = 2
        num_rows = 3
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        col_step = 0.25
        row_step = 0.05
        col_initial = 1
        row_initial = 0.2

        descriptor = []
        for length, name, units, step, initial in zip([num_cols, num_rows], dim_names, dim_units,
                                                      [col_step, row_step], [col_initial, row_initial]):
            descriptor.append(write_utils.Dimension(name, units, initial + step * np.arange(length)))

        new_base_name = 'Overriden'
        spec_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        spec_vals = np.vstack((np.tile(np.arange(num_cols), num_rows) * col_step + col_initial,
                              np.repeat(np.arange(num_rows), num_cols) * row_step + row_initial))

        file_path = 'test_write_ind_val_dsets.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            h5_group = h5_f.create_group("Blah")
            h5_inds, h5_vals = hdf_utils.write_ind_val_dsets(h5_group, descriptor, is_spectral=True,
                                                             base_name=new_base_name)
            self.__validate_aux_dset_pair(h5_group, h5_inds, h5_vals, dim_names, dim_units, spec_inds,
                                          vals_matrix=spec_vals, base_name=new_base_name, is_spectral=True)
        os.remove(file_path)

    def test_write_ind_val_dsets_illegal(self):
        sizes = [3, 2]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        file_path = 'test_write_ind_val_dsets.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path, mode='w') as h5_f:
            pass

        with self.assertRaises(ValueError):
            # h5_f should be valid in terms of type but closed
            _ = hdf_utils.write_ind_val_dsets(h5_f, descriptor)

        os.remove(file_path)

    def test_assign_group_index_existing(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            ret_val = hdf_utils.assign_group_index(h5_group, 'source_main-Fitter')
            self.assertEqual(ret_val, 'source_main-Fitter_002')

    def test_assign_group_index_new(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            ret_val = hdf_utils.assign_group_index(h5_group, 'blah_')
            self.assertEqual(ret_val, 'blah_000')

    def test_write_legal_atts_to_grp(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            h5_group = h5_f.create_group('Blah')

            attrs = {'att_1': 'string_val', 'att_2': 1.234, 'att_3': [1, 2, 3.14, 4],
                     'att_4': ['s', 'tr', 'str_3']}

            hdf_utils.write_simple_attrs(h5_group, attrs)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_val))

        os.remove(file_path)

    def test_write_legal_atts_to_dset_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            h5_dset = h5_f.create_dataset('Test', data=np.arange(3))

            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3']}

            hdf_utils.write_simple_attrs(h5_dset, attrs)

            self.assertEqual(len(h5_dset.attrs), len(attrs))

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_dset, key) == expected_val))

        os.remove(file_path)

    def test_is_editable_h5_read_only(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            self.assertFalse(hdf_utils.is_editable_h5(h5_group))
            self.assertFalse(hdf_utils.is_editable_h5(h5_f))
            self.assertFalse(hdf_utils.is_editable_h5(h5_main))

    def test_is_editable_h5_r_plus(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r+') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            self.assertTrue(hdf_utils.is_editable_h5(h5_group))
            self.assertTrue(hdf_utils.is_editable_h5(h5_f))
            self.assertTrue(hdf_utils.is_editable_h5(h5_main))

    def test_is_editable_h5_w(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.arange(3))
            h5_group = h5_f.create_group('blah')
            self.assertTrue(hdf_utils.is_editable_h5(h5_group))
            self.assertTrue(hdf_utils.is_editable_h5(h5_f))
            self.assertTrue(hdf_utils.is_editable_h5(h5_dset))

        os.remove(file_path)

    def test_is_editable_h5_illegal(self):
        # wrong kind of object
        with self.assertRaises(TypeError):
            _ = hdf_utils.is_editable_h5(np.arange(4))

        # closed file
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']

        with self.assertRaises(ValueError):
            _ = hdf_utils.is_editable_h5(h5_group)

    def test_write_main_dset_small(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']

        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                              np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            pycro_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                      spec_dims, main_dset_attrs=None)
            self.assertIsInstance(pycro_main, PycroDataset)
            self.assertEqual(pycro_main.name.split('/')[-1], main_data_name)
            self.assertEqual(pycro_main.parent, h5_f)
            self.assertTrue(np.allclose(main_data, pycro_main[()]))

            self.__validate_aux_dset_pair(h5_f, pycro_main.h5_pos_inds, pycro_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=pycro_main, is_spectral=False)

            self.__validate_aux_dset_pair(h5_f, pycro_main.h5_spec_inds, pycro_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=pycro_main, is_spectral=True)
        os.remove(file_path)

    def test_write_main_dset_empty(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        main_data = (15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']

        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                              np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            pycro_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                      spec_dims, main_dset_attrs=None)
            self.assertIsInstance(pycro_main, PycroDataset)
            self.assertEqual(pycro_main.name.split('/')[-1], main_data_name)
            self.assertEqual(pycro_main.parent, h5_f)
            self.assertEqual(main_data, pycro_main.shape)

            self.__validate_aux_dset_pair(h5_f, pycro_main.h5_pos_inds, pycro_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=pycro_main, is_spectral=False)

            self.__validate_aux_dset_pair(h5_f, pycro_main.h5_spec_inds, pycro_main.h5_spec_vals, spec_names, spec_units,
                                          spec_data, h5_main=pycro_main, is_spectral=True)
        os.remove(file_path)

    def test_write_main_existing_spec_aux(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']
        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                               np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            h5_spec_inds, h5_spec_vals = hdf_utils.write_ind_val_dsets(h5_f, spec_dims, is_spectral=True)
            self.__validate_aux_dset_pair(h5_f, h5_spec_inds, h5_spec_vals, spec_names, spec_units, spec_data,
                                          is_spectral=True)

            pycro_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                      None, h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                                                      main_dset_attrs=None)

            self.__validate_aux_dset_pair(h5_f, pycro_main.h5_pos_inds, pycro_main.h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=pycro_main, is_spectral=False)

        os.remove(file_path)

    def test_write_main_existing_both_aux(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 3]
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']
        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        pos_data = np.vstack((np.tile(np.arange(5), 3),
                              np.repeat(np.arange(3), 5))).T

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))
        spec_data = np.vstack((np.tile(np.arange(7), 2),
                               np.repeat(np.arange(2), 7)))

        with h5py.File(file_path) as h5_f:
            h5_spec_inds, h5_spec_vals = hdf_utils.write_ind_val_dsets(h5_f, spec_dims, is_spectral=True)
            h5_pos_inds, h5_pos_vals = hdf_utils.write_ind_val_dsets(h5_f, pos_dims, is_spectral=False)

            pycro_main = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, None,
                                                      None, h5_spec_inds=h5_spec_inds, h5_spec_vals=h5_spec_vals,
                                                      h5_pos_vals=h5_pos_vals, h5_pos_inds=h5_pos_inds,
                                                      main_dset_attrs=None)

            self.__validate_aux_dset_pair(h5_f, h5_pos_inds, h5_pos_vals, pos_names, pos_units,
                                          pos_data, h5_main=pycro_main, is_spectral=False)

            self.__validate_aux_dset_pair(h5_f, h5_spec_inds, h5_spec_vals, spec_names,spec_units,
                                          spec_data, h5_main=pycro_main, is_spectral=True)
        os.remove(file_path)

    def test_write_main_dset_prod_sizes_mismatch(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        main_data = np.random.rand(15, 14)
        main_data_name = 'Test_Main'
        quantity = 'Current'
        dset_units = 'nA'

        pos_sizes = [5, 15]  # too many steps in the Y direction
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'um']
        pos_dims = []
        for length, name, units in zip(pos_sizes, pos_names, pos_units):
            pos_dims.append(write_utils.Dimension(name, units, np.arange(length)))

        spec_sizes = [7, 2]
        spec_names = ['Bias', 'Cycle']
        spec_units = ['V', '']
        spec_dims = []
        for length, name, units in zip(spec_sizes, spec_names, spec_units):
            spec_dims.append(write_utils.Dimension(name, units, np.arange(length)))

        with h5py.File(file_path) as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.write_main_dataset(h5_f, main_data, main_data_name, quantity, dset_units, pos_dims,
                                                 spec_dims)
        os.remove(file_path)

    def test_clean_reg_refs_1d(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7))
            reg_ref = (slice(0, None, 2))
            cleaned = hdf_utils.clean_reg_ref(h5_dset, reg_ref)
            self.assertEqual(reg_ref, cleaned[0])
        os.remove(file_path)

    def test_clean_reg_refs_2d(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, None, 2), slice(None))
            cleaned = hdf_utils.clean_reg_ref(h5_dset, reg_ref)
            self.assertTrue(np.all([x == y for x, y in zip(reg_ref, cleaned)]))
        os.remove(file_path)

    def test_clean_reg_refs_illegal_too_many_slices(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, None, 2), slice(None), slice(1, None, 2))
            with self.assertRaises(ValueError):
                _ = hdf_utils.clean_reg_ref(h5_dset, reg_ref)

        os.remove(file_path)

    def test_clean_reg_refs_illegal_too_few_slices(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, None, 2))
            with self.assertRaises(ValueError):
                _ = hdf_utils.clean_reg_ref(h5_dset, reg_ref)

        os.remove(file_path)

    def test_clean_reg_refs_out_of_bounds(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Test', data=np.random.rand(7, 5))
            reg_ref = (slice(0, 13, 2), slice(None))
            expected = (slice(0, 7, 2), slice(None))
            cleaned = hdf_utils.clean_reg_ref(h5_dset, reg_ref, verbose=False)
            self.assertTrue(np.all([x == y for x, y in zip(expected, cleaned)]))
        os.remove(file_path)

    def test_attempt_reg_ref_build_spec(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(2, 5))
            dim_names = ['Bias', 'Cycle']
            expected = {'Bias': (slice(0, 1), slice(None)),
                        'Cycle': (slice(1, 2), slice(None))}
            cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            for key, value in expected.items():
                self.assertEqual(value, cleaned[key])
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias', 'Cycle']
            expected = {'Bias': (slice(None), slice(0, 1)),
                        'Cycle': (slice(None), slice(1, 2))}
            cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            for key, value in expected.items():
                self.assertEqual(value, cleaned[key])
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos_too_many_dims(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias', 'Cycle', 'Blah']
            ret_val = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)

    def test_attempt_reg_ref_build_pos_too_few_dims(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Indices', data=np.random.rand(5, 2))
            dim_names = ['Bias']
            ret_val = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            self.assertEqual(ret_val, dict())
        os.remove(file_path)

    def test_write_reg_ref_main_one_dim(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2)),
                        'odd_rows': (slice(1, None, 2))}
            hdf_utils.write_region_references(h5_dset, reg_refs, add_labels_attr=True)
            self.assertEqual(len(h5_dset.attrs), 1 + len(reg_refs))
            actual = hdf_utils.get_attr(h5_dset, 'labels')
            self.assertTrue(np.all([x == y for x, y in zip(actual, ['even_rows', 'odd_rows'])]))

            expected_data = [data[0:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_reg_ref_main_1st_dim(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            hdf_utils.write_region_references(h5_dset, reg_refs, add_labels_attr=True)
            self.assertEqual(len(h5_dset.attrs), 1 + len(reg_refs))
            actual = hdf_utils.get_attr(h5_dset, 'labels')
            self.assertTrue(np.all([x == y for x, y in zip(actual, ['even_rows', 'odd_rows'])]))

            expected_data = [data[0:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_reg_ref_main_2nd_dim(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=data)
            reg_refs = {'even_rows': (slice(None), slice(0, None, 2)),
                        'odd_rows': (slice(None), slice(1, None, 2))}
            hdf_utils.write_region_references(h5_dset, reg_refs, add_labels_attr=False)
            self.assertEqual(len(h5_dset.attrs), len(reg_refs))
            self.assertTrue('labels' not in h5_dset.attrs.keys())

            expected_data = [data[:, 0:None:2], data[:, 1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_reduced_spec_dsets_2d_to_1d(self):
        self.__ensure_test_h5_file()
        duplicate_path = 'copy_test_hdf_utils.h5'
        self.__delete_existing_file(duplicate_path)
        shutil.copy(test_h5_file_path, duplicate_path)
        with h5py.File(duplicate_path) as h5_f:
            h5_spec_inds_orig = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_spec_vals_orig = h5_f['/Raw_Measurement/Spectroscopic_Values']
            new_base_name = 'Blah'
            cycle_starts = np.where(h5_spec_inds_orig[0] == 0)[0]
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_spec_dsets(h5_spec_inds_orig.parent,
                                                                                    h5_spec_inds_orig,
                                                                                    h5_spec_vals_orig,
                                                                                    np.array([False, True]),
                                                                                    cycle_starts,
                                                                                    basename=new_base_name)

            dim_names = ['Cycle']
            dim_units = ['']
            ref_data = np.expand_dims(np.arange(2), axis=0)
            for h5_dset, exp_dtype, exp_name in zip([h5_spec_inds_new, h5_spec_vals_new],
                                                    [h5_spec_inds_orig.dtype, h5_spec_vals_orig.dtype],
                                                    [new_base_name + '_Indices', new_base_name + '_Values']):
                self.assertIsInstance(h5_dset, h5py.Dataset)
                self.assertEqual(h5_dset.parent, h5_spec_inds_orig.parent)
                self.assertEqual(h5_dset.name.split('/')[-1], exp_name)
                self.assertTrue(np.allclose(ref_data, h5_dset[()]))
                self.assertEqual(h5_dset.dtype, exp_dtype)
                self.assertTrue(np.all([_ in h5_dset.attrs.keys() for _ in ['labels', 'units']]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_names, hdf_utils.get_attr(h5_dset, 'labels'))]))
                self.assertTrue(np.all([x == y for x, y in zip(dim_units, hdf_utils.get_attr(h5_dset, 'units'))]))
                # assert region references
                for dim_ind, curr_name in enumerate(dim_names):
                    self.assertTrue(np.allclose(np.squeeze(ref_data[dim_ind]),
                                                np.squeeze(h5_dset[h5_dset.attrs[curr_name]])))

        os.remove(duplicate_path)

    def test_calc_chunks_no_unit_chunk(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = None
        ret_val = hdf_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)
        self.assertTrue(np.allclose(ret_val, (26, 100)))

    def test_calc_chunks_unit_chunk(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (3, 7)
        ret_val = hdf_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)
        self.assertTrue(np.allclose(ret_val, (27, 98)))

    def test_calc_chunks_no_unit_chunk_max_mem(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = None
        max_mem = 50000
        ret_val = hdf_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks, max_chunk_mem=max_mem)
        self.assertTrue(np.allclose(ret_val, (56, 224)))

    def test_calc_chunks_unit_chunk_max_mem(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (3, 7)
        max_mem = 50000
        ret_val = hdf_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks, max_chunk_mem=max_mem)
        self.assertTrue(np.allclose(ret_val, (57, 224)))

    def test_calc_chunks_unit_not_iterable(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = 4

        with self.assertRaises(TypeError):
            _ = hdf_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)

    def test_calc_chunks_shape_mismatch(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (1, 5, 9)

        with self.assertRaises(ValueError):
            _ = hdf_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)

    def test_get_indices_for_region_ref_corners(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            reg_ref = hdf_utils.get_attr(h5_main, 'even_rows')
            ret_val = hdf_utils.get_indices_for_region_ref(h5_main, reg_ref, 'corners')
            expected_pos = np.repeat(np.arange(h5_main.shape[0])[::2], 2)
            expected_spec = np.tile(np.array([0,h5_main.shape[1]-1]), expected_pos.size // 2)
            expected_corners = np.vstack((expected_pos, expected_spec)).T
            self.assertTrue(np.allclose(ret_val, expected_corners))

    def test_get_indices_for_region_ref_slices(self):
        self.__ensure_test_h5_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            reg_ref = hdf_utils.get_attr(h5_main, 'even_rows')
            ret_val = hdf_utils.get_indices_for_region_ref(h5_main, reg_ref, 'slices')
            spec_slice = slice(0, h5_main.shape[1] - 1, None)
            expected_slices = np.array([[slice(x, x, None), spec_slice] for x in np.arange(h5_main.shape[0])[::2]])
            self.assertTrue(np.all(ret_val==expected_slices))


if __name__ == '__main__':
    unittest.main()
