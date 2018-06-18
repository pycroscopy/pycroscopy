# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import os
import sys
import h5py
import numpy as np
import shutil

import pycroscopy.core.io.write_utils

sys.path.append("../../../pycroscopy/")
from pycroscopy.core.io import hdf_utils, write_utils, io_utils
from pycroscopy.core.io.pycro_data import PycroDataset
from pycroscopy import __version__
from platform import platform
import socket

test_h5_file_path = 'test_hdf_utils.h5'

if sys.version_info.major == 3:
    unicode = str


class TestHDFUtils(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def __write_safe_attrs(h5_object, attrs):
        for key, val in attrs.items():
            h5_object.attrs[key] = val

    @staticmethod
    def __write_string_list_as_attr(h5_object, attrs):
        for key, val in attrs.items():
            h5_object.attrs[key] = np.array(val, dtype='S')

    @staticmethod
    def __write_aux_reg_ref(h5_dset, labels, is_spec=True):
        for index, reg_ref_name in enumerate(labels):
            if is_spec:
                reg_ref_tuple = (slice(index, index + 1), slice(None))
            else:
                reg_ref_tuple = (slice(None), slice(index, index + 1))
            h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]

    @staticmethod
    def __write_main_reg_refs(h5_dset, attrs):
        for reg_ref_name, reg_ref_tuple in attrs.items():
            h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]
        TestHDFUtils.__write_string_list_as_attr(h5_dset, {'labels': list(attrs.keys())})

    def setUp(self):
        if os.path.exists(test_h5_file_path):
            os.remove(test_h5_file_path)

        with h5py.File(test_h5_file_path) as h5_f:

            h5_raw_grp = h5_f.create_group('Raw_Measurement')
            TestHDFUtils.__write_safe_attrs(h5_raw_grp, {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
            TestHDFUtils.__write_string_list_as_attr(h5_raw_grp, {'att_4': ['str_1', 'str_2', 'str_3']})

            _ = h5_raw_grp.create_group('Misc')

            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_dset_name = 'source_main'
            tool_name = 'Fitter'

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_inds, pos_attrs)

            # make the values more interesting:
            source_pos_data = np.vstack((source_pos_data[:, 0] * 50, source_pos_data[:, 1] * 1.25)).T

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack(
                (np.tile(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)),
                         num_cycles),
                 np.repeat(np.arange(num_cycles), num_cycle_pts)))

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            TestHDFUtils.__write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})
            TestHDFUtils.__write_main_reg_refs(h5_source_main, {'even_rows': (slice(0, None, 2), slice(None)),
                                                   'odd_rows': (slice(1, None, 2), slice(None))})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            _ = h5_raw_grp.create_dataset('Ancillary', data=np.arange(5))

            # Now add a few results:

            h5_results_grp_1 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_000')
            TestHDFUtils.__write_safe_attrs(h5_results_grp_1, {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
            TestHDFUtils.__write_string_list_as_attr(h5_results_grp_1, {'att_4': ['str_1', 'str_2', 'str_3']})

            num_cycles = 1
            num_cycle_pts = 7

            results_spec_inds = np.expand_dims(np.arange(num_cycle_pts), 0)
            results_spec_attrs = {'units': ['V'], 'labels': ['Bias']}

            h5_results_1_spec_inds = h5_results_grp_1.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_results_1_spec_inds, results_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_results_1_spec_inds, results_spec_attrs)

            results_spec_vals = np.expand_dims(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)), 0)

            h5_results_1_spec_vals = h5_results_grp_1.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_results_1_spec_vals, results_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_results_1_spec_vals, results_spec_attrs)

            results_1_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_1_main = h5_results_grp_1.create_dataset('results_main', data=results_1_main_data)
            TestHDFUtils.__write_safe_attrs(h5_results_1_main, {'units': 'pF', 'quantity': 'Capacitance'})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_1_spec_inds, h5_results_1_spec_vals]:
                h5_results_1_main.attrs[dset.name.split('/')[-1]] = dset.ref

            # add another result with different parameters

            h5_results_grp_2 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_001')
            TestHDFUtils.__write_safe_attrs(h5_results_grp_2, {'att_1': 'other_string_val', 'att_2': 5.4321, 'att_3': [4, 1, 3]})
            TestHDFUtils.__write_string_list_as_attr(h5_results_grp_2, {'att_4': ['s', 'str_2', 'str_3']})

            results_2_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_2_main = h5_results_grp_2.create_dataset('results_main', data=results_2_main_data)
            TestHDFUtils.__write_safe_attrs(h5_results_2_main, {'units': 'pF', 'quantity': 'Capacitance'})

            h5_results_2_spec_inds = h5_results_grp_2.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_results_2_spec_inds, results_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_results_2_spec_inds, results_spec_attrs)

            h5_results_2_spec_vals = h5_results_grp_2.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_results_2_spec_vals, results_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_results_2_spec_vals, results_spec_attrs)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_2_spec_inds, h5_results_2_spec_vals]:
                h5_results_2_main.attrs[dset.name.split('/')[-1]] = dset.ref

    def tearDown(self):
        os.remove(test_h5_file_path)

    def test_get_attr_illegal_01(self):
        with self.assertRaises(TypeError):
            hdf_utils.get_attr(np.arange(3), 'units')

    def test_get_attr_illegal_02(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 14)

    def test_get_attr_illegal_03(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(TypeError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], ['quantity', 'units'])

    def test_get_region_illegal_01(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                hdf_utils.get_region(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_get_region_legal_01(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_source = h5_f['/Raw_Measurement/source_main']
            returned = hdf_utils.get_region(h5_source, 'even_rows')
            self.assertTrue(np.all(returned == h5_source[range(0, h5_source.shape[0], 2)]))

    def test_get_attr_illegal_04(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(KeyError):
                hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'non_existent')

    def test_get_attr_legal_01(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/source_main'], 'units')
            self.assertEqual(returned, 'A')

    def test_get_attr_legal_02(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            returned = hdf_utils.get_attr(h5_f['/Raw_Measurement/Position_Indices'], 'labels')
            self.assertTrue(np.all(returned == ['X', 'Y']))

    def test_get_attr_legal_03(self):
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            for key, expected_value in attrs.items():
                self.assertTrue(np.all(hdf_utils.get_attr(h5_group, key) == expected_value))

    def test_get_attributes_01(self):
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
        attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                 'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            returned_attrs = hdf_utils.get_attributes(h5_group)
            self.assertIsInstance(returned_attrs, dict)
            for key in attrs.keys():
                self.assertTrue(np.all(returned_attrs[key] == attrs[key]))

    def test_get_attributes_illegal(self):
        sub_attrs = ['att_1', 'att_4', 'does_not_exist']
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/source_main-Fitter_000']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_attributes(h5_group, sub_attrs)

    def test_get_auxillary_datasets_single(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_get_auxillary_datasets_single(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos = h5_f['/Raw_Measurement/Position_Indices']
            [ret_val] = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Position_Indices')
            self.assertEqual(ret_val, h5_pos)

    def test_get_auxillary_datasets_multiple(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            h5_pos_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_pos_vals = h5_f['/Raw_Measurement/Position_Values']
            ret_val = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name=['Position_Indices',
                                                                               'Position_Values'])
            self.assertEqual(set(ret_val), set([h5_pos_inds, h5_pos_vals]))

    def test_get_auxillary_datasets_all(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = [h5_f['/Raw_Measurement/Position_Indices'],
                        h5_f['/Raw_Measurement/Position_Values'],
                        h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Spectroscopic_Values']]
            ret_val = hdf_utils.get_auxiliary_datasets(h5_main)
            self.assertEqual(set(expected), set(ret_val))

    def test_get_auxillary_datasets_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_auxiliary_datasets(h5_main, aux_dset_name='Does_Not_Exist')

    def test_get_data_descriptor_main(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            self.assertEqual(hdf_utils.get_data_descriptor(h5_main), 'Current (A)')

    def test_get_data_descriptor_main(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_pos = h5_f['/Raw_Measurement/Ancillary']
            self.assertEqual(hdf_utils.get_data_descriptor(h5_pos), 'unknown quantity (unknown units)')

    def test_get_dimensionality_legal_no_sort(self):
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_labels = [['Bias (V)', 'Cycle ()'], ['Bias (V)'], ['X (nm)', 'Y (um)']]
            for h5_dset, exp_labs in zip(h5_dsets, expected_labels):
                self.assertTrue(np.all(exp_labs == hdf_utils.get_formatted_labels(h5_dset)))

    def test_get_formatted_labels_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Ancillary'],
                        h5_f['/Raw_Measurement/source_main']]  # This will have labels and units but of different sizes
            for h5_dset, err_type in zip(h5_dsets, [KeyError, ValueError]):
                with self.assertRaises(err_type):
                    _ = hdf_utils.get_formatted_labels(h5_dset)

    def test_get_group_refs_legal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_refs = [h5_f['/Raw_Measurement/Ancillary'],
                       h5_f['/Raw_Measurement/source_main-Fitter_000'],
                       h5_f['/Raw_Measurement/source_main-Fitter_001'],
                       h5_f['/Raw_Measurement/source_main-Fitter_000/results_main']]
            group_prefix = 'source_main-Fitter'
            expected_objs = set([h5_f['/Raw_Measurement/source_main-Fitter_000'],
                                 h5_f['/Raw_Measurement/source_main-Fitter_001']])
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    ret_vals = set(hdf_utils.get_group_refs(group_prefix, h5_refs))
            else:
                ret_vals = set(hdf_utils.get_group_refs(group_prefix, h5_refs))
            self.assertTrue(ret_vals == expected_objs)

    def test_get_group_refs_failure(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_refs = [h5_f['/Raw_Measurement/Ancillary'],
                       h5_f,
                       np.arange(15),
                       h5_f['/Raw_Measurement/source_main-Fitter_000/results_main']]
            group_prefix = 'source_main_Blah'
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    self.assertTrue(hdf_utils.get_group_refs(group_prefix, h5_refs) == [])
            else:
                self.assertTrue(hdf_utils.get_group_refs(group_prefix, h5_refs) == [])

    def test_get_h5_obj_refs_legal_01(self):
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            expected_dsets = [h5_f['/Raw_Measurement/source_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/results_main']]
            for dset in expected_dsets:
                self.assertTrue(hdf_utils.check_if_main(dset))

    def test_check_is_main_illegal_01(self):
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        h5_f['/Raw_Measurement/Position_Indices']]
            expected_order = [[0, 1], [0], [0, 1]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    def test_get_sort_order_reversed(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_dsets = [np.flipud(h5_f['/Raw_Measurement/Spectroscopic_Indices']),
                        h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                        np.fliplr(h5_f['/Raw_Measurement/Position_Indices'])]
            expected_order = [[1, 0], [0], [1, 0]]
            for h5_dset, exp_order in zip(h5_dsets, expected_order):
                self.assertTrue(np.all(exp_order == hdf_utils.get_sort_order(h5_dset)))

    def test_get_source_dataset_legal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                        h5_f['/Raw_Measurement/source_main-Fitter_001']]
            h5_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            for h5_grp in h5_groups:
                self.assertEqual(h5_main, hdf_utils.get_source_dataset(h5_grp))

    def test_get_source_dataset_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_source_dataset(h5_f['/Raw_Measurement/Misc'])

    def test_get_unit_values_source_spec_all(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float32(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False))),
                        'Cycle': [0., 1.]}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_spec_all_explicit(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float32(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False))),
                        'Cycle': [0., 1.]}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Bias'])
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_illegal_key(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            with self.assertRaises(KeyError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Does not exist'])

    def test_get_unit_values_illegal_dset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Ancillary']
            with self.assertRaises(ValueError):
                _ = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names=['Cycle', 'Bias'])

    def test_get_unit_values_source_spec_single(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_vals = h5_f['/Raw_Measurement/Spectroscopic_Values']
            num_cycle_pts = 7
            expected = {'Bias': np.float32(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)))}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals, dim_names='Bias')
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_pos_all(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_inds = h5_f['/Raw_Measurement/Position_Indices']
            h5_vals = h5_f['/Raw_Measurement/Position_Values']
            num_rows = 3
            num_cols = 5
            expected = {'X': np.float32(np.arange(num_cols) * 50),
                        'Y': np.float32(np.arange(num_rows) * 1.25)}
            ret_val = hdf_utils.get_unit_values(h5_inds, h5_vals)
            self.assertEqual(len(expected), len(ret_val))
            for key, exp in expected.items():
                self.assertTrue(np.allclose(exp, ret_val[key]))

    def test_get_unit_values_source_pos_single(self):
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            expected_dsets = [h5_f['/Raw_Measurement/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices'],
                              h5_f['/Raw_Measurement/source_main-Fitter_001/Spectroscopic_Indices']]
            ret_val = hdf_utils.find_dataset(h5_group, 'Spectroscopic_Indices')
            self.assertEqual(set(ret_val), set(expected_dsets))

    def test_find_dataset_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement/']
            ret_val = hdf_utils.find_dataset(h5_group, 'Does_Not_Exist')
            self.assertEqual(len(ret_val), 0)

    def test_find_results_groups_legal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected_groups = [h5_f['/Raw_Measurement/source_main-Fitter_000'],
                               h5_f['/Raw_Measurement/source_main-Fitter_001']]
            ret_val = hdf_utils.find_results_groups(h5_main, 'Fitter')
            self.assertEqual(set(ret_val), set(expected_groups))

    def test_find_results_groups_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            ret_val = hdf_utils.find_results_groups(h5_main, 'Blah')
            self.assertEqual(len(ret_val), 0)

    def test_check_for_matching_attrs_dset_no_attrs(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=None))

    def test_check_for_matching_attrs_dset_matching_attrs(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'quantity':'Current'}
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_dset_one_mismatched_attrs(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'units': 'A', 'blah': 'meh'}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            self.assertTrue(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_01(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': 'string_val'}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_02(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_1': ['str_1', 'str_2', 'str_3']}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_03(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 'str_3']}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_matching_attrs_grp_mismatched_types_04(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main-Fitter_000']
            attrs = {'att_4': [1, 4.234, 45]}
            self.assertFalse(hdf_utils.check_for_matching_attrs(h5_main, new_parms=attrs))

    def test_check_for_old_exact_match(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'string_val', 'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_check_for_old_subset_but_match(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_000'])

    def test_check_for_old_exact_match_02(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': 'other_string_val', 'att_2': 5.4321,
                     'att_3': [4, 1, 3], 'att_4': ['s', 'str_2', 'str_3']}
            [h5_ret_grp] = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertEqual(h5_ret_grp, h5_f['/Raw_Measurement/source_main-Fitter_001'])

    def test_check_for_old_fail_01(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            attrs = {'att_1': [4, 1, 3], 'att_2': ['s', 'str_2', 'str_3'],
                     'att_3': 'other_string_val', 'att_4': 5.4321}
            ret_val = hdf_utils.check_for_old(h5_main, 'Fitter', new_parms=attrs, target_dset=None)
            self.assertIsInstance(ret_val, list)
            self.assertEqual(len(ret_val), 0)

    def test_check_for_old_fail_02(self):
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
            h5_raw_grp = h5_f.create_group('Raw_Measurement')

            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_dset_name = 'source_main'

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_inds, pos_attrs)

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            TestHDFUtils.__write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})

            self.assertFalse(hdf_utils.check_if_main(h5_source_main))

            # Now need to link as main!
            hdf_utils.link_as_main(h5_source_main, h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals)

            # Finally:
            self.assertTrue(hdf_utils.check_if_main(h5_source_main))

        os.remove(file_path)

    def test_link_as_main_size_mismatch(self):
        file_path = 'link_as_main.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_raw_grp = h5_f.create_group('Raw_Measurement')

            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_dset_name = 'source_main'

            source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                         np.repeat(np.arange(num_rows), num_cols))).T
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_inds, pos_attrs)

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                          np.repeat(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            TestHDFUtils.__write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})

            self.assertFalse(hdf_utils.check_if_main(h5_source_main))

            # Swap the order of the datasets to cause a clash in the shapes
            with self.assertRaises(ValueError):
                hdf_utils.link_as_main(h5_source_main, h5_source_spec_inds, h5_pos_vals, h5_pos_inds,
                                       h5_source_spec_vals)

        os.remove(file_path)

    def test_link_h5_obj_as_alias(self):
        file_path = 'link_as_alias.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            h5_main = h5_f.create_dataset('main', data=np.arange(5))
            h5_anc = h5_f.create_dataset('Ancillary', data=np.arange(3))
            h5_group = h5_f.create_group('Results')

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

            h5_main = h5_f.create_dataset('main', data=np.arange(5))
            h5_anc = h5_f.create_dataset('Ancillary', data=np.arange(3))
            h5_group = h5_f.create_group('Results')

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
        n_dim, success = hdf_utils.reshape_to_n_dims(source_main_data, h5_pos=source_pos_data,
                                                     h5_spec=source_spec_data, get_labels=False)
        expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
        self.assertTrue(np.allclose(expected_n_dim, n_dim))

    def test_reshape_to_n_dim_sort_required(self):
        file_path = 'reshape_to_n_dim_sort_required.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_raw_grp = h5_f.create_group('Raw_Measurement')

            num_rows = 3
            num_cols = 5
            num_cycles = 2
            num_cycle_pts = 7

            source_dset_name = 'source_main'

            # arrange as slow, fast instead of fast, slow
            source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                         np.tile(np.arange(num_cols), num_rows))).T
            pos_attrs = {'units': ['nm', 'um'], 'labels': ['X', 'Y']}

            h5_pos_inds = h5_raw_grp.create_dataset('Position_Indices', data=source_pos_data, dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_inds, pos_attrs)

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            TestHDFUtils.__write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_main_data = np.zeros(shape=(num_rows * num_cols, num_cycle_pts * num_cycles), dtype=np.float16)
            for row_ind in range(num_rows):
                for col_ind in range(num_cols):
                    for cycle_ind in range(num_cycles):
                        for bias_ind in range(num_cycle_pts):
                            val = 1E+3*row_ind + 1E+2*col_ind + 1E+1*cycle_ind + bias_ind
                            source_main_data[row_ind*num_cols + col_ind, cycle_ind*num_cycle_pts + bias_ind] = val

            # source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            TestHDFUtils.__write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})

            # make spectroscopic slow, fast instead of fast, slow
            source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                          np.tile(np.arange(num_cycle_pts), num_cycles)))
            source_spec_attrs = {'units': ['', 'V'], 'labels': ['Cycle', 'Bias']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            TestHDFUtils.__write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            TestHDFUtils.__write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            n_dim, success, labels = hdf_utils.reshape_to_n_dims(h5_source_main, get_labels=True, sort_dims=True)
            self.assertTrue(np.all([x == y for x, y in zip(labels, ['Y', 'X', 'Bias', 'Cycle'])]))
            expected_n_dim = np.reshape(source_main_data, (num_rows, num_cols, num_cycles, num_cycle_pts))
            expected_n_dim = np.transpose(expected_n_dim, [1, 0, 3, 2])
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
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))

        ret_2d, success = hdf_utils.reshape_from_n_dims(source_nd, h5_pos=source_pos_data, h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 2: Only Pos arranged as slow to fast:
        main_pos_sorted = np.transpose(source_nd, (0, 1, 3, 2))
        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                      np.repeat(np.arange(num_cycles), num_cycle_pts),))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_pos_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 3: only Spec arranged as slow to fast:
        main_spec_sorted = np.transpose(source_nd, (1, 0, 2, 3))
        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))

        ret_2d, success = hdf_utils.reshape_from_n_dims(main_spec_sorted, h5_pos=source_pos_data,
                                                        h5_spec=source_spec_data)
        self.assertTrue(success)
        self.assertTrue(np.allclose(ret_2d, expected_2d))

        # case 4: neither pos nor spec arranged as slow to fast:
        main_not_sorted = np.transpose(source_nd, (1, 0, 3, 2))
        source_pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                                     np.repeat(np.arange(num_rows), num_cols))).T
        source_spec_data = np.vstack((np.tile(np.arange(num_cycle_pts), num_cycles),
                                      np.repeat(np.arange(num_cycles), num_cycle_pts),))

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

        source_pos_data = np.vstack((np.repeat(np.arange(num_rows), num_cols),
                                     np.tile(np.arange(num_cols), num_rows))).T
        source_spec_data = np.vstack((np.repeat(np.arange(num_cycles), num_cycle_pts),
                                      np.tile(np.arange(num_cycle_pts), num_cycles)))

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
            data = np.random.rand(5, 7)
            h5_orig_dset = h5_f.create_dataset('test', data=data)
            self.assertIsInstance(h5_orig_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}}

            TestHDFUtils.__write_main_reg_refs(h5_orig_dset, attrs['labels'])
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
            h5_new_dset = h5_f.create_dataset('other', data=data)
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            ret_val = hdf_utils.assign_group_index(h5_group, 'source_main-Fitter')
            self.assertEqual(ret_val, 'source_main-Fitter_002')

    def test_assign_group_index_new(self):
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_group = h5_f['/Raw_Measurement']
            h5_main = h5_f['/Raw_Measurement/Ancillary']
            self.assertFalse(hdf_utils.is_editable_h5(h5_group))
            self.assertFalse(hdf_utils.is_editable_h5(h5_f))
            self.assertFalse(hdf_utils.is_editable_h5(h5_main))

    def test_is_editable_h5_r_plus(self):
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
                                                      spec_dims, dtype=np.float16, main_dset_attrs=None)
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
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            else:
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
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    cleaned = hdf_utils.attempt_reg_ref_build(h5_dset, dim_names)
            else:
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
        duplicate_path = 'copy_test_hdf_utils.h5'
        self.__delete_existing_file(duplicate_path)
        shutil.copy(test_h5_file_path, duplicate_path)
        with h5py.File(duplicate_path) as h5_f:
            h5_spec_inds_orig = h5_f['/Raw_Measurement/Spectroscopic_Indices']
            h5_spec_vals_orig = h5_f['/Raw_Measurement/Spectroscopic_Values']
            new_base_name = 'Blah'
            # cycle_starts = np.where(h5_spec_inds_orig[0] == 0)[0]
            h5_spec_inds_new, h5_spec_vals_new = hdf_utils.write_reduced_spec_dsets(h5_spec_inds_orig.parent,
                                                                                    h5_spec_inds_orig,
                                                                                    h5_spec_vals_orig,
                                                                                    'Bias',
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

    def test_get_indices_for_region_ref_corners(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            reg_ref = hdf_utils.get_attr(h5_main, 'even_rows')
            ret_val = hdf_utils.get_indices_for_region_ref(h5_main, reg_ref, 'corners')
            expected_pos = np.repeat(np.arange(h5_main.shape[0])[::2], 2)
            expected_spec = np.tile(np.array([0,h5_main.shape[1]-1]), expected_pos.size // 2)
            expected_corners = np.vstack((expected_pos, expected_spec)).T
            self.assertTrue(np.allclose(ret_val, expected_corners))

    def test_get_indices_for_region_ref_slices(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            reg_ref = hdf_utils.get_attr(h5_main, 'even_rows')
            ret_val = hdf_utils.get_indices_for_region_ref(h5_main, reg_ref, 'slices')
            spec_slice = slice(0, h5_main.shape[1] - 1, None)
            expected_slices = np.array([[slice(x, x, None), spec_slice] for x in np.arange(h5_main.shape[0])[::2]])
            self.assertTrue(np.all(ret_val == expected_slices))

    def __verify_book_keeping_attrs(self, h5_obj):
        time_stamp = io_utils.get_time_stamp()
        in_file = h5_obj.attrs['timestamp']
        self.assertEqual(time_stamp[:time_stamp.rindex('_')], in_file[:in_file.rindex('_')])
        self.assertEqual(__version__, h5_obj.attrs['pycroscopy_version'])
        self.assertEqual(socket.getfqdn(), h5_obj.attrs['machine_id'])
        self.assertEqual(platform(), h5_obj.attrs['platform'])

    def test_write_book_keeping_attrs_file(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            hdf_utils.write_book_keeping_attrs(h5_f)
            self.__verify_book_keeping_attrs(h5_f)
        os.remove(file_path)

    def test_write_book_keeping_attrs_group(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_g = h5_f.create_group('group')
            hdf_utils.write_book_keeping_attrs(h5_g)
            self.__verify_book_keeping_attrs(h5_g)
        os.remove(file_path)

    def test_write_book_keeping_attrs_dset(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('dset', data=[1, 2, 3])
            hdf_utils.write_book_keeping_attrs(h5_dset)
            self.__verify_book_keeping_attrs(h5_dset)
        os.remove(file_path)

    def test_write_book_keeping_attrs_invalid(self):
        with self.assertRaises(TypeError):
            hdf_utils.write_book_keeping_attrs(np.arange(4))

    def test_copy_attributes_file_dset(self):
        file_path = 'test.h5'
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        also_easy_attr = {'N_numbers': [1, -53.6, 0.000463]}
        hard_attrs = {'N_strings': np.array(['a', 'bc', 'def'], dtype='S')}
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_f.attrs.update(easy_attrs)
            h5_f.attrs.update(also_easy_attr)
            h5_f.attrs.update(hard_attrs)
            h5_dset = h5_f.create_dataset('Main_01', data=[1, 2, 3])
            hdf_utils.copy_attributes(h5_f, h5_dset)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset.attrs[key])
            for key, val in also_easy_attr.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
            for key, val in hard_attrs.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
        os.remove(file_path)

    def test_copy_attributes_group_dset(self):
        file_path = 'test.h5'
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        also_easy_attr = {'N_numbers': [1, -53.6, 0.000463]}
        hard_attrs = {'N_strings': np.array(['a', 'bc', 'def'], dtype='S')}
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_group = h5_f.create_group('Group')
            h5_group.attrs.update(easy_attrs)
            h5_group.attrs.update(also_easy_attr)
            h5_group.attrs.update(hard_attrs)
            h5_dset = h5_f.create_dataset('Main_01', data=[1, 2, 3])
            hdf_utils.copy_attributes(h5_group, h5_dset)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset.attrs[key])
            for key, val in also_easy_attr.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
            for key, val in hard_attrs.items():
                self.assertTrue(np.all([x == y for x, y in zip(val, h5_dset.attrs[key])]))
        os.remove(file_path)

    def test_copy_attributes_dset_w_reg_ref_group_but_skipped(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(easy_attrs)
            h5_dset_sink = h5_f.create_dataset('Sink', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            hdf_utils.copy_attributes(h5_dset_source, h5_dset_sink, skip_refs=True)

            self.assertEqual(len(h5_dset_sink.attrs), len(easy_attrs))
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset_sink.attrs[key])

        os.remove(file_path)

    def test_copy_attributes_dset_w_reg_ref_group_to_file(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(easy_attrs)
            reg_refs = {'even_rows': (slice(None), slice(0, None, 2)),
                        'odd_rows': (slice(None), slice(1, None, 2))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    hdf_utils.copy_attributes(h5_dset_source, h5_f, skip_refs=False)
            else:
                hdf_utils.copy_attributes(h5_dset_source, h5_f, skip_refs=False)

            self.assertEqual(len(h5_f.attrs), len(easy_attrs))
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_f.attrs[key])

        os.remove(file_path)

    def test_copy_attributes_dset_w_reg_ref_group(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(easy_attrs)
            h5_dset_sink = h5_f.create_dataset('Sink', data=data)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            hdf_utils.copy_attributes(h5_dset_source, h5_dset_sink, skip_refs=False)

            self.assertEqual(len(h5_dset_sink.attrs), len(reg_refs) + len(easy_attrs))
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_dset_sink.attrs[key])

            self.assertTrue('labels' not in h5_dset_sink.attrs.keys())

            expected_data = [data[0:None:2, :], data[1:None:2, :]]
            written_data = [h5_dset_sink[h5_dset_sink.attrs['even_rows']],
                            h5_dset_sink[h5_dset_sink.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_copy_attributes_illegal_to_from_reg_ref(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_dest = h5_f.create_dataset('Sink', data=data[:-1, :-1])
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}
            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    hdf_utils.copy_attributes(h5_dset_source, h5_dset_dest, skip_refs=False)
            else:
                hdf_utils.copy_attributes(h5_dset_source, h5_dset_dest, skip_refs=False)

    def test_copy_main_attributes_valid(self):
        file_path = 'test.h5'
        main_attrs = {'quantity': 'Current', 'units': 'nA'}
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Main_01', data=[1, 23])
            h5_dset_source.attrs.update(main_attrs)
            h5_group = h5_f.create_group('Group')
            h5_dset_sink = h5_group.create_dataset('Main_02', data=[4, 5])
            hdf_utils.copy_main_attributes(h5_dset_source, h5_dset_sink)
            for key, val in main_attrs.items():
                self.assertEqual(val, h5_dset_sink.attrs[key])
        os.remove(file_path)

    def test_copy_main_attributes_no_main_attrs(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Main_01', data=[1, 23])
            h5_group = h5_f.create_group('Group')
            h5_dset_sink = h5_group.create_dataset('Main_02', data=[4, 5])
            with self.assertRaises(KeyError):
                hdf_utils.copy_main_attributes(h5_dset_source, h5_dset_sink)
        os.remove(file_path)

    def test_copy_main_attributes_wrong_objects(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Main_01', data=[1, 23])
            h5_group = h5_f.create_group('Group')
            with self.assertRaises(TypeError):
                hdf_utils.copy_main_attributes(h5_dset_source, h5_group)
            with self.assertRaises(TypeError):
                hdf_utils.copy_main_attributes(h5_group, h5_dset_source)
        os.remove(file_path)

    def test_create_empty_dataset_same_group_new_attrs(self):
        file_path = 'test.h5'
        existing_attrs = {'a': 1, 'b': 'Hello'}
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_source.attrs.update(existing_attrs)
            h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Duplicate', new_attrs=easy_attrs)
            self.assertIsInstance(h5_duplicate, h5py.Dataset)
            self.assertEqual(h5_duplicate.parent, h5_dset_source.parent)
            self.assertEqual(h5_duplicate.name, '/Duplicate')
            self.assertEqual(h5_duplicate.dtype, np.float16)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])
            for key, val in existing_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])

        os.remove(file_path)

    def test_create_empty_dataset_diff_groups(self):
        file_path = 'test.h5'
        existing_attrs = {'a': 1, 'b': 'Hello'}
        easy_attrs = {'1_string': 'Current', '1_number': 35.23}
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            h5_dset_source.attrs.update(existing_attrs)
            h5_group = h5_f.create_group('Group')
            h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Duplicate',
                                                          h5_group=h5_group, new_attrs=easy_attrs)
            self.assertIsInstance(h5_duplicate, h5py.Dataset)
            self.assertEqual(h5_duplicate.parent, h5_group)
            self.assertEqual(h5_duplicate.name, '/Group/Duplicate')
            self.assertEqual(h5_duplicate.dtype, np.float16)
            for key, val in easy_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])
            for key, val in existing_attrs.items():
                self.assertEqual(val, h5_duplicate.attrs[key])

        os.remove(file_path)

    def test_create_empty_dataset_w_region_refs(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        main_attrs = {'quantity': 'Current', 'units': 'nA'}
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_source.attrs.update(main_attrs)
            reg_refs = {'even_rows': (slice(0, None, 2), slice(None)),
                        'odd_rows': (slice(1, None, 2), slice(None))}

            for reg_ref_name, reg_ref_tuple in reg_refs.items():
                h5_dset_source.attrs[reg_ref_name] = h5_dset_source.regionref[reg_ref_tuple]

            h5_copy = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Existing')

            for reg_ref_name in reg_refs.keys():
                self.assertTrue(isinstance(h5_copy.attrs[reg_ref_name], h5py.RegionReference))
                self.assertTrue(h5_dset_source[h5_dset_source.attrs[reg_ref_name]].shape == h5_copy[h5_copy.attrs[reg_ref_name]].shape)

        os.remove(file_path)

    def test_create_empty_dataset_existing_dset_name(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=[1, 2, 3])
            _ = h5_f.create_dataset('Existing', data=[4, 5, 6])
            if sys.version_info.major == 3:
                with self.assertWarns(UserWarning):
                    h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Existing')
            else:
                h5_duplicate = hdf_utils.create_empty_dataset(h5_dset_source, np.float16, 'Existing')
            self.assertIsInstance(h5_duplicate, h5py.Dataset)
            self.assertEqual(h5_duplicate.name, '/Existing')
            self.assertTrue(np.allclose(h5_duplicate[()], np.zeros(3)))
            self.assertEqual(h5_duplicate.dtype, np.float16)
        os.remove(file_path)

    def test_create_index_group_first_group(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_group = hdf_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Hello_000')
            self.assertEqual(h5_group.parent, h5_f)
            self.__verify_book_keeping_attrs(h5_group)

            h5_sub_group = hdf_utils.create_indexed_group(h5_group, 'Test')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Hello_000/Test_000')
            self.assertEqual(h5_sub_group.parent, h5_group)
            self.__verify_book_keeping_attrs(h5_sub_group)
        os.remove(file_path)

    def test_create_index_group_second(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_group_1 = hdf_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group_1, h5py.Group)
            self.assertEqual(h5_group_1.name, '/Hello_000')
            self.assertEqual(h5_group_1.parent, h5_f)
            self.__verify_book_keeping_attrs(h5_group_1)

            h5_group_2 = hdf_utils.create_indexed_group(h5_f, 'Hello')
            self.assertIsInstance(h5_group_2, h5py.Group)
            self.assertEqual(h5_group_2.name, '/Hello_001')
            self.assertEqual(h5_group_2.parent, h5_f)
            self.__verify_book_keeping_attrs(h5_group_2)
        os.remove(file_path)

    def test_create_index_group_w_suffix_(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_group = hdf_utils.create_indexed_group(h5_f, 'Hello_')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Hello_000')
            self.assertEqual(h5_group.parent, h5_f)
            self.__verify_book_keeping_attrs(h5_group)
        os.remove(file_path)

    def test_create_index_group_empty_base_name(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            with self.assertRaises(ValueError):
                _ = hdf_utils.create_indexed_group(h5_f, '    ')

        os.remove(file_path)

    def test_create_results_group_first(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1,2,3])
            h5_group = hdf_utils.create_results_group(h5_dset, 'Tool')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Main-Tool_000')
            self.assertEqual(h5_group.parent, h5_f)
            self.__verify_book_keeping_attrs(h5_group)

            h5_dset = h5_group.create_dataset('Main_Dataset', data=[1, 2, 3])
            h5_sub_group = hdf_utils.create_results_group(h5_dset, 'SHO_Fit')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Main-Tool_000/Main_Dataset-SHO_Fit_000')
            self.assertEqual(h5_sub_group.parent, h5_group)
            self.__verify_book_keeping_attrs(h5_sub_group)
        os.remove(file_path)

    def test_create_results_group_second(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            h5_group = hdf_utils.create_results_group(h5_dset, 'Tool')
            self.assertIsInstance(h5_group, h5py.Group)
            self.assertEqual(h5_group.name, '/Main-Tool_000')
            self.assertEqual(h5_group.parent, h5_f)
            self.__verify_book_keeping_attrs(h5_group)

            h5_sub_group = hdf_utils.create_results_group(h5_dset, 'Tool')
            self.assertIsInstance(h5_sub_group, h5py.Group)
            self.assertEqual(h5_sub_group.name, '/Main-Tool_001')
            self.assertEqual(h5_sub_group.parent, h5_f)
            self.__verify_book_keeping_attrs(h5_sub_group)
        os.remove(file_path)

    def test_create_results_group_empty_tool_name(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Main', data=[1, 2, 3])
            with self.assertRaises(ValueError):
                _ = hdf_utils.create_results_group(h5_dset, '   ')
        os.remove(file_path)

    def test_create_results_group_not_dataset(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            with self.assertRaises(TypeError):
                _ = hdf_utils.create_results_group(h5_f, 'Tool')
                
        os.remove(file_path)

    def test_create_region_ref(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(5, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset = h5_f.create_dataset('Source', data=data)
            pos_inds = np.arange(0, h5_dset.shape[0], 2)
            ref_inds = [((pos_start, 0), (pos_start, h5_dset.shape[1]-1)) for pos_start in pos_inds]
            ref_inds = np.array(ref_inds)
            reg_ref = hdf_utils.create_region_reference(h5_dset, ref_inds)
            ref_slices = list()
            for start, stop in ref_inds:
                ref_slices.append([slice(start[0], stop[0]+1), slice(start[1], None)])

            h5_reg = h5_dset[reg_ref]

            h5_slice = np.vstack([h5_dset[pos_slice, spec_slice] for (pos_slice, spec_slice) in ref_slices])

            self.assertTrue(np.allclose(h5_reg, h5_slice))

        os.remove(file_path)

    def test_copy_region_refs(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        data = np.random.rand(11, 7)
        with h5py.File(file_path) as h5_f:
            h5_dset_source = h5_f.create_dataset('Source', data=data)
            h5_dset_dest = h5_f.create_dataset('Target', data=data)
            source_ref = h5_dset_source.regionref[0:-1:2]
            h5_dset_source.attrs['regref'] = source_ref

            hdf_utils.copy_region_refs(h5_dset_source, h5_dset_dest)

            self.assertTrue(np.allclose(h5_dset_source[h5_dset_source.attrs['regref']],
                                        h5_dset_dest[h5_dset_dest.attrs['regref']]))

        os.remove(file_path)

    def test_copy_reg_ref_reduced_dim(self):
        # TODO: Fill this test in at earliest convenience. Overriden temporarily
        assert True


if __name__ == '__main__':
    unittest.main()
