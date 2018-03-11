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
from pycroscopy.core.io import PycroDataset, VirtualDataset, VirtualGroup, hdf_utils, HDFwriter

if sys.version_info.major == 3:
    unicode = str

test_h5_file_path = 'test_hdf_utils.h5'


class TestPycroDataset(unittest.TestCase):

    @staticmethod
    def _create_test_h5_file():
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

    def __ensure_test_file(self):
        if not os.path.exists(test_h5_file_path):
            TestPycroDataset._create_test_h5_file()

    def test_equality_correct_pycrodataset(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            self.assertTrue(expected == expected)

    def test_equality_correct_h5_dataset(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            self.assertTrue(expected == h5_main)

    def test_equality_incorrect_pycrodataset(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            incorrect = PycroDataset(h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'])
            self.assertFalse(expected == incorrect)

    def test_equality_incorrect_h5_dataset(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            incorrect = h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']
            self.assertFalse(expected == incorrect)

    def test_equality_incorrect_object(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            incorrect = np.zeros(shape=(1, 2, 3, 4))
            self.assertFalse(expected == incorrect)

    def test_string_representation(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            pycro_dset = PycroDataset(h5_main)
            actual = pycro_dset.__repr__()
            actual = [line.strip() for line in actual.split("\n")]
            actual = [actual[line_ind] for line_ind in [0, 2, 4, 7, 8, 10, 11]]

            expected = list()
            expected.append(h5_main.__repr__())
            expected.append(h5_main.name)
            expected.append(hdf_utils.get_attr(h5_main, "quantity") + " (" + hdf_utils.get_attr(h5_main, "units") + ")")
            for h5_inds in [pycro_dset.h5_pos_inds, pycro_dset.h5_spec_inds]:
                for dim_name, dim_size in zip(hdf_utils.get_attr(h5_inds, "labels"),
                                              hdf_utils.get_dimensionality(h5_inds)):
                    expected.append(dim_name + ' - size: ' + str(dim_size))
            self.assertTrue(np.all([x == y for x, y in zip(actual, expected)]))

    def test_get_n_dim_form_unsorted(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = np.reshape(h5_main, (3, 5, 2, 7))
            expected = np.transpose(expected, (1, 0, 3, 2))
            pycro_dset = PycroDataset(h5_main)
            self.assertTrue(np.allclose(expected, pycro_dset.get_n_dim_form()))

    def test_get_n_dim_form_sorted(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = np.reshape(h5_main, (3, 5, 2, 7))
            # expected = np.transpose(expected, (1, 0, 3, 2))
            pycro_dset = PycroDataset(h5_main)
            pycro_dset.toggle_sorting()
            self.assertTrue(np.allclose(expected, pycro_dset.get_n_dim_form()))

    def test_get_pos_spec_slices_empty_dict(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({})
            self.assertTrue(np.allclose(np.expand_dims(np.arange(14), axis=1), actual_spec))
            self.assertTrue(np.allclose(np.expand_dims(np.arange(15), axis=1), actual_pos))

    def test_get_pos_spec_slices_non_existent_dim(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = pycro_main._get_pos_spec_slices({'blah': 4, 'X': 3, 'Y': 1})

    def test_get_pos_spec_slices_incorrect_type(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(TypeError):
                _ = pycro_main._get_pos_spec_slices({'X': 'fdfd', 'Y': 1})

    def test_get_pos_spec_slices_negative_index(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = pycro_main._get_pos_spec_slices({'X': -4, 'Y': 1})

    def test_get_pos_spec_slices_out_of_bounds(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = pycro_main._get_pos_spec_slices({'X': 15, 'Y': 1})

    def test_get_pos_spec_slices_one_pos_dim_removed(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            # orig_pos = np.vstack([np.tile(np.arange(5), 3), np.repeat(np.arange(3), 5)]).T
            # orig_spec = np.vstack([np.tile(np.arange(7), 2), np.repeat(np.arange(2), 7)])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': 3})
            # we want every fifth position starting from 3
            expected_pos = np.expand_dims(np.arange(3, 15, 5), axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_get_pos_spec_slices_one_pos_dim_sliced(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': slice(1, 5, 2)})
            # we want every fifth position starting from 3
            positions = []
            for row_ind in range(3):
                for col_ind in range(1, 5, 2):
                    positions.append(5 * row_ind + col_ind)
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_get_pos_spec_slices_two_pos_dim_sliced(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': slice(1, 5, 2), 'Y': 1})
            # we want every fifth position starting from 3
            positions = []
            for row_ind in range(1, 2):
                for col_ind in range(1, 5, 2):
                    positions.append(5 * row_ind + col_ind)
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_get_pos_spec_slices_two_pos_dim_sliced_list(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': [1, 2, 4], 'Y': 1})
            # we want every fifth position starting from 3
            positions = []
            for row_ind in range(1, 2):
                for col_ind in [1, 2, 4]:
                    positions.append(5 * row_ind + col_ind)
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_get_pos_spec_slices_both_pos_removed(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': 3, 'Y': 1})
            # we want every fifth position starting from 3
            expected_pos = np.expand_dims([1 * 5 + 3], axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_get_pos_spec_slices_pos_and_spec_sliced_list(self):
        self.__ensure_test_file()
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)})
            # we want every fifth position starting from 3
            positions = []
            for row_ind in range(3):
                for col_ind in [1, 2, 4]:
                    positions.append(5 * row_ind + col_ind)
            specs = []
            for cycle_ind in range(2):
                for bias_ind in range(1, 7, 3):
                    specs.append(7 * cycle_ind + bias_ind)
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(specs, axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))






if __name__ == '__main__':
    unittest.main()
