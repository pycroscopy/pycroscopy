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
from io import StringIO
from contextlib import contextmanager

sys.path.append("../../../pycroscopy/")
from pycroscopy.core.io import PycroDataset, hdf_utils

if sys.version_info.major == 3:
    unicode = str

test_h5_file_path = 'test_pycro_dataset.h5'


@contextmanager
def capture_stdout():
    """
    context manager encapsulating a pattern for capturing stdout writes
    and restoring sys.stdout even upon exceptions

    Examples:
    >>> with capture_stdout() as get_value:
    >>>     print("here is a print")
    >>>     captured = get_value()
    >>> print('Gotcha: ' + captured)

    >>> with capture_stdout() as get_value:
    >>>     print("here is a print")
    >>>     raise Exception('oh no!')
    >>> print('Does printing still work?')
    """
    # Redirect sys.stdout
    out = StringIO()
    sys.stdout = out
    # Yield a method clients can use to obtain the value
    try:
        yield out.getvalue
    finally:
        # Restore the normal stdout
        sys.stdout = sys.__stdout__


class TestPycroDataset(unittest.TestCase):

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
        TestPycroDataset.__write_string_list_as_attr(h5_dset, {'labels': list(attrs.keys())})

    def setUp(self):

        if os.path.exists(test_h5_file_path):
            os.remove(test_h5_file_path)
        with h5py.File(test_h5_file_path) as h5_f:

            h5_raw_grp = h5_f.create_group('Raw_Measurement')
            TestPycroDataset.__write_safe_attrs(h5_raw_grp, {'att_1': 'string_val', 'att_2': 1.2345,
                                                             'att_3': [1, 2, 3, 4]})
            TestPycroDataset.__write_string_list_as_attr(h5_raw_grp, {'att_4': ['str_1', 'str_2', 'str_3']})

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
            TestPycroDataset.__write_aux_reg_ref(h5_pos_inds, pos_attrs['labels'], is_spec=False)
            TestPycroDataset.__write_string_list_as_attr(h5_pos_inds, pos_attrs)

            # make the values more interesting:
            source_pos_data = np.vstack((source_pos_data[:, 0] * 50, source_pos_data[:, 1] * 1.25)).T

            h5_pos_vals = h5_raw_grp.create_dataset('Position_Values', data=source_pos_data, dtype=np.float32)
            TestPycroDataset.__write_aux_reg_ref(h5_pos_vals, pos_attrs['labels'], is_spec=False)
            TestPycroDataset.__write_string_list_as_attr(h5_pos_vals, pos_attrs)

            source_spec_data = np.vstack((np.repeat(np.arange(num_cycle_pts), num_cycles),
                                          np.tile(np.arange(num_cycles), num_cycle_pts)))
            source_spec_attrs = {'units': ['V', ''], 'labels': ['Bias', 'Cycle']}

            h5_source_spec_inds = h5_raw_grp.create_dataset('Spectroscopic_Indices', data=source_spec_data,
                                                            dtype=np.uint16)
            TestPycroDataset.__write_aux_reg_ref(h5_source_spec_inds, source_spec_attrs['labels'], is_spec=True)
            TestPycroDataset.__write_string_list_as_attr(h5_source_spec_inds, source_spec_attrs)

            # make spectroscopic axis interesting as well
            source_spec_data = np.vstack(
                (np.repeat(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)),
                         num_cycles),
                 np.tile(np.arange(num_cycles), num_cycle_pts)))

            h5_source_spec_vals = h5_raw_grp.create_dataset('Spectroscopic_Values', data=source_spec_data,
                                                            dtype=np.float32)
            TestPycroDataset.__write_aux_reg_ref(h5_source_spec_vals, source_spec_attrs['labels'], is_spec=True)
            TestPycroDataset.__write_string_list_as_attr(h5_source_spec_vals, source_spec_attrs)

            source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_source_main = h5_raw_grp.create_dataset(source_dset_name, data=source_main_data)
            TestPycroDataset.__write_safe_attrs(h5_source_main, {'units': 'A', 'quantity': 'Current'})
            TestPycroDataset.__write_main_reg_refs(h5_source_main, {'even_rows': (slice(0, None, 2), slice(None)),
                                                                    'odd_rows': (slice(1, None, 2), slice(None))})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_source_spec_inds, h5_source_spec_vals]:
                h5_source_main.attrs[dset.name.split('/')[-1]] = dset.ref

            _ = h5_raw_grp.create_dataset('Ancillary', data=np.arange(5))

            # Now add a few results:

            h5_results_grp_1 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_000')
            TestPycroDataset.__write_safe_attrs(h5_results_grp_1,
                                                {'att_1': 'string_val', 'att_2': 1.2345, 'att_3': [1, 2, 3, 4]})
            TestPycroDataset.__write_string_list_as_attr(h5_results_grp_1, {'att_4': ['str_1', 'str_2', 'str_3']})

            num_cycles = 1
            num_cycle_pts = 7

            results_spec_inds = np.expand_dims(np.arange(num_cycle_pts), 0)
            results_spec_attrs = {'units': ['V'], 'labels': ['Bias']}

            h5_results_1_spec_inds = h5_results_grp_1.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            TestPycroDataset.__write_aux_reg_ref(h5_results_1_spec_inds, results_spec_attrs['labels'], is_spec=True)
            TestPycroDataset.__write_string_list_as_attr(h5_results_1_spec_inds, results_spec_attrs)

            results_spec_vals = np.expand_dims(2.5 * np.sin(np.linspace(0, np.pi, num_cycle_pts, endpoint=False)), 0)

            h5_results_1_spec_vals = h5_results_grp_1.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            TestPycroDataset.__write_aux_reg_ref(h5_results_1_spec_vals, results_spec_attrs['labels'], is_spec=True)
            TestPycroDataset.__write_string_list_as_attr(h5_results_1_spec_vals, results_spec_attrs)

            results_1_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_1_main = h5_results_grp_1.create_dataset('results_main', data=results_1_main_data)
            TestPycroDataset.__write_safe_attrs(h5_results_1_main, {'units': 'pF', 'quantity': 'Capacitance'})

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_1_spec_inds, h5_results_1_spec_vals]:
                h5_results_1_main.attrs[dset.name.split('/')[-1]] = dset.ref

            # add another result with different parameters

            h5_results_grp_2 = h5_raw_grp.create_group(source_dset_name + '-' + tool_name + '_001')
            TestPycroDataset.__write_safe_attrs(h5_results_grp_2,
                                                {'att_1': 'other_string_val', 'att_2': 5.4321, 'att_3': [4, 1, 3]})
            TestPycroDataset.__write_string_list_as_attr(h5_results_grp_2, {'att_4': ['s', 'str_2', 'str_3']})

            results_2_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)
            h5_results_2_main = h5_results_grp_2.create_dataset('results_main', data=results_2_main_data)
            TestPycroDataset.__write_safe_attrs(h5_results_2_main, {'units': 'pF', 'quantity': 'Capacitance'})

            h5_results_2_spec_inds = h5_results_grp_2.create_dataset('Spectroscopic_Indices',
                                                                     data=results_spec_inds, dtype=np.uint16)
            TestPycroDataset.__write_aux_reg_ref(h5_results_2_spec_inds, results_spec_attrs['labels'], is_spec=True)
            TestPycroDataset.__write_string_list_as_attr(h5_results_2_spec_inds, results_spec_attrs)

            h5_results_2_spec_vals = h5_results_grp_2.create_dataset('Spectroscopic_Values', data=results_spec_vals,
                                                                     dtype=np.float32)
            TestPycroDataset.__write_aux_reg_ref(h5_results_2_spec_vals, results_spec_attrs['labels'], is_spec=True)
            TestPycroDataset.__write_string_list_as_attr(h5_results_2_spec_vals, results_spec_attrs)

            # Now need to link as main!
            for dset in [h5_pos_inds, h5_pos_vals, h5_results_2_spec_inds, h5_results_2_spec_vals]:
                h5_results_2_main.attrs[dset.name.split('/')[-1]] = dset.ref

    def tearDown(self):
        os.remove(test_h5_file_path)

    def test_equality_correct_pycrodataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            self.assertTrue(expected == expected)

    def test_equality_correct_h5_dataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            self.assertTrue(expected == h5_main)

    def test_equality_incorrect_pycrodataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            incorrect = PycroDataset(h5_f['/Raw_Measurement/source_main-Fitter_000/results_main'])
            self.assertFalse(expected == incorrect)

    def test_equality_incorrect_h5_dataset(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            incorrect = h5_f['/Raw_Measurement/source_main-Fitter_000/Spectroscopic_Indices']
            self.assertFalse(expected == incorrect)

    def test_equality_incorrect_object(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = PycroDataset(h5_main)
            incorrect = np.zeros(shape=(1, 2, 3, 4))
            self.assertFalse(expected == incorrect)

    def test_string_representation(self):
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = np.reshape(h5_main, (3, 5, 7, 2))
            expected = np.transpose(expected, (1, 0, 2, 3))
            pycro_dset = PycroDataset(h5_main)
            self.assertTrue(np.allclose(expected, pycro_dset.get_n_dim_form()))

    def test_get_n_dim_form_sorted(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            h5_main = h5_f['/Raw_Measurement/source_main']
            expected = np.reshape(h5_main, (3, 5, 7, 2))
            expected = np.transpose(expected, (1, 0, 3, 2))
            pycro_dset = PycroDataset(h5_main)
            pycro_dset.toggle_sorting()
            self.assertTrue(np.allclose(expected, pycro_dset.get_n_dim_form()))

    def test_get_pos_spec_slices_empty_dict(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({})
            self.assertTrue(np.allclose(np.expand_dims(np.arange(14), axis=1), actual_spec))
            self.assertTrue(np.allclose(np.expand_dims(np.arange(15), axis=1), actual_pos))

    def test_get_pos_spec_slices_non_existent_dim(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = pycro_main._get_pos_spec_slices({'blah': 4, 'X': 3, 'Y': 1})

    def test_get_pos_spec_slices_incorrect_type(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(TypeError):
                _ = pycro_main._get_pos_spec_slices({'X': 'fdfd', 'Y': 1})

    def test_get_pos_spec_slices_negative_index(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = pycro_main._get_pos_spec_slices({'X': -4, 'Y': 1})

    def test_get_pos_spec_slices_out_of_bounds(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = pycro_main._get_pos_spec_slices({'X': 15, 'Y': 1})

    def test_get_pos_spec_slices_one_pos_dim_removed(self):
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
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': 3, 'Y': 1})
            # we want every fifth position starting from 3
            expected_pos = np.expand_dims([1 * 5 + 3], axis=1)
            expected_spec = np.expand_dims(np.arange(14), axis=1)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_get_pos_spec_slices_pos_and_spec_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            h5_pos_inds = pycro_main.h5_pos_inds
            h5_spec_inds = pycro_main.h5_spec_inds
            actual_pos, actual_spec = pycro_main._get_pos_spec_slices({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)})
            # we want every fifth position starting from 3
            positions = []
            for col_ind in [1, 2, 4]:
                positions += np.argwhere(h5_pos_inds[h5_pos_inds.attrs['X']] == col_ind)[:, 0].tolist()
            specs = []
            for bias_ind in range(1, 7, 3):
                specs += np.argwhere(h5_spec_inds[h5_spec_inds.attrs['Bias']] == bias_ind)[:, 1].tolist()
            expected_pos = np.expand_dims(positions, axis=1)
            expected_spec = np.expand_dims(specs, axis=1)
            expected_pos.sort(axis=0)
            expected_spec.sort(axis=0)
            self.assertTrue(np.allclose(expected_spec, actual_spec))
            self.assertTrue(np.allclose(expected_pos, actual_pos))

    def test_get_pos_values(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            expected = pycro_main.h5_pos_vals[:5, 0]
            actual = pycro_main.get_pos_values('X')
            self.assertTrue(np.allclose(expected, actual))
            expected = pycro_main.h5_pos_vals[0:None:5, 1]
            actual = pycro_main.get_pos_values('Y')
            self.assertTrue(np.allclose(expected, actual))

    def test_get_pos_values_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = pycro_main.get_pos_values('blah')
            with self.assertRaises(TypeError):
                _ = pycro_main.get_pos_values(np.array(5))

    def test_get_spec_values(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            expected = pycro_main.h5_spec_vals[0, ::2]
            actual = pycro_main.get_spec_values('Bias')
            self.assertTrue(np.allclose(expected, actual))
            expected = pycro_main.h5_spec_vals[1, 0:None:7]
            actual = pycro_main.get_spec_values('Cycle')
            self.assertTrue(np.allclose(expected, actual))

    def test_get_spec_values_illegal(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = pycro_main.get_spec_values('blah')
            with self.assertRaises(TypeError):
                _ = pycro_main.get_spec_values(np.array(5))

    def test_slice_empty(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice(None)
            expected = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            self.assertTrue(np.allclose(expected, actual))

    def test_slice_non_existent_dim(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(KeyError):
                _ = pycro_main.slice({'blah': 4, 'X': 3, 'Y': 1})

    def test_slice_incorrect_type(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(TypeError):
                _ = pycro_main.slice({'X': 'fdfd', 'Y': 1})

    def test_slice_negative_index(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = pycro_main.slice({'X': -4, 'Y': 1})

    def test_slice_out_of_bounds(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            with self.assertRaises(ValueError):
                _ = pycro_main.slice({'X': 15, 'Y': 1})

    def test_slice_one_pos_dim_removed(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice(slice_dict={'X': 3})
            n_dim_form = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[3, :, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_slice_one_pos_dim_sliced(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice({'X': slice(1, 5, 2)})
            n_dim_form = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[slice(1, 5, 2), :, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_slice_two_pos_dim_sliced(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice({'X': slice(1, 5, 2), 'Y': 1})
            n_dim_form = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[slice(1, 5, 2), 1, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_slice_two_pos_dim_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice({'X': [1, 2, 4], 'Y': 1})
            n_dim_form = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[[1, 2, 4], 1, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_slice_both_pos_removed(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice({'X': 3, 'Y': 1})
            n_dim_form = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[3, 1, :, :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_slice_pos_and_spec_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice({'X': [1, 2, 4], 'Bias': slice(1, 7, 3)})
            n_dim_form = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[[1, 2, 4], :, slice(1, 7, 3), :]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_slice_all_dims_sliced_list(self):
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            actual, success = pycro_main.slice({'X': [1, 2, 4], 'Y': 2, 'Bias': slice(1, 7, 3), 'Cycle': 1})
            n_dim_form = np.transpose(np.reshape(pycro_main[()], (3, 5, 7, 2)), (1, 0, 2, 3))
            expected = n_dim_form[[1, 2, 4], 2, slice(1, 7, 3), 1]
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(success)

    def test_toggle_sorting(self):
        # Need to change data file so that sorting actually does something
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            for label, size in zip(pycro_main.n_dim_labels, pycro_main.n_dim_sizes):
                print('{}: {}'.format(label, size))
            self.assertTrue(pycro_main.n_dim_labels == ['X', 'Y', 'Bias', 'Cycle'])

            pycro_main.toggle_sorting()

            for label, size in zip(pycro_main.n_dim_labels, pycro_main.n_dim_sizes):
                print('{}: {}'.format(label, size))
            # TODO: Fix this test. Overriden only temporarily
            # self.assertTrue(pycro_main.n_dim_labels==['X', 'Y', 'Cycle', 'Bias'])

    def test_get_current_sorting(self):
        # Need to change data file so that sorting actually does something
        with h5py.File(test_h5_file_path, mode='r') as h5_f:
            pycro_main = PycroDataset(h5_f['/Raw_Measurement/source_main'])
            unsorted_str = 'Data dimensions are in the order they occur in the file.\n'
            sorted_str = 'Data dimensions are sorted in order from fastest changing dimension to slowest.\n'
            # Initial state should be unsorted
            self.assertFalse(pycro_main._PycroDataset__sort_dims)
            with capture_stdout() as get_value:
                pycro_main.get_current_sorting()
                test_str = get_value()
            self.assertTrue(test_str == unsorted_str)
            # Toggle sorting.  Sorting should now be true.
            pycro_main.toggle_sorting()
            self.assertTrue(pycro_main._PycroDataset__sort_dims)
            with capture_stdout() as get_value:
                pycro_main.get_current_sorting()
                test_str = get_value()
            self.assertTrue(test_str == sorted_str)


if __name__ == '__main__':
    unittest.main()
