# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

import unittest
import sys
import numpy as np

sys.path.append("../../../pycroscopy/")
from pycroscopy.core.io import write_utils

test_h5_file_path = 'test_hdf_utils.h5'


class TestWriteUtils(unittest.TestCase):

    def test_make_indices_dim_w_val_1(self):
        with self.assertRaises(AssertionError):
            _ = write_utils.make_indices_matrix([1, 2, 3])

    def test_make_indices_non_int_dim_sizes(self):
        with self.assertRaises(AssertionError):
            _ = write_utils.make_indices_matrix([1.233, 2.4, 3])

    def test_make_indices_not_list(self):
        with self.assertRaises(AssertionError):
            _ = write_utils.make_indices_matrix(1)

    def test_make_indices_weird_inputs(self):
        with self.assertRaises(AssertionError):
            _ = write_utils.make_indices_matrix([2, 'hello', 3])

    def test_make_indices_matrix_1_dims(self):
        expected = np.expand_dims(np.arange(4), axis=0)
        ret_val = write_utils.make_indices_matrix([4], is_position=False)
        self.assertTrue(np.allclose(expected, ret_val))
        ret_val = write_utils.make_indices_matrix([4], is_position=True)
        self.assertTrue(np.allclose(expected.T, ret_val))

    def test_make_indices_matrix_2_dims(self):
        expected = np.vstack((np.tile(np.arange(2), 3),
                              np.repeat(np.arange(3), 2)))
        ret_val = write_utils.make_indices_matrix([2, 3], is_position=False)
        self.assertTrue(np.allclose(expected, ret_val))
        ret_val = write_utils.make_indices_matrix([2, 3], is_position=True)
        self.assertTrue(np.allclose(expected.T, ret_val))

    def test_make_indices_matrix_3_dims(self):
        expected = np.vstack((np.tile(np.arange(2), 3 * 4),
                              np.tile(np.repeat(np.arange(3), 2), 4),
                              np.repeat(np.arange(4), 6)))
        ret_val = write_utils.make_indices_matrix([2, 3, 4], is_position=False)
        self.assertTrue(np.allclose(expected, ret_val))
        ret_val = write_utils.make_indices_matrix([2, 3, 4], is_position=True)
        self.assertTrue(np.allclose(expected.T, ret_val))

    def test_get_aux_dset_slicing_legal_single_dim(self):
        ret_val = write_utils.get_aux_dset_slicing(['X'], is_spectroscopic=True)
        expected = {'X': (slice(0, 1), slice(None))}
        self.assertEqual(ret_val, expected)

        ret_val = write_utils.get_aux_dset_slicing(['X'], is_spectroscopic=False)
        expected = {'X': (slice(None), slice(0, 1))}
        self.assertEqual(ret_val, expected)

    def test_get_aux_dset_slicing_legal_multi_dim(self):
        ret_val = write_utils.get_aux_dset_slicing(['X', 'Y'], is_spectroscopic=True)
        expected = {'X': (slice(0, 1), slice(None)), 'Y': (slice(1, 2), slice(None))}
        self.assertEqual(ret_val, expected)

        ret_val = write_utils.get_aux_dset_slicing(['X', 'Y'], is_spectroscopic=False)
        expected = {'X': (slice(None), slice(0, 1)), 'Y': (slice(None), slice(1, 2))}
        self.assertEqual(ret_val, expected)

    def test_get_aux_dset_slicing_odd_input(self):
        with self.assertRaises(AssertionError):
            _ = write_utils.get_aux_dset_slicing([1, 'Y'], is_spectroscopic=True)
        with self.assertRaises(AssertionError):
            _ = write_utils.get_aux_dset_slicing([], is_spectroscopic=True)
            
    def test_clean_string_att_float(self):
        expected = 5.321
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_clean_string_att_str(self):
        expected = 'test'
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_clean_string_att_num_array(self):
        expected = [1, 2, 3.456]
        self.assertEqual(expected, write_utils.clean_string_att(expected))

    def test_clean_string_att_str_list(self):
        expected = ['a', 'bc', 'def']
        returned = write_utils.clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)

    def test_clean_string_att_str_tuple(self):
        expected = ('a', 'bc', 'def')
        returned = write_utils.clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)

    def test_aux_dset_descriptor_minimum(self):
        sizes = [3, 2]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        steps = [1, 1]
        inits = [0, 0]

        descriptor = write_utils.AuxillaryDescriptor(sizes, dim_names, dim_units)
        for expected, actual in zip([sizes, dim_names, dim_units, steps, inits],
                                    [descriptor.sizes, descriptor.names, descriptor.units,
                                     descriptor.steps, descriptor.initial_vals]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_aux_dset_descriptor_full_legal(self):
        sizes = [3, 2]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        steps = [0.25, -0.1]
        inits = [1500, 1E-6]

        descriptor = write_utils.AuxillaryDescriptor(sizes, dim_names, dim_units, dim_step_sizes=steps,
                                                     dim_initial_vals=inits)
        for expected, actual in zip([sizes, dim_names, dim_units, steps, inits],
                                    [descriptor.sizes, descriptor.names, descriptor.units,
                                     descriptor.steps, descriptor.initial_vals]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_aux_dset_descriptor_illegal(self):
        sizes = [3, 2]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        steps = [0.25, -0.1]
        inits = [1500, 1E-6]
        
        with self.assertRaises(ValueError):
            # too few steps
            _ = write_utils.AuxillaryDescriptor(sizes, dim_names, dim_units, dim_step_sizes=[5], 
                                                dim_initial_vals=inits)

        with self.assertRaises(ValueError):
            # too few dimension sizes
            _ = write_utils.AuxillaryDescriptor([5], dim_names, dim_units,
                                                dim_step_sizes=steps, dim_initial_vals=inits)

        with self.assertRaises(ValueError):
            # too few names
            _ = write_utils.AuxillaryDescriptor(sizes, [dim_names[0]], dim_units,
                                                dim_step_sizes=steps, dim_initial_vals=inits)

        with self.assertRaises(ValueError):
            # too few names and units
            _ = write_utils.AuxillaryDescriptor(sizes, [dim_names[0]], [dim_units[1]],
                                                dim_step_sizes=steps, dim_initial_vals=inits)

        with self.assertRaises(AssertionError):
            # Swapped names (strs) with sizes (uints)
            _ = write_utils.AuxillaryDescriptor(dim_names, sizes, dim_units,
                                                dim_step_sizes=steps, dim_initial_vals=inits)

        with self.assertRaises(AssertionError):
            # Swapped names (strs) with sizes (uints)
            _ = write_utils.AuxillaryDescriptor(dim_names, sizes, steps,
                                                dim_step_sizes=dim_units, dim_initial_vals=dim_names)


if __name__ == '__main__':
    unittest.main()