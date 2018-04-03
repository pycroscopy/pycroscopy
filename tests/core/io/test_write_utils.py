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
from pycroscopy import VirtualDataset

test_h5_file_path = 'test_hdf_utils.h5'

if sys.version_info.major == 3:
    unicode = str


class TestWriteUtils(unittest.TestCase):

    def test_make_indices_dim_w_val_1(self):
        with self.assertRaises(AssertionError):
            _ = write_utils.make_indices_matrix([1, 2, 3])

    def test_make_indices_non_int_dim_sizes(self):
        with self.assertRaises(ValueError):
            _ = write_utils.make_indices_matrix([1.233, 2.4, 3])

    def test_make_indices_not_list(self):
        with self.assertRaises(TypeError):
            _ = write_utils.make_indices_matrix(1)

    def test_make_indices_weird_inputs(self):
        with self.assertRaises(ValueError):
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
        with self.assertRaises(TypeError):
            _ = write_utils.get_aux_dset_slicing([1, 'Y'], is_spectroscopic=True)
        with self.assertRaises(ValueError):
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

        with self.assertRaises(TypeError):
            # Swapped names (strs) with sizes (uints)
            _ = write_utils.AuxillaryDescriptor(dim_names, sizes, dim_units,
                                                dim_step_sizes=steps, dim_initial_vals=inits)

        with self.assertRaises(TypeError):
            # Swapped names (strs) with sizes (uints)
            _ = write_utils.AuxillaryDescriptor(dim_names, sizes, steps,
                                                dim_step_sizes=dim_units, dim_initial_vals=dim_names)

    def __validate_aux_virtual_dset_pair(self, ds_inds, ds_vals, dim_names, dim_units, inds_matrix,
                                         vals_matrix=None, base_name=None, is_spectral=True):
        if vals_matrix is None:
            vals_matrix = inds_matrix
        if base_name is None:
            if is_spectral:
                base_name = 'Spectroscopic'
            else:
                base_name = 'Position'
        else:
            self.assertIsInstance(base_name, (str, unicode))

        for vr_dset, exp_dtype, exp_name, ref_data in zip([ds_inds, ds_vals],
                                                          [write_utils.INDICES_DTYPE, write_utils.VALUES_DTYPE],
                                                          [base_name + '_Indices', base_name + '_Values'],
                                                          [inds_matrix, vals_matrix]):

            self.assertIsInstance(vr_dset, VirtualDataset)
            self.assertEqual(vr_dset.name, exp_name)
            self.assertTrue(np.allclose(ref_data, vr_dset.data))
            self.assertEqual(vr_dset.dtype, exp_dtype)
            self.assertTrue(np.all([_ in vr_dset.attrs.keys() for _ in ['labels', 'units']]))
            self.assertTrue(np.all([x == y for x, y in zip(dim_units, vr_dset.attrs['units'])]))

            # assert region references
            self.assertTrue(isinstance(vr_dset.attrs['labels'], dict))
            for dim_ind, curr_name in enumerate(dim_names):
                if is_spectral:
                    expected = (slice(dim_ind, dim_ind + 1, None), slice(None))
                else:
                    expected = (slice(None), slice(dim_ind, dim_ind + 1, None))
                actual = vr_dset.attrs['labels'][curr_name]
                self.assertEqual(expected, actual)

    def test_build_ind_val_dsets_legal_bare_minimum_pos(self):
        num_cols = 3
        num_rows = 2
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = write_utils.AuxillaryDescriptor([num_cols, num_rows], dim_names, dim_units)

        pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T

        ds_inds, ds_vals = write_utils.build_ind_val_dsets(descriptor, is_spectral=False)

        self.__validate_aux_virtual_dset_pair(ds_inds, ds_vals, dim_names, dim_units, pos_data, is_spectral=False)

    def test_build_ind_val_dsets_legal_bare_minimum_spec(self):
        num_cols = 3
        num_rows = 2
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = write_utils.AuxillaryDescriptor([num_cols, num_rows], dim_names, dim_units)

        spec_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                               np.repeat(np.arange(num_rows), num_cols)))

        ds_inds, ds_vals = write_utils.build_ind_val_dsets(descriptor, is_spectral=True)

        self.__validate_aux_virtual_dset_pair(ds_inds, ds_vals, dim_names, dim_units, spec_data, is_spectral=True)

    def test_build_ind_val_dsets_legal_override_steps_offsets_base_name(self):
        num_cols = 2
        num_rows = 3
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        col_step = 0.25
        row_step = 0.05
        col_initial = 1
        row_initial = 0.2

        descriptor = write_utils.AuxillaryDescriptor([num_cols, num_rows], dim_names, dim_units,
                                                     dim_step_sizes=[col_step, row_step],
                                                     dim_initial_vals=[col_initial, row_initial])

        new_base_name = 'Overriden'
        spec_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                               np.repeat(np.arange(num_rows), num_cols)))
        spec_vals = np.vstack((np.tile(np.arange(num_cols), num_rows) * col_step + col_initial,
                               np.repeat(np.arange(num_rows), num_cols) * row_step + row_initial))

        ds_inds, ds_vals = write_utils.build_ind_val_dsets(descriptor, is_spectral=True, base_name=new_base_name)
        self.__validate_aux_virtual_dset_pair(ds_inds, ds_vals, dim_names, dim_units, spec_inds,
                                              vals_matrix=spec_vals, base_name=new_base_name, is_spectral=True)


if __name__ == '__main__':
    unittest.main()