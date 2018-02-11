# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

import unittest
import sys
import numpy as np
sys.path.append("../../../pycroscopy/")
from pycroscopy import VirtualDataset
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

    def test_build_ind_val_dsets_legal_bare_minimum(self):
        num_cols = 2
        num_rows = 3
        spec_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        with self.assertWarns(DeprecationWarning):
            ds_inds, ds_vals = write_utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=True)
        exp_inds = VirtualDataset('Spectroscopic_Indices', write_utils.INDICES_DTYPE(spec_data),
                                  attrs={'units': ['Arb Unit 0', 'Arb Unit 1'],
                                       'labels': {'Unknown Dimension 0': (slice(0, 1), slice(None)),
                                                  'Unknown Dimension 1': (slice(1, 2), slice(None))}})
        exp_vals = VirtualDataset('Spectroscopic_Values', write_utils.VALUES_DTYPE(spec_data),
                                  attrs={'units': ['Arb Unit 0', 'Arb Unit 1'],
                                       'labels': {'Unknown Dimension 0': (slice(0, 1), slice(None)),
                                                  'Unknown Dimension 1': (slice(1, 2), slice(None))}})
        self.assertEqual(exp_inds, ds_inds)
        self.assertEqual(exp_vals, ds_vals)

    def test_build_ind_val_dsets_legal_override_steps_offsets(self):
        num_cols = 2
        num_rows = 3
        col_step = 0.25
        row_step = 0.05
        col_initial = 1
        row_initial = 0.2
        spec_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        spec_vals = np.vstack((np.tile(np.arange(num_cols), num_rows) * col_step + col_initial,
                              np.repeat(np.arange(num_rows), num_cols) * row_step + row_initial))
        with self.assertWarns(DeprecationWarning):
            ds_inds, ds_vals = write_utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=True,
                                                               steps=[col_step, row_step],
                                                               initial_values=[col_initial, row_initial])
        exp_inds = VirtualDataset('Spectroscopic_Indices', write_utils.INDICES_DTYPE(spec_inds),
                                  attrs={'units': ['Arb Unit 0', 'Arb Unit 1'],
                                       'labels': {'Unknown Dimension 0': (slice(0, 1), slice(None)),
                                                  'Unknown Dimension 1': (slice(1, 2), slice(None))}})
        exp_vals = VirtualDataset('Spectroscopic_Values', write_utils.VALUES_DTYPE(spec_vals),
                                  attrs={'units': ['Arb Unit 0', 'Arb Unit 1'],
                                       'labels': {'Unknown Dimension 0': (slice(0, 1), slice(None)),
                                                  'Unknown Dimension 1': (slice(1, 2), slice(None))}})
        self.assertEqual(exp_inds, ds_inds)
        self.assertEqual(exp_vals, ds_vals)

    def test_build_ind_val_dsets_legal_all_inputs_spec(self):
        num_cols = 2
        num_rows = 3
        col_step = 0.25
        row_step = 0.05
        col_initial = 1
        row_initial = 0.2
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        spec_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols)))
        spec_vals = np.vstack((np.tile(np.arange(num_cols), num_rows) * col_step + col_initial,
                              np.repeat(np.arange(num_rows), num_cols) * row_step + row_initial))
        ds_inds, ds_vals = write_utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=True,
                                                           steps=[col_step, row_step],
                                                           initial_values=[col_initial, row_initial],
                                                           labels=dim_names, units=dim_units)
        exp_inds = VirtualDataset('Spectroscopic_Indices', write_utils.INDICES_DTYPE(spec_inds),
                                  attrs={'units': dim_units,
                                       'labels': {dim_names[0]: (slice(0, 1), slice(None)),
                                                  dim_names[1]: (slice(1, 2), slice(None))}})
        exp_vals = VirtualDataset('Spectroscopic_Values', write_utils.VALUES_DTYPE(spec_vals),
                                  attrs={'units': dim_units,
                                       'labels': {dim_names[0]: (slice(0, 1), slice(None)),
                                                  dim_names[1]: (slice(1, 2), slice(None))}})
        self.assertEqual(exp_inds, ds_inds)
        self.assertEqual(exp_vals, ds_vals)

    def test_build_ind_val_dsets_legal_all_inputs_pos(self):
        num_cols = 2
        num_rows = 3
        col_step = 0.25
        row_step = -0.05
        col_initial = 1
        row_initial = 0.2
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        pos_inds = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T
        pos_vals = np.vstack((np.tile(np.arange(num_cols), num_rows) * col_step + col_initial,
                              np.repeat(np.arange(num_rows), num_cols) * row_step + row_initial)).T
        ds_inds, ds_vals = write_utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=False,
                                                           steps=[col_step, row_step],
                                                           initial_values=[col_initial, row_initial],
                                                           labels=dim_names, units=dim_units)
        exp_inds = VirtualDataset('Spectroscopic_Indices', write_utils.INDICES_DTYPE(pos_inds),
                                  attrs={'units': dim_units,
                                       'labels': {dim_names[0]: (slice(None), slice(0, 1)),
                                                  dim_names[1]: (slice(None), slice(1, 2))}})
        exp_vals = VirtualDataset('Spectroscopic_Values', write_utils.VALUES_DTYPE(pos_vals),
                                  attrs={'units': dim_units,
                                       'labels': {dim_names[0]: (slice(None), slice(0, 1)),
                                                  dim_names[1]: (slice(None), slice(1, 2))}})
        self.assertEqual(exp_inds, ds_inds)
        self.assertEqual(exp_vals, ds_vals)

    def test_build_ind_val_dsets_illegal_input_sizes(self):
        num_cols = 2
        num_rows = 3
        col_step = 0.25
        row_step = 0.05
        col_initial = 1
        row_initial = 0.2
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']
        with self.assertRaises(ValueError):
            _ = write_utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=True, steps=[col_step],
                                                initial_values=[col_initial, row_initial], labels=dim_names,
                                                units=dim_units)

        with self.assertRaises(ValueError):
            _ = write_utils.build_ind_val_dsets([num_cols], is_spectral=True, steps=[col_step, row_step],
                                                initial_values=[col_initial, row_initial], labels=dim_names,
                                                units=dim_units)

        with self.assertRaises(ValueError):
            _ = write_utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=True, steps=[col_step, row_step],
                                                initial_values=[col_initial, row_initial], labels=[dim_names[0]],
                                                units=dim_units)

        with self.assertRaises(ValueError):
            _ = write_utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=True, steps=[col_step, row_step],
                                                initial_values=[col_initial, row_initial], labels=[dim_names[0]],
                                                units=[dim_units[1]])


if __name__ == '__main__':
    unittest.main()