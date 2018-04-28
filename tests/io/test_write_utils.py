# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import numpy as np

sys.path.append("../../../pycroscopy/")
from pycroscopy.io import write_utils
from pycroscopy.io.virtual_data import VirtualDataset

if sys.version_info.major == 3:
    unicode = str


class TestDeprecatedWriteUtils(unittest.TestCase):

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
        sizes = [num_cols, num_rows]
        dim_names = ['X', 'Y']
        dim_units = ['nm', 'um']

        descriptor = []
        for length, name, units in zip(sizes, dim_names, dim_units):
            descriptor.append(write_utils.Dimension(name, units, np.arange(length)))

        pos_data = np.vstack((np.tile(np.arange(num_cols), num_rows),
                              np.repeat(np.arange(num_rows), num_cols))).T

        ds_inds, ds_vals = write_utils.build_ind_val_dsets(descriptor, is_spectral=False)

        self.__validate_aux_virtual_dset_pair(ds_inds, ds_vals, dim_names, dim_units, pos_data, is_spectral=False)

    def test_build_ind_val_dsets_legal_bare_minimum_spec(self):
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

        descriptor = []
        for length, name, units, step, initial in zip([num_cols, num_rows], dim_names, dim_units,
                                                      [col_step, row_step], [col_initial, row_initial]):
            descriptor.append(write_utils.Dimension(name, units, initial + step * np.arange(length)))

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