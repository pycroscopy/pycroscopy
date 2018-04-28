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
from pycroscopy.core.io import write_utils

if sys.version_info.major == 3:
    unicode = str


class TestWriteUtils(unittest.TestCase):

    def test_make_indices_dim_w_val_1(self):
        with self.assertRaises(ValueError):
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

    def test_dimension_array(self):
        name = 'Bias'
        units = 'V'
        values = np.random.rand(5)

        descriptor = write_utils.Dimension(name, units, values)
        for expected, actual in zip([name, units, values],
                                    [descriptor.name, descriptor.units, descriptor.values]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))

    def test_dimension_length(self):
        name = 'Bias'
        units = 'V'
        values = np.arange(5)

        descriptor = write_utils.Dimension(name, units, len(values))
        for expected, actual in zip([name, units],
                                    [descriptor.name, descriptor.units]):
            self.assertTrue(np.all([x == y for x, y in zip(expected, actual)]))
        self.assertTrue(np.allclose(values, descriptor.values))

    def test_aux_dset_descriptor_illegal(self):

        with self.assertRaises(TypeError):
            _ = write_utils.Dimension('Name', 14, np.arange(4))

        with self.assertRaises(TypeError):
            _ = write_utils.Dimension(14, 'nm', np.arange(4))

        with self.assertRaises(ValueError):
            _ = write_utils.Dimension('Name', 'unit', 0)

        with self.assertRaises(TypeError):
            _ = write_utils.Dimension('Name', 'unit', 'invalid')

    def test_build_ind_val_matrices_empty(self):
        inds, vals = write_utils.build_ind_val_matrices([[0]], is_spectral=True)
        self.assertTrue(np.allclose(inds, write_utils.INDICES_DTYPE(np.expand_dims(np.arange(1), 0))))
        self.assertTrue(np.allclose(vals, write_utils.VALUES_DTYPE(np.expand_dims(np.arange(1), 0))))

    def test_build_ind_val_matrices_1D(self):
        sine_val = np.sin(np.linspace(0, 2*np.pi, 128))
        inds, vals = write_utils.build_ind_val_matrices([sine_val], is_spectral=True)
        self.assertTrue(np.allclose(inds, write_utils.INDICES_DTYPE(np.expand_dims(np.arange(len(sine_val)), axis=0))))
        self.assertTrue(np.allclose(vals, write_utils.VALUES_DTYPE(np.expand_dims(sine_val, axis=0))))

    def test_build_ind_val_matrices_1D_pos(self):
        sine_val = np.sin(np.linspace(0, 2 * np.pi, 128))
        inds, vals = write_utils.build_ind_val_matrices([sine_val], is_spectral=False)
        self.assertTrue(np.allclose(inds, write_utils.INDICES_DTYPE(np.expand_dims(np.arange(len(sine_val)), axis=1))))
        self.assertTrue(np.allclose(vals, write_utils.VALUES_DTYPE(np.expand_dims(sine_val, axis=1))))

    def test_build_ind_val_matrices_3D(self):
        max_v = 4
        half_pts = 8
        bi_triang = np.roll(np.hstack((np.linspace(-max_v, max_v, half_pts, endpoint=False),
                                       np.linspace(max_v, -max_v, half_pts, endpoint=False))), -half_pts // 2)
        cycles = [0, 1, 2]
        fields = [0, 1]
        exp_vals = np.vstack((np.tile(bi_triang, 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))
        exp_inds = np.vstack((np.tile(np.arange(2 * half_pts), 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))
        inds, vals = write_utils.build_ind_val_matrices([bi_triang, fields, cycles])
        self.assertTrue(np.allclose(exp_inds, inds))
        self.assertTrue(np.allclose(exp_vals, vals))

    def test_create_spec_inds_from_vals(self):
        max_v = 4
        half_pts = 8
        bi_triang = np.roll(np.hstack((np.linspace(-max_v, max_v, half_pts, endpoint=False),
                                       np.linspace(max_v, -max_v, half_pts, endpoint=False))), -half_pts // 2)
        cycles = [0, 1, 2]
        fields = [0, 1]
        exp_vals = np.vstack((np.tile(bi_triang, 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))
        exp_inds = np.vstack((np.tile(np.arange(2 * half_pts), 6), np.tile(np.repeat(fields, 2 * half_pts), 3),
                              np.repeat(cycles, 2 * 2 * half_pts)))

        inds = write_utils.create_spec_inds_from_vals(exp_vals)
        self.assertTrue(np.allclose(inds, exp_inds))

    def test_calc_chunks_no_unit_chunk(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = None
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)
        self.assertTrue(np.allclose(ret_val, (26, 100)))

    def test_calc_chunks_unit_chunk(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (3, 7)
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)
        self.assertTrue(np.allclose(ret_val, (27, 98)))

    def test_calc_chunks_no_unit_chunk_max_mem(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = None
        max_mem = 50000
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks, max_chunk_mem=max_mem)
        self.assertTrue(np.allclose(ret_val, (56, 224)))

    def test_calc_chunks_unit_chunk_max_mem(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (3, 7)
        max_mem = 50000
        ret_val = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks, max_chunk_mem=max_mem)
        self.assertTrue(np.allclose(ret_val, (57, 224)))

    def test_calc_chunks_unit_not_iterable(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = 4

        with self.assertRaises(TypeError):
            _ = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)

    def test_calc_chunks_shape_mismatch(self):
        dimensions = (16384, 16384 * 4)
        dtype_bytesize = 4
        unit_chunks = (1, 5, 9)

        with self.assertRaises(ValueError):
            _ = write_utils.calc_chunks(dimensions, dtype_bytesize, unit_chunks=unit_chunks)


if __name__ == '__main__':
    unittest.main()