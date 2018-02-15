# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

import unittest
import sys
import numpy as np
sys.path.append("../../../pycroscopy/")
from pycroscopy.core.io import dtype_utils

struc_dtype = np.dtype({'names': ['r', 'g', 'b'],
                        'formats': [np.float32, np.uint16, np.float64]})


def compare_structured_arrays(arr_1, arr_2):
    """
    if not isinstance(arr_1, np.ndarray):
        raise TypeError("arr_1 was not a numpy array")
    if not isinstance(arr_2, np.ndarray):
        raise TypeError("arr_2 was not a numpy array")
    """
    if arr_1.dtype != arr_2.dtype:
        return False
    if arr_1.shape != arr_2.shape:
        return False
    tests = []
    for name in arr_1.dtype.names:
        tests.append(np.allclose(arr_1[name], arr_2[name]))
    return np.all(tests)


class TestDtypeUtils(unittest.TestCase):

    def test_contains_integers(self):
        self.assertTrue(dtype_utils.contains_integers([1, 2, -3, 4]))
        self.assertTrue(dtype_utils.contains_integers(range(5)))
        self.assertTrue(dtype_utils.contains_integers([2, 5, 8, 3], min_val=2))
        self.assertTrue(dtype_utils.contains_integers(np.arange(5)))
        self.assertFalse(dtype_utils.contains_integers(np.arange(5), min_val=2))
        self.assertFalse(dtype_utils.contains_integers([1, 4.5, 2.2, -1]))
        self.assertFalse(dtype_utils.contains_integers([1, -2, 5], min_val=1))
        self.assertFalse(dtype_utils.contains_integers(['dsss', 34, 1.23, None]))
        self.assertFalse(dtype_utils.contains_integers([]))
        
        with self.assertRaises(AssertionError):
            _ = dtype_utils.contains_integers(None)
        with self.assertRaises(AssertionError):
            _ = dtype_utils.contains_integers(14)

    def test_stack_real_to_complex_single(self):
        expected = 4.32 + 5.67j
        real_val = [np.real(expected), np.imag(expected)]
        actual = dtype_utils.stack_real_to_complex(real_val)
        self.assertTrue(np.allclose(actual, expected))

    def test_complex_to_real_1d_array(self):
        complex_array = 5 * np.random.rand(5) + 7j * np.random.rand(5)
        actual = dtype_utils.flatten_complex_to_real(complex_array)
        expected = np.hstack([np.real(complex_array), np.imag(complex_array)])
        self.assertTrue(np.allclose(actual, expected))

    def test_stack_real_1d_to_complex_array(self):
        expected = 5 * np.random.rand(6) + 7j * np.random.rand(6)
        real_val = np.hstack([np.real(expected), np.imag(expected)])
        actual = dtype_utils.stack_real_to_complex(real_val)
        self.assertTrue(np.allclose(actual, expected))

    def test_complex_to_real_2d_array(self):
        complex_array = 5 * np.random.rand(2, 3) + 7j * np.random.rand(2, 3)
        actual = dtype_utils.flatten_complex_to_real(complex_array)
        expected = np.hstack([np.real(complex_array), np.imag(complex_array)])
        self.assertTrue(np.allclose(actual, expected))

    def test_stack_real_2d_to_complex_array(self):
        expected = 5 * np.random.rand(2, 8) + 7j * np.random.rand(2, 8)
        real_val = np.hstack([np.real(expected), np.imag(expected)])
        actual = dtype_utils.stack_real_to_complex(real_val)
        self.assertTrue(np.allclose(actual, expected))

    def test_complex_to_real_nd_array(self):
        complex_array = 5 * np.random.rand(2, 3, 5, 7) + 7j * np.random.rand(2, 3, 5, 7)
        actual = dtype_utils.flatten_complex_to_real(complex_array)
        expected = np.concatenate([np.real(complex_array), np.imag(complex_array)], axis=3)
        self.assertTrue(np.allclose(actual, expected))

    def test_stack_real_nd_to_complex_array(self):
        expected = 5 * np.random.rand(2, 3, 5, 8) + 7j * np.random.rand(2, 3, 5, 8)
        real_val = np.concatenate([np.real(expected), np.imag(expected)], axis=3)
        actual = dtype_utils.stack_real_to_complex(real_val)
        self.assertTrue(np.allclose(actual, expected))

    def test_complex_to_real_compound_illegal(self):
        num_elems = 5
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = np.random.random(size=num_elems)
        structured_array['g'] = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = np.random.random(size=num_elems)
        with self.assertRaises(TypeError):
            _ = dtype_utils.flatten_complex_to_real(structured_array)

    def test_stack_real_to_complex_illegal_odd_last_dim(self):
        expected = 5 * np.random.rand(2, 3, 5, 7) + 7j * np.random.rand(2, 3, 5, 7)
        real_val = np.concatenate([np.real(expected), np.imag(expected)[..., :-1]], axis=3)
        with self.assertRaises(AssertionError):
            _ = dtype_utils.stack_real_to_complex(real_val)

    def test_get_compund_sub_dtypes_legal(self):
        self.assertEqual({'r': np.float32, 'g': np.uint16, 'b': np.float64},
                         dtype_utils.get_compound_sub_dtypes(struc_dtype))

    def test_get_compound_sub_dtypes_illegal(self):
        with self.assertRaises(TypeError):
            _ = dtype_utils.get_compound_sub_dtypes(np.float16)

        with self.assertRaises(TypeError):
            _ = dtype_utils.get_compound_sub_dtypes(16)

        with self.assertRaises(TypeError):
            _ = dtype_utils.get_compound_sub_dtypes(np.arange(4))

    def test_compound_to_real_single(self):
        num_elems = 1
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        expected = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.flatten_compound_to_real(structured_array[0])
        self.assertTrue(np.allclose(actual, expected))

    def test_real_to_compound_single(self):
        num_elems = 1
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.stack_real_to_compound(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array[0]))

    def test_compound_to_real_1d(self):
        num_elems = 5
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        expected = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.flatten_compound_to_real(structured_array)
        self.assertTrue(np.allclose(actual, expected))

    def test_real_to_compound_1d(self):
        num_elems = 5
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals))
        actual = dtype_utils.stack_real_to_compound(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array))

    def test_compound_to_real_nd(self):
        num_elems = (5, 7, 2, 3)
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        expected = np.concatenate((r_vals, g_vals, b_vals), axis=len(num_elems) - 1)
        actual = dtype_utils.flatten_compound_to_real(structured_array)
        self.assertTrue(np.allclose(actual, expected))

    def test_real_to_compound_nd(self):
        num_elems = (2, 3, 5, 7)
        structured_array = np.zeros(shape=num_elems, dtype=struc_dtype)
        structured_array['r'] = r_vals = np.random.random(size=num_elems)
        structured_array['g'] = g_vals = np.random.randint(0, high=1024, size=num_elems)
        structured_array['b'] = b_vals = np.random.random(size=num_elems)
        real_val = np.concatenate((r_vals, g_vals, b_vals), axis=len(num_elems) - 1)
        actual = dtype_utils.stack_real_to_compound(real_val, struc_dtype)
        self.assertTrue(compare_structured_arrays(actual, structured_array))

    def test_compound_to_real_illegal(self):
        num_elems = (2, 3)
        r_vals = np.random.random(size=num_elems)
        with self.assertRaises(TypeError):
            _ = dtype_utils.flatten_compound_to_real(r_vals)

        with self.assertRaises(TypeError):
            _ = dtype_utils.flatten_compound_to_real(14)

    def test_real_to_compound_illegal(self):
        num_elems = (3, 5)
        r_vals = np.random.random(size=num_elems)
        with self.assertRaises(TypeError):
            _ = dtype_utils.stack_real_to_compound(r_vals, np.float32)
        with self.assertRaises(ValueError):
            _ = dtype_utils.stack_real_to_compound(r_vals, struc_dtype)


if __name__ == '__main__':
    unittest.main()
