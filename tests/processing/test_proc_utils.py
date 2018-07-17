"""
Created on Thur July 12

@author: Christopher Smith
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import numpy as np
import sys
sys.path.append("../../../pycroscopy/")
from pycroscopy.processing.proc_utils import get_component_slice, to_ranges

total_comps = 100
int_comps = 10
pair_comps = (27, 93)
array_comps = np.arange(0, total_comps, 3)
slice_comps = slice(5, 97, 3)

class TestProcUtils(unittest.TestCase):

    def setUp(self):

        pass

    def tearDown(self):

        pass

    def test_to_ranges(self):

        pass

    '''
    The following set of tests is for the get_component_slice function
    '''
    # Tests on integer components
    def test_get_component_slice_int_1(self):
        # Component, no total comps
        comp_slice, num_comps = get_component_slice(int_comps)

        self.assertTrue(comp_slice == slice(0, int_comps, None))
        self.assertTrue(num_comps == int_comps)

    def test_get_component_slice_int_2(self):
        # Component with total comps
        comp_slice, num_comps = get_component_slice(int_comps, total_comps)

        self.assertTrue(comp_slice == slice(0, int_comps, None))
        self.assertTrue(num_comps == int_comps)

    def test_get_component_slice_int_3(self):
        # Component outside total comps
        comp_slice, num_comps = get_component_slice(total_comps+5, total_comps)

        self.assertTrue(comp_slice == slice(0, total_comps, None))
        self.assertTrue(num_comps == total_comps)

    def test_get_component_slice_float(self):
        # Float component, should error
        with self.assertRaises(TypeError):
            comp_slice, num_comps = get_component_slice(float(int_comps), total_comps)

    # Test pairs of integer components
    def test_get_component_slice_pair_1(self):
        # Components, no total comps
        comp_slice, num_comps = get_component_slice(pair_comps)

        self.assertTrue(comp_slice == slice(pair_comps[0], pair_comps[1], None))
        self.assertTrue(num_comps == pair_comps[1] - pair_comps[0])

    def test_get_component_slice_pair_2(self):
        # Components, total comps
        comp_slice, num_comps = get_component_slice(pair_comps, total_comps)

        self.assertTrue(comp_slice == slice(pair_comps[0], pair_comps[1], None))
        self.assertTrue(num_comps == pair_comps[1] - pair_comps[0])

    def test_get_component_slice_pair_3(self):
        # Components reversed, no total comps
        comp_slice, num_comps = get_component_slice(pair_comps[::-1])

        self.assertTrue(comp_slice == slice(pair_comps[1], pair_comps[0], None))
        self.assertTrue(num_comps == pair_comps[1] - pair_comps[0])

    def test_get_component_slice_pair_4(self):
        # Components reversed, total comps
        comp_slice, num_comps = get_component_slice(pair_comps[::-1], total_comps)

        self.assertTrue(comp_slice == slice(pair_comps[1], pair_comps[0], None))
        self.assertTrue(num_comps == pair_comps[1] - pair_comps[0])


    # Test longer iterables of components
    def test_get_component_slice_list_1(self):
        # List of integers, no total comps
        comp_slice, num_comps = get_component_slice(array_comps.tolist())

        self.assertTrue(comp_slice == array_comps.tolist())
        self.assertTrue(num_comps == array_comps.size)

    def test_get_component_slice_list_2(self):
        # List of integers, total comps
        comp_slice, num_comps = get_component_slice(array_comps.tolist(), total_comps)

        self.assertTrue(comp_slice == array_comps.tolist())
        self.assertTrue(num_comps == array_comps.size)

    def test_get_component_slice_tuple_1(self):
        # Tuple of integers, no total comps
        comp_slice, num_comps = get_component_slice(tuple(array_comps))

        self.assertTrue(comp_slice == array_comps.tolist())
        self.assertTrue(num_comps == array_comps.size)

    def test_get_component_slice_tuple_2(self):
        # Tuple of integers, total comps
        comp_slice, num_comps = get_component_slice(tuple(array_comps), total_comps)

        self.assertTrue(comp_slice == array_comps.tolist())
        self.assertTrue(num_comps == array_comps.size)

    def test_get_component_slice_array_1(self):
        # Array of integers, no total comps
        comp_slice, num_comps = get_component_slice(array_comps)

        self.assertTrue(comp_slice == array_comps.tolist())
        self.assertTrue(num_comps == array_comps.size)

    def test_get_component_slice_array_2(self):
        # Array of integers, total comps
        comp_slice, num_comps = get_component_slice(array_comps, total_comps)

        self.assertTrue(comp_slice == array_comps.tolist())
        self.assertTrue(num_comps == array_comps.size)

    # Test component slices
    def test_get_component_slice_slice_1(self):
        # Slice, no total comps
        comp_slice, num_comps = get_component_slice(slice_comps)

        self.assertTrue(comp_slice == slice_comps)
        self.assertTrue(num_comps == np.arange(slice_comps.stop+1)[slice_comps].size)

