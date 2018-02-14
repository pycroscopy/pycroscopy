# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

import unittest
import sys
from multiprocessing import cpu_count
sys.path.append("../../../pycroscopy/")
from pycroscopy.core.io import io_utils
import numpy as np

MAX_CPU_CORES = cpu_count()


class TestIOUtils(unittest.TestCase):

    def test_recommend_cores_many_small_jobs(self):
        num_jobs = 14035
        ret_val = io_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
        self.assertEqual(ret_val, MAX_CPU_CORES-2)
        ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=1, lengthy_computation=False)
        self.assertEqual(ret_val, 1)
        ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES, lengthy_computation=False)
        self.assertEqual(ret_val, MAX_CPU_CORES)
        ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=5000, lengthy_computation=False)
        self.assertEqual(ret_val, MAX_CPU_CORES)

    def test_reccomend_cores_few_small_jobs(self):
        num_jobs = 13
        ret_val = io_utils.recommend_cpu_cores(num_jobs, lengthy_computation=False)
        self.assertEqual(ret_val, 1)
        ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES, lengthy_computation=False)
        self.assertEqual(ret_val, 1)

    def test_recommed_cores_few_large_jobs(self):
        num_jobs = 13
        ret_val = io_utils.recommend_cpu_cores(num_jobs, lengthy_computation=True)
        self.assertEqual(ret_val, MAX_CPU_CORES-2)
        ret_val = io_utils.recommend_cpu_cores(num_jobs, requested_cores=MAX_CPU_CORES - 1, lengthy_computation=True)
        self.assertEqual(ret_val, MAX_CPU_CORES - 1)

    def test_formatted_str_to_number(self):
        self.assertEqual(io_utils.formatted_str_to_number("4.32 MHz", ["MHz", "kHz"], [1E+6, 1E+3]), 4.32E+6)

    def test_formatted_str_to_number_invalid(self):
        with self.assertRaises(AssertionError):
            _ = io_utils.formatted_str_to_number("4.32 MHz", ["MHz"], [1E+6, 1E+3])
        with self.assertRaises(AssertionError):
            _ = io_utils.formatted_str_to_number("4.32 MHz", ["MHz", "kHz"], [1E+3])
        with self.assertRaises(AssertionError):
            _ = io_utils.formatted_str_to_number("4.32-MHz", ["MHz", "kHz"], [1E+6, 1E+3])
        with self.assertRaises(ValueError):
            _ = io_utils.formatted_str_to_number("haha MHz", ["MHz", "kHz"], [1E+6, 1E+3])
        with self.assertRaises(ValueError):
            _ = io_utils.formatted_str_to_number("1.2.3.4 MHz", ["MHz", "kHz"], [1E+6, 1E+3])
        with self.assertRaises(AssertionError):
            _ = io_utils.formatted_str_to_number("MHz", ["MHz", "kHz"], [1E+6, 1E+3])

    def test_format_quantity(self):
        qty_names = ['sec', 'mins', 'hours', 'days']
        qty_factors = [1, 60, 3600, 3600*24]
        ret_val = io_utils.format_quantity(315, qty_names, qty_factors)
        self.assertEqual(ret_val, '5.25 mins')
        ret_val = io_utils.format_quantity(6300, qty_names, qty_factors)
        self.assertEqual(ret_val, '1.75 hours')

    def test_format_quantity_illegal(self):
        with self.assertRaises(AssertionError):
            _ = io_utils.format_quantity(315, ['sec', 'mins', 'hours'], [1, 60, 3600, 3600*24])
        with self.assertRaises(AssertionError):
            _ = io_utils.format_quantity(315, ['sec', 'mins', 'hours'], [1, 60])
        with self.assertRaises(AssertionError):
            _ = io_utils.format_quantity(315, ['sec', 14, 'hours'], [1, 60, 3600*24])
        with self.assertRaises(TypeError):
            _ = io_utils.format_quantity('hello', ['sec', 'mins', 'hours'], [1, 60, 3600])

    def test_format_time(self):
        ret_val = io_utils.format_time(315)
        self.assertEqual(ret_val, '5.25 mins')
        ret_val = io_utils.format_time(6300)
        self.assertEqual(ret_val, '1.75 hours')

    def test_format_size(self):
        ret_val = io_utils.format_size(15.23)
        self.assertEqual(ret_val, '15.23 bytes')
        ret_val = io_utils.format_size(5830418104.32)
        self.assertEqual(ret_val, '5.43 GB')


if __name__ == '__main__':
    unittest.main()