# -*- coding: utf-8 -*-
"""
Unit Test created by Christopher Smith on 7/19/18
"""
import sys
import os
import unittest
import numpy as np
import h5py
from pyUSID import USIDataset, Dimension
from pyUSID.io.hdf_utils import write_simple_attrs, write_main_dataset
sys.path.append("../../../pycroscopy/")
from pycroscopy.analysis.fitter import Fitter

test_h5_file_path = 'temp_base_fitter.h5'


class TestBaseFitterClass(unittest.TestCase):
    """
    Test class for the base Fitter class
    """
    def setUp(self):
        self.h5_f = h5py.File(test_h5_file_path)
        h5_raw_grp = self.h5_f.create_group('Raw_Measurement')

        num_rows = 3
        num_cols = 5
        num_cycles = 2
        num_cycle_pts = 7

        # Create Main dataset and ancillaries
        source_dset_name = 'source_main'

        pos_dims = [Dimension('X', 'nm', num_rows),
                    Dimension('Y', 'nm', num_cols)]

        spec_dims = [Dimension('Bias', 'V', num_cycle_pts),
                     Dimension('Cycle', 'a.u.', num_cycles)]

        source_main_data = np.random.rand(num_rows * num_cols, num_cycle_pts * num_cycles)

        h5_source_main = write_main_dataset(h5_raw_grp, source_main_data, source_dset_name,
                                            'Current', 'A',
                                            pos_dims, spec_dims)

        # Create Guess dataset and ancillaries
        h5_guess_grp = h5_raw_grp.create_group(source_dset_name+'-Fitter_000')

        guess_data = np.random.rand(num_rows * num_cols, num_cycles)

        guess_spec_dims = spec_dims[1]

        self.h5_guess = write_main_dataset(h5_guess_grp, guess_data, 'Guess',
                                           'Guess', 'a.u.',
                                           pos_dims, guess_spec_dims)

        self.fitter = Fitter(h5_source_main, variables=['Bias'])
        self.h5_main = h5_source_main

        self.h5_f.flush()

    def tearDown(self):
        if os.path.exists(test_h5_file_path):
            os.remove(test_h5_file_path)

    def test_setup(self):
        """
        Test that everything was set-up properly during init
        """
        self.assertEqual(self.fitter.h5_main, self.h5_main)
        self.assertTrue(self.fitter._parallel)
        self.assertFalse(self.fitter._verbose)
        self.assertEqual(self.fitter._start_pos, 0)
        self.assertEqual(self.fitter._end_pos, self.h5_main.shape[0])

        self.assertIsNotNone(self.fitter._maxCpus)
        self.assertIsNotNone(self.fitter._maxMemoryMB)
        self.assertIsNotNone(self.fitter._maxDataChunk)
        self.assertIsNotNone(self.fitter._max_pos_per_read)

    def test_get_data_chunk(self):
        # Test initial call for first position
        self.fitter._start_pos = 0

        end_pos = int(min(self.h5_main.shape[0], self.fitter._start_pos + self.fitter._max_pos_per_read))
        data_chunk = self.h5_main[0:end_pos, :]

        self.fitter._get_data_chunk()

        self.assertEqual(self.fitter._end_pos, end_pos)
        self.assertTrue(np.array_equal(self.fitter.data, data_chunk))

        # Test all final call after all positions fitted

        self.fitter._start_pos = self.h5_main.shape[0]
        self.fitter._get_data_chunk()

        self.assertIsNone(self.fitter.data)

    def test_get_guess_chunk(self):
        # Test initial call for first position
        self.fitter._start_pos = 0
        end_pos = int(min(self.h5_main.shape[0], self.fitter._start_pos + self.fitter._max_pos_per_read))

        guess_chunk = self.h5_guess[0:end_pos, :]

        self.fitter.h5_guess = self.h5_guess
        self.fitter._get_guess_chunk()

        self.assertEqual(self.fitter._end_pos, end_pos)
        self.assertTrue(np.array_equal(self.fitter.guess, guess_chunk))

        # Test all final call after all positions fitted

        self.fitter._start_pos = self.h5_main.shape[0]
        self.fitter._get_guess_chunk()

        self.assertTrue(np.array_equal(np.array([], dtype=np.float64).reshape([0, 2]),
                                       self.fitter.guess))

    def test_check_for_old_guess_no_last_pixel(self):
        self.fitter._fitter_name = 'Fitter'
        self.fitter._parms_dict = dict()

        # Initial check, guess has no last_pixel attribute
        partial, completed = self.fitter._check_for_old_guess()

        self.assertEqual(partial, [])
        self.assertEqual(completed, [])

    def test_check_for_old_guess_incomplete(self):
        self.fitter._fitter_name = 'Fitter'
        # Set last_pixel to less than number of positions
        write_simple_attrs(self.h5_guess, {'last_pixel': np.random.randint(self.h5_guess.shape[0]-1)})

        partial, completed = self.fitter._check_for_old_guess()

        self.assertEqual(USIDataset(partial[0]), self.h5_guess)
        self.assertEqual(completed, [])

    def test_check_for_old_guess_complete(self):
        self.fitter._fitter_name = 'Fitter'
        # Set last_pixel to number of positions
        write_simple_attrs(self.h5_guess, {'last_pixel': self.h5_guess.shape[0]})

        partial, completed = self.fitter._check_for_old_guess()

        self.assertEqual(partial, [])
        self.assertEqual(USIDataset(completed[0]), self.h5_guess)

    def test_do_guess_bad_strategy(self):
        self.fitter._fitter_name = 'Fitter'
        bad_strategy = 'do_a_fit'

        with self.assertRaises(KeyError):
            self.fitter.do_guess(strategy=bad_strategy)

    def test_do_guess_bad_previous(self):
        self.fitter._fitter_name = 'Fitter'
        strategy = 'absolute_maximum'
        bad_previous = self.h5_guess

        with self.assertRaises(NotImplementedError):
            self.fitter.do_guess(strategy=strategy, h5_partial_guess=bad_previous)

    def test_do_guess_previous_not_dset(self):
        self.fitter._fitter_name = 'Fitter'
        strategy = 'absolute_maximum'
        bad_previous = 5

        with self.assertRaises(NotImplementedError):
            self.fitter.do_guess(strategy=strategy, h5_partial_guess=bad_previous)

    def test_do_guess_no_previous(self):
        self.fitter._fitter_name = 'Fitter'
        strategy = 'absolute_maximum'
        bad_previous = self.h5_guess

        with self.assertRaises(NotImplementedError):
            self.fitter.do_guess(strategy=strategy)


if __name__ == '__main__':
    unittest.main()
