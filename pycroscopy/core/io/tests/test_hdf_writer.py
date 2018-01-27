import unittest
import os
import h5py
import numpy as np

import sys
sys.path.append("../../../")
from pycroscopy import MicroDataGroup, MicroDataset
from pycroscopy import HDFwriter


class TestHDFWriter(unittest.TestCase):

    @staticmethod
    def __delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def test_init_invalid_input(self):
        with self.assertRaises(TypeError):
            _ = HDFwriter(4)

    def test_init_path_non_existant_file_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)

        writer = HDFwriter(file_path)
        self.assertIsInstance(writer, HDFwriter, "writer should be an HDFwriter")
        writer.close()
        os.remove(file_path)

    def test_init_path_existing_file_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        h5_f = h5py.File(file_path)
        h5_f.close()
        # Existing h5 file
        writer = HDFwriter(file_path)
        self.assertIsInstance(writer, HDFwriter, "writer should be an HDFwriter")
        writer.close()
        os.remove(file_path)

    def test_init_h5_handle_r_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        h5_f = h5py.File(file_path)
        h5_f.close()
        h5_f = h5py.File(file_path, mode='r')
        # hdf handle but of mode r
        with self.assertRaises(TypeError):
            _ = HDFwriter(h5_f)
        os.remove(file_path)

    def test_init_h5_handle_r_plus_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        h5_f = h5py.File(file_path)
        h5_f.close()
        h5_f = h5py.File(file_path, mode='r+')
        # open h5 file handle or mode r+

        writer = HDFwriter(h5_f)
        self.assertIsInstance(writer, HDFwriter, "writer should be an HDFwriter")
        writer.close()
        os.remove(file_path)

    def test_init_h5_handle_w_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        h5_f = h5py.File(file_path)
        h5_f.close()
        h5_f = h5py.File(file_path, mode='w')
        # open h5 file handle or mode w

        writer = HDFwriter(h5_f)
        self.assertIsInstance(writer, HDFwriter, "writer should be an HDFwriter")
        writer.close()
        os.remove(file_path)

    def test_init_h5_handle_closed(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        h5_f = h5py.File(file_path)
        h5_f.close()
        # Existing h5 file but closed
        with self.assertRaises(ValueError):
            _ = HDFwriter(h5_f)
        os.remove(file_path)

    def test_simple_dset_write_success_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            dtype = np.uint16
            dset_name = 'test'
            data = np.random.randint(0, high=15, size=5, dtype=dtype)
            microdset = MicroDataset(dset_name, data)

            writer = HDFwriter(h5_f)
            h5_d = writer._create_simple_dset(h5_f, microdset)
            self.assertEqual(h5_d.name, '/' + dset_name)
            self.assertTrue(np.allclose(h5_d[()], data))
            self.assertEqual(h5_d.dtype, dtype)

        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
