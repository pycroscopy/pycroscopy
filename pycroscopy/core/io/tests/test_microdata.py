import unittest
import numpy as np
from pycroscopy import MicroDataGroup, MicroDataset


class TestMicroDataSet(unittest.TestCase):

    def test_insufficient_inputs(self):
        name = 'test'
        with self.assertRaises(ValueError):
            _ = MicroDataset(name, data=None)

    def test_simple_correct_01(self):
        data = np.arange(5)
        name = 'test'
        dset = MicroDataset(name, data)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)

    def test_correct_compression_simple_dset(self):
        data = np.arange(5)
        name = 'test'
        compression = 'gzip'
        dset = MicroDataset(name, data, compression=compression)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)
        self.assertEqual(dset.compression, compression)

    def test_incorrect_compression_simple_dset(self):
        data = np.arange(5)
        name = 'test'
        compression = 'blah'
        with self.assertRaises(ValueError):
            _ = MicroDataset(name, data, compression=compression)

    def test_large_empty_correct_01(self):
        data = None
        name = 'test'
        maxshape = (1024, 16384)
        dset = MicroDataset(name, data, maxshape=maxshape)
        self.assertEqual(dset.data, data)
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, maxshape)

    def test_resizable_correct_01(self):
        dtype = np.uint16
        data = np.zeros(shape=(1, 7), dtype=dtype)
        name = 'test'
        resizable = True
        dset = MicroDataset(name, data, )
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)


if __name__ == '__main__':
    unittest.main()
