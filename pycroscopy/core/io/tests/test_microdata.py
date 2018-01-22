import unittest
import numpy as np
from pycroscopy import MicroDataGroup, MicroDataset


class TestMicroDataSet(unittest.TestCase):
    def test_simple(self):
        data = np.arange(5)
        name = 'test'
        dset = MicroDataset(name, data)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)

    def test_large_empty(self):
        data = None
        name = 'test'
        maxshape = (1024, 16384)
        dset = MicroDataset(name, data, maxshape=maxshape)
        self.assertEqual(dset.data, data)
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, maxshape)


if __name__ == '__main__':
    unittest.main()
