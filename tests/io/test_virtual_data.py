# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
sys.path.append("../../../pycroscopy/")
import numpy as np
from pycroscopy.io.virtual_data import VirtualGroup, VirtualDataset


class MyOutput(object):
    # http://pragmaticpython.com/2017/03/23/unittesting-print-statements/

    def __init__(self):
        self.data = []

    def write(self, s):
        self.data.append(s)

    def __str__(self):
        return "".join(self.data)


class TestVirtualDataSet(unittest.TestCase):

    def test_insufficient_inputs(self):
        name = 'test'
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, data=None)

    def test_children_for_virtualgroup_legal(self):
        child = VirtualGroup('blah')
        mg = VirtualGroup('test_group', children=child)
        self.assertEqual(len(mg.children), 1)
        self.assertEqual(child, mg.children[0])

    def test_children_for_virtualgroup_illegal(self):
        mg = VirtualGroup('test_group', children=np.arange(4))
        self.assertEqual(len(mg.children), 0)

    def test_attrs_for_virtualgroup_legal(self):
        attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3']}
        mg = VirtualGroup('test_group', attrs=attrs)
        self.assertEqual(len(mg.attrs), len(attrs))
        for key, expected_val in attrs.items():
            self.assertEqual(expected_val, mg.attrs[key])

    def test_invalid_chunking_argument_01(self):
        name = 'test'
        chunking = (-1, 128)
        data = np.random.rand(2, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, data, chunking=chunking)

    def test_invalid_chunking_argument_02(self):
        name = 'test'
        chunking = ('a', range(5))
        data = np.random.rand(2, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, data, chunking=chunking)

    def test_incompatible_chunking_data(self):
        name = 'test'
        chunking = (4, 128)
        data = np.random.rand(2, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, data, chunking=chunking)

    def test_incompatible_chunking_data_02(self):
        name = 'test'
        chunking = (4, 128)
        maxshape = (2, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, None, maxshape=maxshape, chunking=chunking)

    def test_incompatible_chunking_data_03(self):
        name = 'test'
        chunking = (4, 128)
        maxshape = (None, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, None, maxshape=maxshape, chunking=chunking)

    def test_incompatible_maxshape_chunking_01(self):
        name = 'test'
        chunking = (3, 128)
        maxshape = (1,129)
        data = np.random.rand(5, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, data, chunking=chunking, maxshape=maxshape)

    def test_only_chunking_provided(self):
        name = 'test'
        chunking = (1, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, None, chunking=chunking)

    def test_chunking_w_none(self):
        name = 'test'
        chunking = (None, 128)
        data = np.random.rand(2, 128)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, data, chunking=chunking)

    def test_incompatible_maxshape_data_shapes(self):
        name = 'test'
        maxshape = 128
        data = np.random.rand(2, 128)
        with self.assertRaises(ValueError):
           _ = VirtualDataset(name, data, maxshape=maxshape)

    def test_simple_correct_01(self):
        data = np.arange(5)
        name = 'test'
        dset = VirtualDataset(name, data)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)

    def test_correct_compression_simple_dset(self):
        data = np.arange(5)
        name = 'test'
        compression = 'gzip'
        dset = VirtualDataset(name, data, compression=compression)
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
            _ = VirtualDataset(name, data, compression=compression)

    def test_simple_dset_inherit_dtype(self):
        data = np.arange(5)
        name = 'test'
        dset = VirtualDataset(name, data)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)
        self.assertEqual(dset.dtype, data.dtype)

    def test_simple_dset_independent_dtype(self):
        data = np.arange(5, dtype=np.complex64)
        name = 'test'
        dtype = np.uint8
        dset = VirtualDataset(name, data, dtype=dtype)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)
        self.assertEqual(dset.dtype, dtype)

    def test_simple_dset_str_dtype_valid(self):
        data = np.arange(5, dtype=np.complex64)
        name = 'test'
        dtype = 'uint8'
        dset = VirtualDataset(name, data, dtype=dtype)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, None)
        self.assertEqual(dset.dtype, np.dtype(dtype))

    def test_large_empty_correct_01(self):
        data = None
        name = 'test'
        maxshape = (1024, 16384)
        dset = VirtualDataset(name, data, maxshape=maxshape)
        self.assertEqual(dset.data, data)
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, False)
        self.assertEqual(dset.maxshape, maxshape)

    def test_empty_incorrect_01(self):
        data = None
        name = 'test'
        maxshape = (None, 16384)
        with self.assertRaises(ValueError):
            _ = VirtualDataset(name, data, maxshape=maxshape)

    def test_resizable_correct_01(self):
        dtype = np.uint16
        data = np.zeros(shape=(1, 7), dtype=dtype)
        name = 'test'
        resizable = True
        dset = VirtualDataset(name, data, resizable=resizable)
        self.assertTrue(np.all(np.equal(dset.data, data)))
        self.assertEqual(dset.name, name)
        self.assertEqual(dset.resizable, resizable)
        self.assertEqual(dset.maxshape, None)


class TestVirtualGroup(unittest.TestCase):

    def test_creation_non_indexed(self):
        name = 'test_group'
        group = VirtualGroup(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.parent, '/')
        self.assertEqual(group.children, [])
        self.assertEqual(group.indexed, False)

    def test_creation_indexed(self):
        name = 'indexed_group_'
        group = VirtualGroup(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.parent, '/')
        self.assertEqual(group.children, [])
        self.assertEqual(group.indexed, True)

    def test_add_single_child_legal_01(self):
        dset_name_1 = 'dset_1'
        data_1 = np.arange(3)
        dset_1 = VirtualDataset(dset_name_1, data_1)
        group_name = 'indexed_group_'
        group = VirtualGroup(group_name)
        group.add_children(dset_1)
        group.show_tree()
        self.assertEqual(len(group.children), 1)
        in_dset = group.children[0]
        self.assertIsInstance(in_dset, VirtualDataset)
        self.assertEqual(in_dset.name, dset_name_1)
        self.assertTrue(np.all(np.equal(in_dset.data, data_1)))

    def test_add_single_child_illegal_01(self):
        group_name = 'indexed_group_'
        group = VirtualGroup(group_name)
        # with self.assertWarns('Children must be of type VirtualData. child ignored'):
        group.add_children(np.arange(4))
        self.assertEqual(len(group.children), 0)

    def test_add_children_legal_01(self):
        group_name = 'indexed_group_'
        group = VirtualGroup(group_name)
        dset_name_1 = 'dset_1'
        data_1 = np.arange(3)
        dset_1 = VirtualDataset(dset_name_1, data_1)
        dset_name_2 = 'dset_2'
        data_2 = np.random.rand(2, 3)
        dset_2 = VirtualDataset(dset_name_2, data_2)
        group.add_children([dset_1, dset_2])
        self.assertEqual(len(group.children), 2)

    def test_print(self):
        group_name = 'indexed_group_'
        group = VirtualGroup(group_name)
        dset_name_1 = 'dset_1'
        data_1 = np.arange(3)
        dset_1 = VirtualDataset(dset_name_1, data_1)
        dset_name_2 = 'dset_2'
        data_2 = np.random.rand(2, 3)
        dset_2 = VirtualDataset(dset_name_2, data_2)
        inner_grp_name = 'other_indexed_group_'
        inner_group = VirtualGroup(inner_grp_name)
        inner_group.add_children(dset_2)
        group.add_children([dset_1, inner_group])

        # http://pragmaticpython.com/2017/03/23/unittesting-print-statements/
        stdout_org = sys.stdout
        my_stdout = MyOutput()
        try:
            sys.stdout = my_stdout
            group.show_tree()
        finally:
            sys.stdout = stdout_org

        self.assertEqual(str(my_stdout), "/indexed_group_/dset_1\n/indexed_group_/other_indexed_group_\n/indexed_group_"
                                         "/other_indexed_group_/dset_2\n")


if __name__ == '__main__':
    unittest.main()
