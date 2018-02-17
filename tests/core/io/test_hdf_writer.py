# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2017

@author: Suhas Somnath
"""

import unittest
import os
import h5py
import numpy as np

import sys
sys.path.append("../../../pycroscopy/")
from pycroscopy import VirtualGroup, VirtualDataset
from pycroscopy import HDFwriter
from pycroscopy.core.io.write_utils import clean_string_att
from pycroscopy.core.io.hdf_utils import get_attr, get_h5_obj_refs  # Until an elegant solution presents itself


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
            microdset = VirtualDataset(dset_name, data)

            writer = HDFwriter(h5_f)
            h5_d = writer._create_simple_dset(h5_f, microdset)
            self.assertIsInstance(h5_d, h5py.Dataset)
            self.assertEqual(h5_d.parent, h5_f)
            self.assertEqual(h5_d.name, '/' + dset_name)
            self.assertEqual(h5_d.shape, data.shape)
            self.assertTrue(np.allclose(h5_d[()], data))
            self.assertEqual(h5_d.dtype, dtype)

        os.remove(file_path)

    def test_simple_dset_write_success_more_options_02(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            dset_name = 'test'
            data = np.random.rand(16, 1024)
            dtype = data.dtype
            compression = 'gzip'
            chunking=(1, 1024)
            microdset = VirtualDataset(dset_name, data, dtype=dtype, compression=compression, chunking=chunking)

            writer = HDFwriter(h5_f)
            h5_d = writer._create_simple_dset(h5_f, microdset)
            self.assertIsInstance(h5_d, h5py.Dataset)
            self.assertEqual(h5_d.parent, h5_f)
            self.assertEqual(h5_d.name, '/' + dset_name)
            self.assertEqual(h5_d.shape, data.shape)
            self.assertTrue(np.allclose(h5_d[()], data))
            self.assertEqual(h5_d.dtype, dtype)
            self.assertEqual(h5_d.compression, compression)
            self.assertEqual(h5_d.chunks, chunking)

        os.remove(file_path)

    def test_simple_dset_write_success_more_options_03(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            dset_name = 'test'
            data = np.random.rand(16, 1024)
            dtype = np.float16
            compression = 'gzip'
            chunking=(1, 1024)
            microdset = VirtualDataset(dset_name, data, dtype=dtype, compression=compression, chunking=chunking)

            writer = HDFwriter(h5_f)
            h5_d = writer._create_simple_dset(h5_f, microdset)
            self.assertIsInstance(h5_d, h5py.Dataset)
            self.assertEqual(h5_d.parent, h5_f)
            self.assertEqual(h5_d.name, '/' + dset_name)
            self.assertEqual(h5_d.shape, data.shape)
            self.assertEqual(h5_d.dtype, dtype)
            self.assertEqual(h5_d.compression, compression)
            self.assertEqual(h5_d.chunks, chunking)
            self.assertTrue(np.all(h5_d[()] - data < 1E-3))

        os.remove(file_path)

    def test_empty_dset_write_success_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            dset_name = 'test'
            maxshape = (16, 1024)
            microdset = VirtualDataset(dset_name, None, maxshape=maxshape)

            writer = HDFwriter(h5_f)
            h5_d = writer._create_empty_dset(h5_f, microdset)
            self.assertIsInstance(h5_d, h5py.Dataset)
            self.assertEqual(h5_d.parent, h5_f)
            self.assertEqual(h5_d.name, '/' + dset_name)
            self.assertEqual(h5_d.shape, maxshape)
            self.assertEqual(h5_d.maxshape, maxshape)
            # dtype is assigned automatically by h5py. Not to be tested here

        os.remove(file_path)

    def test_empty_dset_write_success_w_options_02(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            dset_name = 'test'
            maxshape = (16, 1024)
            chunking = (1, 1024)
            compression = 'gzip'
            dtype = np.float16
            microdset = VirtualDataset(dset_name, None, maxshape=maxshape,
                                       dtype=dtype, compression=compression, chunking=chunking)

            writer = HDFwriter(h5_f)
            h5_d = writer._create_empty_dset(h5_f, microdset)
            self.assertIsInstance(h5_d, h5py.Dataset)
            self.assertEqual(h5_d.parent, h5_f)
            self.assertEqual(h5_d.name, '/' + dset_name)
            self.assertEqual(h5_d.dtype, dtype)
            self.assertEqual(h5_d.compression, compression)
            self.assertEqual(h5_d.chunks, chunking)
            self.assertEqual(h5_d.shape, maxshape)
            self.assertEqual(h5_d.maxshape, maxshape)

        os.remove(file_path)

    def test_expandable_dset_write_success_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            dset_name = 'test'
            maxshape = (None, 1024)
            data = np.random.rand(1, 1024)
            microdset = VirtualDataset(dset_name, data, maxshape=maxshape)

            writer = HDFwriter(h5_f)
            h5_d = writer._create_resizeable_dset(h5_f, microdset)
            self.assertIsInstance(h5_d, h5py.Dataset)
            self.assertEqual(h5_d.parent, h5_f)
            self.assertEqual(h5_d.name, '/' + dset_name)
            self.assertEqual(h5_d.shape, data.shape)
            self.assertEqual(h5_d.maxshape, maxshape)
            self.assertTrue(np.allclose(h5_d[()], data))

            # Now test to make sure that the dataset can be expanded:
            # TODO: add this to the example!

            expansion_axis = 0
            h5_d.resize(h5_d.shape[expansion_axis] + 1, axis=expansion_axis)

            self.assertEqual(h5_d.shape, (data.shape[0]+1, data.shape[1]))
            self.assertEqual(h5_d.maxshape, maxshape)

            # Finally try checking to see if this new data is also present in the file
            new_data = np.random.rand(1024)
            h5_d[1] = new_data

            data = np.vstack((np.squeeze(data), new_data))
            self.assertTrue(np.allclose(h5_d[()], data))

        os.remove(file_path)

    # TODO: will have to check to see if the parent is correctly declared for the group

    def test_group_create_non_indexed_simple_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            grp_name = 'test'
            micro_group = VirtualGroup(grp_name)

            writer = HDFwriter(h5_f)
            h5_grp = writer._create_group(h5_f, micro_group)
            self.assertIsInstance(h5_grp, h5py.Group)
            self.assertEqual(h5_grp.parent, h5_f)
            self.assertEqual(h5_grp.name, '/' + grp_name)
            # self.assertEqual(len(h5_grp.items), 0)

        os.remove(file_path)

    def test_group_create_indexed_simple_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            grp_name = 'test_'
            micro_group = VirtualGroup(grp_name)

            writer = HDFwriter(h5_f)
            h5_grp = writer._create_group(h5_f, micro_group)
            self.assertIsInstance(h5_grp, h5py.Group)
            self.assertEqual(h5_grp.parent, h5_f)
            self.assertEqual(h5_grp.name, '/' + grp_name + '000')
            # self.assertEqual(len(h5_grp.items), 0)

        os.remove(file_path)

    def test_group_create_root_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            grp_name = ''
            micro_group = VirtualGroup(grp_name)

            writer = HDFwriter(h5_f)
            with self.assertRaises(ValueError):
                _ = writer._create_group(h5_f, micro_group)

        os.remove(file_path)

    def test_group_create_indexed_nested_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            outer_grp_name = 'outer_'
            micro_group = VirtualGroup(outer_grp_name)

            writer = HDFwriter(h5_f)
            h5_outer_grp = writer._create_group(h5_f, micro_group)
            self.assertIsInstance(h5_outer_grp, h5py.Group)
            self.assertEqual(h5_outer_grp.parent, h5_f)
            self.assertEqual(h5_outer_grp.name, '/' + outer_grp_name + '000')

            inner_grp_name = 'inner_'
            micro_group = VirtualGroup(inner_grp_name)

            h5_inner_grp = writer._create_group(h5_outer_grp, micro_group)
            self.assertIsInstance(h5_inner_grp, h5py.Group)
            self.assertEqual(h5_inner_grp.parent, h5_outer_grp)
            self.assertEqual(h5_inner_grp.name, h5_outer_grp.name + '/' + inner_grp_name + '000')

        os.remove(file_path)

    def test_write_legal_reg_ref_multi_dim_data(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}}

            writer._write_dset_attributes(h5_dset, attrs.copy())
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:

            self.assertTrue(np.all([x in list(attrs['labels'].keys()) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_legal_reg_ref_multi_dim_data_2nd_dim(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 3)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(None), slice(0, None, 2)),
                                'odd_rows': (slice(None), slice(1, None, 2))}}

            writer._write_dset_attributes(h5_dset, attrs.copy())
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:

            self.assertTrue(np.all([x in list(attrs['labels'].keys()) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[:, 0:None:2], data[:, 1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_legal_reg_ref_one_dim_data(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2)),
                                'odd_rows': (slice(1, None, 2))}}

            writer._write_dset_attributes(h5_dset, attrs.copy())
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:
            self.assertTrue(np.all([x in list(attrs['labels'].keys()) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_generate_and_write_reg_ref_legal(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(2, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': ['row_1', 'row_2']}

            writer._write_dset_attributes(h5_dset, attrs.copy())
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:

            self.assertTrue(np.all([x in list(attrs['labels']) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[0], data[1]]
            written_data = [h5_dset[h5_dset.attrs['row_1']], h5_dset[h5_dset.attrs['row_2']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(np.squeeze(exp), np.squeeze(act)))

        os.remove(file_path)

    def test_generate_and_write_reg_ref_illegal(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(3, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            # with self.assertWarns(UserWarning):
            writer._write_dset_attributes(h5_dset, {'labels': ['row_1', 'row_2']})

            self.assertEqual(len(h5_dset.attrs), 0)

            h5_f.flush()

        os.remove(file_path)

    def test_generate_and_write_reg_ref_illegal(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(2, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            # with self.assertWarns(UserWarning):
            with self.assertRaises(TypeError):
                writer._write_dset_attributes(h5_dset, {'labels': [1, np.arange(3)]})

        os.remove(file_path)

    def test_write_illegal_reg_ref_too_many_slices(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), slice(None), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None), slice(None))}}

            with self.assertRaises(ValueError):
                writer._write_dset_attributes(h5_dset, attrs.copy())

        os.remove(file_path)

    def test_write_illegal_reg_ref_too_few_slices(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2)),
                                'odd_rows': (slice(1, None, 2))}}

            with self.assertRaises(ValueError):
                writer._write_dset_attributes(h5_dset, attrs.copy())

        os.remove(file_path)

    def test_write_reg_ref_slice_dim_larger_than_data(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, 15, 2), slice(None)),
                                'odd_rows': (slice(1, 15, 2), slice(None))}}

            writer._write_dset_attributes(h5_dset, attrs.copy())
            h5_f.flush()

            # two atts point to region references. one for labels
            self.assertEqual(len(h5_dset.attrs), 1 + len(attrs['labels']))

            # check if the labels attribute was written:

            self.assertTrue(np.all([x in list(attrs['labels'].keys()) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_illegal_reg_ref_not_slice_objs(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), 15),
                                'odd_rows': (slice(1, None, 2), 'hello')}}

            with self.assertRaises(TypeError):
                writer._write_dset_attributes(h5_dset, attrs.copy())

        os.remove(file_path)

    def test_write_simple_atts_reg_ref_to_dset(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, VirtualDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3'],
                     'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}
                     }

            writer._write_dset_attributes(h5_dset, attrs.copy())

            reg_ref = attrs.pop('labels')

            self.assertEqual(len(h5_dset.attrs), len(attrs) + 1 + len(reg_ref))

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(get_attr(h5_dset, key) == expected_val))

            self.assertTrue(np.all([x in list(reg_ref.keys()) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_invalid_input(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            with self.assertRaises(TypeError):
               _ = writer.write(np.arange(5))

    def test_write_dset_under_root(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3'],
                     'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}
                     }
            micro_dset = VirtualDataset('test', data)
            micro_dset.attrs = attrs.copy()
            [h5_dset] = writer.write(micro_dset)
            self.assertIsInstance(h5_dset, h5py.Dataset)

            reg_ref = attrs.pop('labels')

            self.assertEqual(len(h5_dset.attrs), len(attrs) + 1 + len(reg_ref))

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(get_attr(h5_dset, key) == expected_val))

            self.assertTrue(np.all([x in list(reg_ref.keys()) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_dset_under_existing_group(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            h5_g = writer._create_group(h5_f, VirtualGroup('test_group'))

            self.assertIsInstance(h5_g, h5py.Group)

            data = np.random.rand(5, 7)
            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3'],
                     'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                'odd_rows': (slice(1, None, 2), slice(None))}
                     }
            micro_dset = VirtualDataset('test', data, parent='/test_group')
            micro_dset.attrs = attrs.copy()
            [h5_dset] = writer.write(micro_dset)
            self.assertIsInstance(h5_dset, h5py.Dataset)

            self.assertEqual(h5_dset.parent, h5_g)

            reg_ref = attrs.pop('labels')

            self.assertEqual(len(h5_dset.attrs), len(attrs) + 1 + len(reg_ref))

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(get_attr(h5_dset, key) == expected_val))

            self.assertTrue(np.all([x in list(reg_ref.keys()) for x in get_attr(h5_dset, 'labels')]))

            expected_data = [data[:None:2], data[1:None:2]]
            written_data = [h5_dset[h5_dset.attrs['even_rows']], h5_dset[h5_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

        os.remove(file_path)

    def test_write_dset_under_invalid_group(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)

            with self.assertRaises(KeyError):
                _ = writer.write(VirtualDataset('test', np.random.rand(5, 7), parent='/does_not_exist'))

        os.remove(file_path)

    def test_write_root(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3']}

            micro_group = VirtualGroup('')
            micro_group.attrs = attrs
            writer = HDFwriter(h5_f)
            [ret_val] = writer.write(micro_group)

            self.assertIsInstance(ret_val, h5py.File)
            self.assertEqual(h5_f, ret_val)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(get_attr(h5_f, key) == expected_val))

        os.remove(file_path)

    def test_write_single_group(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3']}

            micro_group = VirtualGroup('Test_')
            micro_group.attrs = attrs
            writer = HDFwriter(h5_f)
            [h5_group] = writer.write(micro_group)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(get_attr(h5_group, key) == expected_val))

        os.remove(file_path)

    def test_group_indexing_sequential(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            writer = HDFwriter(h5_f)
            micro_group_0 = VirtualGroup('Test_', attrs={'att_1': 'string_val', 'att_2': 1.2345})
            [h5_group_0] = writer.write(micro_group_0)

            _ = writer.write(VirtualGroup('blah'))

            self.assertIsInstance(h5_group_0, h5py.Group)
            self.assertEqual(h5_group_0.name, '/Test_000')
            for key, expected_val in micro_group_0.attrs.items():
                self.assertTrue(np.all(get_attr(h5_group_0, key) == expected_val))

            micro_group_1 = VirtualGroup('Test_', attrs={'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']})
            [h5_group_1] = writer.write(micro_group_1)

            self.assertIsInstance(h5_group_1, h5py.Group)
            self.assertEqual(h5_group_1.name, '/Test_001')
            for key, expected_val in micro_group_1.attrs.items():
                self.assertTrue(np.all(get_attr(h5_group_1, key) == expected_val))

        os.remove(file_path)

    def test_group_indexing_simultaneous(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            micro_group_0 = VirtualGroup('Test_', attrs = {'att_1': 'string_val', 'att_2': 1.2345})
            micro_group_1 = VirtualGroup('Test_', attrs={'att_3': [1, 2, 3, 4], 'att_4': ['str_1', 'str_2', 'str_3']})
            root_group = VirtualGroup('', children=[VirtualGroup('blah'), micro_group_0,
                                                    VirtualGroup('meh'), micro_group_1])

            writer = HDFwriter(h5_f)
            h5_refs_list = writer.write(root_group)

            [h5_group_1] = get_h5_obj_refs(['Test_001'], h5_refs_list)
            [h5_group_0] = get_h5_obj_refs(['Test_000'], h5_refs_list)

            self.assertIsInstance(h5_group_0, h5py.Group)
            self.assertEqual(h5_group_0.name, '/Test_000')
            for key, expected_val in micro_group_0.attrs.items():
                self.assertTrue(np.all(get_attr(h5_group_0, key) == expected_val))

            self.assertIsInstance(h5_group_1, h5py.Group)
            self.assertEqual(h5_group_1.name, '/Test_001')
            for key, expected_val in micro_group_1.attrs.items():
                self.assertTrue(np.all(get_attr(h5_group_1, key) == expected_val))

        os.remove(file_path)

    def test_write_simple_tree(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            inner_dset_data = np.random.rand(5, 7)
            inner_dset_attrs = {'att_1': 'string_val',
                                'att_2': 1.2345,
                                'att_3': [1, 2, 3, 4],
                                'att_4': ['str_1', 'str_2', 'str_3'],
                                'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                           'odd_rows': (slice(1, None, 2), slice(None))}
                                }
            inner_dset = VirtualDataset('inner_dset', inner_dset_data)
            inner_dset.attrs = inner_dset_attrs.copy()

            attrs_inner_grp = {'att_1': 'string_val',
                               'att_2': 1.2345,
                               'att_3': [1, 2, 3, 4],
                               'att_4': ['str_1', 'str_2', 'str_3']}
            inner_group = VirtualGroup('indexed_inner_group_')
            inner_group.attrs = attrs_inner_grp
            inner_group.add_children(inner_dset)

            outer_dset_data = np.random.rand(5, 7)
            outer_dset_attrs = {'att_1': 'string_val',
                                'att_2': 1.2345,
                                'att_3': [1, 2, 3, 4],
                                'att_4': ['str_1', 'str_2', 'str_3'],
                                'labels': {'even_rows': (slice(0, None, 2), slice(None)),
                                           'odd_rows': (slice(1, None, 2), slice(None))}
                                }
            outer_dset = VirtualDataset('test', outer_dset_data, parent='/test_group')
            outer_dset.attrs = outer_dset_attrs.copy()

            attrs_outer_grp = {'att_1': 'string_val',
                               'att_2': 1.2345,
                               'att_3': [1, 2, 3, 4],
                               'att_4': ['str_1', 'str_2', 'str_3']}
            outer_group = VirtualGroup('unindexed_outer_group')
            outer_group.attrs = attrs_outer_grp
            outer_group.add_children([inner_group, outer_dset])

            writer = HDFwriter(h5_f)
            h5_refs_list = writer.write(outer_group)

            # I don't know of a more elegant way to do this:
            [h5_outer_dset] = get_h5_obj_refs([outer_dset.name], h5_refs_list)
            [h5_inner_dset] = get_h5_obj_refs([inner_dset.name], h5_refs_list)
            [h5_outer_group] = get_h5_obj_refs([outer_group.name], h5_refs_list)
            [h5_inner_group] = get_h5_obj_refs(['indexed_inner_group_000'], h5_refs_list)

            self.assertIsInstance(h5_outer_dset, h5py.Dataset)
            self.assertIsInstance(h5_inner_dset, h5py.Dataset)
            self.assertIsInstance(h5_outer_group, h5py.Group)
            self.assertIsInstance(h5_inner_group, h5py.Group)

            # check assertions for the inner dataset first
            self.assertEqual(h5_inner_dset.parent, h5_inner_group)

            reg_ref = inner_dset_attrs.pop('labels')

            self.assertEqual(len(h5_inner_dset.attrs), len(inner_dset_attrs) + 1 + len(reg_ref))

            for key, expected_val in inner_dset_attrs.items():
                self.assertTrue(np.all(get_attr(h5_inner_dset, key) == expected_val))

            self.assertTrue(np.all([x in list(reg_ref.keys()) for x in get_attr(h5_inner_dset, 'labels')]))

            expected_data = [inner_dset_data[:None:2], inner_dset_data[1:None:2]]
            written_data = [h5_inner_dset[h5_inner_dset.attrs['even_rows']], h5_inner_dset[h5_inner_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

            # check assertions for the inner data group next:
            self.assertEqual(h5_inner_group.parent, h5_outer_group)
            for key, expected_val in attrs_inner_grp.items():
                self.assertTrue(np.all(get_attr(h5_inner_group, key) == expected_val))

            # check the outer dataset next:
            self.assertEqual(h5_outer_dset.parent, h5_outer_group)

            reg_ref = outer_dset_attrs.pop('labels')

            self.assertEqual(len(h5_outer_dset.attrs), len(outer_dset_attrs) + 1 + len(reg_ref))

            for key, expected_val in outer_dset_attrs.items():
                self.assertTrue(np.all(get_attr(h5_outer_dset, key) == expected_val))

            self.assertTrue(np.all([x in list(reg_ref.keys()) for x in get_attr(h5_outer_dset, 'labels')]))

            expected_data = [outer_dset_data[:None:2], outer_dset_data[1:None:2]]
            written_data = [h5_outer_dset[h5_outer_dset.attrs['even_rows']],
                            h5_outer_dset[h5_outer_dset.attrs['odd_rows']]]

            for exp, act in zip(expected_data, written_data):
                self.assertTrue(np.allclose(exp, act))

            # Finally check the outer group:
            self.assertEqual(h5_outer_group.parent, h5_f)
            for key, expected_val in attrs_outer_grp.items():
                self.assertTrue(np.all(get_attr(h5_outer_group, key) == expected_val))

        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
