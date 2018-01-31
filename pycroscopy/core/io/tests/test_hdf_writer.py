import unittest
import os
import h5py
import numpy as np

import sys
sys.path.append("../../../")
from pycroscopy import MicroDataGroup, MicroDataset
from pycroscopy import HDFwriter
from pycroscopy.core.io.hdf_writer import clean_string_att
from pycroscopy.core.io.hdf_utils import get_attr


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
            microdset = MicroDataset(dset_name, data, dtype=dtype, compression=compression, chunking=chunking)

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
            microdset = MicroDataset(dset_name, data, dtype=dtype, compression=compression, chunking=chunking)

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
            microdset = MicroDataset(dset_name, None, maxshape=maxshape)

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
            microdset = MicroDataset(dset_name, None, maxshape=maxshape,
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
            microdset = MicroDataset(dset_name, data, maxshape=maxshape)

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
            micro_group = MicroDataGroup(grp_name)

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
            micro_group = MicroDataGroup(grp_name)

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
            micro_group = MicroDataGroup(grp_name)

            writer = HDFwriter(h5_f)
            with self.assertRaises(ValueError):
                _ = writer._create_group(h5_f, micro_group)

        os.remove(file_path)

    def test_group_create_indexed_nested_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            outer_grp_name = 'outer_'
            micro_group = MicroDataGroup(outer_grp_name)

            writer = HDFwriter(h5_f)
            h5_outer_grp = writer._create_group(h5_f, micro_group)
            self.assertIsInstance(h5_outer_grp, h5py.Group)
            self.assertEqual(h5_outer_grp.parent, h5_f)
            self.assertEqual(h5_outer_grp.name, '/' + outer_grp_name + '000')

            inner_grp_name = 'inner_'
            micro_group = MicroDataGroup(inner_grp_name)

            h5_inner_grp = writer._create_group(h5_outer_grp, micro_group)
            self.assertIsInstance(h5_inner_grp, h5py.Group)
            self.assertEqual(h5_inner_grp.parent, h5_outer_grp)
            self.assertEqual(h5_inner_grp.name, h5_outer_grp.name + '/' + inner_grp_name + '000')

        os.remove(file_path)

    def test_clean_string_att_float(self):
        expected = 5.321
        self.assertEqual(expected, clean_string_att(expected))

    def test_clean_string_att_str(self):
        expected = 'test'
        self.assertEqual(expected, clean_string_att(expected))

    def test_clean_string_att_num_array(self):
        expected = [1, 2, 3.456]
        self.assertEqual(expected, clean_string_att(expected))

    def test_clean_string_att_str_list(self):
        expected = ['a', 'bc', 'def']
        returned = clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)

    def test_clean_string_att_str_tuple(self):
        expected = ('a', 'bc', 'def')
        returned = clean_string_att(expected)
        expected = np.array(expected, dtype='S')
        for exp, act in zip(expected, returned):
            self.assertEqual(exp, act)

    def test_write_legal_atts_to_grp(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            grp_name = 'test_group_'
            micro_group = MicroDataGroup(grp_name)

            writer = HDFwriter(h5_f)
            h5_group = writer._create_group(h5_f, micro_group)
            self.assertIsInstance(h5_group, h5py.Group)

            attrs = {'att_1': 'string_val', 'att_2': 1.234, 'att_3': [1, 2, 3.14, 4],
                     'att_4': ['s', 'tr', 'str_3']}

            writer._write_simple_attrs(h5_group, attrs)

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(get_attr(h5_group, key) == expected_val))

        os.remove(file_path)

    def test_write_legal_atts_to_dset_01(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', np.arange(3)))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'att_1': 'string_val',
                     'att_2': 1.2345,
                     'att_3': [1, 2, 3, 4],
                     'att_4': ['str_1', 'str_2', 'str_3']}

            writer._write_simple_attrs(h5_dset, attrs)

            self.assertEqual(len(h5_dset.attrs), len(attrs))

            for key, expected_val in attrs.items():
                self.assertTrue(np.all(get_attr(h5_dset, key) == expected_val))

        os.remove(file_path)

    def test_write_legal_reg_ref_multi_dim_data(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:

            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', data))
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
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', data))
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
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', data))
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

    def test_write_illegal_reg_ref_too_many_slices(self):
        file_path = 'test.h5'
        self.__delete_existing_file(file_path)
        with h5py.File(file_path) as h5_f:
            writer = HDFwriter(h5_f)
            data = np.random.rand(5, 7)
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', data))
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
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', data))
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
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', data))
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
            h5_dset = writer._create_simple_dset(h5_f, MicroDataset('test', data))
            self.assertIsInstance(h5_dset, h5py.Dataset)

            attrs = {'labels': {'even_rows': (slice(0, None, 2), 15),
                                'odd_rows': (slice(1, None, 2), 'hello')}}

            with self.assertRaises(TypeError):
                writer._write_dset_attributes(h5_dset, attrs.copy())

        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
