# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import h5py
import six
from warnings import warn
import numpy as np
from .hdf_utils import checkIfMain, getAuxData, get_attr, get_data_descriptor, get_formatted_labels, \
    get_dimensionality, get_sort_order, get_unit_values, reshape_to_Ndims


class PycroDataset(h5py.Dataset):

    def __init__(self, h5_ref, sort_dims=False):

        if not checkIfMain(h5_ref):
            raise TypeError('Supply a h5py.Dataset object that is a pycroscopy main dataset')

        super(PycroDataset, self).__init__(h5_ref.id)

        # User accessible properties
        # The required Position and Spectroscopic datasets
        self.h5_spec_vals = getAuxData(h5_ref, 'Spectroscopic_Values')[-1]
        self.h5_spec_inds = getAuxData(h5_ref, 'Spectroscopic_Indices')[-1]
        self.h5_pos_vals = getAuxData(h5_ref, 'Position_Values')[-1]
        self.h5_pos_inds = getAuxData(h5_ref, 'Position_Indices')[-1]

        # The dimension labels
        self.__pos_dim_labels = get_attr(self.h5_pos_inds, 'labels')
        self.__spec_dim_labels = get_attr(self.h5_spec_inds, 'labels')

        # Data desciptors
        self.data_descriptor = get_data_descriptor(self)
        self.pos_dim_descriptors = get_formatted_labels(self.h5_pos_inds)
        self.spec_dim_descriptors = get_formatted_labels(self.h5_spec_inds)

        # The size of each dimension
        self.__pos_dim_sizes = get_dimensionality(np.transpose(self.h5_pos_inds))
        self.__spec_dim_sizes = get_dimensionality(self.h5_spec_inds)

        # Sorted dimension order
        self.__pos_sort_order = get_sort_order(np.transpose(self.h5_pos_inds))
        self.__spec_sort_order = get_sort_order(self.h5_spec_inds)

        # iternal book-keeping / we don't want users to mess with these?
        self.__n_dim_sizes = np.append(self.__pos_dim_sizes, self.__spec_dim_sizes)
        self.__n_dim_labs = np.append(self.__pos_dim_labels, self.__spec_dim_labels)
        self.__n_dim_sort_order = np.append(self.__pos_sort_order, self.__spec_sort_order)
        self.__n_dim_data = None

        # Should the dimensions be sorted from fastest to slowest
        self.__sort_dims = sort_dims

    def __eq__(self, other):
        if isinstance(other, PycroDataset):
            if isinstance(other, h5py.Dataset):
                warn('Comparing PycroData object with h5py.Dataset')

            return super(PycroDataset, self).__eq__(other)

        return False

    def __repr__(self):
        h5_str = super(PycroDataset, self).__repr__()

        pos_str = ' \n'.join(['{} - size: {}'.format(dim_name, str(dim_size)) for dim_name, dim_size in
                              zip(self.__pos_dim_labels, self.__pos_dim_sizes)])
        spec_str = ' \n'.join(['{} - size: {}'.format(dim_name, str(dim_size)) for dim_name, dim_size in
                               zip(self.__spec_dim_labels, self.__spec_dim_sizes)])

        pycro_str = ' \n'.join(['located at:',
                                self.name,
                                'Data contains:', self.data_descriptor,
                                'Data dimensions and original shape:',
                                'Position Dimensions:',
                                pos_str,
                                'Spectroscopic Dimensions:',
                                spec_str])

        r = '\n'.join([h5_str, pycro_str])

        if six.PY2:
            pycro_str = pycro_str.encode('utf8')

        return '\n'.join([h5_str, pycro_str])

    def pos_dim_labels(self):
        if self.__sort_dims:
            return self.__pos_dim_labels[self.__pos_sort_order].tolist()
        else:
            return self.__pos_dim_labels.tolist()

    def spec_dim_labels(self):
        if self.__sort_dims:
            return self.__spec_dim_labels[self.__spec_sort_order].tolist()
        else:
            return self.__spec_dim_labels.tolist()

    def pos_dim_sizes(self):
        if self.__sort_dims:
            return self.__pos_dim_sizes[self.__pos_sort_order].tolist()
        else:
            return self.__pos_dim_sizes.tolist()

    def spec_dim_sizes(self):
        if self.__sort_dims:
            return self.__spec_dim_sizes[self.__spec_sort_order].tolist()
        else:
            return self.__spec_dim_sizes.tolist()

    def n_dim_labels(self):
        if self.__sort_dims:
            return self.__n_dim_labs[self.__n_dim_sort_order].tolist()
        else:
            return self.__n_dim_labs.tolist()

    def n_dim_sizes(self):
        if self.__sort_dims:
            return self.__n_dim_sizes[self.__n_dim_sort_order].tolist()
        else:
            return self.__n_dim_sizes.tolist()

    def get_pos_values(self, dim_name):
        return get_unit_values(self.h5_pos_inds, self.h5_pos_vals)[dim_name]

    def get_spec_values(self, dim_name):
        return get_unit_values(self.h5_spec_inds, self.h5_spec_vals)[dim_name]

    def current_sorting(self):
        if self.__sort_dims:
            print('Data dimensions are sorted in order from fastest changing dimension to slowest.')
        else:
            print('Data dimensions are in the order they occur in the file.')

    def toggle_sorting(self):
        if self.__n_dim_data is not None:
            if self.__sort_dims:
                nd_sort = self.__pos_sort_order[::-1] + self.__spec_sort_order[::-1]
            else:
                nd_sort = self.__n_dim_sort_order

            self.__n_dim_data = np.transpose(self.__n_dim_data, nd_sort)

        self.__sort_dims = not self.__sort_dims

    def get_n_dim_form(self):

        n_dim_data, success = reshape_to_Ndims(self, sort_dims=self.__sort_dims)

        if success is not True:
            raise ValueError('Unable to reshape data to N-dimensional form.')

        self.__n_dim_data = n_dim_data

        return self.__n_dim_data

    def slice(self, **kwargs):
        """

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        # Ensure that the n_dimensional data exists
        if self.__n_dim_data is None:
            warn('N-dimensional form of the dataset has not yet been extracted.  '
                 'This will be done before slicing.')
            _ = self.get_n_dim_form()

        # Create default slices that include the entire dimension
        n_dim_slices = [slice(None) for _ in self.__n_dim_labs]

        # Loop over all the keyword arguments and create slices for each.
        for key, val in kwargs.items():
            # Make sure the dimension is valid
            if key not in self.__n_dim_labs:

                raise KeyError('Cannot slice on dimension {}.  '
                               'Valid dimensions are {}.'.format(key, self.__n_dim_labs.tolist()))

            # Check the value and convert to a slice object if possible.
            # Use a list if not.
            if isinstance(val, slice) or isinstance(val, list):
                pass
            elif isinstance(val, np.ndarray):
                val = val.flatten().tolist()
            elif isinstance(val, tuple):
                val = list(val)
            else:
                raise TypeError('The slices must be array-likes or slice objects.')

            idim = self.n_dim_labels().index(key)

            n_dim_slices[idim] = val

        # Now that the slices are built, we just need to apply them to the data
        # This method is slow and memory intensive but shouldn't fail if multiple lists are given.
        # TODO: More elegant slicing method for PycroDataset objects
        data_slice = self.__n_dim_data[n_dim_slices[0]]

        for idim, this_slice in enumerate(n_dim_slices[1:]):
            idim += 1
            print(idim, this_slice)
            base_slice = [slice(None) for _ in self.__n_dim_labs]

            base_slice[idim] = this_slice
            print(base_slice)
            data_slice = data_slice[base_slice]

        return data_slice
