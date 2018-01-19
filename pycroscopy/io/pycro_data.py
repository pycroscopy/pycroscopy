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
from .hdf_utils import checkIfMain, get_attr, get_data_descriptor, get_formatted_labels, \
    get_dimensionality, get_sort_order, get_unit_values, reshape_to_Ndims
from .io_utils import transformToReal
# from ..viz.jupyter_utils import simple_ndim_visualizer


class PycroDataset(h5py.Dataset):

    def __init__(self, h5_ref, sort_dims=False):
        """
        New data object that extends the h5py.Dataset.

        Parameters
        ----------
        h5_ref : hdf5.Dataset
            The base dataset to be extended
        sort_dims : bool
            Should the dimensions be sorted internally from fastest changing to slowest.

        Methods
        -------
        self.get_current_sorting
        self.toggle_sorting
        self.get_pos_values
        self.get_spec_values
        self.get_n_dim_form
        self.slice


        Attributes
        ----------
        self.h5_spec_vals : h5py.Dataset
            Associated Spectroscopic Values dataset
        self.h5_spec_inds : h5py.Dataset
            Associated Spectroscopic Indices dataset
        self.h5_pos_vals : h5py.Dataset
            Associated Position Values dataset
        self.h5_pos_inds : h5py.Dataset
            Associated Position Indices dataset
        self.pos_dim_labels : list of str
            The labels for the position dimensions.
        self.spec_dim_labels : list of str
            The labels for the spectroscopic dimensions.
        self.n_dim_labels : list of str
            The labels for the n-dimensional dataset.
        self.pos_dim_sizes : list of int
            A list of the sizes of each position dimension.
        self.spec_dim_sizes : list of int
            A list of the sizes of each spectroscopic dimension.
        self.n_dim_sizes : list of int
            A list of the sizes of each dimension.

        Notes
        -----
        The order of all labels and sizes attributes is determined by the current value of `sort_dims`.

        """

        if not checkIfMain(h5_ref):
            raise TypeError('Supply a h5py.Dataset object that is a pycroscopy main dataset')

        super(PycroDataset, self).__init__(h5_ref.id)

        # User accessible properties
        # The required Position and Spectroscopic datasets
        self.h5_spec_vals = self.file[self.attrs['Spectroscopic_Values']]
        self.h5_spec_inds = self.file[self.attrs['Spectroscopic_Indices']]
        self.h5_pos_vals = self.file[self.attrs['Position_Values']]
        self.h5_pos_inds = self.file[self.attrs['Position_Indices']]

        # The dimension labels
        self.__pos_dim_labels = get_attr(self.h5_pos_inds, 'labels')
        self.__spec_dim_labels = get_attr(self.h5_spec_inds, 'labels')

        # Data desciptors
        self.data_descriptor = get_data_descriptor(self)
        self.pos_dim_descriptors = get_formatted_labels(self.h5_pos_inds)
        self.spec_dim_descriptors = get_formatted_labels(self.h5_spec_inds)

        # The size of each dimension
        self.__pos_dim_sizes = np.array(get_dimensionality(np.transpose(self.h5_pos_inds)))
        self.__spec_dim_sizes = np.array(get_dimensionality(np.atleast_2d(self.h5_spec_inds)))

        # Sorted dimension order
        self.__pos_sort_order = get_sort_order(np.transpose(self.h5_pos_inds))
        self.__spec_sort_order = get_sort_order(np.atleast_2d(self.h5_spec_inds))

        # iternal book-keeping / we don't want users to mess with these?
        self.__n_dim_sizes = np.append(self.__pos_dim_sizes, self.__spec_dim_sizes)
        self.__n_dim_labs = np.append(self.__pos_dim_labels, self.__spec_dim_labels)
        self.__n_dim_sort_order = np.append(self.__pos_sort_order, self.__spec_sort_order)
        self.__n_dim_data = None

        # Should the dimensions be sorted from fastest to slowest
        self.__sort_dims = sort_dims

        self.__set_labels_and_sizes()

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

    def __set_labels_and_sizes(self):
        """
        Sets the labels and sizes attributes to the correct values based on
        the value of `self.__sort_dims`

        Returns
        -------
        None

        """
        if self.__sort_dims:
            self.pos_dim_labels = self.__pos_dim_labels[self.__pos_sort_order].tolist()
            self.spec_dim_labels = self.__spec_dim_labels[self.__spec_sort_order].tolist()
            self.pos_dim_sizes = self.__pos_dim_sizes[self.__pos_sort_order].tolist()
            self.spec_dim_sizes = self.__spec_dim_sizes[self.__spec_sort_order].tolist()
            self.n_dim_labels = self.__n_dim_labs[self.__n_dim_sort_order].tolist()
            self.n_dim_sizes = self.__n_dim_sizes[self.__n_dim_sort_order].tolist()

        else:
            self.pos_dim_labels = self.__pos_dim_labels.tolist()
            self.spec_dim_labels = self.__spec_dim_labels.tolist()
            self.pos_dim_sizes = self.__pos_dim_sizes.tolist()
            self.spec_dim_sizes = self.__spec_dim_sizes.tolist()
            self.n_dim_labels = self.__n_dim_labs.tolist()
            self.n_dim_sizes = self.__n_dim_sizes.tolist()

    def get_pos_values(self, dim_name):
        """
        Extract the values for the specified position dimension

        Parameters
        ----------
        dim_name : str
            Name of one of the dimensions in `self.pos_dim_labels`

        Returns
        -------
        dim_values : numpy.ndarray
            Array containing the unit values of the dimension `dim_name`

        """
        return get_unit_values(self.h5_pos_inds, self.h5_pos_vals)[dim_name]

    def get_spec_values(self, dim_name):
        """
        Extract the values for the specified spectroscopic dimension

        Parameters
        ----------
        dim_name : str
            Name of one of the dimensions in `self.spec_dim_labels`

        Returns
        -------
        dim_values : numpy.ndarray
            Array containing the unit values of the dimension `dim_name`

        """
        return get_unit_values(self.h5_spec_inds, self.h5_spec_vals)[dim_name]

    def get_current_sorting(self):
        """
        Prints the current sorting method.

        """
        if self.__sort_dims:
            print('Data dimensions are sorted in order from fastest changing dimension to slowest.')
        else:
            print('Data dimensions are in the order they occur in the file.')

    def toggle_sorting(self):
        """
        Toggles between sorting from the fastest changing dimension to the slowest and sorting based on the
        order of the labels

        """
        if self.__n_dim_data is not None:
            if self.__sort_dims:
                nd_sort = self.__pos_sort_order[::-1] + self.__spec_sort_order[::-1]
            else:
                nd_sort = self.__n_dim_sort_order

            self.__n_dim_data = np.transpose(self.__n_dim_data, nd_sort)

        self.__sort_dims = not self.__sort_dims

        self.__set_labels_and_sizes()

    def get_n_dim_form(self, as_scalar=False):
        """
        Reshapes the dataset to an N-dimensional array

        Returns
        -------
        n_dim_data : numpy.ndarray
            N-dimensional form of the dataset

        """

        n_dim_data, success = reshape_to_Ndims(self, sort_dims=self.__sort_dims)

        if success is not True:
            raise ValueError('Unable to reshape data to N-dimensional form.')

        if as_scalar:
            self.__n_dim_data = transformToReal(n_dim_data)
        else:
            self.__n_dim_data = n_dim_data

        return self.__n_dim_data

    def slice(self, as_scalar=False, slice_dict=dict()):
        """
        Slice the dataset based on an input dictionary of 'str': slice pairs.
        Each string should correspond to a dimension label.  The slices can be
        array-likes or slice objects.

        Parameters
        ----------
        as_scalar : bool
            Should the data be returned as scalar values only.
        slice_dict : dict
            Dictionary of array-likes.

        Returns
        -------
        data_slice : numpy.ndarray
            Slice of the dataset.  Dataset has been reshaped to N-dimensions if `success` is True, only
            by Position dimensions if `success` is 'Positions', or not reshape at all if `success`
            is False.
        success : str or bool
            Informs the user as to how the data_slice has been shaped.

        """
        # Convert the slice dictionary into lists of indices for each dimension
        pos_slice, spec_slice = self.get_pos_spec_slices(slice_dict)

        # Now that the slices are built, we just need to apply them to the data
        # This method is slow and memory intensive but shouldn't fail if multiple lists are given.
        if len(pos_slice) <= len(spec_slice):
            # Fewer final positions that spectra (Most common case)
            data_slice = np.atleast_2d(self[pos_slice, :])[:, spec_slice]
        else:
            data_slice = np.atleast_2d(self[:, spec_slice])[pos_slice, :]

        pos_inds = self.h5_pos_inds[pos_slice, :]
        spec_inds = self.h5_spec_inds[:, spec_slice].reshape([self.h5_spec_inds.shape[0], -1])
        data_slice, success = reshape_to_Ndims(data_slice,
                                               h5_pos=pos_inds,
                                               h5_spec=spec_inds)

        if as_scalar:
            return transformToReal(data_slice), success
        else:
            return data_slice, success

    def get_pos_spec_slices(self, slice_dict):
        """
        Convert the slice dictionary into two lists of indices, one each for the position and spectroscopic
        dimensions.

        Parameters
        ----------
        slice_dict : dict
            Dictionary of array-likes.

        Returns
        -------
        pos_slice : list of uints
            Position indices included in the slice
        spec_slice : list of uints
            Spectroscopic indices included in the slice

        """
        # Create default slices that include the entire dimension
        n_dim_slices = dict()
        n_dim_slices_sizes = dict()
        for dim_lab, dim_size in zip(self.n_dim_labels, self.n_dim_sizes):
            n_dim_slices[dim_lab] = list(range(dim_size))
            n_dim_slices_sizes[dim_lab] = len(n_dim_slices[dim_lab])
        # Loop over all the keyword arguments and create slices for each.
        for key, val in slice_dict.items():
            # Make sure the dimension is valid
            if key not in self.__n_dim_labs:
                raise KeyError('Cannot slice on dimension {}.  '
                               'Valid dimensions are {}.'.format(key, self.__n_dim_labs.tolist()))

            # Check the value and convert to a slice object if possible.
            # Use a list if not.
            if isinstance(val, slice):
                val = n_dim_slices[key][val]
            elif isinstance(val, list):
                pass
            elif isinstance(val, np.ndarray):
                val = val.flatten().tolist()
            elif isinstance(val, tuple):
                val = list(val)
            else:
                raise TypeError('The slices must be array-likes or slice objects.')

            n_dim_slices[key] = val

            n_dim_slices_sizes[key] = len(val)

        # Build the list of position slice indices
        for pos_ind, pos_lab in enumerate(self.__pos_dim_labels):
            n_dim_slices[pos_lab] = np.isin(self.h5_pos_inds[:, pos_ind], n_dim_slices[pos_lab])
            if pos_ind == 0:
                pos_slice = n_dim_slices[pos_lab]
            else:
                pos_slice = np.logical_and(pos_slice, n_dim_slices[pos_lab])
        pos_slice = np.argwhere(pos_slice)

        # Do the same for the spectroscopic slice
        for spec_ind, spec_lab in enumerate(self.__spec_dim_labels):
            n_dim_slices[spec_lab] = np.isin(self.h5_spec_inds[spec_ind], n_dim_slices[spec_lab])
            if spec_ind == 0:
                spec_slice = n_dim_slices[spec_lab]
            else:
                spec_slice = np.logical_and(spec_slice, n_dim_slices[spec_lab])
        spec_slice = np.argwhere(spec_slice)

        return pos_slice, spec_slice

    # def visualize(self, slice_dict=None, **kwargs):
    #     """
    #     Interactive visualization of this dataset. Only available on jupyter notebooks
    #
    #     Parameters
    #     ----------
    #     slice_dict : dictionary, optional
    #         Slicing instructions
    #     """
    #     # TODO: Robust implementation that allows slicing
    #     if len(self.pos_dim_labels + self.spec_dim_labels) > 4:
    #         raise NotImplementedError('Unable to support visualization of more than 4 dimensions. Try slicing')
    #     data_mat = self.get_n_dim_form()
    #     pos_dim_names = self.pos_dim_labels[::-1]
    #     spec_dim_names = self.spec_dim_labels
    #     pos_dim_units_old = get_attr(self.h5_pos_inds, 'units')
    #     spec_dim_units_old = get_attr(self.h5_spec_inds, 'units')
    #     pos_ref_vals = get_unit_values(self.h5_pos_inds, self.h5_pos_vals, is_spec=False)
    #     spec_ref_vals = get_unit_values(self.h5_spec_inds, self.h5_spec_vals, is_spec=True)
    #
    #     simple_ndim_visualizer(data_mat, pos_dim_names, pos_dim_units_old, spec_dim_names, spec_dim_units_old,
    #                            pos_ref_vals=pos_ref_vals, spec_ref_vals=spec_ref_vals, **kwargs)
