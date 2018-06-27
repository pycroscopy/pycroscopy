# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Chris Smith
"""

from warnings import warn
import sys
import h5py
import numpy as np

from .virtual_data import VirtualDataset
from pyUSID.io.write_utils import INDICES_DTYPE, VALUES_DTYPE, Dimension, build_ind_val_matrices, get_aux_dset_slicing

if sys.version_info.major == 3:
    unicode = str


def build_reduced_spec_dsets(h5_parent_group, h5_spec_inds, h5_spec_vals, keep_dim, step_starts,
                             basename='Spectroscopic'):
    """
    Creates new Spectroscopic Indices and Values datasets from the input datasets
    and keeps the dimensions specified in keep_dim

    Parameters
    ----------
    h5_parent_group : h5py.Group or h5py.File
        Group under which the indices and values datasets will be created
    h5_spec_inds : HDF5 Dataset
            Spectroscopic indices dataset
    h5_spec_vals : HDF5 Dataset
            Spectroscopic values dataset
    keep_dim : Numpy Array, Boolean
            Array designating which rows of the input spectroscopic datasets to keep
    step_starts : Numpy Array, Unsigned Integers
            Array specifying the start of each step in the reduced datasets
    basename : str / unicode
            String to which '_Indices' and '_Values' will be appended to get the names
            of the new datasets

    Returns
    -------
    ds_inds : VirtualDataset
            Reduced Spectroscopic indices dataset
    ds_vals : VirtualDataset
            Reduces Spectroscopic values dataset
    """
    warn('build_reduced_spec_dsets is available only for legacy purposes and will be REMOVED in a future release.\n'
         'Please consider using write_reduced_spec_dsets instead', DeprecationWarning)

    if not isinstance(h5_parent_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or h5py.Group object')
    if basename is not None:
        if not isinstance(basename, (str, unicode)):
            raise TypeError('basename should be a string')

    for sub_name in ['_Indices', '_Values']:
        if basename + sub_name in h5_parent_group.keys():
            raise KeyError('Dataset: {} already exists in provided group: {}'.format(basename + sub_name,
                                                                                     h5_parent_group.name))

    for param, param_name in zip([h5_spec_inds, h5_spec_vals], ['h5_spec_inds', 'h5_spec_vals']):
        if not isinstance(param, h5py.Dataset):
            raise TypeError(param_name + ' should be a h5py.Dataset object')
    if not isinstance(keep_dim, (bool, np.ndarray, list, tuple)):
        raise TypeError('keep_dim should be a bool, np.ndarray, list, or tuple')
    if not isinstance(step_starts, (list, np.ndarray, list, tuple)):
        raise TypeError('step_starts should be a list, np.ndarray, list, or tuple')

    if h5_spec_inds.shape[0] > 1:
        '''
        Extract all rows that we want to keep from input indices and values
        '''
        # TODO: handle TypeError: Indexing elements must be in increasing order
        ind_mat = h5_spec_inds[keep_dim, :][:, step_starts]
        val_mat = h5_spec_vals[keep_dim, :][:, step_starts]
        '''
        Create new Datasets to hold the data
        Name them based on basename
        '''
        ds_inds = VirtualDataset(basename + '_Indices', ind_mat, dtype=h5_spec_inds.dtype)
        ds_vals = VirtualDataset(basename + '_Values', val_mat, dtype=h5_spec_vals.dtype)

        # Extracting the labels from the original spectroscopic data sets
        labels = h5_spec_inds.attrs['labels'][keep_dim]
        # Creating the dimension slices for the new spectroscopic data sets
        reg_ref_slices = dict()
        for row_ind, row_name in enumerate(labels):
            reg_ref_slices[row_name] = (slice(row_ind, row_ind + 1), slice(None))

        # Adding the labels and units to the new spectroscopic data sets
        for dset in [ds_inds, ds_vals]:
            dset.attrs['labels'] = reg_ref_slices
            dset.attrs['units'] = h5_spec_inds.attrs['units'][keep_dim]

    else:  # Single spectroscopic dimension:
        ds_inds = VirtualDataset(basename + '_Indices', np.array([[0]], dtype=INDICES_DTYPE))
        ds_vals = VirtualDataset(basename + '_Values', np.array([[0]], dtype=VALUES_DTYPE))

        for dset in [ds_inds, ds_vals]:
            dset.attrs['labels'] = {'Single_Step': (slice(0, None), slice(None))}
            dset.attrs['units'] = ''

    return ds_inds, ds_vals


def build_ind_val_dsets(dimensions, is_spectral=True, verbose=False, base_name=None):
    """
    Creates VirtualDatasets for the position or spectroscopic indices and values of the data.
    Remember that the contents of the dataset can be changed if need be after the creation of the datasets.
    For example if one of the spectroscopic dimensions (e.g. - Bias) was sinusoidal and not linear, The specific
    dimension in the Spectroscopic_Values dataset can be manually overwritten.

    Parameters
    ----------
    dimensions : Dimension or array-like of Dimension objects
        Sequence of Dimension objects that provides all necessary instructions for constructing the indices and values
        datasets
    is_spectral : bool, optional. default = True
        Spectroscopic (True) or Position (False)
    verbose : Boolean, optional
        Whether or not to print statements for debugging purposes
    base_name : str / unicode, optional
        Prefix for the datasets. Default: 'Position\_' when is_spectral is False, 'Spectroscopic\_' otherwise

    Returns
    -------
    ds_inds : VirtualDataset
            Reduced Spectroscopic indices dataset
    ds_vals : VirtualDataset
            Reduces Spectroscopic values dataset

    Notes
    -----
    `steps`, `initial_values`, `labels`, and 'units' must be the same length as
    `dimensions` when they are specified.

    Dimensions should be in the order from fastest varying to slowest.
    """

    warn('build_ind_val_dsets is available only for legacy purposes and will be REMOVED in a future release.\n'
         'Please consider using write_ind_val_dsets in hdf_utils instead', DeprecationWarning)

    if isinstance(dimensions, Dimension):
        dimensions = [dimensions]
    if not isinstance(dimensions, (list, np.ndarray, tuple)):
        raise TypeError('dimensions should be array-like ')
    if not np.all([isinstance(x, Dimension) for x in dimensions]):
        raise TypeError('dimensions should be a sequence of Dimension objects')

    if base_name is not None:
        if not isinstance(base_name, (str, unicode)):
            raise TypeError('base_name should be a string')
        if not base_name.endswith('_'):
            base_name += '_'
    else:
        base_name = 'Position_'
        if is_spectral:
            base_name = 'Spectroscopic_'

    unit_values = [x.values for x in dimensions]

    indices, values = build_ind_val_matrices(unit_values, is_spectral=is_spectral)

    if verbose:
        print('Indices:')
        print(indices)
        print('Values:')
        print(values)

    # Create the slices that will define the labels
    region_slices = get_aux_dset_slicing([x.name for x in dimensions], is_spectroscopic=is_spectral)

    # Create the VirtualDataset for both Indices and Values
    ds_indices = VirtualDataset(base_name + 'Indices', indices, dtype=INDICES_DTYPE)
    ds_values = VirtualDataset(base_name + 'Values', VALUES_DTYPE(values), dtype=VALUES_DTYPE)

    for dset in [ds_indices, ds_values]:
        dset.attrs['labels'] = region_slices
        dset.attrs['units'] = [x.units for x in dimensions]

    return ds_indices, ds_values
