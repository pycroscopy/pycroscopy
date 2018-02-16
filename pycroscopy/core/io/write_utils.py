import sys
from warnings import warn

import h5py
import numpy as np
from collections import Iterable

from .dtype_utils import contains_integers
import warnings

__all__ = ['build_ind_val_dsets', 'get_aux_dset_slicing', 'make_indices_matrix', 'INDICES_DTYPE', 'VALUES_DTYPE']

if sys.version_info.major == 3:
    unicode = str

INDICES_DTYPE = np.uint32
VALUES_DTYPE = np.float32


def build_ind_val_dsets(h5_parent_group, dimensions, is_spectral=True, steps=None, initial_values=None, labels=None,
                        units=None, verbose=False, base_name=None):
    """
    Creates h5py.Datasets for the position OR spectroscopic indices and values of the data.

    Parameters
    ----------
    h5_parent_group : h5py.Group or h5py.File
        Group under which the indices and values datasets will be created
    dimensions : array_like of numpy.uint
        Integer values for the length of each dimension
    is_spectral : bool, optional. default = True
        Spectroscopic (True) or Position (False)
    steps : array_like of float, optional
        Floating point values for the step-size in each dimension.  One
        if not specified.
    initial_values : array_like of float, optional
        Floating point for the zeroth value in each dimension.  Zero if
        not specified.
    labels : array_like of str, optional
        The names of each dimension.  Empty strings will be used if not
        specified.
    units : array_like of str, optional
        The units of each dimension.  Empty strings will be used if not
        specified.
    verbose : Boolean, optional
        Whether or not to print statements for debugging purposes
    base_name : str / unicode, optional
        Prefix for the datasets. Default: 'Position_' when is_spectral is False, 'Spectroscopic_' otherwise

    Returns
    -------
    h5_spec_inds : h5py.Dataset
        Dataset containing the position indices
    h5_spec_vals : h5py.Dataset
        Dataset containing the value at each position

    Notes
    -----
    `steps`, `initial_values`, `labels`, and 'units' must be the same length as
    `dimensions` when they are specified.

    Dimensions should be in the order from fastest varying to slowest.
    """
    assert contains_integers(dimensions, min_val=2)
    assert isinstance(h5_parent_group, (h5py.Group, h5py.File))

    if base_name is not None:
        assert isinstance(base_name, (str, unicode))
        if not base_name.endswith('_'):
            base_name += '_'
    else:
        base_name = 'Position_'
        if is_spectral:
            base_name = 'Spectroscopic_'

    # check if the datasets already exist. If they do, there's no point in going any further
    for sub_name in ['Indices', 'Values']:
        if base_name + sub_name in h5_parent_group.keys():
            raise KeyError('Dataset: {} already exists in provided group: {}'.format(base_name + sub_name,
                                                                                       h5_parent_group.name))

    if labels is None:
        warnings.warn('Arbitrary names provided to dimensions. Please provide legitimate values for parameter - labels',
                      DeprecationWarning)
        labels = ['Unknown Dimension {}'.format(ind) for ind in range(len(dimensions))]
    else:
        assert isinstance(labels, Iterable)
        if len(labels) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')

    if units is None:
        warnings.warn('Arbitrary units provided to dimensions. Please provide legitimate values for parameter - units',
                      DeprecationWarning)
        units = ['Arb Unit {}'.format(ind) for ind in range(len(dimensions))]
    else:
        assert isinstance(units, Iterable)
        if len(units) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')

    if steps is None:
        steps = np.ones_like(dimensions)
    else:
        assert isinstance(steps, Iterable)
        if len(steps) != len(dimensions):
            raise ValueError('The arrays for step sizes and dimension sizes must be the same.')
        steps = np.atleast_2d(steps)

    if verbose:
        print('Steps')
        print(steps.shape)
        print(steps)

    if initial_values is None:
        initial_values = np.zeros_like(dimensions)
    else:
        assert isinstance(initial_values, Iterable)
        if len(initial_values) != len(dimensions):
            raise ValueError('The arrays for initial values and dimension sizes must be the same.')
        initial_values = np.atleast_2d(initial_values)

    if verbose:
        print('Initial Values')
        print(initial_values.shape)
        print(initial_values)

    # Get the indices for all dimensions
    indices = make_indices_matrix(dimensions)
    assert isinstance(indices, np.ndarray)
    if verbose:
        print('Indices')
        print(indices.shape)
        print(indices)

    # Convert the indices to values
    values = initial_values + VALUES_DTYPE(indices)*steps

    # Create the slices that will define the labels
    if is_spectral:
        indices = indices.transpose()
        values = values.transpose()

    region_slices = get_aux_dset_slicing(labels, is_spectroscopic=is_spectral)

    # Create the Datasets for both Indices and Values
    h5_indices = h5_parent_group.create_dataset(base_name + 'Indices', data=INDICES_DTYPE(indices), dtype=INDICES_DTYPE)
    h5_values = h5_parent_group.create_dataset(base_name + 'Values', data=VALUES_DTYPE(values), dtype=VALUES_DTYPE)

    for h5_dset in [h5_indices, h5_values]:
        write_region_references(h5_dset, region_slices, print_log=verbose)
        h5_dset.attrs['units'] = units
        h5_dset.attrs['labels'] = labels

    return h5_indices, h5_values


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
    ds_inds : h5py.Dataset
            Reduced Spectroscopic indices dataset
    ds_vals : h5py.Dataset
            Reduces Spectroscopic values dataset
    """
    assert isinstance(h5_parent_group, (h5py.Group, h5py.File))
    if basename is not None:
        assert isinstance(basename, (str, unicode))

    for sub_name in ['_Indices', '_Values']:
        if basename + sub_name in h5_parent_group.keys():
            raise KeyError('Dataset: {} already exists in provided group: {}'.format(basename + sub_name,
                                                                                     h5_parent_group.name))

    for param in [h5_spec_inds, h5_spec_vals]:
        assert isinstance(param, h5py.Dataset)
    assert isinstance(keep_dim, (bool, np.ndarray))
    assert isinstance(step_starts, (list, np.ndarray))

    if h5_spec_inds.shape[0] > 1:
        '''
        Extract all rows that we want to keep from input indices and values
        '''
        ind_mat = h5_spec_inds[keep_dim, :][:, step_starts]
        val_mat = h5_spec_vals[keep_dim, :][:, step_starts]
        '''
        Create new Datasets to hold the data
        Name them based on basename
        '''
        ds_inds = h5_parent_group.create_dataset(basename + '_Indices', data=ind_mat, dtype=h5_spec_inds.dtype)
        ds_vals = h5_parent_group.create_dataset(basename + '_Values', data=val_mat, dtype=h5_spec_vals.dtype)
        # Extracting the labels from the original spectroscopic data sets
        labels = h5_spec_inds.attrs['labels'][keep_dim]
        # Creating the dimension slices for the new spectroscopic data sets
        reg_ref_slices = dict()
        for row_ind, row_name in enumerate(labels):
            reg_ref_slices[row_name] = (slice(row_ind, row_ind + 1), slice(None))

        # Adding the labels and units to the new spectroscopic data sets
        for dset in [ds_inds, ds_vals]:
            write_region_references(dset, reg_ref_slices, print_log=False)
            dset.attrs['labels'] = labels
            dset.attrs['units'] = h5_spec_inds.attrs['units'][keep_dim]

    else:  # Single spectroscopic dimension:
        ds_inds = h5_parent_group.create_dataset(basename + '_Indices', data=np.array([[0]]), dtype=INDICES_DTYPE)
        ds_vals = h5_parent_group.create_dataset(basename + '_Values', data=np.array([[0]]), dtype=VALUES_DTYPE)

        reg_ref_slices = {'Single_Step': (slice(0, None), slice(None))}

        for dset in [ds_inds, ds_vals]:
            write_region_references(dset, reg_ref_slices, print_log=False)
            dset.attrs['labels'] = 'Single_Step'
            dset.attrs['units'] = ['']

    return ds_inds, ds_vals


def get_aux_dset_slicing(dim_names, last_ind=None, is_spectroscopic=False):
    """
    Returns a dictionary of slice objects to help in creating region references in the position or spectroscopic
    indices and values datasets

    Parameters
    ------------
    dim_names : iterable
        List of strings denoting the names of the position axes or spectroscopic dimensions arranged in the same order
        that matches the dimensions in the indices / values dataset
    last_ind : (Optional) unsigned int, default = None
        Last pixel in the positon or spectroscopic matrix. Useful in experiments where the
        parameters have changed (eg. BEPS new data format) during the experiment.
    is_spectroscopic : bool, optional. default = True
        set to True for position datasets and False for spectroscopic datasets
    Returns
    ------------
    slice_dict : dictionary
        Dictionary of tuples containing slice objects corresponding to
        each position axis.
    """
    assert isinstance(dim_names, Iterable)
    assert len(dim_names) > 0
    assert np.all([isinstance(x, (str, unicode)) for x in dim_names])

    slice_dict = dict()
    for spat_ind, curr_dim_name in enumerate(dim_names):
        val = (slice(last_ind), slice(spat_ind, spat_ind + 1))
        if is_spectroscopic:
            val = val[::-1]
        slice_dict[str(curr_dim_name)] = val
    return slice_dict


def make_indices_matrix(num_steps, is_position=True):
    """
    Makes an ancillary indices matrix given the number of steps in each dimension. In other words, this function builds
    a matrix whose rows correspond to unique combinations of the multiple dimensions provided.

    Parameters
    ------------
    num_steps : List / numpy array
        Number of steps in each spatial or spectral dimension
        Note that the axes must be ordered from fastest varying to slowest varying
    is_position : bool, optional, default = True
        Whether the returned matrix is meant for position (True) indices (tall and skinny) or spectroscopic (False)
        indices (short and wide)

    Returns
    --------------
    indices_matrix : 2D unsigned int numpy array
        arranged as [steps, spatial dimension]
    """
    assert contains_integers(num_steps, min_val=2)

    num_steps = np.array(num_steps)
    spat_dims = max(1, len(np.where(num_steps > 1)[0]))

    indices_matrix = np.zeros(shape=(np.prod(num_steps), spat_dims), dtype=INDICES_DTYPE)
    dim_ind = 0

    for indx, curr_steps in enumerate(num_steps):
        if curr_steps > 1:

            part1 = np.prod(num_steps[:indx+1])

            if indx > 0:
                part2 = np.prod(num_steps[:indx])
            else:
                part2 = 1

            if indx+1 == len(num_steps):
                part3 = 1
            else:
                part3 = np.prod(num_steps[indx+1:])

            indices_matrix[:, dim_ind] = np.tile(np.floor(np.arange(part1)/part2), part3)
            dim_ind += 1

    if not is_position:
        indices_matrix = indices_matrix.T

    return indices_matrix


def clean_string_att(att_val):
    """
    Replaces any unicode objects within lists with their string counterparts to ensure compatibility with python 3.
    If the attribute is indeed a list of unicodes, the changes will be made in-place

    Parameters
    ----------
    att_val : object
        Attribute object

    Returns
    -------
    att_val : object
        Attribute object
    """
    try:
        if isinstance(att_val, Iterable):
            if type(att_val) in [unicode, str]:
                return att_val
            elif np.any([type(x) in [str, unicode, bytes] for x in att_val]):
                return np.array(att_val, dtype='S')
        if type(att_val) == np.str_:
            return str(att_val)
        return att_val
    except TypeError:
        raise TypeError('Failed to clean: {}'.format(att_val))


def attempt_reg_ref_build(h5_dset, dim_names, print_log=False):
    """

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references need to be added as attributes
    dim_names : list or tuple
        List of the names of the region references (typically names of dimensions)
    print_log : bool, optional. Default=False
        Whether or not to print debugging statements

    Returns
    -------
    labels_dict : dict
        The slicing information must be formatted using tuples of slice objects.
        For example {'region_1':(slice(None, None), slice (0,1))}
    """
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object but is instead of type '
                        '{}.'.format(type(h5_dset)))
    if not isinstance(dim_names, (list, tuple)):
        raise TypeError('slices should be a list or tuple but is instead of type '
                        '{}'.format(type(dim_names)))

    if len(h5_dset.shape) != 2:
        return dict()

    if not np.all([isinstance(obj, (str, unicode)) for obj in dim_names]):
        raise TypeError('Unable to automatically generate region references for dataset: {} since one or more names'
                        ' of the region references was not a string'.format(h5_dset.name))

    labels_dict = dict()
    if len(dim_names) == h5_dset.shape[0]:
        if print_log:
            print('Most likely a spectroscopic indices / values dataset')
        for dim_index, curr_name in enumerate(dim_names):
            labels_dict[curr_name] = (slice(dim_index, dim_index+1), slice(None))
    elif len(dim_names) == h5_dset.shape[1]:
        if print_log:
            print('Most likely a position indices / values dataset')
        for dim_index, curr_name in enumerate(dim_names):
            labels_dict[curr_name] = (slice(None), slice(dim_index, dim_index + 1))

    if len(labels_dict) > 0:
        warn('Attempted to automatically build region reference dictionary for dataset: {}.\n'
             'Please specify region references as a tuple of slice objects for each attribute'.format(h5_dset.name))
    else:
        if print_log:
            print('Could not build region references since dataset had shape:{} and number of region references is '
                  '{}'.format(h5_dset.shape, len(dim_names)))
    return labels_dict


def write_region_references(h5_dset, reg_ref_dict, print_log=False):
    """
    Creates attributes of a h5py.Dataset that refer to regions in the dataset

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references will be added as attributes
    reg_ref_dict : dict
        The slicing information must be formatted using tuples of slice objects.
        For example {'region_1':(slice(None, None), slice (0,1))}
    print_log : Boolean (Optional. Default = False)
        Whether or not to print status messages
    """
    if not isinstance(reg_ref_dict, dict):
        raise TypeError('slices should be a dictionary but is instead of type '
                        '{}'.format(type(reg_ref_dict)))
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object but is instead of type '
                        '{}'.format(type(h5_dset)))

    if print_log:
        print('Starting to write Region References to Dataset', h5_dset.name, 'of shape:', h5_dset.shape)
    for reg_ref_name, reg_ref_tuple in reg_ref_dict.items():
        if print_log:
            print('About to write region reference:', reg_ref_name, ':', reg_ref_tuple)

        reg_ref_tuple = clean_reg_ref(h5_dset, reg_ref_tuple, print_log=print_log)

        h5_dset.attrs[reg_ref_name] = h5_dset.regionref[reg_ref_tuple]

        if print_log:
            print('Wrote Region Reference:%s' % reg_ref_name)


def clean_reg_ref(h5_dset, reg_ref_tuple, print_log=False):
    """
    Makes sure that the provided instructions for a region reference are indeed valid
    This method has become necessary since h5py allows the writing of region references larger than the maxshape

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references will be added as attributes
    reg_ref_tuple : list / tuple
        The slicing information formatted using tuples of slice objects.
    print_log : Boolean (Optional. Default = False)
        Whether or not to print status messages

    Returns
    -------
    is_valid : bool
        Whether or not this
    """
    if not isinstance(reg_ref_tuple, (tuple, dict, slice)):
        raise TypeError('slices should be a tuple, list, or slice but is instead of type '
                        '{}'.format(type(reg_ref_tuple)))
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object but is instead of type '
                        '{}'.format(type(h5_dset)))

    if isinstance(reg_ref_tuple, slice):
        # 1D dataset
        reg_ref_tuple = [reg_ref_tuple]

    if len(reg_ref_tuple) != len(h5_dset.shape):
        raise ValueError('Region reference tuple did not have the same dimensions as the h5 dataset')

    if print_log:
        print('Comparing {} with h5 dataset maxshape of {}'.format(reg_ref_tuple, h5_dset.maxshape))

    new_reg_refs = list()

    for reg_ref_slice, max_size in zip(reg_ref_tuple, h5_dset.maxshape):
        if not isinstance(reg_ref_slice, slice):
            raise TypeError('slices should be a tuple or a list but is instead of type '
                            '{}'.format(type(reg_ref_slice)))

        # For now we will simply make sure that the end of the slice is <= maxshape
        if max_size is not None and reg_ref_slice.stop is not None:
            reg_ref_slice = slice(reg_ref_slice.start, min(reg_ref_slice.stop, max_size), reg_ref_slice.step)

        new_reg_refs.append(reg_ref_slice)

    if print_log:
        print('Region reference tuple now: {}'.format(new_reg_refs))

    return tuple(new_reg_refs)


def assign_group_index(h5_parent_group, base_name, print_log=False):
    """
    Searches the parent h5 group to find the next available index for the group

    Parameters
    ----------
    h5_parent_group : h5py.Group object
        Parent group under which the new group object will be created
    base_name : str / unicode
        Base name of the new group without index
    print_log : bool, optional. Default=False
        Whether or not to print debugging statements

    Returns
    -------
    base_name : str / unicode
        Base name of the new group with the next available index as a suffix
    """
    temp = [key for key in h5_parent_group.keys()]
    if print_log:
        print('Looking for group names starting with {} in parent containing items: '
              '{}'.format(base_name, temp))
    previous_indices = []
    for item_name in temp:
        if isinstance(h5_parent_group[item_name], h5py.Group) and item_name.startswith(base_name):
            previous_indices.append(int(item_name.replace(base_name, '')))
    previous_indices = np.sort(previous_indices)
    if print_log:
        print('indices of existing groups with the same prefix: {}'.format(previous_indices))
    if len(previous_indices) == 0:
        index = 0
    else:
        index = previous_indices[-1] + 1
    return base_name + '{:03d}'.format(index)


def write_simple_attrs(h5_obj, attrs, obj_type='', print_log=False):
    """
    Writes attributes to a h5py object

    Parameters
    ----------
    h5_obj : h5py.File, h5py.Group, or h5py.Dataset object
        h5py object to which the attributes will be written to
    attrs : dict
        Dictionary containing the attributes as key-value pairs
    obj_type : str / unicode, optional. Default = ''
        type of h5py.obj. Examples include 'group', 'file', 'dataset
    print_log : bool, optional. Default=False
        Whether or not to print debugging statements
    """
    if not isinstance(attrs, dict):
        raise TypeError('attrs should be a dictionary but is instead of type '
                        '{}'.format(type(attrs)))
    if not isinstance(h5_obj, (h5py.File, h5py.Group, h5py.Dataset)):
        raise TypeError('h5_obj should be a h5py File, Group or Dataset object but is instead of type '
                        '{}. UNABLE to safely abort'.format(type(h5_obj)))

    for key, val in attrs.items():
        if val is None:
            continue
        if print_log:
            print('Writing attribute: {} with value: {}'.format(key, val))
        h5_obj.attrs[key] = clean_string_att(val)
    if print_log:
        print('Wrote all (simple) attributes to {}: {}\n'.format(obj_type, h5_obj.name.split('/')[-1]))
