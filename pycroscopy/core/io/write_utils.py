import sys

import numpy as np
from collections import Iterable
import numbers

from .virtual_data import VirtualDataset

from .dtype_utils import contains_integers

__all__ = ['clean_string_att', 'get_aux_dset_slicing', 'make_indices_matrix',
           'INDICES_DTYPE', 'VALUES_DTYPE', 'AuxillaryDescriptor', 'build_ind_val_dsets']

if sys.version_info.major == 3:
    unicode = str

INDICES_DTYPE = np.uint32
VALUES_DTYPE = np.float32


class AuxillaryDescriptor(object):
    def __init__(self, dim_sizes, dim_names, dim_units, dim_step_sizes=None, dim_initial_vals=None):
        """
        Object that provides the instructions necessary for building ancillary datasets

        Parameters
        ----------
        dim_sizes : list / tuple of of unsigned ints.
            Sizes of all dimensions arranged from fastest to slowest.
            For example - [5, 3], if the data had 5 units along X (changing faster) and 3 along Y (changing slower)
        dim_names : list / tuple of str / unicode
            Names corresponding to each dimension in 'sizes'. For example - ['X', 'Y']
        dim_units : list / tuple of str / unicode
            Units corresponding to each dimension in 'sizes'. For example - ['nm', 'um']
        dim_step_sizes : list / tuple of numbers, optional
            step-size in each dimension.  One if not specified.
        dim_initial_vals : list / tuple of numbers, optional
            Floating point for the zeroth value in each dimension.  Zero if not specified.

        """
        lengths = []
        for val, var_name, elem_type, required in zip([dim_sizes, dim_names, dim_units, dim_step_sizes,
                                                       dim_initial_vals],
                                                      ['dim_sizes', 'dim_names', 'dim_units', 'dim_step_sizes',
                                                       'dim_initial_vals'],
                                                      [0, 2, 2, 1, 1],
                                                      [True, True, True, False, False]):
            if not required and val is None:
                continue
            if not isinstance(val, (list, tuple, np.ndarray)):
                raise TypeError(var_name + ' should be a list / tuple / numpy array')
            lengths.append(len(val))
            if elem_type == 2:
                if not np.all([isinstance(_, (str, unicode)) for _ in val]):
                    raise TypeError(var_name + ' should be a iterable containing strings')
            elif elem_type == 0:
                if not contains_integers(val, min_val=1 + len(val) > 1):
                    raise TypeError(var_name + ' should be a iterable containing integers > 1')
            else:
                if not np.all([isinstance(_, numbers.Number) for _ in val]):
                    raise TypeError(var_name + ' should be a iterable containing Numbers')
        num_elems = np.unique(lengths)
        if len(num_elems) != 1:
            raise ValueError('All the arguments should have the same number of elements')
        if num_elems[0] == 0:
            raise ValueError('Argument should not be empty')

        self.sizes = dim_sizes
        self.names = dim_names
        self.units = dim_units
        if dim_step_sizes is None:
            dim_step_sizes = np.ones_like(dim_sizes)
        self.steps = dim_step_sizes
        if dim_initial_vals is None:
            dim_initial_vals = np.zeros_like(dim_sizes)
        self.initial_vals = dim_initial_vals


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
    if not isinstance(dim_names, Iterable):
        raise TypeError('dim_names should be Iterables')
    if not len(dim_names) > 0:
        raise ValueError('dim_names should not be empty')
    if not np.all([isinstance(x, (str, unicode)) for x in dim_names]):
        raise TypeError('dim_names should contain strings')

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
    if not isinstance(num_steps, (tuple, list, np.ndarray)):
        raise TypeError('num_steps should be a list / tuple / numpy array')
    if not contains_integers(num_steps, min_val=1 + len(num_steps) > 0):
        raise ValueError('num_steps should contain integers greater than equal to 1 (empty dimension) or 2')

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
            elif np.any([type(x) in [str, unicode, bytes, np.str_] for x in att_val]):
                return np.array(att_val, dtype='S')
        if type(att_val) == np.str_:
            return str(att_val)
        return att_val
    except TypeError:
        raise TypeError('Failed to clean: {}'.format(att_val))


def build_ind_val_dsets(descriptor, is_spectral=True, verbose=False, base_name=None):
    """
    Creates VirtualDatasets for the position OR spectroscopic indices and values of the data.
    Remember that the contents of the dataset can be changed if need be after the creation of the datasets.
    For example if one of the spectroscopic dimensions (e.g. - Bias) was sinusoidal and not linear, The specific
    dimension in the Spectroscopic_Values dataset can be manually overwritten.

    Parameters
    ----------
    descriptor : AuxillaryDescriptor
        Object that provides all necessary instructions for constructing the indices and values datasets
    is_spectral : bool, optional. default = True
        Spectroscopic (True) or Position (False)
    verbose : Boolean, optional
        Whether or not to print statements for debugging purposes
    base_name : str / unicode, optional
        Prefix for the datasets. Default: 'Position_' when is_spectral is False, 'Spectroscopic_' otherwise

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
    if not isinstance(descriptor, AuxillaryDescriptor):
        raise TypeError('descriptor should be an AuxillaryDescriptor object')

    if base_name is not None:
        if not isinstance(base_name, (str, unicode)):
            raise TypeError('base_name should be a string')
        if not base_name.endswith('_'):
            base_name += '_'
    else:
        base_name = 'Position_'
        if is_spectral:
            base_name = 'Spectroscopic_'

    steps = np.atleast_2d(descriptor.steps)

    if verbose:
        print('Steps')
        print(steps.shape)
        print(steps)

    initial_values = np.atleast_2d(descriptor.initial_vals)

    if verbose:
        print('Initial Values')
        print(initial_values.shape)
        print(initial_values)

    # Get the indices for all dimensions
    indices = make_indices_matrix(descriptor.sizes)
    if not isinstance(indices, np.ndarray):
        raise TypeError('indices should be a numpy array')
    if verbose:
        print('Indices')
        print(indices.shape)
        print(indices)

    # Convert the indices to values
    values = initial_values + VALUES_DTYPE(indices)*steps

    if is_spectral:
        indices = indices.transpose()
        values = values.transpose()

    # Create the slices that will define the labels
    region_slices = get_aux_dset_slicing(descriptor.names, is_spectroscopic=is_spectral)

    # Create the VirtualDataset for both Indices and Values
    ds_indices = VirtualDataset(base_name + 'Indices', indices, dtype=INDICES_DTYPE)
    ds_values = VirtualDataset(base_name + 'Values', VALUES_DTYPE(values), dtype=VALUES_DTYPE)

    for dset in [ds_indices, ds_values]:
        dset.attrs['labels'] = region_slices
        dset.attrs['units'] = descriptor.units

    return ds_indices, ds_values