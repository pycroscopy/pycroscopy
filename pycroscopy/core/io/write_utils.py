import sys
import numpy as np
from collections import Iterable
from .microdata import MicroDataset
from .dtype_utils import contains_integers
import warnings

__all__ = ['build_ind_val_dsets', 'get_aux_dset_slicing', 'make_indices_matrix']

if sys.version_info.major == 3:
    unicode = str


def build_ind_val_dsets(dimensions, is_spectral=True, steps=None, initial_values=None, labels=None,
                        units=None, verbose=False):
    """
    Builds the MicroDatasets for the position OR spectroscopic indices and values
    of the data.

    Parameters
    ----------
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

    Returns
    -------
    ds_spec_inds : Microdataset of numpy.uint
        Dataset containing the position indices
    ds_spec_vals : Microdataset of float
        Dataset containing the value at each position

    Notes
    -----
    `steps`, `initial_values`, `labels`, and 'units' must be the same length as
    `dimensions` when they are specified.

    Dimensions should be in the order from fastest varying to slowest.
    """
    assert contains_integers(dimensions)

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

    # Get the indices for all dimensions
    indices = make_indices_matrix(dimensions)
    assert isinstance(indices, np.ndarray)
    if verbose:
        print('Indices')
        print(indices.shape)
        print(indices)

    # Convert the indices to values
    values = initial_values + np.float32(indices)*steps

    # Create the slices that will define the labels
    if is_spectral:
        mode = 'Spectroscopic_'
        indices = indices.transpose()
        values = values.transpose()
    else:
        mode = 'Position_'

    region_slices = get_aux_dset_slicing(labels, is_spectroscopic=is_spectral)

    # Create the MicroDatasets for both Indices and Values
    ds_indices = MicroDataset(mode + 'Indices', indices, dtype=np.uint32)
    ds_indices.attrs['labels'] = region_slices

    ds_values = MicroDataset(mode + 'Values', np.float32(values), dtype=np.float32)
    ds_values.attrs['labels'] = region_slices

    ds_indices.attrs['units'] = units
    ds_values.attrs['units'] = units

    return ds_indices, ds_values


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

    indices_matrix = np.zeros(shape=(np.prod(num_steps), spat_dims), dtype=np.uint32)
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
