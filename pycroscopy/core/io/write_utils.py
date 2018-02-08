import numpy as np
from collections import Iterable
from .microdata import MicroDataset
import warnings

__all__ = ['build_ind_val_dsets', 'get_position_slicing', 'get_spectral_slicing', 'make_indices_matrix']


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
    assert isinstance(dimensions, Iterable)

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
        warnings.warn('Arbitrary names provided to dimensions. Please provide legitimate values for parameter - labels')
        labels = ['Unknown Dimension {}'.format(ind) for ind in range(len(dimensions))]
    else:
        assert isinstance(labels, Iterable)
        if len(labels) != len(dimensions):
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
        region_slices = get_spectral_slicing(labels)
    else:
        mode = 'Position_'
        region_slices = get_position_slicing(labels)

    # Create the MicroDatasets for both Indices and Values
    ds_indices = MicroDataset(mode + 'Indices', indices, dtype=np.uint32)
    ds_indices.attrs['labels'] = region_slices

    ds_values = MicroDataset(mode + 'Values', np.float32(values), dtype=np.float32)
    ds_values.attrs['labels'] = region_slices

    if units is None:
        warnings.warn('Arbitrary units provided to dimensions. Please provide legitimate values for parameter - units')
        units = ['Arb Unit {}'.format(ind) for ind in range(len(dimensions))]
    else:
        assert isinstance(units, Iterable)
        if len(units) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')

    ds_indices.attrs['units'] = units
    ds_values.attrs['units'] = units

    return ds_indices, ds_values


def get_position_slicing(pos_lab, curr_pix=None):
    """
    Returns a dictionary of slice objects to help in creating region references
    to the position indices and values H5 datasets

    Parameters
    ------------
    pos_lab : List of strings
        Labels of each of the position axes
    curr_pix : (Optional) unsigned int
        Last pixel in the positon matrix. Useful in experiments where the
        parameters have changed (eg. BEPS new data format)

    Returns
    ------------
    slice_dict : dictionary
        Dictionary of tuples containing slice objects corresponding to
        each position axis.
    """
    assert isinstance(pos_lab, Iterable)

    slice_dict = dict()
    for spat_ind, spat_dim in enumerate(pos_lab):
        slice_dict[spat_dim] = (slice(curr_pix), slice(spat_ind, spat_ind+1))
    return slice_dict


def get_spectral_slicing(spec_lab, curr_spec=None):
    """
    Returns a dictionary of slice objects to help in creating region references
    to the spectroscopic indices and values H5 datasets

    Parameters
    ------------
    spec_lab : List of strings
        Labels of each of the Spectroscopic axes
    curr_spec : (Optional) unsigned int
        Last position in the spectroscopic matrix. Useful in experiments where the
        parameters have changed (eg. BEPS new data format)

    Returns
    ------------
    slice_dict : dictionary
        Dictionary of tuples containing slice objects corresponding to
        each Spectroscopic axis.
    """
    assert isinstance(spec_lab, Iterable)

    slice_dict = dict()
    for spat_ind, spat_dim in enumerate(spec_lab):
        slice_dict[spat_dim] = (slice(spat_ind, spat_ind + 1), slice(curr_spec))
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
    assert isinstance(num_steps, Iterable)
    # assert np.all([isinstance(x, int) for x in num_steps])

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
