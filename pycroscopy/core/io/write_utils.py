import sys

import numpy as np
from collections import Iterable
import numbers

from .dtype_utils import contains_integers

__all__ = ['clean_string_att', 'get_aux_dset_slicing', 'make_indices_matrix',
           'INDICES_DTYPE', 'VALUES_DTYPE']

if sys.version_info.major == 3:
    unicode = str

INDICES_DTYPE = np.uint32
VALUES_DTYPE = np.float32


class AuxillaryDescriptor(object):
    def __init__(self, dim_sizes, dim_names, dim_units, dim_step_sizes=None, dim_initial_vals=None):
        lengths = []
        for val, elem_type, required in zip([dim_sizes, dim_names, dim_units, dim_step_sizes, dim_initial_vals],
                                            [0, 2, 2, 1, 1],
                                            [True, True, True, False, False]):
            if not required and val is None:
                continue
            assert isinstance(val, (list, tuple, np.ndarray))
            lengths.append(len(val))
            if elem_type == 2:
                assert np.all([isinstance(_, (str, unicode)) for _ in val])
            elif elem_type == 0:
                assert contains_integers(val, min_val=2)
            else:
                assert np.all([isinstance(_, numbers.Number) for _ in val])
        num_elems = np.unique(lengths)
        if len(num_elems) != 1:
            raise ValueError('All the arguments should have the same number of elements')
        if num_elems[0] == 0:
            raise ValueError('Argument should not be empty')


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


