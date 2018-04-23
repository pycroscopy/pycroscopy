# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

import h5py
import numpy as np
from collections import Iterable

__all__ = ['flatten_complex_to_real', 'get_compound_sub_dtypes', 'flatten_compound_to_real', 'check_dtype',
           'stack_real_to_complex', 'validate_dtype',
           'stack_real_to_compound', 'stack_real_to_target_dtype', 'flatten_to_real', 'contains_integers']


def contains_integers(iter_int, min_val=None):
    """
    Checks if the provided object is iterable (list, tuple etc.) and contains integers optionally greater than equal to
    the provided min_val

    Parameters
    ----------
    iter_int : Iterable of integers
        Iterable of integers
    min_val : int, optional, default = None
        The value above which each element of iterable must possess. By default, this is ignored.

    Returns
    -------
    bool
        Whether or not the provided object is an iterable of integers
    """
    if not isinstance(iter_int, Iterable):
        raise TypeError('iter_int should be an Iterable')
    if len(iter_int) == 0:
        return False
    try:
        if min_val is not None:
            if min_val % 1 != 0:
                raise ValueError('min_val should be an integer')
            return np.all([x % 1 == 0 and x >= min_val for x in iter_int])
        else:
            return np.all([x % 1 == 0 for x in iter_int])
    except TypeError:
        return False


def flatten_complex_to_real(ds_main):
    """
    Stacks the real values followed by the imaginary values in the last dimension of the given N dimensional matrix.
    Thus a complex matrix of shape (2, 3, 5) will turn into a matrix of shape (2, 3, 10)

    Parameters
    ----------
    ds_main : complex array-like or h5py.Dataset
        Dataset of interest

    Returns
    -------
    retval : ND real numpy array
    """
    if not isinstance(ds_main, (h5py.Dataset, np.ndarray)):
        raise TypeError('ds_main should either be a h5py.Dataset or numpy array')
    if not is_complex_dtype(ds_main.dtype):
        raise TypeError("Expected a complex valued matrix")

    axis = np.array(ds_main).ndim - 1
    if axis == -1:
        return np.hstack([np.real(ds_main), np.imag(ds_main)])
    else:  # along the last axis
        return np.concatenate([np.real(ds_main), np.imag(ds_main)], axis=axis)


def flatten_compound_to_real(ds_main):
    """
    Flattens the individual components in a structured array or compound valued hdf5 dataset along the last axis to form
    a real valued array. Thus a compound h5py.Dataset or structured numpy matrix of shape (2, 3, 5) having 3 components
    will turn into a real valued matrix of shape (2, 3, 15), assuming that all the sub-dtypes of the matrix are real
    valued. ie - this function does not handle structured dtypes having complex values


    Parameters
    ----------
    ds_main : numpy array that is a structured array or h5py.Dataset of compound dtype
        Dataset of interest

    Returns
    -------
    retval : n-dimensional real numpy array
        real valued dataset
    """
    if isinstance(ds_main, h5py.Dataset):
        if len(ds_main.dtype) == 0:
            raise TypeError("Expected compound h5py dataset")
        return np.concatenate([np.array(ds_main[name]) for name in ds_main.dtype.names], axis=len(ds_main.shape) - 1)
    elif isinstance(ds_main, np.ndarray):
        if len(ds_main.dtype) == 0:
            raise TypeError("Expected structured numpy array")
        if ds_main.ndim > 0:
            return np.concatenate([ds_main[name] for name in ds_main.dtype.names], axis=ds_main.ndim - 1)
        else:
            return np.hstack([ds_main[name] for name in ds_main.dtype.names])
    elif isinstance(ds_main, np.void):
        return np.hstack([ds_main[name] for name in ds_main.dtype.names])
    else:
        raise TypeError('Datatype {} not supported in struct_to_scalar'.format(type(ds_main)))


def flatten_to_real(ds_main):
    """
    Flattens complex / compound / real valued arrays to real valued arrays

    Parameters
    ----------
    ds_main : nD compound, complex or real numpy array or HDF5 dataset
        Data that could be compound, complex or real

    Returns
    ----------
    ds_main : nD numpy array
        Data raveled to a float data type
    """
    if not isinstance(ds_main, (h5py.Dataset, np.ndarray)):
        ds_main = np.array(ds_main)
    if is_complex_dtype(ds_main.dtype):
        return flatten_complex_to_real(ds_main)
    elif len(ds_main.dtype) > 0:
        return flatten_compound_to_real(ds_main)
    else:
        return ds_main


def get_compound_sub_dtypes(struct_dtype):
    """
    Returns a dictionary of the dtypes of each of the fields in the given structured array dtype

    Parameters
    ----------
    struct_dtype : numpy.dtype object
        dtype of a structured array

    Returns
    -------
    dtypes : dict
        Dictionary whose keys are the field names and values are the corresponding dtypes
    """
    if not isinstance(struct_dtype, np.dtype):
        raise TypeError('Provided object must be a structured array dtype')
    dtypes = dict()
    for field_name in struct_dtype.fields:
        dtypes[field_name] = struct_dtype.fields[field_name][0]
    return dtypes


def check_dtype(h5_dset):
    """
    Checks the datatype of the input HDF5 dataset and provides the appropriate
    function calls to convert it to a float

    Parameters
    ----------
    h5_dset : HDF5 Dataset
        Dataset of interest

    Returns
    -------
    func : function
        function that will convert the dataset to a float
    is_complex : Boolean
        is the input dataset complex?
    is_compound : Boolean
        is the input dataset compound?
    n_features : Unsigned integer, the length of the 2nd dimension of
        the data after func is called on it
    n_samples : Unsigned integer
        the length of the 1st dimension of the data
    type_mult : Unsigned integer
        multiplier that converts from the typesize of the input dtype to the
        typesize of the data after func is run on it
    """
    # TODO: avoid assuming 2d shape
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object')
    is_complex = False
    is_compound = False
    in_dtype = h5_dset.dtype
    n_samples, n_features = h5_dset.shape
    if is_complex_dtype(h5_dset.dtype):
        is_complex = True
        new_dtype = np.real(h5_dset[0, 0]).dtype
        type_mult = new_dtype.itemsize * 2
        func = flatten_complex_to_real
        n_features *= 2
    elif len(h5_dset.dtype) > 0:
        """
        Some form of structured numpy is in use
        We only support real scalars for the component types at the current time
        """
        is_compound = True
        # TODO: Avoid hard-coding to float32
        new_dtype = np.float32
        type_mult = len(in_dtype) * new_dtype(0).itemsize
        func = flatten_compound_to_real
        n_features *= len(in_dtype)
    else:
        if h5_dset.dtype not in [np.float32, np.float64]:
            new_dtype = np.float32
        else:
            new_dtype = h5_dset.dtype.type

        type_mult = new_dtype(0).itemsize

        func = new_dtype

    return func, is_complex, is_compound, n_features, n_samples, type_mult


def stack_real_to_complex(ds_real):
    """
    Puts the real and imaginary sections of the provided matrix (in the last axis) together to make complex matrix

    Parameters
    ------------
    ds_real : n dimensional real-valued numpy array or h5py.Dataset
        Data arranged as [instance, 2 x features]
        where the first half of the features are the real component and the
        second half contains the imaginary components

    Returns
    ----------
    ds_compound : 2D complex numpy array
        Data arranged as [sample, features]
    """
    if not isinstance(ds_real, (np.ndarray, h5py.Dataset)):
        if not isinstance(ds_real, Iterable):
            raise TypeError("Expected at least an iterable like a list or tuple")
        ds_real = np.array(ds_real)
    if isinstance(ds_real.dtype, np.void):
        raise TypeError("Array cannot have a compound dtype")
    if is_complex_dtype(ds_real.dtype):
        raise TypeError("Array cannot have complex dtype")

    if ds_real.shape[-1] / 2 != ds_real.shape[-1] // 2:
        raise ValueError("Last dimension must be even sized")
    half_point = ds_real.shape[-1] // 2
    return ds_real[..., :half_point] + 1j * ds_real[..., half_point:]


def stack_real_to_compound(ds_real, compound_type):
    """
    Converts a real-valued dataset to a compound dataset (along the last axis) of the provided compound d-type

    Parameters
    ------------
    ds_real : n dimensional real-valued numpy array or h5py.Dataset
        Data arranged as [instance, features]
    compound_type : dtype
        Target complex datatype

    Returns
    ----------
    ds_compound : ND complex numpy array
        Data arranged as [sample, features]
    """
    if not isinstance(ds_real, (np.ndarray, h5py.Dataset)):
        if not isinstance(ds_real, Iterable):
            raise TypeError("Expected at least an iterable like a list or tuple")
        ds_real = np.array(ds_real)
    if len(ds_real.dtype) > 0:
        raise TypeError("Array cannot have a compound dtype")
    elif is_complex_dtype(ds_real.dtype):
        raise TypeError("Array cannot have complex dtype")
    if not isinstance(compound_type, np.dtype):
        raise TypeError('Provided object must be a structured array dtype')

    new_spec_length = ds_real.shape[-1] / len(compound_type)
    if new_spec_length % 1:
        raise ValueError('Provided compound type was not compatible by number of elements')

    new_spec_length = int(new_spec_length)
    new_shape = list(ds_real.shape)  # Make mutable
    new_shape[-1] = new_spec_length
    ds_compound = np.empty(new_shape, dtype=compound_type)
    for name_ind, name in enumerate(compound_type.names):
        i_start = name_ind * new_spec_length
        i_end = (name_ind + 1) * new_spec_length
        ds_compound[name] = ds_real[..., i_start:i_end]

    return np.squeeze(ds_compound)


def stack_real_to_target_dtype(ds_real, new_dtype):
    """
    Transforms real data into the target dtype

    Parameters
    ----------
    ds_real : nD real numpy array or HDF5 dataset
        Source dataset
    new_dtype : dtype
        Target data type

    Returns
    ----------
    ret_val : nD numpy array
        Data of the target data type
    """
    if is_complex_dtype(new_dtype):
        return stack_real_to_complex(ds_real)
    elif len(new_dtype) > 0:
        return stack_real_to_compound(ds_real, new_dtype)
    else:
        return new_dtype.type(ds_real)


def validate_dtype(dtype):
    """
    Checks the provided object to ensure that it is a valid dtype that can be written to an HDF5 file.
    Raises a type error if invalid. Returns True if the object passed the tests

    Parameters
    ----------
    dtype : object
        Object that is hopefully a h5py.Datatype, np.dtype object.

    Returns
    -------
    status : bool
        True if the object was a valid dtype
    """
    if isinstance(dtype, (h5py.Datatype, np.dtype)):
        pass
    elif isinstance(np.dtype(dtype), np.dtype):
        # This should catch all those instances when dtype is something familiar like - np.float32
        pass
    else:
        raise TypeError('dtype should either be a numpy or h5py dtype')
    return True


def is_complex_dtype(dtype):
    """
    Checks if the provided dtype is a complex dtype

    Parameters
    ----------
    dtype : object
        Object that is a h5py.Datatype, np.dtype object.

    Returns
    -------
    is_complex : bool
        True if the dtype was a complex dtype. Else returns False
    """
    validate_dtype(dtype)
    if dtype in [np.complex, np.complex64, np.complex128, np.complex256]:
        return True
    return False
