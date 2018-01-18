# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

import h5py
import numpy as np

__all__ = ['complex_to_real', 'get_compound_subdtypes', 'compound_to_real', 'check_dtype', 'real_to_complex',
           'real_to_compound', 'transform_to_target_dtype', 'transform_to_real']


def complex_to_real(ds_main):
    """
    Stacks the real values followed by the imaginary values in an N dimensional matrix

    Parameters
    ----------
    ds_main : complex array-like or h5py.Dataset
        Dataset of interest

    Returns
    -------
    retval : ND real numpy array
    """
    return np.hstack([np.real(ds_main), np.imag(ds_main)])


def get_compound_subdtypes(struct_dtype):
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
    dtypes = dict()
    for field_name in struct_dtype.fields:
        dtypes[field_name] = struct_dtype.fields[field_name][0]
    return dtypes


def compound_to_real(ds_main):
    """
    Stacks the individual components in a structured array or compound valued hdf5 dataset to form a real valued array

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
        # TODO: Avoid hard-coding to float32
        return np.hstack([np.float32(ds_main[name]) for name in ds_main.dtype.names])
    elif isinstance(ds_main, np.ndarray):
        return np.hstack([ds_main[name] for name in ds_main.dtype.names])
    else:
        raise TypeError('Datatype {} not supported in struct_to_scalar'.format(type(ds_main)))


def check_dtype(ds_main):
    """
    Checks the datatype of the input dataset and provides the appropriate
    function calls to convert it to a float

    Parameters
    ----------
    ds_main : HDF5 Dataset
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
    # TODO: Avoid hard-coding to float32
    is_complex = False
    is_compound = False
    in_dtype = ds_main.dtype
    new_dtype = ds_main.dtype
    n_samples, n_features = ds_main.shape
    if ds_main.dtype in [np.complex64, np.complex128, np.complex]:
        is_complex = True
        new_dtype = np.real(ds_main[0, 0]).dtype
        type_mult = new_dtype.itemsize * 2
        func = complex_to_real
        n_features *= 2
    elif len(ds_main.dtype) > 0:
        """
        Some form of structured numpy is in use
        We only support real scalars for the component types at the current time
        """
        is_compound = True
        new_dtype = np.float32
        type_mult = len(in_dtype) * new_dtype(0).itemsize
        func = compound_to_real
        n_features *= len(in_dtype)
    else:
        if ds_main.dtype not in [np.float32, np.float64]:
            new_dtype = np.float32
        else:
            new_dtype = ds_main.dtype.type

        type_mult = new_dtype(0).itemsize

        func = new_dtype

    return func, is_complex, is_compound, n_features, n_samples, type_mult


def real_to_complex(ds_real):
    """
    Puts the real and imaginary sections together to make complex dataset

    Parameters
    ------------
    ds_real : 2D real numpy array or HDF5 dataset
        Data arranged as [instance, 2 x features]
        where the first half of the features are the real component and the
        second half contains the imaginary components

    Returns
    ----------
    ds_compound : 2D complex numpy array
        Data arranged as [sample, features]
    """
    return ds_real[:, :int(0.5 * ds_real.shape[1])] + 1j * ds_real[:, int(0.5 * ds_real.shape[1]):]


def real_to_compound(ds_real, compound_type):
    """
    Converts a real dataset to a compound dataset of the provided compound d-type

    Parameters
    ------------
    ds_real : 2D real numpy array or HDF5 dataset
        Data arranged as [instance, features]
    compound_type : dtype
        Target complex datatype

    Returns
    ----------
    ds_compound : 2D complex numpy array
        Data arranged as [sample, features]
    """
    # TODO: More robust check to ensure that we are not inserting int string / boolean / other valued dtypes
    # TODO: Handle inserting into complex valued dtypes
    new_spec_length = ds_real.shape[1] / len(compound_type)
    if new_spec_length % 1:
        raise TypeError('Provided compound type was not compatible by number of elements')

    new_spec_length = int(new_spec_length)
    ds_compound = np.empty([ds_real.shape[0], new_spec_length], dtype=compound_type)
    for iname, name in enumerate(compound_type.names):
        istart = iname * ds_compound.shape[1]
        iend = (iname + 1) * ds_compound.shape[1]
        ds_compound[name] = ds_real[:, istart:iend]

    return np.squeeze(ds_compound)


def transform_to_target_dtype(ds_real, new_dtype):
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
    if new_dtype in [np.complex64, np.complex128, np.complex]:
        return real_to_complex(ds_real)
    elif len(new_dtype) > 0:
        return real_to_compound(ds_real, new_dtype)
    else:
        return new_dtype.type(ds_real)


def transform_to_real(ds_main):
    """
    Transforms complex / compound / real valued arrays to real valued arrays

    Parameters
    ----------
    ds_main : nD compound, complex or real numpy array or HDF5 dataset
        Data that could be compound, complex or real

    Returns
    ----------
    ds_main : nD numpy array
        Data raveled to a float data type
    """
    if ds_main.dtype in [np.complex64, np.complex128, np.complex]:
        return complex_to_real(ds_main)
    elif len(ds_main.dtype) > 0:
        return compound_to_real(ds_main)
    else:
        return ds_main
