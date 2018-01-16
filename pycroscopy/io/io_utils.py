# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import os
from multiprocessing import cpu_count
from time import strftime
from psutil import virtual_memory as vm
from warnings import warn
import h5py
import numpy as np
import itertools

__all__ = ['getAvailableMem', 'getTimeStamp', 'transformToTargetType', 'transformToReal',
           'complex_to_float', 'compound_to_scalar', 'realToComplex', 'realToCompound', 'check_dtype',
           'recommendCores', 'uiGetFile']


def check_ssh():
    return 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ


def uiGetFile(file_filter='H5 file (*.h5)', caption='Select File'):
    """
    Presents a File dialog used for selecting the .mat file
    and returns the absolute filepath of the selecte file\n

    Parameters
    ----------
    file_filter : String or list of strings
        file extensions to look for
    caption : (Optional) String
        Title for the file browser window

    Returns
    -------
    file_path : String
        Absolute path of the chosen file
    """
    # Only try to use the GUI options if not over an SSH connection.
    if not check_ssh():
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            warn('The required package PyQt5 could not be imported.\n',
                 'The code will check for PyQt4.')
        except:
            raise
        else:
            app = QtWidgets.QApplication([])
            path = QtWidgets.QFileDialog.getOpenFileName(caption=caption, filter=file_filter)[0]
            app.closeAllWindows()
            app.exit()
            del app

            return str(path)

        try:
            from PyQt4 import QtGui
        except ImportError:
            warn('PyQt4 also not found.  Will use standard text input.')
        except:
            raise
        else:
            app = QtGui.QApplication([])
            path = QtGui.QFileDialog.getOpenFileName(caption=caption, filter=file_filter)
            app.exit()
            del app

            return str(path)

    path = input('Enter path to datafile.  Raw Data (*.txt, *.mat, *.xls, *.xlsx) or Translated file (*.h5)')

    return str(path)


def getTimeStamp():
    """
    Teturns the current date and time as a string formatted as:
    Year_Month_Dat-Hour_Minute_Second

    Parameters
    ----------

    Returns
    -------
    String
    """
    return strftime('%Y_%m_%d-%H_%M_%S')


def getAvailableMem():
    """
    Returns the available memory

    Chris Smith -- csmith55@utk.edu

    Parameters
    ----------

    Returns
    -------
    mem : unsigned int
        Memory in bytes
    """
    import sys
    mem = vm().available

    if sys.maxsize <= 2 ** 32:
        mem = min([mem, sys.maxsize])

    return mem


def recommendCores(num_jobs, requested_cores=None, lengthy_computation=False):
    """
    Decides the number of cores to use for parallel computing

    Parameters
    ----------
    num_jobs : unsigned int
        Number of times a parallel operation needs to be performed
    requested_cores : unsigned int (Optional. Default = None)
        Number of logical cores to use for computation
    lengthy_computation : Boolean (Optional. Default = False)
        Whether or not each computation takes a long time. If each computation is quick, it may not make sense to take
        a hit in terms of starting and using a larger number of cores, so use fewer cores instead.
        Eg- BE SHO fitting is fast (<1 sec) so set this value to False,
        Eg- Bayesian Inference is very slow (~ 10-20 sec)so set this to True

    Returns
    -------
    requested_cores : unsigned int
        Number of logical cores to use for computation
    """

    max_cores = max(1, cpu_count() - 2)

    if requested_cores is None:
        # conservative allocation
        requested_cores = max_cores
    else:
        # Respecting the explicit request
        requested_cores = min(int(abs(requested_cores)), cpu_count())

    recom_chunks = max(int(num_jobs / requested_cores), 1)

    if not lengthy_computation:
        if requested_cores > 1 and recom_chunks < 10:
            recom_chunks = 20
            # intelligently set the cores now.
            requested_cores = min(requested_cores, int(num_jobs / recom_chunks))
            # print('Not enough jobs per core. Reducing cores to {}'.format(recom_cores))

    return requested_cores


def complex_to_float(ds_main):
    """
    Function to convert a complex ND numpy array or HDF5 dataset into a scalar dataset

    Parameters
    ----------
    ds_main : complex ND numpy array or ND HDF5 dataset
        Dataset of interest

    Returns
    -------
    retval : ND real numpy array
    """
    return np.hstack([np.real(ds_main), np.imag(ds_main)])


def compound_to_scalar(ds_main):
    """
    Converts a compound ND numpy array or HDF5 dataset into a real scalar dataset

    Parameters
    ----------
    ds_main : ND numpy array or ND HDF5 dataset object of compound datatype
        Dataset of interest

    Returns
    -------
    retval : ND real numpy array

    """
    if isinstance(ds_main, h5py.Dataset):
        return np.hstack([np.float32(ds_main[name]) for name in ds_main.dtype.names])
    elif isinstance(ds_main, np.ndarray):
        return np.hstack([ds_main[name] for name in ds_main.dtype.names])
    else:
        raise TypeError('Datatype {} not supported in compound_to_scalar'.format(type(ds_main)))


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
    is_complex = False
    is_compound = False
    in_dtype = ds_main.dtype
    new_dtype = ds_main.dtype
    n_samples, n_features = ds_main.shape
    if ds_main.dtype in [np.complex64, np.complex128, np.complex]:
        is_complex = True
        new_dtype = np.real(ds_main[0, 0]).dtype
        type_mult = new_dtype.itemsize * 2
        func = complex_to_float
        n_features *= 2
    elif len(ds_main.dtype) > 0:
        """
        Some form of compound datatype is in use
        We only support real scalars for the component types at the current time
        """
        is_compound = True
        new_dtype = np.float32
        type_mult = len(in_dtype) * new_dtype(0).itemsize
        func = compound_to_scalar
        n_features *= len(in_dtype)
    else:
        if ds_main.dtype not in [np.float32, np.float64]:
            new_dtype = np.float32
        else:
            new_dtype = ds_main.dtype.type

        type_mult = new_dtype(0).itemsize

        func = new_dtype

    return func, is_complex, is_compound, n_features, n_samples, type_mult


def realToComplex(ds_real):
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


def realToCompound(ds_real, compound_type):
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
    new_spec_length = ds_real.shape[1] / len(compound_type)
    if new_spec_length % 1:
        raise TypeError('Provided compound type was not compatible by numbr of elements')

    new_spec_length = int(new_spec_length)
    ds_compound = np.empty([ds_real.shape[0], new_spec_length], dtype=compound_type)
    for iname, name in enumerate(compound_type.names):
        istart = iname * ds_compound.shape[1]
        iend = (iname + 1) * ds_compound.shape[1]
        ds_compound[name] = ds_real[:, istart:iend]

    return np.squeeze(ds_compound)


def transformToTargetType(ds_real, new_dtype):
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
        return realToComplex(ds_real)
    elif len(new_dtype) > 0:
        return realToCompound(ds_real, new_dtype)
    else:
        return new_dtype.type(ds_real)


def transformToReal(ds_main):
    """
    Transforms real data into the target dtype

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
        return complex_to_float(ds_main)
    elif len(ds_main.dtype) > 0:
        return compound_to_scalar(ds_main)
    else:
        return ds_main


def to_ranges(iterable):
    """
    Converts a sequence of iterables to range tuples
    
    From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
    
    Credits: @juanchopanza and @luca
    
    Parameters
    ----------
    iterable : collections.Iterable object
        iterable object like a list
    
    Returns
    -------
    iterable : generator object
        Cast to list or similar to use
    """
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]
