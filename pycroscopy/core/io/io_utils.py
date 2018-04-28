# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import os
import sys
from collections import Iterable
from multiprocessing import cpu_count
from time import strftime
from psutil import virtual_memory as vm
from warnings import warn
import numpy as np

if sys.version_info.major == 3:
    unicode = str

__all__ = ['get_available_memory', 'get_time_stamp', 'recommend_cpu_cores', 'file_dialog', 'format_quantity',
           'format_time', 'format_size']


def check_ssh():
    """
    Checks whether or not the python kernel is running locally (False) or remotely (True)

    Returns
    -------
    output : bool
        Whether or not the kernel is running over SSH (remote machine)
    """
    return 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ


def file_dialog(file_filter='H5 file (*.h5)', caption='Select File'):
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
    for param in [file_filter, caption]:
        if param is not None:
            if not isinstance(param, (str, unicode)):
                raise TypeError('param must be a string')

    # Only try to use the GUI options if not over an SSH connection.
    if not check_ssh():
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            warn('The required package PyQt5 could not be imported.\n',
                 'The code will check for PyQt4.')

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

        else:
            app = QtGui.QApplication([])
            path = QtGui.QFileDialog.getOpenFileName(caption=caption, filter=file_filter)
            app.exit()
            del app

            return str(path)

    path = input('Enter path to datafile.  Raw Data (*.txt, *.mat, *.xls, *.xlsx) or Translated file (*.h5)')

    return str(path)


def get_time_stamp():
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


def format_quantity(value, unit_names, factors, decimals=2):
    """
    Formats the provided quantity such as time or size to appropriate strings

    Parameters
    ----------
    value : number
        value in some base units. For example - time in seconds
    unit_names : array-like
        List of names of units for each scale of the value
    factors : array-like
        List of scaling factors for each scale of the value
    decimals : uint, optional. default = 2
        Number of decimal places to which the value needs to be formatted

    Returns
    -------
    str
        String with value formatted correctly
    """
    # assert isinstance(value, (int, float))
    if not isinstance(unit_names, Iterable):
        raise TypeError('unit_names must an Iterable')
    if not isinstance(factors, Iterable):
        raise TypeError('factors must be an Iterable')
    if len(unit_names) != len(factors):
        raise ValueError('unit_names and factors must be of the same length')
    if not np.all([isinstance(_, (str, unicode)) for _ in unit_names]):
        raise TypeError('unit_names must be strings')
    index = None

    for index, val in enumerate(factors):
        if value < val:
            index -= 1
            break

    index = max(0, index)  # handles sub msec

    return '{} {}'.format(np.round(value / factors[index], decimals), unit_names[index])


def format_time(time_in_seconds, decimals=2):
    """
    Formats the provided time in seconds to seconds, minutes, or hours

    Parameters
    ----------
    time_in_seconds : number
        Time in seconds
    decimals : uint, optional. default = 2
        Number of decimal places to which the time needs to be formatted

    Returns
    -------
    str
        String with time formatted correctly
    """
    units = ['msec', 'sec', 'mins', 'hours']
    factors = [0.001, 1, 60, 3600]
    return format_quantity(time_in_seconds, units, factors, decimals=decimals)


def format_size(size_in_bytes, decimals=2):
    """
    Formats the provided size in bytes to kB, MB, GB, TB etc.

    Parameters
    ----------
    size_in_bytes : number
        size in bytes
    decimals : uint, optional. default = 2
        Number of decimal places to which the size needs to be formatted

    Returns
    -------
    str
        String with size formatted correctly
    """
    units = ['bytes', 'kB', 'MB', 'GB', 'TB']
    factors = 1024 ** np.arange(len(units))
    return format_quantity(size_in_bytes, units, factors, decimals=decimals)


def get_available_memory():
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


def recommend_cpu_cores(num_jobs, requested_cores=None, lengthy_computation=False):
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
        requested_cores = max(min(int(abs(requested_cores)), cpu_count()), 1)

    jobs_per_core = max(int(num_jobs / requested_cores), 1)
    min_jobs_per_core = 20  # I don't like to hard-code things here but I don't have a better idea for now

    if not lengthy_computation:
        if requested_cores > 1 and jobs_per_core < min_jobs_per_core:
            # cut down the number of cores if there are too few jobs
            jobs_per_core = 2 * min_jobs_per_core
            # intelligently set the cores now.
            requested_cores = max(1, min(requested_cores, int(num_jobs / jobs_per_core)))
            # print('Not enough jobs per core. Reducing cores to {}'.format(recom_cores))

    return int(requested_cores)


def formatted_str_to_number(str_val, magnitude_names, magnitude_values, separator=' '):
    """
    Takes a formatted string like '4.32 MHz' to 4.32 E+6

    Parameters
    ----------
    str_val : str / unicode
        String value of the quantity. Example '4.32 MHz'
    magnitude_names : Iterable
        List of names of units like ['seconds', 'minutes', 'hours']
    magnitude_values : Iterable
        List of values (corresponding to magnitude_names) that scale the numeric value. Example [1, 60, 3600]
    separator : str / unicode, optional. Default = ' ' (space)
        The text that separates the numeric value and the units.

    Returns
    -------
    number
        Numeric value of the string
    """
    if not isinstance(str_val, (str, unicode)):
        raise TypeError('str_val must be a string')
    if not isinstance(separator, (str, unicode)):
        raise TypeError('separator must be a string')
    if not isinstance(magnitude_names, Iterable):
        raise TypeError('magnitude_names must be an Iterable')
    if not isinstance(magnitude_values, Iterable):
        raise TypeError('magnitude_values must be an Iterable')
    if not np.all([isinstance(_, (str, unicode)) for _ in magnitude_names]):
        raise TypeError('magnitude_names should contain strings')
    if len(magnitude_names) != len(magnitude_values):
        raise ValueError('magnitude_names and magnitude_values should be of the same length')

    components = str_val.split(separator)
    if len(components) != 2:
        raise ValueError('String value should be of format "123.45<separator>Unit')

    for unit_name, scaling in zip(magnitude_names, magnitude_values):
        if unit_name == components[1]:
            # Let it raise an exception. Don't catch
            return scaling * float(components[0])
