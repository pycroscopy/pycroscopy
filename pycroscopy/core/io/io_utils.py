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
            assert isinstance(param, (str, unicode))

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


def format_quantity(value, units, factors, decimals=2):
    """
    Formats the provided quantity such as time or size to appropriate strings

    Parameters
    ----------
    value : number
        value in some base units. For example - time in seconds
    units : array-like
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
    assert isinstance(unicode, Iterable)
    assert isinstance(factors, Iterable)
    index = None

    for index, val in enumerate(factors):
        if value < val:
            index -= 1
            break

    index = max(0, index)  # handles sub msec

    return '{} {}'.format(np.round(value / factors[index], decimals), units[index])


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
    units = ['msec', 'sec', 'min', 'hours']
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
        requested_cores = min(int(abs(requested_cores)), cpu_count())

    recom_chunks = max(int(num_jobs / requested_cores), 1)

    if not lengthy_computation:
        if requested_cores > 1 and recom_chunks < 10:
            recom_chunks = 20
            # intelligently set the cores now.
            requested_cores = max(1, min(requested_cores, int(num_jobs / recom_chunks)))
            # print('Not enough jobs per core. Reducing cores to {}'.format(recom_cores))

    return int(requested_cores)


def interpret_frequency(freq_str):
    """
    Interprets a string denoting frequency into its numerical equivalent.
    For example "4 MHz" is translated to 4E+6

    Parameters
    ----------
    freq_str : unicode / string
        Frequency as a string - eg '4 MHz'

    Returns
    -------
    frequency : float
        Frequency in hertz
    """
    components = freq_str.split()
    if components[1] == 'MHz':
        return int(components[0])*1.0E+6
    elif components[1] == 'kHz':
        return int(components[0])*1.0E+3