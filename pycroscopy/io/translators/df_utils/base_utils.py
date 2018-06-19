# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:04:34 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import numpy as np
from os.path import exists

if sys.version_info.major == 3:
    unicode = str


def read_binary_data(file_path, offset, num_bytes, bytes_per_data_point):
    """
    Reads numeric data encoded in binary files. Uses instructions such as offset, length and precision

    Parameters
    ----------
    file_path : str
    offset : uint
        Byte offset into the file
    num_bytes : uint
        Number of bytes to read from the offset
    bytes_per_data_point : uint
        Precision of stored data. Currently accounting only for half-precision (16 bit / 2 byte) and full precision
        (32 bit / 4 byte) data

    Returns
    -------
    value : np.ndarray
        1D array of numeric values
    """
    if not isinstance(file_path, (str, unicode)):
        raise TypeError('file_path must be a string')
    if not isinstance(offset, int) or offset < 0:
        raise TypeError('offset must be an unsigned integer')
    if not isinstance(num_bytes, int) or num_bytes < 0:
        raise TypeError('num_bytes must be an unsigned integer')
    if not isinstance(bytes_per_data_point, int) or bytes_per_data_point < 0 or np.log2(bytes_per_data_point) % 1 != 0:
        raise TypeError('bytes_per_data_point must be an unsigned integer which is a power of 2')
    if not exists(file_path):
        raise FileNotFoundError('Provided file: {} does not exist. Check path'.format(file_path))

    if bytes_per_data_point == 2:
        dtype = 'h'
    elif bytes_per_data_point == 4:
        dtype = 'f'
    else:
        raise NotImplementedError('Currently only supporting half and full precision')

    with open(file_path, "rb") as file_handle:
        file_handle.seek(offset)
        value = np.fromstring(file_handle.read(num_bytes), dtype=dtype)
    return value
