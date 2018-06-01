# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:24:12 2015

@author: Carlo Dri
"""
from __future__ import division, print_function
import numpy as np

__all__ = ['gsf_read']


def gsf_read(file_name):
    """
    Read a Gwyddion Simple Field 1.0 file format
    http://gwyddion.net/documentation/user-guide-en/gsf.html
    
   Parameters
    ----------
    file_name : string
        path to the file

    Returns
    -------
    metadata : dict)
        Additional metadata to be included in the file
    data : numpy.ndarray
        An arbitrarily sized 2D array of arbitrary numeric type
    """
    if file_name.rpartition('.')[1] == '.':
        file_name = file_name[0:file_name.rfind('.')]
    
    gsf_file = open(file_name + '.gsf', 'rb')
    
    metadata = {}
    
    # check if header is OK
    if not(gsf_file.readline().decode('UTF-8') == 'Gwyddion Simple Field 1.0\n'):
        gsf_file.close()
        raise ValueError('File has wrong header')
        
    term = b'00'
    # read metadata header
    while term != b'\x00':
        line_string = gsf_file.readline().decode('UTF-8')
        metadata[line_string.rpartition(' = ')[0]] = line_string.rpartition('=')[2]
        term = gsf_file.read(1)
        gsf_file.seek(-1, 1)
    
    gsf_file.read(4 - gsf_file.tell() % 4)
    
    # fix known metadata types from .gsf file specs
    # first the mandatory ones...
    metadata['XRes'] = np.int(metadata['XRes'])
    metadata['YRes'] = np.int(metadata['YRes'])
    
    # now check for the optional ones
    if 'XReal' in metadata:
        metadata['XReal'] = np.float(metadata['XReal'])
    
    if 'YReal' in metadata:
        metadata['YReal'] = np.float(metadata['YReal'])
                
    if 'XOffset' in metadata:
        metadata['XOffset'] = np.float(metadata['XOffset'])
    
    if 'YOffset' in metadata:
        metadata['YOffset'] = np.float(metadata['YOffset'])
    
    data = np.frombuffer(gsf_file.read(), dtype='float32').reshape(metadata['YRes'], metadata['XRes'])
    
    gsf_file.close()
    
    return metadata, data
