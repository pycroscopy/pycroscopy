# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Chris Smith
"""
import os
from . import nanonispy as nap

# TODO: Consider moving this to NanonisTranslator
def read_nanonis_file(file_path):
    """
    Function to read the nanonis data file and return the header dictionary containing the parameters
    and the signal dictionary containing the data

    Parameters
    ----------
    file_path : str
        Path to the file to be read

    Returns
    -------
    header : dict
        Dictionary of parameters
    signals : dict
        Dictionary of data

    """

    file_path = os.path.abspath(file_path)
    folder, filename = os.path.split(file_path)
    basename, file_ext = os.path.splitext(filename)

    if file_ext == '.3ds':
        reader = nap.read.Grid
    elif file_ext == '.sxm':
        reader = nap.read.Scan
    elif file_ext == '.dat':
        reader = nap.read.Spec
    else:
        raise ValueError("Nanosis file must be either '.3ds', '.sxm', or '.dat'.")

    data = reader(file_path)

    return data.header, data.signals

