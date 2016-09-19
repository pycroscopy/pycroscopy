# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith, Nouamane Laanait
"""

import os
from multiprocessing import cpu_count
from time import strftime
from PyQt4 import QtGui

def getTimeStamp():
    '''
    Teturns the current date and time as a string formatted as:
    Year_Month_Dat-Hour_Minute_Second

    Parameters
    ------
    None

    Returns
    ---------
    String
    '''
    return strftime('%Y_%m_%d-%H_%M_%S')


def uiGetFile(extension, caption='Select File'):
    '''
    Presents a File dialog used for selecting the .mat file
    and returns the absolute filepath of the selecte file\n

    Parameters
    ---------
    extension : String or list of strings
        file extensions to look for
    caption : (Optional) String
        Title for the file browser window

    Returns
    --------
    file_path : String
        Absolute path of the chosen file
    '''

    return QtGui.QFileDialog.getOpenFileName(caption=caption, filter=extension)


def getAvailableMem():
    '''
    Returns the available memory

    Chris Smith -- csmith55@utk.edu

    Parameters
    ------
    None

    Returns
    --------
    mem : unsigned int
        Memory in bytes
    '''
    from psutil import virtual_memory as vm
    mem = vm()
    return getattr(mem, 'available')


def recommendCores(num_jobs, requested_cores=None):
    '''
    Decides the number of cores to use for parallel computing

    Parameters
    ----------
    num_jobs : unsigned int
        Number of times a parallel operation needs to be performed
    requested_cores : unsigned int (Optional. Default = None)
        Number of logical cores to use for computation

    Returns
    --------
    requested_cores : unsigned int
        Number of logical cores to use for computation
    '''

    max_cores = max(1, cpu_count() - 2)

    if requested_cores == None:
        # conservative allocation
        requested_cores = max_cores
    else:
        # Respecting the explicit request
        requested_cores = min(int(abs(requested_cores)), cpu_count())

    recom_chunks = int(num_jobs / requested_cores)

    if requested_cores > 1 and recom_chunks < 10:
        recom_chunks = 20
        # intelligently set the cores now.
        requested_cores = min(requested_cores, int(num_jobs / recom_chunks))
        # print 'Not enough jobs per core. Reducing cores to', recom_cores

    return requested_cores
