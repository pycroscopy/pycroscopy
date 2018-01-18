"""
Created on Mar 1, 2016

@author: Chris Smith -- cmsith55@utk.edu
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import sys

if sys.version_info.major == 3 and sys.version_info.minor == 6:
    disable_histogram = True
else:
    disable_histogram = False
    from numpy_groupies import aggregate_np


def build_histogram(x_hist, data_mat, N_x_bins, N_y_bins, weighting_vec=1, min_resp=None, max_resp=None, func=None,
                    debug=False, *args, **kwargs):
    """
    Creates histogram for a single block of pixels

    Parameters
    ----------
    x_hist : 1D numpy array
        bins for x-axis of 2d histogram
    data_mat : numpy array
        data to be binned for y-axis of 2d histogram
    weighting_vec : 1D numpy array or float
        weights. If setting all to one value, can be a scalar
    N_x_bins : integer
        number of bins in the x-direction
    N_y_bins : integer
        number of bins in the y-direction
    min_resp : float
        minimum value for y binning
    max_resp : float
        maximum value for y binning
    func : function
        function to be used to bin data_vec.  All functions should take as input data_vec.
        Arguments should be passed properly to func.  This has not been heavily tested.
    debug : bool, optional
        If True, extra debugging statements are printed.  Default False

    Returns
    -------
    pixel_hist : 2D numpy array
        contains the histogram of the input data

    Apply func to input data, convert to 1D array, and normalize
    """
    if func is not None:
        y_hist = func(data_mat, *args, **kwargs)
    else:
        y_hist = data_mat

    '''
    Get the min_resp and max_resp from y_hist if they are none
    '''
    if min_resp is None:
        min_resp = np.min(y_hist)
    if max_resp is None:
        max_resp = np.max(y_hist)
    if debug:
        print('min_resp', min_resp, 'max_resp', max_resp)

    y_hist = __scale_and_discretize(y_hist, N_y_bins, max_resp, min_resp, debug)

    '''
    Combine x_hist and y_hist into one matrix
    '''
    if debug:
        print(np.shape(x_hist))
        print(np.shape(y_hist))

    try:
        group_idx = np.zeros((2, x_hist.size), dtype=np.int32)
        group_idx[0, :] = x_hist
        group_idx[1, :] = y_hist
    except:
        raise

    '''
    Aggregate matrix for histogram of current chunk
    '''
    if debug:
        print(np.shape(group_idx))
        print(np.shape(weighting_vec))
        print(N_x_bins, N_y_bins)

    try:
        if not disable_histogram:
            pixel_hist = aggregate_np(group_idx, weighting_vec, func='sum', size=(N_x_bins, N_y_bins), dtype=np.int32)
        else:
            pixel_hist = None
    except:
        raise

    return pixel_hist


def __scale_and_discretize(y_hist, N_y_bins, max_resp, min_resp, debug=False):
    """
    Normalizes and discretizes the `y_hist` array 
    
    Parameters
    ----------
    y_hist : numpy.ndarray
    N_y_bins : int
    max_resp : float
    min_resp : float
    debug : bool

    Returns
    -------
    y_hist numpy.ndarray
    """
    y_hist = y_hist.flatten()
    y_hist = np.clip(y_hist, min_resp, max_resp)
    y_hist = np.add(y_hist, -min_resp)
    y_hist = np.dot(y_hist, 1.0 / (max_resp - min_resp))
    '''
    Discretize y_hist
    '''
    y_hist = np.rint(y_hist * (N_y_bins - 1))
    if debug:
        print('ymin', min(y_hist), 'ymax', max(y_hist))

    return y_hist
