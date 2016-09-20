# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 16:08:20 2016

@author: Suhas Somnath
"""

from __future__ import division # int/int = float
from os import path
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from ..io.hdf_utils import getDataSet, getAuxData, findH5group
import h5py

#%%
def plotLoops(dc_vec, resp_mat, x_label='', y_label='', title=None, save_path=None):
    ''' 
    Plots BE loops from up to 9 positions (evenly separated)
    
    Parameters
    -----------
    dc_vec : 1D numpy array
        X axis - DC offset / AC amplitude
    resp_mat : real 2D numpy array
        containing quantity such as amplitude or phase organized as 
        [position, spectroscopic index]
    x_label : (optional) String
        X Label for all plots
    y_label : (optional) String
        Y label for all plots
    title : (optional) String
        Main plot title
    save_path : (Optional) String
        Absolute path to write the figure to
        
    Returns
    -----------
    None
    '''
    num_pos = resp_mat.shape[0]
    if num_pos >= 9:
        tot_plots = 9
    elif num_pos >= 4:
        tot_plots = 4
    else:
        tot_plots = 1
    delta_pos = int(np.ceil(num_pos/tot_plots))
    
    fig, axes = plt.subplots(nrows=int(tot_plots**0.5),ncols=int(tot_plots**0.5),
                             figsize=(12, 12))
    if tot_plots > 1:    
        axes_lin = axes.reshape(tot_plots)
    else:
        axes_lin = axes
    
    for count, posn in enumerate(xrange(0,num_pos, delta_pos)):
        
        axes_lin[count].plot(dc_vec,np.squeeze(resp_mat[posn,:]))
        axes_lin[count].set_title('Pixel #' + str(posn))
        axes_lin[count].set_xlabel(x_label) 
        axes_lin[count].set_ylabel(y_label)
        axes_lin[count].axis('tight')
        axes_lin[count].set_aspect('auto')
    
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='png', dpi=300)


def getGoodLims(resp_mat):
    '''
    Returns the mean and standard deviation of the provided numpy array        
    
    Parameters
    ------------
    resp_mat : numpy ndarray
        N dimensional array containing homogenous data
        
    Returns
    ---------
    mean: float
        Mean of the complete dataset       
    std: float
        Standard deviation of the dataset
    '''
    return np.mean(resp_mat), np.std(resp_mat)
    


    
