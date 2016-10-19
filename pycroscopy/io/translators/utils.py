# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:11:33 2015
Basic functions that can be used by any SPM translator class

@author: Suhas Somnath
"""

from __future__ import division; # int/int = float
import numpy as np; # For array operations
import time as tm; # for getting time stamps


def interpretFreq(freq_str):
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


def generateDummyMainParms():
    """
    Generates a (dummy) dictionary of parameters that will be used at the root level of the h5 file
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    main_parms : dictionary
        Dictionary containing basic descriptors that describe a dataset
    """        
    main_parms = {};   
    main_parms['translate_date'] = tm.strftime("%Y_%m_%d");
    main_parms['instrument'] = 'cypher_west';
    main_parms['xcams_id'] = 'abc';
    main_parms['user_name'] = 'John Doe';
    main_parms['sample_name'] = 'PZT';
    main_parms['sample_description'] = 'Thin Film';
    main_parms['project_name'] = 'Band Excitation';
    main_parms['project_id'] = 'CNMS_2015B_X0000';
    main_parms['comments'] = 'Band Excitation data';
    main_parms['data_tool'] = 'be_analyzer';
    # This parameter actually need not be a dummy and can be extracted from the parms file
    main_parms['experiment_date'] = '2015-10-05 14:55:05';
    main_parms['experiment_unix_time'] = tm.time();         
    # Need to fill in the x and y grid size here
    main_parms['grid_size_x'] = 1;
    main_parms['grid_size_y'] = 1;        
    # Need to fill in the current X, Y, Z, Laser position here
    main_parms['current_position_x'] = 1;
    main_parms['current_position_y'] = 1;
    
    return main_parms;
    
def makePositionMat(num_steps):
    """
    Sets the position index matrices and labels for each of the spatial dimensions.
    It is intentionally generic so that it works for any SPM dataset.
    
    Parameters
    ------------
    num_steps : List / numpy array
        Steps in each spatial direction
    Note: that the axes must be ordered from fastest varying to slowest varying

    Returns
    --------------
    pos_mat : 2D unsigned int numpy array
        arranged as [steps, spatial dimension]  
    """

    num_steps = np.array(num_steps); 
    spat_dims = len(np.where(num_steps > 1)[0]);
    
    pos_mat = np.zeros(shape=(np.prod(num_steps),spat_dims),dtype=np.uint32);
    pos_ind = 0;
    
    for indx, curr_steps in enumerate(num_steps):
        if curr_steps > 1:
            
            part1 = np.prod(num_steps[:indx+1])
            
            if indx > 0:
                part2 = np.prod(num_steps[:indx])
            else:
                part2 = 1
            
            if indx+1 == len(num_steps):
                part3 = 1
            else:
                part3 = np.prod(num_steps[indx+1:])           
                                
            pos_mat[:,pos_ind] = np.tile(np.floor(np.arange(part1)/(part2)),part3)
            pos_ind+= 1
    
    return pos_mat


def getPositionSlicing(pos_lab, curr_pix=None):
    """
    Returns a dictionary of slice objects to help in creating region references 
    to the position indices and values H5 datasets 
    
    Parameters
    ------------
    pos_lab : List of strings
        Labels of each of the position axes
    curr_pix : (Optional) unsigned int
        Last pixel in the positon matrix. Useful in experiments where the 
        parameters have changed (eg. BEPS new data format)
    
    Returns
    ------------
    slice_dict : dictionary
        Dictionary of tuples containing slice objects corresponding to 
        each position axis.
    """
    slice_dict = dict();
    for spat_ind, spat_dim in enumerate(pos_lab):
        slice_dict[spat_dim[0]] = (slice(curr_pix), slice(spat_ind,spat_ind+1))
    return slice_dict


def getSpectralSlicing(spec_lab, curr_spec=None):
    """
    Returns a dictionary of slice objects to help in creating region references
    to the spectroscopic indices and values H5 datasets

    Parameters
    ------------
    spec_lab : List of strings
        Labels of each of the Spectroscopic axes
    curr_spec : (Optional) unsigned int
        Last position in the spectroscopic matrix. Useful in experiments where the
        parameters have changed (eg. BEPS new data format)

    Returns
    ------------
    slice_dict : dictionary
        Dictionary of tuples containing slice objects corresponding to
        each Spectroscopic axis.
    """
    slice_dict = dict()
    for spat_ind, spat_dim in enumerate(spec_lab):
        slice_dict[spat_dim[0]] = (slice(spat_ind, spat_ind + 1), slice(curr_spec))
    return slice_dict

