# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:10:37 2015

@author: Suhas Somnath
"""
import numpy as np

from .hdf_utils import getAuxData

__all__ = [
    'maxReadPixels', 'getActiveUDVSsteps', 'getDataIndicesForUDVSstep', 'getForExcitWfm',
    'getIndicesforPlotGroup', 'getSliceForExcWfm', 'generateTestSpectroscopicData', 'getSpecSliceForUDVSstep',
    'isSimpleDataset', 'reshapeToNsteps', 'reshapeToOneStep'
    ]


def maxReadPixels(max_memory, tot_pix, bins_per_step, bytes_per_bin=4):
    """
    Calculates the maximum number of pixels that can be loaded into the 
    specified memory size. This is particularly useful when applying a 
    (typically parallel) operation / processing on each pixel. 
    Example - Fitting response to a model.
    
    Parameters
    -------------
    max_memory : unsigned int
        Maximum memory (in bytes) that can be used. 
        For example 4 GB would be = 4*((2**10)**3) bytes
    tot_pix : unsigned int
        Total number of pixels in dataset
    bins_per_step : unsigned int
        Number of bins that will be read (can be portion of each pixel)
    bytes_per_bin : (Optional) unsigned int
        size of each bin - set to 4 bytes
    
    Returns
    -------
    max_pix : unsigned int
        Maximum number of pixels that will be loaded
    """
    # alternatively try .nbytes
    bytes_per_step = bins_per_step * bytes_per_bin
    max_pix = np.rint(max_memory / bytes_per_step)
    #print('Allowed to read {} of {} pixels'.format(max_pix,tot_pix))
    max_pix = max(1,min(tot_pix, max_pix))
    return np.uint(max_pix)

def getActiveUDVSsteps(h5_raw):
    """
    Returns all the active UDVS steps in the data
    
    Parameters
    ----------
    h5_raw : HDF5 dataset reference
        Reference to the raw data
        
    Returns
    -----------
    steps : 1D numpy array
        Active UDVS steps
    """
    udvs_step_vec = getAuxData(h5_raw, auxDataName =['UDVS_Indices'])[0].value
    return np.unique(udvs_step_vec)
    
def getSliceForExcWfm(h5_bin_wfm, excit_wfm):
    """
    Returns the indices that correspond to the given excitation waveform
    that can be used to slice the bin datasets
    * Developer note - Replace the first parameter with the Raw_Data dataset
    
    Parameters
    ----------------
    h5_bin_wfm : Reference to HDF5 dataset
        Bin Waveform Indices
    excit_wfm : integer
        excitation waveform / wave type
    
    Returns
    --------------
    slc : slice object
        Slice with the start and end indices
    """
    temp = np.where(h5_bin_wfm.value == excit_wfm)[0]
    return slice(temp[0],temp[-1]+1) # Need to add one additional index otherwise, the last index will be lost
    
def getDataIndicesForUDVSstep(h5_udvs_inds, udvs_step_index):
    """
    Returns the spectroscopic indices that correspond to the given udvs_step_index
    that can be used to slice the main data matrix.
    * Developer note - Replace the first parameter with the Raw_Data dataset
    
    Parameters
    -------------
    h5_udvs_inds : Reference to HDF5 dataset 
        UDVS_Indices dataset
    udvs_step_index : usigned int
        UDVS step index (base 0)
    
    Returns
    --------------
    ans : 1D numpy array 
        Spectroscopic indices
    """
    spec_ind_udvs_step_col = h5_udvs_inds[h5_udvs_inds.attrs.get('UDVS_Step')]
    return np.where(spec_ind_udvs_step_col == udvs_step_index)[0]
    
def getSpecSliceForUDVSstep(h5_udvs_inds, udvs_step_index):
    """
    Returns the spectroscopic indices that correspond to the given udvs_step_index
    that can be used to slice the main data matrix
    * Developer note - Replace the first parameter with the Raw_Data dataset
    
    Parameters
    -------------
    h5_udvs_inds : Reference to HDF5 dataset
        UDVS_Indices dataset
    udvs_step_index : unsigned int
        UDVS step index (base 0)
    
    Returns
    ----------
    slc : slice object
        Object containing the start and end indices
    """
    temp = np.where(h5_udvs_inds.value == udvs_step_index)[0]
    return slice(temp[0],temp[-1]+1) # Need to add one additional index otherwise, the last index will be lost
    

def getForExcitWfm(h5_main, h5_other, wave_type):
    """
    Slices the provided H5 dataset by the provided wave type. 
    Note that this is applicable to only certain H5 datasets such as the bin frequences, bin FFT etc.
    
    Parameters
    ----------
    h5_main : Reference to HDF5 dataset 
        Raw_Data dataset
    h5_other :Reference to HDF5 dataset 
        The dataset that needs to be sliced such as bin frequencies
    wave_type : unsigned int
        Excitation waveform type

    Returns
    ---------
    freq_vec : 1D numpy array
        data specific to specified excitation waveform
    """    
    h5_bin_wfm_type = getAuxData(h5_main, auxDataName=['Bin_Wfm_Type'])[0]
    inds = np.where(h5_bin_wfm_type.value == wave_type)[0]
    return h5_other[slice(inds[0],inds[-1]+1)]

    
def getIndicesforPlotGroup(h5_udvs_inds, ds_udvs, plt_grp_name):
    """
    For a provided plot group name in the udvs table, this function 
    returns the corresponding spectroscopic indices that can be used to index / slice the main data set
    and the data within the udvs table for the requested plot group
    * Developer note - Replace the first parameter with the Raw_Data dataset
        
    Parameters
    ------------
    h5_udvs_inds : Reference to HDF5 dataset
        containing the UDVS indices
    ds_udvs : Reference to HDF5 dataset 
        containing the UDVS table
    plt_grp_name : string 
        name of the plot group in the UDVS table
            
    Returns
    -----------
    step_bin_indices : 2D numpy array
        Indices arranged as [step, bin] in the spectroscopic_indices table
        This is useful for knowing the number of bins and steps in this plot group.
        We are allowed to assume that the number of bins does NOT change within the plot group
    oneD_indices : 1D numpy array
        spectroscopic indices corresponding to the requested plot group
    udvs_plt_grp_col : 1D numpy array
        data contained within the udvs table for the requested plot group        
    """
    
    # working on the UDVS table first:
    # getting the numpy array corresponding the requested plot group
    udvs_col_data = np.squeeze(ds_udvs[ds_udvs.attrs.get(plt_grp_name)])
    # All UDVS steps that are NOT part of the plot grop are empty cells in the table
    # and hence assume a nan value.
    # getting the udvs step indices that belong to this plot group:
    step_inds = np.where(np.isnan(udvs_col_data) == False)[0]    
    # Getting the values in that plot group that were non NAN
    udvs_plt_grp_col = udvs_col_data[step_inds]
    
    #---------------------------------
    
    # Now we use the udvs step indices calculated above to get 
    # the indices in the spectroscopic indices table
    spec_ind_udvs_step_col = h5_udvs_inds[h5_udvs_inds.attrs.get('UDVS_Step')]
    num_bins = len(np.where(spec_ind_udvs_step_col == step_inds[0])[0])
    # Stepehen says that we can assume that the number of bins will NOT change in a plot group
    step_bin_indices = np.zeros(shape=(len(step_inds),num_bins), dtype=int)

    for indx, step in enumerate(step_inds):
        step_bin_indices[indx,:] = np.where(spec_ind_udvs_step_col == step)[0]
    
    oneD_indices = step_bin_indices.reshape((step_bin_indices.shape[0]*step_bin_indices.shape[1]))
    return (step_bin_indices, oneD_indices, udvs_plt_grp_col)
    
def reshapeToOneStep(raw_mat, num_steps):
    """
    Reshapes provided data from (pos, step * bin) to (pos * step, bin). 
    This is useful when unraveling data for parallel processing.
    
    Parameters
    -------------
    raw_mat : 2D numpy array
        Data organized as (positions, step * bins)
    num_steps : unsigned int
        Number of spectroscopic steps per pixel (eg - UDVS steps)
        
    Returns
    --------------
    twoD : 2D numpy array
        Data rearranged as (positions * step, bin)
    """
    num_pos = raw_mat.shape[0]
    num_bins = int(raw_mat.shape[1]/num_steps)
    oneD = raw_mat
    oneD = oneD.reshape((num_bins * num_steps * num_pos))
    twoD = oneD.reshape((num_steps * num_pos, num_bins))
    return twoD
    
def reshapeToNsteps(raw_mat, num_steps):
    """
    Reshapes provided data from (positions * step, bin) to (positions, step * bin).  
    Use this to restructure data back to its original form after parallel computing 
    
    Parameters
    --------------
    raw_mat : 2D numpy array 
        Data organized as (positions * step, bin)
    num_steps : unsigned int
         Number of spectroscopic steps per pixel (eg - UDVS steps)
        
    Returns
    --------------- 
    twoD : 2D numpy array
        Data rearranged as (positions, step * bin)
    """
    num_bins = raw_mat.shape[1]
    num_pos = int(raw_mat.shape[0]/num_steps)
    oneD = raw_mat
    oneD = oneD.reshape(num_bins * num_steps * num_pos)
    twoD = oneD.reshape((num_pos, num_steps * num_bins)) 
    return twoD
    
def generateTestSpectroscopicData(num_bins=7, num_steps=3, num_pos=4):
    """
    Generates a (preferably small) test data set using the given parameters.
    Data is filled with indices (with base 1 for simplicity). 
    Use this for testing reshape operations etc.
    
    Parameters
    ----------
    num_bins : unsigned int (Optional. Default = 7)
        Number of bins
    num_steps : unsigned int (Optional. Default = 3)
        Number of spectroscopic steps  
    num_pos : unsigned int (Optional. Default = 4)
        Number of fictional positions
    
    Returns
    --------------
    full_data : 2D numpy array
        Data organized as [steps x bins, positions]
    """
    full_data = np.zeros((num_steps * num_bins, num_pos))
    for pos in xrange(num_pos):
        bin_count=0
        for step in xrange(num_steps):
            for bind in xrange(num_bins):
                full_data[bin_count,pos] = (pos+1)*100 + (step+1)*10 + (bind+1)
                bin_count+=1
    return full_data


def isSimpleDataset(h5_main, isBEPS=True):
    """
    This function figures out if a single number defines the bins for all UDVS steps
    In such cases (udvs_steps x bins, pos) can be reshaped to (bins, positions x steps)
    for (theoretically) faster computation, especially for large datasets

    Actually, things are a lot simpler. Only need to check if number of bins for all excitation waveforms are equal
    
    Parameters
    -------------
    h5_main : Reference to HDF5 dataset
        Raw_Data dataset
    isBEPS : Boolean (default = True)
        Whether or not this dataset is BEPS
        
    Returns
    ----------
    data_type : Boolean
        Whether or not this dataset can be unraveled / flattened
    """
    
    if isBEPS:
        if h5_main.parent.parent.attrs['VS_mode'] in ['DC modulation mode','AC modulation mode with time reversal','current mode','Relaxation']:
            # I am pretty sure that AC modulation also is simple
            return True
        else:
            # Could be user defined or some other kind I am not aware of
            # In many cases, some of these datasets could also potentially be simple datasets
            ds_udvs = getAuxData(h5_main, auxDataName=['UDVS'])[0]                
            excit_wfms = ds_udvs[ds_udvs.attrs.get('wave_mod')]
            wfm_types = np.unique(excit_wfms)
            if len(wfm_types) == 1:
                # BEPS with single excitation waveform
                print('Single BEPS excitation waveform')
                return True
            else:
                # Multiple waveform types here
                harm_types = np.unique(np.abs(wfm_types))
                if len(harm_types) != 1:
                    # eg - excitaiton waveforms 1, 2, 3 NOT -1, +1
                    return False
                # In this case a single excitation waveform with forward and reverse was used.
                h5_bin_wfm_type = getAuxData(h5_main, auxDataName=['Bin_Wfm_Type'])[0]
                # Now for each wfm type, count number of bins.
                wfm_bin_count = []
                for wfm in wfm_types:
                    wfm_bin_count.append(np.where(h5_bin_wfm_type.value == wfm)[0])
                wfm_lengths = np.unique(np.array(wfm_bin_count))
                if len(wfm_lengths) == 1:
                    # BEPS with multiple excitation waveforms but each excitation waveform has same number of bins
                    print('All BEPS excitation waves have same number of bins')
                    return True
            return False   
    else:
        # BE-Line
        return True


def isReshapable(h5_main, step_start_inds=None):
    """
    A BE dataset is said to be reshape-able if the number of bins per steps is constant. Even if the dataset contains
    multiple excitation waveforms (harmonics), We know that the measurement is always at the resonance peak, so the
    frequency vector should not change.

    Parameters
    ----------
    h5_main : h5py.Dataset object
        Reference to the main dataset
    step_start_inds : list or 1D array
        Indices that correspond to the start of each BE pulse / UDVS step

    Returns
    ---------
    reshapable : Boolean
        Whether or not the number of bins per step are constant in this dataset
    """
    if step_start_inds is None:
        h5_spec_inds = getAuxData(h5_main, auxDataName=['Spectroscopic_Indices'])[0]
        step_start_inds = np.where(h5_spec_inds[0] == 0)[0]
    # Adding the size of the main dataset as the last (virtual) step
    step_start_inds = np.hstack((step_start_inds, h5_main.shape[1]))
    num_bins = np.diff(step_start_inds)
    step_types = np.unique(num_bins)
    return len(step_types) == 1