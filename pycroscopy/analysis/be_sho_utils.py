# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 16:08:20 2016

@author: Suhas Somnath
"""

from __future__ import division # int/int = float

from os import path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

# TODO: Get rid of relative imports.
from ..io.hdf_utils import getDataSet, getAuxData, findH5group
from ..viz.plot_utils import plotVSsnapshots, plotSHOMaps
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


# def getGoodLims(resp_mat):
#     '''
#     Returns the mean and standard deviation of the provided numpy array        
#     
#     Parameters
#     ------------
#     resp_mat : numpy ndarray
#         N dimensional array containing homogenous data
#         
#     Returns
#     ---------
#     mean: float
#         Mean of the complete dataset       
#     std: float
#         Standard deviation of the dataset
#     '''
#     return np.mean(resp_mat), np.std(resp_mat)
#     

#%%

def visualizeSHOResults(h5_path, mode=0, save_plots=True, show_plots = True):
    '''
    Plots some loops, amplitude, phase maps for BE-Line and BEPS datasets.\n
    Note: The file MUST contain SHO fit gusses at the very least
    
    Parameters:
    ------------
    h5_path : String
        absolute path of the h5 file containing SHO fit guesses
    mode : unsigned integer (0 or 1)
        0 - SHO Guess, 1 - SHO Fit
    save_plots : (Optional) Boolean
        Whether or not to save plots to files in the same directory as the h5 file
    show_plots : (Optional) Boolean
        Whether or not to display the plots on the screen
    
    Returns:
    ---------
    None
    '''

    h5_file = h5py.File(h5_path, mode ='r')
    
    expt_type = h5_file.attrs['data_type']
    if expt_type not in ['BEPSData','BELineData']:
        warn('Unsupported data format')
        return
    isBEPS = expt_type == 'BEPSData' 
    
    (folder_path, basename) = path.split(h5_path)
    basename = basename[:-3] # remove .h5
    
    datasets = getDataSet(h5_file,'Raw_Data')
    
    plt_path = None
        
    for h5_main in datasets:
        
        chan_grp = h5_main.parent
        grp_name = '_'.join(chan_grp.name[1:].split('/')) + '_'
        try:
            num_rows = len(np.unique(chan_grp['Position_Indices'][:,0]))
        except:
            num_rows = np.floor((np.sqrt(h5_main.size)))
        try:
            num_cols = len(np.unique(chan_grp['Position_Indices'][:,1]))
        except:
            num_cols = np.size(chan_grp['Position_Indices'])/num_rows
        
        sho_grp = findH5group(h5_main,'SHO_Fit')[-1]
        
        if mode == 0:
            ds_sho = sho_grp['Guess']
            grp_name+= 'Guess'
        else:
            ds_sho = sho_grp['Fit']
            grp_name+= 'Fit'
        
        # Assume that there's enough memory to load all the guesses into memory
        amp_mat = ds_sho['Amplitude [V]']*1000 # convert to mV ahead of time
        freq_mat = ds_sho['Frequency [Hz]']/1000 
        q_mat = ds_sho['Quality Factor']
        phase_mat = ds_sho['Phase [rad]']
        rsqr_mat = ds_sho['R2 Criterion']
        
        if isBEPS:
            meas_type = chan_grp.parent.attrs['VS_mode']
            # basically 3 kinds for now - DC/current, AC, UD - lets ignore this
            if meas_type == 'load user defined VS Wave from file':
                warn('Not handling custom experiments for now')
                h5_file.close()
                return
                
            # Plot amplitude and phase maps at one or more UDVS steps
                    
            h5_udvs = (getAuxData(h5_main, auxDataName =['UDVS'])[0])        
            
            if meas_type == 'AC modulation mode with time reversal':
                center = int(h5_udvs.shape[0]*0.5)
                ac_vec = h5_udvs[h5_udvs.attrs['ac_amp']][0:center]            
                forw_resp = np.squeeze(amp_mat[:,slice(0,center)])
                plt_title = grp_name + '_Forward_Loops'
                if save_plots: 
                    plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')                
                plotLoops(ac_vec, forw_resp, 'AC Amplitude' , 'Amplitude',title=plt_title, save_path=plt_path)
                rev_resp = np.squeeze(amp_mat[:,slice(center,None)])
                plt_title = grp_name + '_Reverse_Loops'
                if save_plots: 
                    plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                plotLoops(ac_vec, rev_resp, 'AC Amplitude' , 'Amplitude',title=plt_title, save_path=plt_path)
                plt_title = grp_name + '_Forward_Snaps'
                if save_plots: 
                    plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                plotVSsnapshots(forw_resp.reshape(num_rows,num_cols,forw_resp.shape[1]), title=plt_title, save_path=plt_path)
                plt_title = grp_name + '_Reverse_Snaps'
                if save_plots: 
                    plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                plotVSsnapshots(rev_resp.reshape(num_rows,num_cols,rev_resp.shape[1]), title=plt_title, save_path=plt_path)
            else: 
                # plot loops at a few locations
                dc_vec = np.squeeze(h5_udvs[h5_udvs.attrs['dc_offset']])
                dc_vec = dc_vec[slice(0,None,2)] # Need to take only half as many 
                if chan_grp.parent.attrs['VS_measure_in_field_loops'] == 'in and out-of-field':
                    in_phase = np.squeeze(phase_mat[:,slice(0,None,2)])
                    in_amp = np.squeeze(amp_mat[:,slice(0,None,2)])
                    plt_title = grp_name + '_In_Field_Loops'
                    if save_plots: 
                        plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                    plotLoops(dc_vec, in_phase*in_amp, 'DC Bias' , 'Piezoresponse (a.u.)',title=plt_title, save_path=plt_path)
                    out_phase = np.squeeze(phase_mat[:,slice(1,None,2)])
                    out_amp = np.squeeze(amp_mat[:,slice(1,None,2)])
                    plt_title = grp_name + '_Out_of_Field_Loops'
                    if save_plots: 
                        plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                    plotLoops(dc_vec, out_phase*out_amp, 'DC Bias' , 'Piezoresponse (a.u.)',title=plt_title, save_path=plt_path)
                    # print 'trying to reshape', in_phase.shape, 'into', in_phase.shape[0],',',num_rows,',',num_cols
                    plt_title = grp_name + '_In_Field_Snaps'
                    if save_plots: 
                        plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                    plotVSsnapshots(in_phase.reshape(num_rows,num_cols,in_phase.shape[1]), title=plt_title, save_path=plt_path)
                    plt_title = grp_name + '_Out_of_Field_Snaps'
                    if save_plots: 
                        plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                    plotVSsnapshots(out_phase.reshape(num_rows,num_cols,out_phase.shape[1]), title=plt_title, save_path=plt_path)
                else:
                    plt_title = grp_name + '_Loops'
                    if save_plots: 
                        plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                    plotLoops(dc_vec, phase_mat*amp_mat, 'DC Bias' , 'Piezoresponse (a.u.)', title=plt_title, save_path=plt_path)
                    plt_title = grp_name + '_Snaps'
                    if save_plots: 
                        plt_path = path.join(folder_path,basename + '_' + plt_title + '.png')
                    plotVSsnapshots(phase_mat.reshape(num_rows,num_cols,phase_mat.shape[1]), title=plt_title, save_path=plt_path)
            
        else: # BE-Line can only visualize the amplitude and phase maps:
            amp_mat = amp_mat.reshape(num_rows,num_cols)
            freq_mat = freq_mat.reshape(num_rows,num_cols)  
            q_mat = q_mat.reshape(num_rows,num_cols)  
            phase_mat = phase_mat.reshape(num_rows,num_cols) 
            rsqr_mat = rsqr_mat.reshape(num_rows,num_cols)  
            if save_plots: 
                plt_path = path.join(folder_path,basename + '_' + grp_name + 'Maps.png')
            plotSHOMaps([amp_mat*1E+3, freq_mat, q_mat, phase_mat, rsqr_mat],
                        ['Amplitude (mV)', 'Frequency (kHz)','Quality Factor', 
                        'Phase (deg)', 'R^2 Criterion'], title=grp_name, save_path=plt_path) 
    
    if show_plots:
        plt.show()
    
    plt.close('all')
    
    h5_file.close()
    
