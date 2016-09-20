# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""
from __future__ import division # int/int = float
from os import path
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from ..io.hdfutils import getDataSet, getAuxData, findH5group
from ..analysis.be_sho_utils import getGoodLims
import h5py

###############################################################################

def rainbowPlot(ax, ao_vec, ai_vec, num_steps=32):
    """
    Plots the input against the output waveform (typically loops).
    The color of the curve changes as a function of time using the jet colorscheme

    Inputs:
    ---------
    ax : axis handle
        Axis to plot the curve
    ao_vec : 1D float numpy array
        vector that forms the X axis
    ai_vec : 1D float numpy array
        vector that forms the Y axis
    num_steps : unsigned int (Optional)
        Number of discrete color steps
    """
    pts_per_step = int(len(ai_vec) / num_steps)
    for step in xrange(num_steps - 1):
        ax.plot(ao_vec[step * pts_per_step:(step + 1) * pts_per_step],
                ai_vec[step * pts_per_step:(step + 1) * pts_per_step],
                color=plt.cm.jet(255 * step / num_steps))
    # plot the remainder:
    ax.plot(ao_vec[(num_steps - 1) * pts_per_step:],
            ai_vec[(num_steps - 1) * pts_per_step:],
            color=plt.cm.jet(255 * num_steps / num_steps))
    """
    CS3=plt.contourf([[0,0],[0,0]], range(0,310),cmap=plt.cm.jet)
    fig.colorbar(CS3)"""


###############################################################################

def plotLoops(excit_wfm, h5_loops, h5_pos=None, central_resp_size=None,
              evenly_spaced=True, plots_on_side=5, rainbow_plot=True,
              x_label='', y_label='', subtitles='Eigenvector', title=None):
    """
    Plots loops from up to 25 evenly spaced positions

    Parameters
    -----------
    excit_wfm : 1D numpy float array
        Excitation waveform in the time domain
    h5_loops : float HDF5 dataset reference or 2D numpy array
        Dataset containing data arranged as (pixel, time)
    h5_pos : HDF5 dataset reference or 2D numpy array
        Dataset containing position indices
    central_resp_size : (optional) unsigned integer
        Number of responce sample points from the center of the waveform to show in plots. Useful for SPORC
    evenly_spaced : boolean
        Evenly spaced positions or first N positions
    plots_on_side : unsigned int
        Number of plots on each side
    rainbow_plot : (optional) Boolean
        Plot the lines as a function of spectral index (eg. time)
    x_label : (optional) String
        X Label for all plots
    y_label : (optional) String
        Y label for all plots
    subtitles : (optional) String
        prefix for title over each plot
    title : (optional) String
        Main plot title

    Returns
    ---------
    fig, axes
    """

    plots_on_side = min(abs(plots_on_side), 5)
    num_pos = h5_loops.shape[0]
    sq_num_plots = min(plots_on_side, int(round(num_pos ** 0.5)))
    if evenly_spaced:
        chosen_pos = np.linspace(0, num_pos - 1, sq_num_plots ** 2, dtype=int)
    else:
        chosen_pos = np.arange(sq_num_plots ** 2, dtype=int)

    fig, axes = plt.subplots(nrows=sq_num_plots, ncols=sq_num_plots, figsize=(12, 12))
    axes_lin = axes.flat

    cent_ind = int(0.5 * h5_loops.shape[1])
    if central_resp_size:
        sz = int(0.5 * central_resp_size)
        l_resp_ind = cent_ind - sz
        r_resp_ind = cent_ind + sz
    else:
        l_resp_ind = 0
        r_resp_ind = h5_loops.shape[1]

    for count, posn in enumerate(chosen_pos):
        if rainbow_plot:
            rainbowPlot(axes_lin[count], excit_wfm[l_resp_ind:r_resp_ind], h5_loops[posn, l_resp_ind:r_resp_ind])
        else:
            axes_lin[count].plot(excit_wfm[l_resp_ind:r_resp_ind], h5_loops[posn, l_resp_ind:r_resp_ind])

        if type(h5_pos) != type(None):
            # print 'Row ' + str(h5_pos[posn,1]) + ' Col ' + str(h5_pos[posn,0])
            axes_lin[count].set_title('Row ' + str(h5_pos[posn, 1]) + ' Col ' + str(h5_pos[posn, 0]), fontsize=12)
        else:
            axes_lin[count].set_title(subtitles + ' ' + str(posn), fontsize=12)

        if count % sq_num_plots == 0:
            axes_lin[count].set_ylabel(y_label, fontsize=12)
        if count >= (sq_num_plots - 1) * sq_num_plots:
            axes_lin[count].set_xlabel(x_label, fontsize=12)
        axes_lin[count].axis('tight')
        axes_lin[count].set_aspect('auto')
        axes_lin[count].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig, axes


def plotSHOMaps(sho_maps, map_names, stdevs=2, title='', save_path=None): 
    '''
    Plots the SHO quantity maps for a single UDVS step
    
    Parameters
    ------------
    sho_maps : List of 2D numpy arrays
        Each SHO map is structured as [row, col]
    map_names: List of strings
        Titles for each of the SHO maps
    stdevs : (Optional) Unsigned int
        Number of standard deviations from the mean to be used to clip the color axis
    title : (Optional) String
        Title for the entire figure. Group name is most appropriate here
    save_path : (Optional) String
        Absolute path to write the figure to
        
    Returns
    ----------
    None
    '''
    fig,axes=plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(15, 10)) 
    
    for index, ax_hand, data_mat, qty_name in zip(range(len(map_names)), axes.flat, sho_maps, map_names):
        (amp_mean, amp_std) = getGoodLims(data_mat)          
        
        pcol0 = ax_hand.pcolor(data_mat, vmin=amp_mean-stdevs*amp_std, 
                               vmax=amp_mean+stdevs*amp_std) 
        ax_hand.axis('tight') 
        fig.colorbar(pcol0, ax=ax_hand) 
        ax_hand.set_title(qty_name) 
         
    plt.setp([ax.get_xticklabels() for ax in axes[0,:]], visible=True) 
    axes[1,2].axis('off') 
    
    plt.tight_layout()   
    if save_path:
        fig.savefig(save_path, format='png', dpi=300)


def plotVSsnapshots(resp_mat, title='', stdevs=2, save_path=None):
    '''
    Plots the spatial distribution of the response at evenly spaced UDVS steps
    
    Parameters
    -------------
    resp_mat : 3D numpy array
        SHO responses arranged as [udvs_step, rows, cols]
    title : (Optional) String
        Super title for the plots - Preferably the group name
    stdevs : (Optional) string
        Number of standard deviations from the mean to be used to clip the color axis
    save_path : (Optional) String
        Absolute path to write the figure to
        
    Returns
    ----------
    None
    '''
    
    num_udvs = resp_mat.shape[2]
    if num_udvs >= 9:
        tot_plots = 9
    elif num_udvs >= 4:
        tot_plots = 4
    else:
        tot_plots = 1
    delta_pos = int(np.ceil(num_udvs/tot_plots)) 
    
    fig, axes = plt.subplots(nrows=int(tot_plots**0.5),ncols=int(tot_plots**0.5),
                             sharex=True, sharey=True, figsize=(12, 12)) 
    if tot_plots > 1:    
        axes_lin = axes.reshape(tot_plots)
    else:
        axes_lin = axes
    
    for count, posn in enumerate(xrange(0,num_udvs, delta_pos)):
        
        snapshot = np.squeeze(resp_mat[:,:,posn])
        (amp_mean, amp_std) = getGoodLims(snapshot)
        ndims = len(snapshot.shape)
        if ndims == 2:
            axes_lin[count].imshow(snapshot, vmin=amp_mean-stdevs*amp_std, vmax=amp_mean+stdevs*amp_std)
        elif ndims == 1:
            np.clip(snapshot,amp_mean-stdevs*amp_std,amp_mean+stdevs*amp_std,snapshot)
            axes_lin[count].plot(snapshot)
        axes_lin[count].axis('tight')
        axes_lin[count].set_aspect('auto')
        axes_lin[count].set_title('UDVS Step #' + str(posn))
    
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format='png', dpi=300)

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