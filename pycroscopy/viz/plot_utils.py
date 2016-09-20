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

