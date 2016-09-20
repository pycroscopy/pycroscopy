# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""

import numpy as np
import matplotlib.pyplot as plt


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
