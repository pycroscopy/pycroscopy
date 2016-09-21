# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""
from __future__ import division # int/int = float

from warnings import warn

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from ..analysis.utils.be_loop import loopFitFunction


def plotLoopFitNGuess(Vdc, ds_proj_loops, ds_guess, ds_fit, title=''):
    '''
    Plots the loop guess, fit, source projected loops for a single cycle

    Parameters
    ----------
    Vdc - 1D float numpy array
        DC offset vector (unshifted)
    ds_proj_loops - 2D numpy array
        Projected loops arranged as [position, Vdc]
    ds_guess - 1D compound numpy array
        Loop guesses arranged as [position]
    ds_fit - 1D compound numpy array
        Loop fits arranged as [position]
    title - (Optional) String / unicode
        Title for the figure

    Returns
    ----------
    fig - matplotlib.pyplot.figure object
        Figure handle
    axes - 2D array of matplotlib.pyplot.axis handles
        handles to axes in the 2d figure
    '''
    shift_ind = int(-1 * len(Vdc) / 4)
    Vdc_shifted = np.roll(Vdc, shift_ind)

    num_plots = np.min([5, int(np.sqrt(ds_proj_loops.shape[0]))])
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots, figsize=(18, 18))
    positions = np.linspace(0, ds_proj_loops.shape[0] - 1, num_plots ** 2, dtype=np.int)
    for ax, pos in zip(axes.flat, positions):
        ax.plot(Vdc, ds_proj_loops[pos, :], 'k', label='Raw')
        ax.plot(Vdc_shifted, loopFitFunction(Vdc_shifted, np.array(list(ds_guess[pos]))), 'g', label='guess')
        ax.plot(Vdc_shifted, loopFitFunction(Vdc_shifted, np.array(list(ds_fit[pos]))), 'r--', label='Fit')
        ax.set_xlabel('V_DC (V)')
        ax.set_ylabel('PR (a.u.)')
        ax.set_title('Loop ' + str(pos))
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes

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
    """
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
    """
    fig,axes=plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(15, 10)) 
    
    for index, ax_hand, data_mat, qty_name in zip(range(len(map_names)), axes.flat, sho_maps, map_names):
        amp_mean = np.mean(data_mat)
        amp_std = np.std(data_mat)          
        
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
    """
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
    """
    
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
        amp_mean = np.mean(snapshot) 
        amp_std = np.std(snapshot)
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


def plotSpectrograms(eigenvectors, num_comps=4, title='Eigenvectors', xlabel='Step', stdevs=2,
                     show_colorbar=True):
    """
    Plots the provided spectrograms from SVD V vector

    Parameters:
    -------------
    eigenvectors : 3D numpy complex matrices
        Eigenvectors rearranged as - [row, col, component]


    xaxis : 1D real numpy array
        The vector to plot against
    num_comps : int
        Number of components to plot
    title : String
        Title to plot above everything else
    xlabel : String
        Label for x axis
    stdevs : int
        Number of standard deviations to consider for plotting

    Returns:
    ---------
    fig, axes
    """
    import matplotlib.pyplot as plt
    fig_h, fig_w = (4, 4 + show_colorbar * 1.00)
    p_rows = int(np.ceil(np.sqrt(num_comps)))
    p_cols = int(np.floor(num_comps / p_rows))
    fig201, axes201 = plt.subplots(p_rows, p_cols, figsize=(p_cols * fig_w, p_rows * fig_h))
    fig201.subplots_adjust(hspace=0.4, wspace=0.4)
    fig201.canvas.set_window_title(title)

    for index in xrange(num_comps):
        cur_map = np.transpose(eigenvectors[index, :, :])
        ax = axes201.flat[index]
        mean = np.mean(cur_map)
        std = np.std(cur_map)
        ax.imshow(cur_map, cmap='jet',
                  vmin=mean - stdevs * std,
                  vmax=mean + stdevs * std)
        ax.set_title('Eigenvector: %d' % (index + 1))
        ax.set_aspect('auto')
        ax.set_xlabel(xlabel)
        ax.axis('tight')

    return fig201, axes201


###############################################################################

def plotBEspectrograms(eigenvectors, num_comps=4, title='Eigenvectors', xlabel='UDVS Step', stdevs=2):
    """
    Plots the provided spectrograms from SVD V vector

    Parameters:
    -------------
    eigenvectors : 3D numpy complex matrices
        Eigenvectors rearranged as - [row, col, component]


    xaxis : 1D real numpy array
        The vector to plot against
    num_comps : int
        Number of components to plot
    title : String
        Title to plot above everything else
    xlabel : String
        Label for x axis
    stdevs : int
        Number of standard deviations to consider for plotting

    Returns:
    ---------
    fig, axes
    """
    fig201, axes201 = plt.subplots(2, num_comps, figsize=(4 * num_comps, 8))
    fig201.subplots_adjust(hspace=0.4, wspace=0.4)
    fig201.canvas.set_window_title(title)

    for index in xrange(num_comps):
        cur_map = np.transpose(eigenvectors[index, :, :])
        axes = [axes201.flat[index], axes201.flat[index + num_comps]]
        funcs = [np.abs, np.angle]
        labels = ['Amplitude', 'Phase']
        for func, lab, ax in zip(funcs, labels, axes):
            amp_mean = np.mean(func(cur_map))
            amp_std = np.std(func(cur_map))
            ax.imshow(func(cur_map), cmap='inferno',
                      vmin=amp_mean - stdevs * amp_std,
                      vmax=amp_mean + stdevs * amp_std)
            ax.set_title('Eigenvector: %d - %s' % (index + 1, lab))
            ax.set_aspect('auto')
        ax.set_xlabel(xlabel)

    return fig201, axes201


###############################################################################

def plotBEeigenvectors(eigenvectors, num_comps=4, xlabel=''):
    """
    Plots the provided spectrograms from SVD V vector

    Parameters:
    -------------
    eigenvectors : 3D numpy complex matrices
        Eigenvectors rearranged as - [row, col, component]


    xaxis : 1D real numpy array
        The vector to plot against
    num_comps : int
        Number of components to plot
    title : String
        Title to plot above everything else
    xlabel : String
        Label for x axis
    stdevs : int
        Number of standard deviations to consider for plotting

    Returns:
    ---------
    fig, axes
    """
    funcs = [np.abs, np.angle]
    labels = ['Amplitude', 'Phase']

    fig201, axes201 = plt.subplots(len(funcs), num_comps, figsize=(num_comps * 4, 4 * len(funcs)))
    fig201.subplots_adjust(hspace=0.4, wspace=0.4)
    fig201.canvas.set_window_title("Eigenvectors")

    for index in xrange(num_comps):
        cur_map = eigenvectors[index, :]
        #         axes = [axes201.flat[index], axes201.flat[index+num_comps], axes201.flat[index+2*num_comps], axes201.flat[index+3*num_comps]]
        axes = [axes201.flat[index], axes201.flat[index + num_comps]]
        for func, lab, ax in zip(funcs, labels, axes):
            ax.plot(func(cur_map))
            ax.set_title('Eigenvector: %d - %s' % (index + 1, lab))
        ax.set_xlabel(xlabel)
    fig201.tight_layout()

    return fig201, axes201


###############################################################################

def plotBELoops(xaxis, xlabel, amp_mat, phase_mat, num_comps, title=None):
    """
    Plots the provided loops from the SHO. Replace / merge with function in BESHOUtils

    Parameters:
    -------------
    xaxis : 1D real numpy array
        The vector to plot against
    xlabel : string
        Label for x axis
    amp_mat : 2D real numpy array
        Amplitude matrix arranged as [points, component]
    phase_mat : 2D real numpy array
        Phase matrix arranged as [points, component]
    num_comps : int
        Number of components to plot
    title : String
        Title to plot above everything else

    Returns:
    ---------
    fig, axes
    """
    fig201, axes201 = plt.subplots(2, num_comps, figsize=(4 * num_comps, 6))
    fig201.subplots_adjust(hspace=0.4, wspace=0.4)
    fig201.canvas.set_window_title(title)

    for index in xrange(num_comps):
        axes = [axes201.flat[index], axes201.flat[index + num_comps]]
        resp_vecs = [amp_mat[index, :], phase_mat[index, :]]
        resp_titles = ['Amplitude', 'Phase']

        for ax, resp, titl in zip(axes, resp_vecs, resp_titles):
            ax.plot(xaxis, resp)
            ax.set_title('%s %d' % (titl, index + 1))
            ax.set_aspect('auto')
            ax.set_xlabel(xlabel)

    fig201.tight_layout()
    return fig201, axes201


###############################################################################

def plotScree(S, title='Scree'):
    """
    Plots the S or scree

    Parameters:
    -------------
    S : 1D real numpy array
        The S vector from SVD

    Returns:
    ---------
    fig, axes
    """
    fig203 = plt.figure(figsize=(6.5, 6))
    axes203 = fig203.add_axes([0.1, 0.1, .8, .8])  # left, bottom, width, height (range 0 to 1)
    axes203.loglog(np.arange(len(S)) + 1, S, 'b', marker='*')
    axes203.set_xlabel('Principal Component')
    axes203.set_ylabel('Variance')
    axes203.set_title(title)
    axes203.set_xlim(left=1, right=len(S))
    axes203.set_ylim(bottom=np.min(S), top=np.max(S))
    fig203.canvas.set_window_title("Scree")

    return fig203, axes203


###############################################################################

def plotLoadingMaps(loadings, num_comps=4, stdevs=2, colormap='jet', show_colorbar=True):
    """
    Plots the provided loading maps

    Parameters:
    -------------
    loadings : 3D real numpy array
        structured as [rows, cols, component]
    num_comps : int
        Number of components to plot
    stdevs : int
        Number of standard deviations to consider for plotting
    colormap : string or object from matplotlib.colors (Optional. Default = jet or rainbow)
        Colormap for the plots
    show_colorbar : Boolean (Optional. Default = True)
        Whether or not to show the color bar

    Returns:
    ---------
    fig, axes
    """
    fig_h, fig_w = (4, 4 + show_colorbar * 1.00)
    p_rows = int(np.ceil(np.sqrt(num_comps)))
    p_cols = int(np.floor(num_comps / p_rows))
    fig202, axes202 = plt.subplots(p_cols, p_rows, figsize=(p_cols * fig_w, p_rows * fig_h))
    fig202.subplots_adjust(hspace=0.4, wspace=0.4)
    fig202.canvas.set_window_title("Loading Maps")

    for index in xrange(num_comps):
        cur_map = loadings[:, :, index]
        amp_mean = np.mean(cur_map)
        amp_std = np.std(cur_map)
        if show_colorbar:
            pcol0 = axes202.flat[index].pcolor(cur_map, vmin=amp_mean - stdevs * amp_std,
                                               vmax=amp_mean + stdevs * amp_std)
            fig202.colorbar(pcol0, ax=axes202.flat[index])
            axes202.flat[index].axis('tight')
        else:
            axes202.flat[index].imshow(cur_map, cmap=colormap,
                                       interpolation='none',
                                       vmin=amp_mean - stdevs * amp_std,
                                       vmax=amp_mean + stdevs * amp_std)

        axes202.flat[index].set_title('Loading %d' % (index + 1))
        axes202.flat[index].set_aspect('auto')
    fig202.tight_layout()

    return fig202, axes202


###############################################################################

def plotKMeansResults(label_mat, cluster_centroids, spec_val=None, cmap=plt.cm.jet,
                      spec_label='Spectroscopic Value', resp_label='Response'):
    """
    label_mat : 2D int numpy array or h5py.Dataset object
                Spatial map of cluster labels structured as [rows, cols]
    cluster_centroids : 2D array or h5py.Dataset object
                Centroids arranged as [cluster number, features]
    spec_val : (Optional) 1D float numpy array or h5py.Dataset object
                X axis to plot the centroids against
                If no value is specified, the data is plotted against the index
    cmap : plt.cm object (Optional. Default = plt.cm.jet)
                Colormap to use for the labels map and the centroid.
                Advised to pick a map where the centroid plots show clearly
    spec_label : String (Optional. Default = 'Spectroscopic Value')
                Label to use for X axis on cluster centroid plot
    resp_label : String (Optional. Default = 'Response')
                Label to use for Y axis on cluster centroid plot
    """

    def __plotCentroids(centroids, ax, spec_val, spec_label, y_label, cmap, title=None):
        num_clusters = centroids.shape[0]
        for clust in xrange(num_clusters):
            ax.plot(spec_val, centroids[clust],
                    label='Cluster {}'.format(clust),
                    color=cmap(int(255 * clust / (num_clusters - 1))))
        ax.set_ylabel(y_label)
        ax.legend(loc='best')
        if title:
            ax.set_title(title)
            ax.set_xlabel(spec_label)

    if type(spec_val) == type(None):
        spec_val = np.arange(cluster_centroids.shape[1])

    if cluster_centroids.dtype in [np.complex64, np.complex128, np.complex]:
        fig = plt.figure(figsize=(12, 8))
        ax_map = plt.subplot2grid((2, 10), (0, 0), colspan=6, rowspan=2)
        ax_amp = plt.subplot2grid((2, 10), (0, 6), colspan=4)
        ax_phase = plt.subplot2grid((2, 10), (1, 6), colspan=4)
        axes = [ax_map, ax_amp, ax_phase]

        __plotCentroids(np.abs(cluster_centroids), ax_amp, spec_val, spec_label,
                        resp_label + ' - Amplitude', cmap, 'Centroids')
        __plotCentroids(np.angle(cluster_centroids), ax_phase, spec_val, spec_label,
                        resp_label + ' - Phase', cmap)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax_map = axes[0]
        __plotCentroids(cluster_centroids, axes[1], spec_val, spec_label,
                        resp_label, cmap, 'Centroids')

    num_clusters = cluster_centroids.shape[0]
    if isinstance(label_mat, h5py.Dataset):
        """
        Reshape label_mat based on linked positions
        """
        pos = label_mat.file[label_mat.attrs['Position_Indices']]
        nx = len(np.unique(pos[:, 0]))
        ny = len(np.unique(pos[:, 1]))
        label_mat = label_mat[()].reshape(nx, ny)
    im = ax_map.imshow(label_mat, interpolation='none')
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # space for colorbar
    fig.colorbar(im, cax=cax)
    # pcol0 = ax_map.pcolor(label_mat, cmap=cmap)
    # fig.colorbar(pcol0, ax=ax_map, ticks=np.arange(num_clusters))
    ax_map.axis('tight')
    ax_map.set_title('Cluster Label Map')

    fig.tight_layout()
    fig.suptitle('k-Means result')
    fig.canvas.set_window_title('k-Means result')

    return fig, axes


###############################################################################

def plotKMeansClusters(label_mat, cluster_centroids,
                       num_cluster=4):
    """
    Plots the provided label mat and centroids
    from KMeans clustering

    Parameters:
    -------------
    label_mat : 2D int numpy array
                structured as [rows, cols]
    cluster_centroids: 2D real numpy array
                       structured as [cluster,features]
    num_cluster : int
                Number of centroids to plot

    Returns:
    ---------
    fig
    """

    if num_cluster < 5:

        fig501 = plt.figure(figsize=(20, 10))
        fax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
        fax2 = plt.subplot2grid((2, 4), (0, 2))
        fax3 = plt.subplot2grid((2, 4), (0, 3))
        fax4 = plt.subplot2grid((2, 4), (1, 2))
        fax5 = plt.subplot2grid((2, 4), (1, 3))
        fig501.tight_layout()
        axes_handles = [fax1, fax2, fax3, fax4, fax5]

    else:
        fig501 = plt.figure(figsize=(20, 10))
        # make subplot for cluster map
        fax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=3)  # For cluster map
        fax1.set_xmargin(0.50)
        # make subplot for cluster centers
        fax2 = plt.subplot2grid((3, 6), (0, 3))
        fax3 = plt.subplot2grid((3, 6), (0, 4))
        fax4 = plt.subplot2grid((3, 6), (0, 5))
        fax5 = plt.subplot2grid((3, 6), (1, 3))
        fax6 = plt.subplot2grid((3, 6), (1, 4))
        fax7 = plt.subplot2grid((3, 6), (1, 5))
        fax8 = plt.subplot2grid((3, 6), (2, 3))
        fax9 = plt.subplot2grid((3, 6), (2, 4))
        fax10 = plt.subplot2grid((3, 6), (2, 5))
        fig501.tight_layout()
        axes_handles = [fax1, fax2, fax3, fax4, fax5, fax6, fax7, fax8, fax9, fax10]

        # Plot results
    for ax, index in zip(axes_handles[0:num_cluster + 1], np.arange(num_cluster + 1)):
        if index == 0:
            im = ax.imshow(label_mat, interpolation='none')
            ax.set_title('K-means Cluster Map')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)  # space for colorbar
            plt.colorbar(im, cax=cax)
        else:
            #             ax.plot(Vdc_vec, cluster_centroids[index-1,:], 'g-')
            ax.plot(cluster_centroids[index - 1, :], 'g-')
            ax.set_xlabel('Voltage (V)')
            ax.set_ylabel('Current (arb.)')
            ax.set_title('K-means Centroid: %d' % (index))

    fig501.subplots_adjust(hspace=0.60, wspace=0.60)

    return fig501


###############################################################################

def plotClusterDendrograms(label_mat, e_vals, num_comp, num_cluster, mode='Full', last=None,
                           sort_type='distance', sort_mode=True):
    """
    Creates and plots the dendrograms for the given label_mat and
    eigenvalues

    Parameters
    -------------
    label_mat : 2D real numpy array
                structured as [rows, cols], from KMeans clustering
    e_vals: 3D real numpy array of eigenvalues
            structured as [component, rows, cols]
    num_comps : int
                Number of components used to make eigenvalues
    num_cluster: int
                 Number of cluster used to make the label_mat
    mode: str, optional
          How should the dendrograms be created.
          "Full" -- use all clusters when creating the dendrograms
          "Truncated" -- stop showing clusters after 'last'
    last: int, optional - should be provided when using "Truncated"
          How many merged clusters should be shown when using
          "Truncated" mode
    sort_type: str, optional
          What type of sorting should be used when plotting the
          dendrograms.  Options are:
              count    Default
                  Uses the count_sort from scipy.cluster.hierachy.dendrogram
              distance
                  Uses the distance_sort from scipy.cluster.hierachy.dendrogram
    sort_mode: str or bool, optional
          For the chosen sort_type, which mode should be used.
          Options:
              False    Default
                  Does no sorting
              'ascending' or True
                  The child with the minimum of the chosen sort parameter is
                  plotted first
              'descending'
                  The child with the maximum of the chosen sort parameter is
                  plotted first

    Returns
    ---------
    fig
    """
    if mode == 'Truncated' and not last:
        warn('Warning: Truncated dendrograms requested, but no last cluster given.  Reverting to full dendrograms.')
        mode = 'Full'

    if mode == 'Full':
        print 'Creating full dendrogram from clusters'
        mode = None
    elif mode == 'Truncated':
        print 'Creating truncated dendrogram from clusters.  Will stop at {}.'.format(last)
        mode = 'lastp'
        show_contracted = True
    else:
        raise ValueError('Error: Unknown mode requested for plotting dendrograms. mode={}'.format(mode))

    c_sort = False
    d_sort = False
    if sort_type == 'count':
        c_sort = sort_mode
        if c_sort == 'descending':
            c_sort = 'descendent'
    elif sort_type == 'distance':
        d_sort = sort_mode

    centroid_mat = np.zeros([num_cluster, num_comp])
    for k1 in xrange(num_cluster):
        [i_x, i_y] = np.where(label_mat == k1)
        u_stack = np.zeros([len(i_x), num_comp])
        for k2 in xrange(len(i_x)):
            u_stack[k2, :] = np.abs(e_vals[i_x[k2], i_y[k2], :num_comp])

        centroid_mat[k1, :] = np.mean(u_stack, 0)


        # Get the distrance between cluster means
    distance_mat = pdist(centroid_mat)

    # get hierachical pairings of clusters
    linkage_pairing = linkage(distance_mat, 'weighted')
    linkage_pairing[:, 3] = linkage_pairing[:, 3] / max(linkage_pairing[:, 3])

    fig = plt.figure()
    dendrogram(linkage_pairing, p=last, truncate_mode=mode, count_sort=c_sort, distance_sort=d_sort, leaf_rotation=90)

    fig.axes[0].set_title('Dendrogram')
    fig.axes[0].set_xlabel('Index or (cluster size)')
    fig.axes[0].set_ylabel('Distance')

    return fig


def plot1DSpectrum(data_vec, freq, title, figure_path=None):
    """
    Plots the Step averaged BE response

    Parameters
    ------------
    data_vec : 1D numpy array
        Response of one BE pulse
    freq : 1D numpy array
        BE frequency that serves as the X axis of the plot
    title : String
        Plot group name
    figure_path : String / Unicode
        Absolute path of the file to write the figure to

    Returns
    ---------
    fig : Matplotlib.pyplot figure
        Figure handle
    ax : Matplotlib.pyplot axis
        Axis handle
    """
    if len(data_vec) != len(freq):
        #         print '1D:',data_vec.shape, freq.shape
        warn('plot2DSpectrogram: Incompatible data sizes!!!!')
        return
    freq = freq * 1E-3  # to kHz
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True);
    ax[0].plot(freq, np.abs(data_vec) * 1E+3)
    ax[0].set_title('Amplitude (mV)')
    # ax[0].set_xlabel('Frequency (kHz)')
    ax[1].plot(freq, np.angle(data_vec) * 180 / np.pi)
    ax[1].set_title('Phase (deg)')
    ax[1].set_xlabel('Frequency (kHz)')
    fig.suptitle(title + ': mean UDVS, mean spatial response')
    if figure_path:
        plt.savefig(figure_path, format='png', dpi=300)
    return (fig, ax)


###############################################################################

def plot2DSpectrogram(mean_spectrogram, freq, title, figure_path=None):
    """
    Plots the position averaged spectrogram

    Parameters
    ------------
    mean_spectrogram : 2D numpy complex array
        Means spectrogram arranged as [frequency, UDVS step]
    freq : 1D numpy float array
        BE frequency that serves as the X axis of the plot
    title : String
        Plot group name
    figure_path : String / Unicode
        Absolute path of the file to write the figure to

    Returns
    ---------
    fig : Matplotlib.pyplot figure
        Figure handle
    ax : Matplotlib.pyplot axis
        Axis handle
    """
    if mean_spectrogram.shape[1] != len(freq):
        #  print '2D:',mean_spectrogram.shape, freq.shape
        warn('plot2DSpectrogram: Incompatible data sizes!!!!')
        return
    freq = freq * 1E-3  # to kHz
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True);
    # print mean_spectrogram.shape
    # print freq.shape
    ax[0].imshow(np.abs(mean_spectrogram), interpolation='nearest',
                 extent=[freq[0], freq[-1], mean_spectrogram.shape[0], 0])
    ax[0].set_title('Amplitude')
    # ax[0].set_xticks(freq)
    # ax[0].set_ylabel('UDVS Step')
    ax[0].axis('tight')
    ax[1].imshow(np.angle(mean_spectrogram), interpolation='nearest',
                 extent=[freq[0], freq[-1], mean_spectrogram.shape[0], 0])
    ax[1].set_title('Phase')
    ax[1].set_xlabel('Frequency (kHz)')
    # ax[0].set_ylabel('UDVS Step')
    ax[1].axis('tight')
    fig.suptitle(title)
    if figure_path:
        plt.savefig(figure_path, format='png', dpi=300)
    return (fig, ax)


###############################################################################

def plotHistgrams(p_hist, p_hbins, title, figure_path=None):
    """
    Plots the position averaged spectrogram

    Parameters
    ------------
    p_hist : 2D numpy array
        histogram data arranged as [physical quantity, frequency bin]
    p_hbins : 1D numpy array
        BE frequency that serves as the X axis of the plot
    title : String
        Plot group name
    figure_path : String / Unicode
        Absolute path of the file to write the figure to

    Returns
    ---------
    fig : Matplotlib.pyplot figure
        Figure handle
    """

    base_fig_size = 7
    h_fig = base_fig_size
    w_fig = base_fig_size * 4

    fig = plt.figure(figsize=(w_fig, h_fig))
    fig.suptitle(title)
    iplot = 0

    p_Nx, p_Ny = np.amax(p_hbins, axis=1) + 1

    p_hist = np.reshape(p_hist, (4, p_Ny, p_Nx))

    iplot += 1
    p_plot_title = 'Spectral BEHistogram Amp (log10 of counts)'
    p_plot = fig.add_subplot(1, 4, iplot, title=p_plot_title)
    p_im = p_plot.imshow(np.rot90(np.log10(p_hist[0])), interpolation='nearest')
    p_plot.axis('tight')
    fig.colorbar(p_im, fraction=0.1)

    iplot += 1
    p_plot_title = 'Spectral BEHistogram Phase (log10 of counts)'
    p_plot = fig.add_subplot(1, 4, iplot, title=p_plot_title)
    p_im = p_plot.imshow(np.rot90(np.log10(p_hist[1])), interpolation='nearest')
    p_plot.axis('tight')
    fig.colorbar(p_im, fraction=0.1)

    iplot += 1
    p_plot_title = 'Spectral BEHistogram Real (log10 of counts)'
    p_plot = fig.add_subplot(1, 4, iplot, title=p_plot_title)
    p_im = p_plot.imshow(np.rot90(np.log10(p_hist[2])), interpolation='nearest')
    p_plot.axis('tight')
    fig.colorbar(p_im, fraction=0.1)

    iplot += 1
    p_plot_title = 'Spectral BEHistogram Imag (log10 of counts)'
    p_plot = fig.add_subplot(1, 4, iplot, title=p_plot_title)
    p_im = p_plot.imshow(np.rot90(np.log10(p_hist[3])), interpolation='nearest')
    p_plot.axis('tight')
    fig.colorbar(p_im, fraction=0.1)

    if figure_path:
        plt.savefig(figure_path, format='png')

    return fig