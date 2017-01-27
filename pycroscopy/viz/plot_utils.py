# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""
from __future__ import division # int/int = float
from warnings import warn
import os
import h5py
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from ..analysis.utils.be_loop import loop_fit_function
from ..io.hdf_utils import reshape_to_Ndims, get_formatted_labels

def set_tick_font_size(axes, font_size):
    """
    Sets the font size of the ticks in the provided axes

    Parameters
    ----------
    axes : matplotlib.pyplot.axis object or list of axis objects
        axes to set font sizes
    font_size : unigned int
        Font size
    """

    def __set_axis_tick(axis):
        """
        Sets the font sizes to the x and y axis in the given axis object

        Parameters
        ----------
        axis : matplotlib.pyplot.axis object
            axis to set font sizes
        """
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

    if hasattr(axes, '__iter__'):
        for axis in axes:
            __set_axis_tick(axis)
    else:
        __set_axis_tick(axes)

def cmap_jet_white_center():
    """
    Generates the jet colormap with a white center

    Returns
    -------
    white_jet : matplotlib.colors.LinearSegmentedColormap object
        color map object that can be used in place of plt.cm.jet
    """
    # For red - central column is like brightness
    # For blue - last column is like brightness
    cdict = {'red': ((0.00, 0.0, 0.0),
                     (0.30, 0.0, 0.0),
                     (0.50, 1.0, 1.0),
                     (0.90, 1.0, 1.0),
                     (1.00, 0.5, 1.0)),
             'green': ((0.00, 0.0, 0.0),
                       (0.10, 0.0, 0.0),
                       (0.42, 1.0, 1.0),
                       (0.58, 1.0, 1.0),
                       (0.90, 0.0, 0.0),
                       (1.00, 0.0, 0.0)),
             'blue': ((0.00, 0.0, 0.5),
                      (0.10, 1.0, 1.0),
                      (0.50, 1.0, 1.0),
                      (0.70, 0.0, 0.0),
                      (1.00, 0.0, 0.0))
             }
    return LinearSegmentedColormap('white_jet', cdict)

def cmap_hot_desaturated():
    hot_desaturated = [(1, (255, 76, 76, 255)),
                       (0.857, (107, 0, 0, 255)),
                       (0.714, (255, 96, 0, 255)),
                       (0.571, (255, 255, 0, 255)),
                       (0.429, (0, 127, 0, 255)),
                       (0.285, (0, 255, 255, 255)),
                       (0.143, (0, 0, 91, 255)),
                       (0, (71, 71, 219, 255))]

    cdict = {'red': tuple([(dist, colors[0]/255.0, colors[0]/255.0) for (dist, colors) in hot_desaturated][::-1]),
             'green': tuple([(dist, colors[1]/255.0, colors[1]/255.0) for (dist, colors) in hot_desaturated][::-1]),
             'blue': tuple([(dist, colors[2]/255.0, colors[2]/255.0) for (dist, colors) in hot_desaturated][::-1])}

    return LinearSegmentedColormap('hot_desaturated', cdict)



def discrete_cmap(num_bins, base_cmap=plt.cm.jet):
    """
    Create an N-bin discrete colormap from the specified input map

    Parameters
    ----------
    num_bins : unsigned int
        Number of discrete bins
    base_cmap : matplotlib.colors.LinearSegmentedColormap object
        Base color map to discretize

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        Discretized color map

    Credits
    -------
    Jake VanderPlas
    License: BSD-style
    """

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, num_bins))
    cmap_name = base.name + str(num_bins)
    return base.from_list(cmap_name, color_list, num_bins)


def plot_loop_guess_fit(vdc, ds_proj_loops, ds_guess, ds_fit, title=''):
    """
    Plots the loop guess, fit, source projected loops for a single cycle

    Parameters
    ----------
    vdc - 1D float numpy array
        DC offset vector (unshifted)
    ds_proj_loops - 2D numpy array
        Projected loops arranged as [position, vdc]
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
    """
    shift_ind = int(-1 * len(vdc) / 4)
    vdc_shifted = np.roll(vdc, shift_ind)

    num_plots = np.min([5, int(np.sqrt(ds_proj_loops.shape[0]))])
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots, figsize=(18, 18))
    positions = np.linspace(0, ds_proj_loops.shape[0] - 1, num_plots ** 2, dtype=np.int)
    for ax, pos in zip(axes.flat, positions):
        ax.plot(vdc, ds_proj_loops[pos, :], 'k', label='Raw')
        ax.plot(vdc_shifted, loop_fit_function(vdc_shifted, np.array(list(ds_guess[pos]))), 'g', label='guess')
        ax.plot(vdc_shifted, loop_fit_function(vdc_shifted, np.array(list(ds_fit[pos]))), 'r--', label='Fit')
        ax.set_xlabel('V_DC (V)')
        ax.set_ylabel('PR (a.u.)')
        ax.set_title('Loop ' + str(pos))
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes

###############################################################################


def rainbow_plot(ax, ao_vec, ai_vec, num_steps=32, cmap=plt.cm.jet, **kwargs):
    """
    Plots the input against the output waveform (typically loops).
    The color of the curve changes as a function of time using the jet colorscheme

    Parameters
    ----------
    ax : axis handle
        Axis to plot the curve
    ao_vec : 1D float numpy array
        vector that forms the X axis
    ai_vec : 1D float numpy array
        vector that forms the Y axis
    num_steps : unsigned int (Optional)
        Number of discrete color steps
    cmap : matplotlib.colors.LinearSegmentedColormap object
        Colormap to be used
    """
    pts_per_step = int(len(ai_vec) / num_steps)
    for step in range(num_steps - 1):
        ax.plot(ao_vec[step * pts_per_step:(step + 1) * pts_per_step],
                ai_vec[step * pts_per_step:(step + 1) * pts_per_step],
                color=cmap(255 * step / num_steps), **kwargs)
    # plot the remainder:
    ax.plot(ao_vec[(num_steps - 1) * pts_per_step:],
            ai_vec[(num_steps - 1) * pts_per_step:],
            color=cmap(255 * num_steps / num_steps), **kwargs)
    """
    CS3=plt.contourf([[0,0],[0,0]], range(0,310),cmap=plt.cm.jet)
    fig.colorbar(CS3)"""


def plot_line_family(axis, x_axis, line_family, line_names=None, label_prefix='Line', label_suffix='', cmap=plt.cm.jet,
                     **kwargs):
    """
    Plots a family of lines with a sequence of colors

    Parameters
    ----------
    axis : axis handle
        Axis to plot the curve
    x_axis : array-like
        Values to plot against
    line_family : 2D numpy array
        family of curves arranged as [curve_index, features]
    line_names : array-like
        array of string or numbers that represent the identity of each curve in the family
    label_prefix : string / unicode
        prefix for the legend (before the index of the curve)
    label_suffix : string / unicode
        suffix for the legend (after the index of the curve)
    cmap : matplotlib.colors.LinearSegmentedColormap object
        Colormap to be used
    """
    num_lines = line_family.shape[0]

    if line_names is None:
        line_names = ['{} {} {}'.format(label_prefix, line_ind, label_suffix) for line_ind in range(num_lines)]
    else:
        if len(line_names) != num_lines:
            warn('Line names of different length compared to provided dataset')
            line_names = ['{} {} {}'.format(label_prefix, line_ind, label_suffix) for line_ind in range(num_lines)]

    for line_ind in range(num_lines):
        axis.plot(x_axis, line_family[line_ind],
                  label=line_names[line_ind],
                  color=cmap(int(255 * line_ind / (num_lines - 1))), **kwargs)


def plot_map(axis, data, stdevs=2, **kwargs):
    """
    Plots a 2d map with a tight z axis, with or without color bars.
    Note that the direction of the y axis is flipped if the color bar is required

    Parameters
    ----------
    axis : matplotlib.pyplot.axis object
        Axis to plot this map onto
    data : 2D real numpy array
        Data to be plotted
    stdevs : unsigned int (Optional. Default = 2)
        Number of standard deviations to consider for plotting

    Returns
    -------
    """
    data_mean = np.mean(data)
    data_std = np.std(data)
    im = axis.imshow(data, interpolation='none',
                     vmin=data_mean - stdevs * data_std,
                     vmax=data_mean + stdevs * data_std,
                     **kwargs)
    axis.set_aspect('auto')

    return im


def plot_loops(excit_wfm, datasets, line_colors=[], dataset_names=[], evenly_spaced=True, plots_on_side=5, x_label='',
               y_label='', subtitles='Position', title='', central_resp_size=None, use_rainbow_plots=False, h5_pos=None):
    # TODO: Allow multiple excitation waveforms
    """
    Plots loops from multiple datasets from up to 25 evenly spaced positions

    Parameters
    -----------
    excit_wfm : 1D numpy float array
        Excitation waveform in the time domain
    datasets : list of 2D numpy arrays or 2D hyp5.Dataset objects
        Datasets containing data arranged as (pixel, time)
    line_colors : list of strings
        Colors to be used for each of the datasets
    dataset_names : (Optional) list of strings
        Names of the different datasets to be compared
    h5_pos : HDF5 dataset reference or 2D numpy array
        Dataset containing position indices
    central_resp_size : (optional) unsigned integer
        Number of responce sample points from the center of the waveform to show in plots. Useful for SPORC
    evenly_spaced : boolean
        Evenly spaced positions or first N positions
    plots_on_side : unsigned int
        Number of plots on each side
    use_rainbow_plots : (optional) Boolean
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
    if type(datasets) in [h5py.Dataset, np.ndarray]:
        # can be numpy array or h5py.dataset
        num_pos = datasets.shape[0]
        num_points = datasets.shape[1]
        datasets = [datasets]
        line_colors = ['b']
        dataset_names = ['Default']
    else:
        # First check if the datasets are correctly shaped:
        num_pos_es = list()
        num_points_es = list()
        for dataset in datasets:
            num_pos_es.append(dataset.shape[0])
            num_points_es.append(dataset.shape[1])
        num_pos_es = np.array(num_pos_es)
        num_points_es = np.array(num_points_es)
        if np.unique(num_pos_es).size > 1 or np.unique(num_points_es).size > 1:
            warn('Datasets of incompatible sizes')
            return
        num_pos = np.unique(num_pos_es)[0]
        num_points = np.unique(num_points_es)[0]

        # Next the identification of datasets:
        if len(dataset_names) > len(datasets):
            # remove additional titles
            dataset_names = dataset_names[:len(datasets)]
        elif len(dataset_names) < len(datasets):
            # add titles
            dataset_names = dataset_names + ['Dataset' + ' ' + str(x) for x in range(len(dataset_names), len(datasets))]
        if len(line_colors) != len(datasets):
            color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'brown', 'orange']
            if len(datasets) < len(color_list):
                remaining_colors = [x for x in color_list if x not in line_colors]
                line_colors += remaining_colors[:len(datasets) - len(color_list)]
            else:
                warn('Insufficient number of line colors provided')
                return


    if excit_wfm.size != num_points:
        warn('Length of excitation waveform not compatible with second axis of datasets')
        return

    plots_on_side = min(abs(plots_on_side), 5)

    sq_num_plots = min(plots_on_side, int(round(num_pos ** 0.5)))
    if evenly_spaced:
        chosen_pos = np.linspace(0, num_pos - 1, sq_num_plots ** 2, dtype=int)
    else:
        chosen_pos = np.arange(sq_num_plots ** 2, dtype=int)

    fig, axes = plt.subplots(nrows=sq_num_plots, ncols=sq_num_plots, figsize=(12, 12))
    axes_lin = axes.flatten()

    cent_ind = int(0.5 * excit_wfm.size)
    if central_resp_size:
        sz = int(0.5 * central_resp_size)
        l_resp_ind = cent_ind - sz
        r_resp_ind = cent_ind + sz
    else:
        l_resp_ind = 0
        r_resp_ind = excit_wfm.size

    for count, posn in enumerate(chosen_pos):
        if use_rainbow_plots and len(datasets) == 1:
            rainbow_plot(axes_lin[count], excit_wfm[l_resp_ind:r_resp_ind], datasets[0][posn, l_resp_ind:r_resp_ind])
        else:
            for dataset, col_val in zip(datasets, line_colors):
                axes_lin[count].plot(excit_wfm[l_resp_ind:r_resp_ind], dataset[posn, l_resp_ind:r_resp_ind], color=col_val)
        if h5_pos is not None:
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
    if len(datasets) > 1:
        axes_lin[count].legend(dataset_names, loc='best')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig, axes

###############################################################################


def plot_complex_map_stack(map_stack, num_comps=4, title='Eigenvectors', xlabel='UDVS Step', stdevs=2):
    """
    Plots the provided spectrograms from SVD V vector

    Parameters:
    -------------
    map_stack : 3D numpy complex matrices
        Eigenvectors rearranged as - [row, col, component]
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

    for index in range(num_comps):
        cur_map = np.transpose(map_stack[index, :, :])
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

def plot_complex_loop_stack(loop_stack, x_axis, heading='BE Loops', subtitle='Eigenvector', num_comps=4, x_label=''):
    """
    Plots the provided spectrograms from SVD V vector

    Parameters:
    -------------
    loop_stack : 3D numpy complex matrices
        Loops rearranged as - [component, points]
    x_axis : 1D real numpy array
        The vector to plot against
    num_comps : int
        Number of components to plot
    title : String
        Title to plot above everything else
    x_label : String
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
    fig201.canvas.set_window_title(heading)

    for index in range(num_comps):
        cur_map = loop_stack[index, :]
        axes = [axes201.flat[index], axes201.flat[index + num_comps]]
        for func, lab, ax in zip(funcs, labels, axes):
            ax.plot(x_axis, func(cur_map))
            ax.set_title('%s: %d - %s' % (subtitle, index + 1, lab))
        ax.set_xlabel(x_label)
    fig201.tight_layout()

    return fig201, axes201

###############################################################################


def plotScree(scree, title='Scree'):
    """
    Plots the scree or scree

    Parameters:
    -------------
    scree : 1D real numpy array
        The scree vector from SVD

    Returns:
    ---------
    fig, axes
    """
    fig203 = plt.figure(figsize=(6.5, 6))
    axes203 = fig203.add_axes([0.1, 0.1, .8, .8])  # left, bottom, width, height (range 0 to 1)
    axes203.loglog(np.arange(len(scree)) + 1, scree, 'b', marker='*')
    axes203.set_xlabel('Principal Component')
    axes203.set_ylabel('Variance')
    axes203.set_title(title)
    axes203.set_xlim(left=1, right=len(scree))
    axes203.set_ylim(bottom=np.min(scree), top=np.max(scree))
    fig203.canvas.set_window_title("Scree")

    return fig203, axes203


# ###############################################################################


def plot_map_stack(map_stack, num_comps=9, stdevs=2, color_bar_mode=None, evenly_spaced=False,
                   title='Component', heading='Map Stack', **kwargs):
    """
    Plots the provided stack of maps

    Parameters:
    -------------
    map_stack : 3D real numpy array
        structured as [rows, cols, component]
    num_comps : unsigned int
        Number of components to plot
    stdevs : int
        Number of standard deviations to consider for plotting
    color_bar_mode : String, Optional
        Options are None, single or each. Default None
    title : String or list of strings
        The titles for each of the plots.
        If a single string is provided, the plot titles become ['title 01', title 02', ...].
        if a list of strings (equal to the number of components) are provided, these are used instead.

    Returns:
    ---------
    fig, axes
    """
    num_comps = abs(num_comps)
    num_comps = min(num_comps, map_stack.shape[-1])


    if evenly_spaced:
        chosen_pos = np.linspace(0, map_stack.shape[-1] - 1, num_comps, dtype=int)
    else:
        chosen_pos = np.arange(num_comps, dtype=int)

    if isinstance(title, list):
        if len(title) > num_comps:
            # remove additional titles
            title = title[:num_comps]
        elif len(title) < num_comps:
            # add titles
            title = title + ['Component' + ' ' + str(x) for x in range(len(title), num_comps)]
    else:
        if not isinstance(title, str):
            title = 'Component'
        title = [title + ' ' + str(x) for x in chosen_pos]

    fig_h, fig_w = (4, 4)
    p_rows = int(np.floor(np.sqrt(num_comps)))
    p_cols = int(np.ceil(num_comps / p_rows))
    if p_rows*p_cols < num_comps:
        p_cols += 1
    fig202 = plt.figure(figsize=(p_cols * fig_w, p_rows * fig_h))
    axes202 = ImageGrid(fig202, 111, nrows_ncols=(p_rows, p_cols),
                        cbar_mode=color_bar_mode,
                        cbar_pad='1%',
                        cbar_size='5%',
                        axes_pad=(0.1*fig_w, 0.07*fig_h))
    # fig202, axes202 = plt.subplots(p_cols, p_rows, figsize=(p_cols * fig_w, p_rows * fig_h))
    # fig202.subplots_adjust(hspace=0.4, wspace=0.4)
    fig202.canvas.set_window_title(heading)
    fig202.suptitle(heading, fontsize=16)

    for count, index, subtitle in zip(range(chosen_pos.size), chosen_pos, title):
        im = plot_map(axes202[count],
                      map_stack[:, :, index],
                      stdevs=stdevs, **kwargs)
        axes202[count].set_title(subtitle)
        if color_bar_mode is 'each':
            axes202.cbar_axes[count].colorbar(im)

    if color_bar_mode is 'single':
        axes202.cbar_axes[0].colorbar(im)

    return fig202, axes202


def plot_cluster_h5_group(h5_group, y_spec_label, centroids_together=True):
    """
        Plots the cluster labels and mean response for each cluster

        Parameters
        ----------
        h5_group : h5py.Datagroup object
            H5 group containing the labels and mean response
        y_spec_label : str
            Label to use for Y axis on cluster centroid plot
        centroids_together : Boolean, optional - default = True
            Whether or nor to plot all centroids together on the same plot

        Returns
        -------
        fig : Figure
            Figure containing the plots
        axes : 1D array_like of axes objects
            Axes of the individual plots within `fig`
        """
    # TODO: The quantity and units for the main dataset itself are missing in most cases!
    h5_labels = h5_group['Labels']
    try:
        h5_mean_resp = h5_group['Mean_Response']
    except KeyError:
        # old PySPM format:
        h5_mean_resp = h5_group['Centroids']

    # Reshape the mean response to N dimensions
    mean_response, success = reshape_to_Ndims(h5_mean_resp)

    # unfortunately, we cannot use the above function for the labels
    # However, we will assume that the position values are linked to the labels:
    h5_pos_vals = h5_labels.file[h5_labels.attrs['Position_Values']]
    h5_pos_inds = h5_labels.file[h5_labels.attrs['Position_Indices']]

    # Reshape the labels correctly:
    pos_dims = []
    for col in range(h5_pos_inds.shape[1]):
        pos_dims.append(np.unique(h5_pos_inds[:, col]).size)

    pos_ticks = [h5_pos_vals[:pos_dims[0], 0], h5_pos_vals[slice(0,None,pos_dims[0]), 1]]
    # prepare the axes ticks for the map

    pos_dims.reverse()  # go from slowest to fastest
    pos_dims = tuple(pos_dims)
    label_mat = np.reshape(h5_labels.value, pos_dims)

    # Figure out the correct units and labels for mean response:
    h5_spec_vals = h5_mean_resp.file[h5_mean_resp.attrs['Spectroscopic_Values']]
    x_spec_label = get_formatted_labels(h5_spec_vals)[0]

    # Figure out the correct axes labels for label map:
    pos_labels = get_formatted_labels(h5_pos_vals)
    # TODO: cleaner x and y axes labels instead of 0.0000125 etc.

    if centroids_together:
        return plot_cluster_results_together(label_mat, mean_response, spec_val=np.squeeze(h5_spec_vals[0]),
                                             spec_label=x_spec_label, resp_label=y_spec_label,
                                             pos_labels=pos_labels, pos_ticks=pos_ticks)
    else:
        return plot_cluster_results_separate(label_mat, mean_response, max_centroids=4, x_label=x_spec_label,
                                             spec_val=np.squeeze(h5_spec_vals[0]), y_label=y_spec_label)

###############################################################################


def plot_cluster_results_together(label_mat, mean_response, spec_val=None, cmap=plt.cm.jet,
                                  spec_label='Spectroscopic Value', resp_label='Response',
                                  pos_labels=('X', 'Y'), pos_ticks=None):
    """
    Plot the cluster labels and mean response for each cluster in separate plots

    Parameters
    ----------
    label_mat : 2D ndarray or h5py.Dataset of ints
        Spatial map of cluster labels structured as [rows, cols]
    mean_response : 2D array or h5py.Dataset
        Mean value of each cluster over all samples 
        arranged as [cluster number, features]
    spec_val :  1D array or h5py.Dataset of floats, optional
        X axis to plot the centroids against
        If no value is specified, the data is plotted against the index
    cmap : plt.cm object or str, optional
        Colormap to use for the labels map and the centroid.
        Advised to pick a map where the centroid plots show clearly.
        Default = matplotlib.pyplot.cm.jet
    spec_label : str, optional
        Label to use for X axis on cluster centroid plot
        Default = 'Spectroscopic Value'
    resp_label : str, optional
        Label to use for Y axis on cluster centroid plot
         Default = 'Response'
    pos_labels : array_like of str, optional
        Labels to use for the X and Y axes on the Label map
        Default = ('X', 'Y')
    pos_ticks : array_like of int

    Returns
    -------
    fig : Figure
        Figure containing the plots
    axes : 1D array_like of axes objects
        Axes of the individual plots within `fig`
    """

    def __plot_centroids(centroids, ax, spec_val, spec_label, y_label, cmap, title=None):
        plot_line_family(ax, spec_val, centroids, label_prefix='Cluster', cmap=cmap)
        ax.set_ylabel(y_label)
        # ax.legend(loc='best')
        if title:
            ax.set_title(title)
            ax.set_xlabel(spec_label)

    if type(spec_val) == type(None):
        spec_val = np.arange(mean_response.shape[1])

    if mean_response.dtype in [np.complex64, np.complex128, np.complex]:
        fig = plt.figure(figsize=(12, 8))
        ax_map = plt.subplot2grid((2, 12), (0, 0), colspan=6, rowspan=2)
        ax_amp = plt.subplot2grid((2, 12), (0, 6), colspan=4)
        ax_phase = plt.subplot2grid((2, 12), (1, 6), colspan=4)
        axes = [ax_map, ax_amp, ax_phase]

        __plot_centroids(np.abs(mean_response), ax_amp, spec_val, spec_label,
                        resp_label + ' - Amplitude', cmap, 'Mean Response')
        __plot_centroids(np.angle(mean_response), ax_phase, spec_val, spec_label,
                        resp_label + ' - Phase', cmap)
        plot_handles, plot_labels = ax_amp.get_legend_handles_labels()

    else:
        fig = plt.figure(figsize=(12, 8))
        ax_map = plt.subplot2grid((1, 12), (0, 0), colspan=6)
        ax_resp = plt.subplot2grid((1, 12), (0, 6), colspan=4)
        axes = [ax_map, ax_resp]
        __plot_centroids(mean_response, ax_resp, spec_val, spec_label,
                        resp_label, cmap, 'Mean Response')
        plot_handles, plot_labels = ax_resp.get_legend_handles_labels()

    fleg = plt.figlegend(plot_handles, plot_labels, loc='center right',
                         borderaxespad=0.0)
    num_clusters = mean_response.shape[0]

    if isinstance(label_mat, h5py.Dataset):
        """
        Reshape label_mat based on linked positions
        """
        pos = label_mat.file[label_mat.attrs['Position_Indices']]
        nx = len(np.unique(pos[:, 0]))
        ny = len(np.unique(pos[:, 1]))
        label_mat = label_mat[()].reshape(nx, ny)

    # im = ax_map.imshow(label_mat, interpolation='none')
    ax_map.set_xlabel(pos_labels[0])
    ax_map.set_ylabel(pos_labels[1])

    if pos_ticks is not None:
        x_ticks = np.linspace(0, label_mat.shape[1] - 1, 5, dtype=np.uint16)
        y_ticks = np.linspace(0, label_mat.shape[0] - 1, 5, dtype=np.uint16)
        ax_map.set_xticks(x_ticks)
        ax_map.set_yticks(y_ticks)
        ax_map.set_xticklabels(pos_ticks[0][x_ticks])
        ax_map.set_yticklabels(pos_ticks[1][y_ticks])

    """divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # space for colorbar
    fig.colorbar(im, cax=cax, ticks=np.arange(num_clusters),
                 cmap=discrete_cmap(num_clusters, base_cmap=plt.cm.jet))
    ax_map.axis('tight')"""
    pcol0 = ax_map.pcolor(label_mat, cmap=discrete_cmap(num_clusters, base_cmap=plt.cm.jet))
    fig.colorbar(pcol0, ax=ax_map, ticks=np.arange(num_clusters))
    ax_map.axis('tight')
    ax_map.set_aspect('auto')
    ax_map.set_title('Cluster Label Map')

    fig.tight_layout()
    fig.canvas.set_window_title('Cluster results')

    return fig, axes

###############################################################################


def plot_cluster_results_separate(label_mat, cluster_centroids, max_centroids=4,
                                  spec_val=None, x_label='Excitation (a.u.)', y_label='Response (a.u.)'):
    """
    Plots the provided labels mat and centroids from clustering

    Parameters
    ----------
    label_mat : 2D int numpy array
                structured as [rows, cols]
    cluster_centroids: 2D real numpy array
                       structured as [cluster,features]
    max_centroids : unsigned int
                    Number of centroids to plot
    spec_val :  array-like
        X axis to plot the centroids against
        If no value is specified, the data is plotted against the index
    x_label : String / unicode
              X label for centroid plots
    y_label : String / unicode
              Y label for centroid plots

    Returns
    -------
    fig
    """

    if max_centroids < 5:

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

    # First plot the labels map:
    pcol0 = fax1.pcolor(label_mat, cmap=discrete_cmap(cluster_centroids.shape[0],
                                                      base_cmap=plt.cm.jet))
    fig501.colorbar(pcol0, ax=fax1, ticks=np.arange(cluster_centroids.shape[0]))
    fax1.axis('tight')
    fax1.set_aspect('auto')
    fax1.set_title('Cluster Label Map')
    """im = fax1.imshow(label_mat, interpolation='none')
    divider = make_axes_locatable(fax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # space for colorbar
    plt.colorbar(im, cax=cax)"""

    if spec_val is None and cluster_centroids.ndim == 2:
        spec_val = np.arange(cluster_centroids.shape[1])

    # Plot results
    for ax, index in zip(axes_handles[1: max_centroids + 1], np.arange(max_centroids)):
        if cluster_centroids.ndim == 2:
            ax.plot(spec_val, cluster_centroids[index, :],
                    color=plt.cm.jet(int(255 * index / (cluster_centroids.shape[0] - 1))))
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        elif cluster_centroids.ndim == 3:
            plot_map(ax, cluster_centroids[index])
        ax.set_title('Centroid: %d' % index)

    fig501.subplots_adjust(hspace=0.60, wspace=0.60)
    fig501.tight_layout()

    return fig501


###############################################################################

def plot_cluster_dendrogram(label_mat, e_vals, num_comp, num_cluster, mode='Full', last=None,
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
    num_comp : int
        Number of components used to make eigenvalues
    num_cluster : int
        Number of cluster used to make the label_mat
    mode: str, optional
        How should the dendrograms be created.
        "Full" -- use all clusters when creating the dendrograms
        "Truncated" -- stop showing clusters after 'last'
    last: int, optional - should be provided when using "Truncated"
        How many merged clusters should be shown when using
        "Truncated" mode
    sort_type: {'count', 'distance'}, optional
        What type of sorting should be used when plotting the
        dendrograms.  Options are:
        count - Uses the count_sort from scipy.cluster.hierachy.dendrogram
        distance - Uses the distance_sort from scipy.cluster.hierachy.dendrogram
    sort_mode: {False, True, 'ascending', 'descending'}, optional
        For the chosen sort_type, which mode should be used.
        False - Does no sorting
        'ascending' or True - The child with the minimum of the chosen sort
        parameter is plotted first
        'descending' - The child with the maximum of the chosen sort parameter is
        plotted first

    Returns
    ---------
    fig : matplotlib.pyplot Figure object
        Figure containing the dendrogram
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
    for k1 in range(num_cluster):
        [i_x, i_y] = np.where(label_mat == k1)
        u_stack = np.zeros([len(i_x), num_comp])
        for k2 in range(len(i_x)):
            u_stack[k2, :] = np.abs(e_vals[i_x[k2], i_y[k2], :num_comp])

        centroid_mat[k1, :] = np.mean(u_stack, 0)

    # Get the distrance between cluster means
    distance_mat = scipy.spatial.distance.pdist(centroid_mat)

    # get hierachical pairings of clusters
    linkage_pairing = scipy.cluster.hierarchy.linkage(distance_mat, 'weighted')
    linkage_pairing[:, 3] = linkage_pairing[:, 3] / max(linkage_pairing[:, 3])

    fig = plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage_pairing, p=last, truncate_mode=mode,
                                       count_sort=c_sort, distance_sort=d_sort,
                                       leaf_rotation=90)

    fig.axes[0].set_title('Dendrogram')
    fig.axes[0].set_xlabel('Index or (cluster size)')
    fig.axes[0].set_ylabel('Distance')

    return fig


def plot_1d_spectrum(data_vec, freq, title, figure_path=None):
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
        warn('plot_1d_spectrum: Incompatible data sizes!!!!')
        print('1D:', data_vec.shape, freq.shape)
        return
    freq *= 1E-3  # to kHz
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(freq, np.abs(data_vec) * 1E+3)
    ax[0].set_title('Amplitude (mV)')
    ax[1].plot(freq, np.angle(data_vec) * 180 / np.pi)
    ax[1].set_title('Phase (deg)')
    ax[1].set_xlabel('Frequency (kHz)')
    fig.suptitle(title + ': mean UDVS, mean spatial response')
    if figure_path:
        plt.savefig(figure_path, format='png', dpi=300)
    return fig, ax


###############################################################################

def plot_2d_spectrogram(mean_spectrogram, freq, title, figure_path=None):
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
        warn('plot_2d_spectrogram: Incompatible data sizes!!!!')
        print('2D:', mean_spectrogram.shape, freq.shape)
        return
    freq *= 1E-3  # to kHz
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
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
    return fig, ax

###############################################################################


def plot_histgrams(p_hist, p_hbins, title, figure_path=None):
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


def visualize_sho_results(h5_main, save_plots=True, show_plots=True):
    """
    Plots some loops, amplitude, phase maps for BE-Line and BEPS datasets.\n
    Note: The file MUST contain SHO fit gusses at the very least

    Parameters
    ----------
    h5_main : HDF5 Dataset
        dataset to be plotted
    save_plots : (Optional) Boolean
        Whether or not to save plots to files in the same directory as the h5 file
    show_plots : (Optional) Boolean
        Whether or not to display the plots on the screen

    Returns
    -------
    None
    """

    def __plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, spec_var_title, meas_var_title, save_plots,
                          folder_path, basename, num_rows, num_cols):
        plt_title = grp_name + '_' + win_title + '_Loops'
        fig, ax = plot_loops(ac_vec, resp_mat, evenly_spaced=True, plots_on_side=5, use_rainbow_plots=False,
                             x_label=spec_var_title, y_label=meas_var_title, subtitles='Loop', title=plt_title)
        if save_plots:
            fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

        plt_title = grp_name + '_' + win_title + '_Snaps'
        fig, axes = plot_map_stack(resp_mat.reshape(num_rows, num_cols, resp_mat.shape[1]),
                                   color_bar_mode="each", evenly_spaced=True, title='UDVS Step #',
                                   heading=plt_title, cmap=cmap_jet_white_center())
        if save_plots:
            fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

    plt_path = None

    print('Creating plots of SHO Results from {}.'.format(h5_main.name))

    h5_file = h5_main.file

    expt_type = h5_file.attrs['data_type']
    if expt_type not in ['BEPSData', 'BELineData']:
        warn('Unsupported data format')
        return
    isBEPS = expt_type == 'BEPSData'

    (folder_path, basename) = os.path.split(h5_file.filename)
    basename, _ = os.path.splitext(basename)

    sho_grp = h5_main.parent
    chan_grp = sho_grp.parent

    grp_name = '_'.join(chan_grp.name[1:].split('/'))
    grp_name = '_'.join([grp_name, sho_grp.name.split('/')[-1].split('-')[0], h5_main.name.split('/')[-1]])

    try:
        h5_pos = h5_file[h5_main.attrs['Position_Indices']]
    except KeyError:
        print('No Position_Indices found as attribute of {}'.format(h5_main.name))
        print('Rows and columns will be calculated from dataset shape.')
        num_rows = int(np.floor((np.sqrt(h5_main.shape[0]))))
        num_cols = int(np.reshape(h5_main, [num_rows, -1, h5_main.shape[1]]).shape[1])
    else:
        num_rows = len(np.unique(h5_pos[:, 0]))
        num_cols = len(np.unique(h5_pos[:, 1]))

    try:
        h5_spec_vals = h5_file[h5_main.attrs['Spectroscopic_Values']]
    # except KeyError:
    #     warn('No Spectrosocpic Datasets found as attribute of {}'.format(h5_main.name))
    #     raise
    except:
        raise

    # Assume that there's enough memory to load all the guesses into memory
    amp_mat = h5_main['Amplitude [V]'] * 1000  # convert to mV ahead of time
    freq_mat = h5_main['Frequency [Hz]'] / 1000
    q_mat = h5_main['Quality Factor']
    phase_mat = h5_main['Phase [rad]']
    rsqr_mat = h5_main['R2 Criterion']

    if isBEPS:
        meas_type = chan_grp.parent.attrs['VS_mode']
        # basically 3 kinds for now - DC/current, AC, UD - lets ignore this
        if meas_type == 'load user defined VS Wave from file':
            warn('Not handling custom experiments for now')
            h5_file.close()
            return

        # Plot amplitude and phase maps at one or more UDVS steps

        if meas_type == 'AC modulation mode with time reversal':
            center = int(h5_spec_vals.shape[1] * 0.5)
            ac_vec = np.squeeze(h5_spec_vals[h5_spec_vals.attrs['AC_Amplitude']][0:center])

            forw_resp = np.squeeze(amp_mat[:, slice(0, center)])
            rev_resp = np.squeeze(amp_mat[:, slice(center, None)])

            for win_title, resp_mat in zip(['Forward', 'Reverse'], [forw_resp, rev_resp]):
                __plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, 'AC Amplitude', 'Amplitude', save_plots,
                                  folder_path, basename, num_rows, num_cols)
        else:
            # plot loops at a few locations
            dc_vec = np.squeeze(h5_spec_vals[h5_spec_vals.attrs['DC_Offset']])
            if chan_grp.parent.attrs['VS_measure_in_field_loops'] == 'in and out-of-field':

                dc_vec = np.squeeze(dc_vec[slice(0, None, 2)])

                in_phase = np.squeeze(phase_mat[:, slice(0, None, 2)])
                in_amp = np.squeeze(amp_mat[:, slice(0, None, 2)])
                out_phase = np.squeeze(phase_mat[:, slice(1, None, 2)])
                out_amp = np.squeeze(amp_mat[:, slice(1, None, 2)])

                for win_title, resp_mat in zip(['In_Field', 'Out_of_Field'], [in_phase * in_amp, out_phase * out_amp]):
                    __plot_loops_maps(dc_vec, resp_mat, grp_name, win_title, 'DC Bias', 'Piezoresponse (a.u.)',
                                      save_plots, folder_path, basename, num_rows, num_cols)
            else:
                __plot_loops_maps(dc_vec, phase_mat * amp_mat, grp_name, '', 'DC Bias', 'Piezoresponse (a.u.)',
                                  save_plots, folder_path, basename, num_rows, num_cols)

    else:  # BE-Line can only visualize the amplitude and phase maps:
        amp_mat = amp_mat.reshape(num_rows, num_cols)
        freq_mat = freq_mat.reshape(num_rows, num_cols)
        q_mat = q_mat.reshape(num_rows, num_cols)
        phase_mat = phase_mat.reshape(num_rows, num_cols)
        rsqr_mat = rsqr_mat.reshape(num_rows, num_cols)
        if save_plots:
            plt_path = os.path.join(folder_path, basename + '_' + grp_name + 'Maps.png')

        fig_ms, ax_ms = plot_map_stack(np.dstack((amp_mat, freq_mat, q_mat, phase_mat, rsqr_mat)),
                                       num_comps=5, color_bar_mode='each', heading=grp_name,
                                       title=['Amplitude (mV)', 'Frequency (kHz)', 'Quality Factor', 'Phase (deg)',
                                              'R^2 Criterion'], cmap=cmap_jet_white_center())
        fig_ms.savefig(plt_path, format='png', dpi=300)

    if show_plots:
        plt.show()

    plt.close('all')
