# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""
# TODO: All general plotting functions should support data with 1, 2, or 3 spatial dimensions.

from __future__ import division, print_function, absolute_import, unicode_literals

import inspect
import os
import sys
from warnings import warn

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.signal import blackman

from ..io.hdf_utils import reshape_to_Ndims, get_formatted_labels, get_data_descriptor

# mpl.rcParams.keys()  # gets all allowable keys
mpl.rc('figure', figsize=(5,5))
mpl.rc('lines', linewidth=2)
mpl.rc('axes', labelsize=16, titlesize=16)
mpl.rc('figure', titlesize=20)
mpl.rc('font', size=14) # global font size
mpl.rc('legend', fontsize=16, fancybox=True)
mpl.rc('xtick.major', size=6)
mpl.rc('xtick.minor', size=4)
# mpl.rcParams['xtick.major.size'] = 6

if sys.version_info.major == 3:
    unicode = str

default_cmap = plt.cm.viridis


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


def make_scalar_mappable(vmin, vmax, cmap=None):
    """
    Creates a scalar mappable object that can be used to create a colorbar for non-image (e.g. - line) plots

    Parameters
    ----------
    vmin : float
        Minimum value for colorbar
    vmax : float
        Maximum value for colorbar
    cmap : colormap object
        Colormap object to use

    Returns
    -------
    sm : matplotlib.pyplot.cm.ScalarMappable object
        The object that can used to create a colorbar via plt.colorbar(sm)
    """
    if cmap is None:
        cmap = default_cmap

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable
    sm._A = []
    return sm


def cbar_for_line_plot(axis, num_steps, discrete_ticks=True, **kwargs):
    """
    Adds a colorbar next to a line plot axis

    Parameters
    ----------
    axis : axis handle
        Axis with multiple line objects
    num_steps : uint
        Number of steps in the colorbar
    discrete_ticks : (optional) bool
        Whether or not to have the ticks match the number of number of steps. Default = True
    """
    cmap = get_cmap_object(kwargs.pop('cmap', None))
    cmap = discrete_cmap(num_steps, cmap=cmap.name)

    sm = make_scalar_mappable(0, num_steps - 1, cmap=cmap, **kwargs)

    if discrete_ticks:
        kwargs.update({'ticks': np.arange(num_steps)})

    cbar = plt.colorbar(sm, ax=axis, orientation='vertical',
                        pad=0.04, use_gridspec=True, **kwargs)
    return cbar


def get_cmap_object(cmap):
    """
    Get the matplotlib.colors.LinearSegmentedColormap object regardless of the input

    Parameters
    ----------
    cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
        Requested color map
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap object
        Requested / Default colormap object
    """
    if cmap is None:
        return default_cmap
    elif type(cmap) in [str, unicode]:
        return plt.get_cmap(cmap)
    return cmap


def cmap_jet_white_center():
    """
    Generates the jet colormap with a white center

    Returns
    -------
    white_jet : matplotlib.colors.LinearSegmentedColormap object
        color map object that can be used in place of the default colormap
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

def cmap_from_rgba(name, interp_vals, normalization_val):
    """
    Generates a colormap given a matlab-style interpolation table

    Parameters
    ----------
    name : String / Unicode
        Name of the desired colormap
    interp_vals : List of tuples
        Interpolation table that describes the desired color map. Each entry in the table should be described as:
        (position in the colorbar, (red, green, blue, alpha))
        The position in the color bar, red, green, blue, and alpha vary from 0 to the normalization value
    normalization_val : number
        The common maximum value for the position in the color bar, red, green, blue, and alpha

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        desired color map
    """

    normalization_val = np.round(1.0 * normalization_val)

    cdict = {'red': tuple([(dist / normalization_val, colors[0] / normalization_val, colors[0] / normalization_val)
                           for (dist, colors) in interp_vals][::-1]),
             'green': tuple([(dist / normalization_val, colors[1] / normalization_val, colors[1] / normalization_val)
                             for (dist, colors) in interp_vals][::-1]),
             'blue': tuple([(dist / normalization_val, colors[2] / normalization_val, colors[2] / normalization_val)
                            for (dist, colors) in interp_vals][::-1]),
             'alpha': tuple([(dist / normalization_val, colors[3] / normalization_val, colors[3] / normalization_val)
                             for (dist, colors) in interp_vals][::-1])}

    return LinearSegmentedColormap(name, cdict)


def make_linear_alpha_cmap(name, solid_color, normalization_val, min_alpha=0, max_alpha=1):
    """
    Generates a transparent to opaque color map based on a single solid color

    Parameters
    ----------
    name : String / Unicode
        Name of the desired colormap
    solid_color : List of numbers
        red, green, blue, and alpha values for a specific color
    normalization_val : number
        The common maximum value for the red, green, blue, and alpha values. This is 1 in matplotlib
    min_alpha : float (optional. Default = 0 : ie- transparent)
        Lowest alpha value for the bottom of the color bar
    max_alpha : float (optional. Default = 1 : ie- opaque)
        Highest alpha value for the top of the color bar

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        transparent to opaque color map based on the provided color
    """
    solid_color = np.array(solid_color) / normalization_val * 1.0
    interp_table = [(1.0, (solid_color[0], solid_color[1], solid_color[2], max_alpha)),
                    (0, (solid_color[0], solid_color[1], solid_color[2], min_alpha))]
    return cmap_from_rgba(name, interp_table, 1)


def cmap_hot_desaturated():
    """
    Returns a desaturated color map based on the hot colormap

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        Desaturated version of the hot color map
    """
    hot_desaturated = [(255.0, (255, 76, 76, 255)),
                       (218.5, (107, 0, 0, 255)),
                       (182.1, (255, 96, 0, 255)),
                       (145.6, (255, 255, 0, 255)),
                       (109.4, (0, 127, 0, 255)),
                       (72.675, (0, 255, 255, 255)),
                       (36.5, (0, 0, 91, 255)),
                       (0, (71, 71, 219, 255))]

    return cmap_from_rgba('hot_desaturated', hot_desaturated, 255)


def discrete_cmap(num_bins, cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map specified

    Parameters
    ----------
    num_bins : unsigned int
        Number of discrete bins
    cmap : matplotlib.colors.Colormap object
        Base color map to discretize

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap object
        Discretized color map

    Notes
    -----
    Jake VanderPlas License: BSD-style
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    """
    if cmap is None:
        cmap = default_cmap.name

    elif type(cmap) not in [unicode, str]:
        # could not figure out a better type check
        cmap = cmap.name

    return plt.get_cmap(cmap, num_bins)


def rainbow_plot(axis, x_vec, y_vec, num_steps=32, **kwargs):
    """
    Plots the input against the output waveform (typically loops).
    The color of the curve changes as a function of time

    Parameters
    ----------
    axis : axis handle
        Axis to plot the curve
    x_vec : 1D float numpy array
        vector that forms the X axis
    y_vec : 1D float numpy array
        vector that forms the Y axis
    num_steps : unsigned int (Optional)
        Number of discrete color steps
    """
    cmap = kwargs.pop('cmap', default_cmap)
    cmap = get_cmap_object(cmap)

    # Remove any color flag
    _ = kwargs.pop('color', None)

    pts_per_step = len(y_vec) // num_steps

    for step in range(num_steps - 1):
        axis.plot(x_vec[step * pts_per_step:(step + 1) * pts_per_step],
                  y_vec[step * pts_per_step:(step + 1) * pts_per_step],
                  color=cmap(255 * step // num_steps), **kwargs)
    # plot the remainder:
    axis.plot(x_vec[(num_steps - 1) * pts_per_step:],
              y_vec[(num_steps - 1) * pts_per_step:],
              color=cmap(255 * num_steps / num_steps), **kwargs)


def plot_line_family(axis, x_vec, line_family, line_names=None, label_prefix='', label_suffix='',
                     y_offset=0, show_cbar=False, **kwargs):
    """
    Plots a family of lines with a sequence of colors

    Parameters
    ----------
    axis : axis handle
        Axis to plot the curve
    x_vec : array-like
        Values to plot against
    line_family : 2D numpy array
        family of curves arranged as [curve_index, features]
    line_names : array-like
        array of string or numbers that represent the identity of each curve in the family
    label_prefix : string / unicode
        prefix for the legend (before the index of the curve)
    label_suffix : string / unicode
        suffix for the legend (after the index of the curve)
    y_offset : (optional) number
        quantity by which the lines are offset from each other vertically (useful for spectra)
    show_cbar : (optional) bool
        Whether or not to show a colorbar (instead of a legend)
    """
    cmap = get_cmap_object(kwargs.pop('cmap', None))

    num_lines = len(line_family)

    default_names = False

    if line_names is None:
        label_prefix = 'Line '
        default_names = True
    elif len(line_names) != num_lines:
        warn('Line names of different length compared to provided dataset')
        default_names = True

    if default_names:
        line_names = [str(line_ind) for line_ind in range(num_lines)]

    line_names = ['{} {} {}'.format(label_prefix, cur_name, label_suffix) for cur_name in line_names]

    for line_ind in range(num_lines):
        axis.plot(x_vec, line_family[line_ind] + line_ind * y_offset,
                  label=line_names[line_ind],
                  color=cmap(int(255 * line_ind / (num_lines - 1))), **kwargs)

    if show_cbar:
        # put back the cmap parameter:
        kwargs.update({'cmap': cmap})
        _ = cbar_for_line_plot(axis, num_lines, **kwargs)


def plot_map(axis, img, show_xy_ticks=True, show_cbar=True, x_size=None, y_size=None, num_ticks=4,
             stdevs=None, cbar_label=None, tick_font_size=14, origin='lower', **kwargs):
    """
    Plots an image within the given axis with a color bar + label and appropriate X, Y tick labels.
    This is particularly useful to get readily interpretable plots for papers
    Parameters
    ----------
    axis : matplotlib.axis object
        Axis to plot this image onto
    img : 2D numpy array with real values
        Data for the image plot
    show_xy_ticks : bool, Optional, default = None, shown unedited
        Whether or not to show X, Y ticks
    show_cbar : bool, optional, default = True
        Whether or not to show the colorbar
    x_size : float, optional, default = number of pixels in x direction
        Extent of tick marks in the X axis. This could be something like 1.5 for 1.5 microns
    y_size : float, optional, default = number of pixels in y direction
        Extent of tick marks in y axis
    num_ticks : unsigned int, optional, default = 4
        Number of tick marks on the X and Y axes
    stdevs : unsigned int (Optional. Default = None)
        Number of standard deviations to consider for plotting.  If None, full range is plotted.
    cbar_label : str, optional, default = None
        Labels for the colorbar. Use this for something like quantity (units)
    tick_font_size : unsigned int, optional, default = 14
        Font size to apply to x, y, colorbar ticks and colorbar label
    origin : str
        Where should the origin of the image data be located.  'lower' sets the origin to the
        bottom left, 'upper' sets it to the upper left.
        Default 'lower'
    kwargs : dictionary
        Anything else that will be passed on to imshow
    Returns
    -------
    im_handle : handle to image plot
        handle to image plot
    cbar : handle to color bar
        handle to color bar
    """
    if stdevs is not None:
        data_mean = np.mean(img)
        data_std = np.std(img)
        kwargs.update({'clim': [data_mean - stdevs * data_std,
                                data_mean + stdevs * data_std]})

    kwargs.update({'origin': origin})

    im_handle = axis.imshow(img, **kwargs)

    if show_xy_ticks is True:
        if x_size is not None and y_size is not None:
            x_ticks = np.linspace(0, img.shape[1] - 1, num_ticks, dtype=int)
            y_ticks = np.linspace(0, img.shape[0] - 1, num_ticks, dtype=int)
            axis.set_xticks(x_ticks)
            axis.set_yticks(y_ticks)
            axis.set_xticklabels([str(np.round(ind * x_size / (img.shape[1] - 1), 2)) for ind in x_ticks])
            axis.set_yticklabels([str(np.round(ind * y_size / (img.shape[0] - 1), 2)) for ind in y_ticks])
            set_tick_font_size(axis, tick_font_size)
    else:
        axis.set_xticks([])
        axis.set_yticks([])

    cbar = None
    if show_cbar:
        cbar = plt.colorbar(im_handle, ax=axis, orientation='vertical',
                            fraction=0.046, pad=0.04, use_gridspec=True)
        # cbar = axis.cbar_axes[count].colorbar(im_handle)

        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=tick_font_size)
        cbar.ax.tick_params(labelsize=tick_font_size)
    return im_handle, cbar


def plot_loops(excit_wfms, datasets, line_colors=[], dataset_names=[], evenly_spaced=True,
               plots_on_side=5, x_label='', y_label='', subtitle_prefix='Position', title='',
               use_rainbow_plots=False, fig_title_yoffset=1.05, h5_pos=None, **kwargs):
    """
    Plots loops from multiple datasets from up to 25 evenly spaced positions
    Parameters
    -----------
    excit_wfms : 1D numpy float array or list of same
        Excitation waveform in the time domain
    datasets : list of 2D numpy arrays or 2D hyp5.Dataset objects
        Datasets containing data arranged as (pixel, time)
    line_colors : list of strings
        Colors to be used for each of the datasets
    dataset_names : (Optional) list of strings
        Names of the different datasets to be compared
    evenly_spaced : boolean
        Evenly spaced positions or first N positions
    plots_on_side : unsigned int
        Number of plots on each side
    x_label : (optional) String
        X Label for all plots
    y_label : (optional) String
        Y label for all plots
    subtitle_prefix : (optional) String
        prefix for title over each plot
    title : (optional) String
        Main plot title
    use_rainbow_plots : (optional) Boolean
        Plot the lines as a function of spectral index (eg. time)
    fig_title_yoffset : (optional) float
        Y offset for the figure title. Value should be around 1
    h5_pos : HDF5 dataset reference or 2D numpy array
        Dataset containing position indices
    Returns
    ---------
    fig, axes
    """
    mode = 0
    # 0 = one excitation waveform and one dataset
    # 1 = one excitation waveform but many datasets
    # 2 = one excitation waveform for each of many dataset
    if type(datasets) in [h5py.Dataset, np.ndarray]:
        # can be numpy array or h5py.dataset
        num_pos = datasets.shape[0]
        num_points = datasets.shape[1]
        datasets = [datasets]
        excit_wfms = [excit_wfms]
        line_colors = ['b']
        dataset_names = ['Default']
        mode = 0
    else:
        # First check if the datasets are correctly shaped:
        num_pos_es = list()
        num_points_es = list()
        for dataset in datasets:
            num_pos_es.append(dataset.shape[0])
            num_points_es.append(dataset.shape[1])
        num_pos_es = np.array(num_pos_es)
        num_points_es = np.array(num_points_es)
        if np.unique(num_pos_es).size > 1:  # or np.unique(num_points_es).size > 1:
            raise ValueError('The first dimension of the datasets are not matching: ' + str(num_pos_es))
        num_pos = np.unique(num_pos_es)[0]

        if len(excit_wfms) == len(datasets):
            # one excitation waveform per dataset but now verify each size
            if not np.all([len(cur_ex) == cur_dset.shape[1] for cur_ex, cur_dset in zip(excit_wfms, datasets)]):
                raise ValueError('Number of points in the datasets do not match with the excitation waveforms')
            mode = 2
        else:
            # one excitation waveform for all datasets
            if np.unique(num_points_es).size > 1:
                raise ValueError('Datasets don not contain the same number of points: ' + str(num_points_es))
            # datasets of the same size but does this match with the size of excitation waveforms:
            if len(excit_wfms) != np.unique(num_points_es)[0]:
                raise ValueError('Number of points in dataset not matching with shape of excitation waveform')
            excit_wfms = [excit_wfms]
            mode = 1

        # Next the identification of datasets:
        if len(dataset_names) > len(datasets):
            # remove additional titles
            dataset_names = dataset_names[:len(datasets)]
        elif len(dataset_names) < len(datasets):
            # add titles
            dataset_names = dataset_names + ['Dataset' + ' ' + str(x) for x in range(len(dataset_names), len(datasets))]
        if len(line_colors) != len(datasets):
            # TODO: Generate colors from a user-specified colormap
            color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'brown', 'orange']
            if len(datasets) < len(color_list):
                remaining_colors = [x for x in color_list if x not in line_colors]
                line_colors += remaining_colors[:len(datasets) - len(color_list)]
            else:
                raise ValueError('Insufficient number of line colors provided')

    # cannot support rainbows with multiple datasets!
    use_rainbow_plots = use_rainbow_plots and len(datasets) == 1

    if mode != 2:
        # convert it to something like mode 2
        excit_wfms = [excit_wfms[0] for _ in range(len(datasets))]

    if mode != 0:
        # users are not allowed to specify colors
        _ = kwargs.pop('color', None)

    plots_on_side = min(abs(plots_on_side), 5)

    sq_num_plots = min(plots_on_side, int(round(num_pos ** 0.5)))
    if evenly_spaced:
        chosen_pos = np.linspace(0, num_pos - 1, sq_num_plots ** 2, dtype=int)
    else:
        chosen_pos = np.arange(sq_num_plots ** 2, dtype=int)

    fig, axes = plt.subplots(nrows=sq_num_plots, ncols=sq_num_plots, sharex=True, figsize=(12, 12))
    axes_lin = axes.flatten()

    for count, posn in enumerate(chosen_pos):
        if use_rainbow_plots:
            rainbow_plot(axes_lin[count], excit_wfms[0], datasets[0][posn], **kwargs)
        else:
            for dataset, ex_wfm, col_val in zip(datasets, excit_wfms, line_colors):
                axes_lin[count].plot(ex_wfm, dataset[posn], color=col_val, **kwargs)
        if h5_pos is not None:
            # print('Row ' + str(h5_pos[posn,1]) + ' Col ' + str(h5_pos[posn,0]))
            axes_lin[count].set_title('Row ' + str(h5_pos[posn, 1]) + ' Col ' + str(h5_pos[posn, 0]), fontsize=12)
        else:
            axes_lin[count].set_title(subtitle_prefix + ' ' + str(posn), fontsize=12)

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
        fig.suptitle(title, fontsize=14, y=fig_title_yoffset)
    plt.tight_layout()
    return fig, axes


###############################################################################


def plot_complex_map_stack(map_stack, num_comps=4, title=None, x_label='', y_label='',
                           subtitle_prefix='Component', amp_units=None, stdevs=2, **kwargs):
    """
    Plots the provided spectrograms from SVD V vector

    Parameters
    -------------
    map_stack : 3D numpy complex matrices
        Eigenvectors rearranged as - [component, row, col]
    num_comps : int
        Number of components to plot
    title : str, optional
        Title to plot above everything else
    x_label : str, optional
        Label for x axis
    y_label : str, optional
        Label for y axis
    subtitle_prefix : str, optional
        Prefix for the title over each image
    amp_units : str, optional
        Units for amplitude
    stdevs : int
        Number of standard deviations to consider for plotting

    Returns
    ---------
    fig, axes
    """
    if amp_units is None:
        amp_units = 'a.u.'

    figsize = kwargs.pop('figsize', (4, 4))
    figsize = (figsize[0] * num_comps, 8)

    num_comps = min(num_comps, map_stack.shape[0])

    fig, axes = plt.subplots(2, num_comps, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    if title is not None:
        fig.canvas.set_window_title(title)
        fig.suptitle(title, y=1.025)

    title_prefix = ''

    for index in range(num_comps):
        cur_axes = [axes.flat[index], axes.flat[index + num_comps]]
        funcs = [np.abs, np.angle]
        labels = ['Amplitude (' + amp_units + ')', 'Phase (rad)']
        for func, comp_name, axis, std_val in zip(funcs, labels, cur_axes, [stdevs, None]):
            kwargs['stdevs'] = std_val
            _ = plot_map(axis, func(map_stack[index]), **kwargs)

            if num_comps > 1:
                title_prefix = '%s %d - ' % (subtitle_prefix, index)
            axis.set_title('%s%s' % (title_prefix, comp_name))

            axis.set_aspect('auto')
            if index == 0:
                axis.set_ylabel(y_label)
        axis.set_xlabel(x_label)

    fig.tight_layout()

    return fig, axes


###############################################################################

def plot_complex_loop_stack(loop_stack, x_vec, title=None, subtitle_prefix='Component', num_comps=4, x_label='',
                            amp_units=None, **kwargs):
    """
    Plots the provided spectrograms from SVD V vector

    Parameters
    -------------
    loop_stack : 2D numpy complex matrix
        Loops rearranged as - [component, points]
    x_vec : 1D real numpy array
        The vector to plot against
    title : str
        Title to plot above everything else
    subtitle_prefix : str
        Subtile to of Figure
    num_comps : int
        Number of components to plot
    x_label : str
        Label for x axis
    amp_units : str, optional
        Units for amplitude

    Returns
    ---------
    fig, axes
    """
    if amp_units is None:
        amp_units = 'a.u.'

    if min(num_comps, loop_stack.shape[0]) == 1:
        subtitle_prefix = None

    num_comps = min(num_comps, loop_stack.shape[0])

    funcs = [np.abs, np.angle]
    comp_labs = ['Amplitude (' + amp_units + ')', 'Phase (rad)']

    figsize = kwargs.pop('figsize', (4, 4))
    figsize = (figsize[0] * num_comps, figsize[1] * len(funcs))

    fig, axes = plt.subplots(len(funcs), num_comps, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if title is not None:
        fig.canvas.set_window_title(title)
        fig.suptitle(title, y=1.025)

    for index in range(num_comps):
        cur_loop = loop_stack[index, :]
        cur_axes = [axes.flat[index], axes.flat[index + num_comps]]
        for func, y_label, axis in zip(funcs, comp_labs, cur_axes):
            axis.plot(x_vec, func(cur_loop), **kwargs)
            if subtitle_prefix is not None:
                axis.set_title('%s: %d' % (subtitle_prefix, index))
            if index == 0:
                axis.set_ylabel(y_label)
        axis.set_xlabel(x_label)

    fig.tight_layout()

    return fig, axes


###############################################################################


def plot_scree(scree, title='Scree', **kwargs):
    """
    Plots the scree or scree

    Parameters
    -------------
    scree : 1D real numpy array
        The scree vector from SVD
    title : str
        Figure title.  Default Scree

    Returns
    ---------
    fig, axes
    """
    fig = plt.figure(figsize=kwargs.pop('figsize', (6.5, 6)))
    axis = fig.add_axes([0.1, 0.1, .8, .8])  # left, bottom, width, height (range 0 to 1)
    kwargs.update({'color': kwargs.pop('color', 'b')})
    kwargs.update({'marker': kwargs.pop('marker', '*')})
    axis.loglog(np.arange(len(scree)) + 1, scree, **kwargs)
    axis.set_xlabel('Component')
    axis.set_ylabel('Variance')
    axis.set_title(title)
    axis.set_xlim(left=1, right=len(scree))
    axis.set_ylim(bottom=np.min(scree), top=np.max(scree))
    fig.canvas.set_window_title("Scree")

    return fig, axis


# ###############################################################################


def plot_map_stack(map_stack, num_comps=9, stdevs=2, color_bar_mode=None, evenly_spaced=False, reverse_dims=True,
                   title='Component', heading='Map Stack', colorbar_label='', fig_mult=(5, 5), pad_mult=(0.1, 0.07),
                   fig_title_yoffset=None, fig_title_size=None, **kwargs):
    """
    Plots the provided stack of maps

    Parameters
    -------------
    map_stack : 3D real numpy array
        structured as [component, rows, cols]
    num_comps : unsigned int
        Number of components to plot
    stdevs : int
        Number of standard deviations to consider for plotting
    color_bar_mode : String, Optional
        Options are None, single or each. Default None
    evenly_spaced : bool
        Default False
    reverse_dims : Boolean (Optional)
        Set this to False to accept data structured as [component, rows, cols]
    title : String or list of strings
        The titles for each of the plots.
        If a single string is provided, the plot titles become ['title 01', title 02', ...].
        if a list of strings (equal to the number of components) are provided, these are used instead.
    heading : String
        ###Insert description here### Default 'Map Stack'
    colorbar_label : String
        label for colorbar. Default is an empty string.
    fig_mult : length 2 array_like of uints
        Size multipliers for the figure.  Figure size is calculated as (num_rows*`fig_mult[0]`, num_cols*`fig_mult[1]`).
        Default (4, 4)
    pad_mult : length 2 array_like of floats
        Multipliers for the axis padding between plots in the stack.  Padding is calculated as
        (pad_mult[0]*fig_mult[1], pad_mult[1]*fig_mult[0]) for the width and height padding respectively.
        Default (0.1, 0.07)
    fig_title_yoffset : float
        Offset to move the figure title vertically in the figure
    fig_title_size : float
        Size of figure title
    kwargs : dictionary
        Keyword arguments to be passed to either matplotlib.pyplot.figure, mpl_toolkits.axes_grid1.ImageGrid, or
        pycroscopy.vis.plot_utils.plot_map.  See specific function documentation for the relavent options.

    Returns
    ---------
    fig, axes
    """
    if reverse_dims:
        map_stack = np.transpose(map_stack, (2, 0, 1))

    num_comps = abs(num_comps)
    num_comps = min(num_comps, map_stack.shape[0])

    if evenly_spaced:
        chosen_pos = np.linspace(0, map_stack.shape[0] - 1, num_comps, dtype=int)
    else:
        chosen_pos = np.arange(num_comps, dtype=int)

    if isinstance(title, list):
        if len(title) > num_comps:
            # remove additional titles
            title = title[:num_comps]
        elif len(title) < num_comps:
            # add titles
            title += ['Component' + ' ' + str(x) for x in range(len(title), num_comps)]
    else:
        if not isinstance(title, str):
            title = 'Component'
        title = [title + ' ' + str(x) for x in chosen_pos]

    fig_h, fig_w = fig_mult
    p_rows = int(np.floor(np.sqrt(num_comps)))
    p_cols = int(np.ceil(num_comps / p_rows))
    if p_rows * p_cols < num_comps:
        p_cols += 1

    pad_w, pad_h = pad_mult

    '''
    Set defaults for kwargs to the figure creation and extract any non-default values from current kwargs
    '''
    figkwargs = dict()

    if sys.version_info.major == 3:
        inspec_func = inspect.getfullargspec
    else:
        inspec_func = inspect.getargspec

    for key in inspec_func(plt.figure).args:
        if key in kwargs:
            figkwargs.update({key: kwargs.pop(key)})

    fig202 = plt.figure(figsize=(p_cols * fig_w, p_rows * fig_h), **figkwargs)

    '''
    Set defaults for kwargs to the ImageGrid and extract any non-default values from current kwargs
    '''
    igkwargs = {'cbar_pad': '1%',
                'cbar_size': '5%',
                'cbar_location': 'right',
                'direction': 'row',
                'add_all': True,
                'share_all': False,
                'aspect': True,
                'label_mode': 'L'}
    for key in igkwargs.keys():
        if key in kwargs:
            igkwargs.update({key: kwargs.pop(key)})

    axes202 = ImageGrid(fig202, 111, nrows_ncols=(p_rows, p_cols),
                        cbar_mode=color_bar_mode,
                        axes_pad=(pad_w * fig_w, pad_h * fig_h),
                        **igkwargs)

    fig202.canvas.set_window_title(heading)
    # These parameters have not been easy to fix:
    if fig_title_yoffset is None:
        fig_title_yoffset = 0.9
    if fig_title_size is None:
        fig_title_size = 16+(p_rows+ p_cols)
    fig202.suptitle(heading, fontsize=fig_title_size, y=fig_title_yoffset)

    for count, index, subtitle in zip(range(chosen_pos.size), chosen_pos, title):
        im, im_cbar = plot_map(axes202[count],
                               map_stack[index],
                               stdevs=stdevs, show_cbar=False, **kwargs)
        axes202[count].set_title(subtitle)

        if color_bar_mode is 'each':
            cb = axes202.cbar_axes[count].colorbar(im)
            cb.set_label_text(colorbar_label)

    if color_bar_mode is 'single':
        cb = axes202.cbar_axes[0].colorbar(im)
        cb.set_label_text(colorbar_label)

    return fig202, axes202


def plot_cluster_h5_group(h5_group, centroids_together=True, cmap=default_cmap):
    """
    Plots the cluster labels and mean response for each cluster

    Parameters
    ----------
    h5_group : h5py.Datagroup object
        H5 group containing the labels and mean response
    centroids_together : Boolean, optional - default = True
        Whether or nor to plot all centroids together on the same plot
    cmap : plt.cm object or str, optional
        Colormap to use for the labels map and the centroid.

    Returns
    -------
    fig : Figure
        Figure containing the plots
    axes : 1D array_like of axes objects
        Axes of the individual plots within `fig`
    """

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

    pos_ticks = [h5_pos_vals[:pos_dims[0], 0], h5_pos_vals[slice(0, None, pos_dims[0]), 1]]
    # prepare the axes ticks for the map

    pos_dims.reverse()  # go from slowest to fastest
    pos_dims = tuple(pos_dims)
    label_mat = np.reshape(h5_labels.value, pos_dims)

    # Figure out the correct units and labels for mean response:
    h5_spec_vals = h5_mean_resp.file[h5_mean_resp.attrs['Spectroscopic_Values']]
    x_spec_label = get_formatted_labels(h5_spec_vals)[0]

    # Figure out the correct axes labels for label map:
    pos_labels = get_formatted_labels(h5_pos_vals)

    y_spec_label = get_data_descriptor(h5_mean_resp)
    # TODO: cleaner x and y axes labels instead of 0.0000125 etc.

    if centroids_together:
        return plot_cluster_results_together(label_mat, mean_response, spec_val=np.squeeze(h5_spec_vals[0]),
                                             spec_label=x_spec_label, resp_label=y_spec_label,
                                             pos_labels=pos_labels, pos_ticks=pos_ticks, cmap=cmap)
    else:
        return plot_cluster_results_separate(label_mat, mean_response, max_centroids=4, x_label=x_spec_label,
                                             spec_val=np.squeeze(h5_spec_vals[0]), y_label=y_spec_label, cmap=cmap)


###############################################################################


def plot_cluster_results_together(label_mat, mean_response, spec_val=None, cmap=default_cmap,
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
    cmap = get_cmap_object(cmap)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    def __plot_centroids(centroids, ax, spec_val, spec_label, y_label, cmap, title=None):
        plot_line_family(ax, spec_val, centroids, label_prefix='Cluster', cmap=cmap)
        ax.set_ylabel(y_label)
        # ax.legend(loc='best')
        if title:
            ax.set_title(title)
            ax.set_xlabel(spec_label)

    if spec_val is None:
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
                 cmap=discrete_cmap(num_clusters, base_cmap=plt.cm.viridis))
    ax_map.axis('tight')"""
    pcol0 = ax_map.pcolor(label_mat, cmap=discrete_cmap(num_clusters, cmap=cmap))
    fig.colorbar(pcol0, ax=ax_map, ticks=np.arange(num_clusters))
    ax_map.axis('tight')
    ax_map.set_aspect('auto')
    ax_map.set_title('Cluster Label Map')

    fig.tight_layout()
    fig.canvas.set_window_title('Cluster results')

    return fig, axes


###############################################################################


def plot_cluster_results_separate(label_mat, cluster_centroids, max_centroids=4, cmap=default_cmap,
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
    cmap : plt.cm object or str, optional
        Colormap to use for the labels map and the centroids
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

    cmap = get_cmap_object(cmap)

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
    pcol0 = fax1.pcolor(label_mat, cmap=discrete_cmap(cluster_centroids.shape[0], cmap=cmap))
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
                    color=cmap(int(255 * index / (cluster_centroids.shape[0] - 1))))
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
        print('Creating full dendrogram from clusters')
        mode = None
    elif mode == 'Truncated':
        print('Creating truncated dendrogram from clusters.  Will stop at {}.'.format(last))
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


def plot_image_cleaning_results(raw_image, clean_image, stdevs=2, heading='Image Cleaning Results',
                                fig_mult=(4, 4), fig_args={}, **kwargs):
    """
    
    Parameters
    ----------
    raw_image
    clean_image
    stdevs
    fig_mult
    fig_args
    heading

    Returns
    -------

    """
    plot_args = {'cbar_pad': '2.0%', 'cbar_size': '4%', 'hor_axis_pad': 0.115, 'vert_axis_pad': 0.1,
                 'sup_title_size': 26, 'sub_title_size': 22, 'show_x_y_ticks': False, 'show_tick_marks': False,
                 'x_y_tick_font_size': 18, 'cbar_tick_font_size': 18}

    plot_args.update(fig_args)

    fig_h, fig_w = fig_mult
    p_rows = 2
    p_cols = 3

    fig_clean = plt.figure(figsize=(p_cols * fig_w, p_rows * fig_h))
    axes_clean = ImageGrid(fig_clean, 111, nrows_ncols=(p_rows, p_cols), cbar_mode='each',
                           cbar_pad=plot_args['cbar_pad'], cbar_size=plot_args['cbar_size'],
                           axes_pad=(plot_args['hor_axis_pad'] * fig_w, plot_args['vert_axis_pad'] * fig_h))
    fig_clean.canvas.set_window_title(heading)
    fig_clean.suptitle(heading, fontsize=plot_args['sup_title_size'])

    '''
    Calculate the removed noise and the FFT's of the raw, clean, and noise
    '''
    removed_noise = raw_image - clean_image
    blackman_window_rows = scipy.signal.blackman(clean_image.shape[0])
    blackman_window_cols = scipy.signal.blackman(clean_image.shape[1])

    FFT_raw = np.abs(np.fft.fftshift(
        np.fft.fft2(blackman_window_rows[:, np.newaxis] * raw_image * blackman_window_cols[np.newaxis, :]),
        axes=(0, 1)))
    FFT_clean = np.abs(np.fft.fftshift(
        np.fft.fft2(blackman_window_rows[:, np.newaxis] * clean_image * blackman_window_cols[np.newaxis, :]),
        axes=(0, 1)))
    FFT_noise = np.abs(np.fft.fftshift(
        np.fft.fft2(blackman_window_rows[:, np.newaxis] * removed_noise * blackman_window_cols[np.newaxis, :]),
        axes=(0, 1)))

    '''
    Now find the mean and standard deviation of the images
    '''
    raw_mean = np.mean(raw_image)
    clean_mean = np.mean(clean_image)
    noise_mean = np.mean(removed_noise)

    raw_std = np.std(raw_image)
    clean_std = np.std(clean_image)
    noise_std = np.std(removed_noise)
    fft_clean_std = np.std(FFT_clean)

    '''
    Make lists of everything needed to plot
    '''
    plot_names = ['Original Image', 'Cleaned Image', 'Removed Noise',
                  'FFT Original Image', 'FFT Cleaned Image', 'FFT Removed Noise']
    plot_data = [raw_image, clean_image, removed_noise, FFT_raw, FFT_clean, FFT_noise]
    plot_mins = [raw_mean - stdevs * raw_std, clean_mean - stdevs * clean_std, noise_mean - stdevs * noise_std, 0, 0, 0]
    plot_maxes = [raw_mean + stdevs * raw_std, clean_mean + stdevs * clean_std, noise_mean + stdevs * noise_std,
                  2 * stdevs * fft_clean_std, 2 * stdevs * fft_clean_std, 2 * stdevs * fft_clean_std]

    for count, ax, image, title, plot_min, plot_max in zip(range(6), axes_clean, plot_data,
                                                           plot_names, plot_mins, plot_maxes):
        im_handle, cbar_handle = plot_map(ax, image, stdevs, show_cbar=False, **kwargs)
        im_handle.set_clim(vmin=plot_min, vmax=plot_max)
        axes_clean[count].set_title(title, fontsize=plot_args['sub_title_size'])
        cbar = axes_clean.cbar_axes[count].colorbar(im_handle)
        cbar.ax.tick_params(labelsize=plot_args['cbar_tick_font_size'])

        if not plot_args['show_x_y_ticks']:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if not plot_args['show_tick_marks']:
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

    return fig_clean, axes_clean


def export_fig_data(fig, filename, include_images=False):
    """
    Export the data of all plots in the figure `fig` to a plain text file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the data to be exported
    filename : str
        The filename of the output text file
    include_images : bool
        Should images in the figure also be exported

    Returns
    -------

    """
    # Get the data from the figure
    axes = fig.get_axes()
    axes_dict = dict()
    for ax in axes:
        ax_dict = dict()

        ims = ax.get_images()
        if len(ims) != 0 and include_images:
            im_dict = dict()

            for im in ims:
                # Image data
                im_lab = im.get_label()

                # X-Axis
                x_ax = ax.get_xaxis()
                x_lab = x_ax.label.get_label()
                if x_lab == '':
                    x_lab = 'X'

                # Y-Axis
                y_ax = ax.get_yaxis()
                y_lab = y_ax.label.get_label()
                if y_lab == '':
                    y_lab = 'Y'

                im_dict[im_lab] = {'data': im.get_array().data,
                                   x_lab: x_ax.get_data_interval(),
                                   y_lab: y_ax.get_data_interval()}

            ax_dict['Images'] = im_dict

        lines = ax.get_lines()
        if len(lines) != 0:
            line_dict = dict()

            xlab = ax.get_xlabel()
            ylab = ax.get_ylabel()

            if xlab == '':
                xlab = 'X Data'
            if ylab == '':
                ylab = 'Y Data'

            for line in lines:
                line_dict[line.get_label()] = {xlab: line.get_xdata(),
                                               ylab: line.get_ydata()}

            ax_dict['Lines'] = line_dict

        if ax_dict != dict():
            axes_dict[ax.get_title()] = ax_dict

    '''
    Now that we have the data from the figure, we need to write it to file.
    '''

    filename = os.path.abspath(filename)
    basename, ext = os.path.splitext(filename)
    folder, _ = os.path.split(basename)

    spacer = '**********************************************\n'

    data_file = open(filename, 'w')

    data_file.write(fig.get_label() + '\n')
    data_file.write('\n')

    for ax_lab, ax in axes_dict.items():
        data_file.write('Axis: {} \n'.format(ax_lab))

        if 'Images' in ax:
            for im_lab, im in ax['Images'].items():
                data_file.write('Image: {} \n'.format(im_lab))
                data_file.write('\n')
                im_data = im.pop('data')
                for row in im_data:
                    row.tofile(data_file, sep='\t', format='%s')
                    data_file.write('\n')
                data_file.write('\n')

                for key, val in im.items():
                    data_file.write(key + '\n')

                    val.tofile(data_file, sep='\n', format='%s')
                    data_file.write('\n')

                data_file.write(spacer)

        if 'Lines' in ax:
            for line_lab, line_dict in ax['Lines'].items():
                data_file.write('Line: {} \n'.format(line_lab))
                data_file.write('\n')

                dim1, dim2 = line_dict.keys()

                data_file.write('{} \t {} \n'.format(dim1, dim2))
                for val1, val2 in zip(line_dict[dim1], line_dict[dim2]):
                    data_file.write('{} \t {} \n'.format(str(val1), str(val2)))

                data_file.write(spacer)

        data_file.write(spacer)

    data_file.close()
