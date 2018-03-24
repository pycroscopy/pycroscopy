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
from numbers import Number
from warnings import warn
import h5py
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid

from ..io.hdf_utils import reshape_to_n_dims, get_formatted_labels, get_data_descriptor

if sys.version_info.major == 3:
    unicode = str

default_cmap = plt.cm.viridis


def use_nice_plot_params():
    """
    Resets default plot parameters such as figure size, font sizes etc. to values better suited for scientific
    publications
    """
    # mpl.rcParams.keys()  # gets all allowable keys
    mpl.rc('figure', figsize=(5.5, 5))
    mpl.rc('lines', linewidth=2)
    mpl.rc('axes', labelsize=16, titlesize=16)
    mpl.rc('figure', titlesize=20)
    mpl.rc('font', size=14)  # global font size
    mpl.rc('legend', fontsize=16, fancybox=True)
    mpl.rc('xtick.major', size=6)
    mpl.rc('xtick.minor', size=4)
    # mpl.rcParams['xtick.major.size'] = 6


def get_plot_grid_size(num_plots, fewer_rows=True):
    """
    Returns the number of rows and columns ideal for visualizing multiple (identical) plots within a single figure

    Parameters
    ----------
    num_plots : uint
        Number of identical subplots within a figure
    fewer_rows : bool, optional. Default = True
        Set to True if the grid should be short and wide or False for tall and narrow

    Returns
    -------
    nrows : uint
        Number of rows
    ncols : uint
        Number of columns
    """
    assert isinstance(num_plots, Number), 'num_plots must be a number'
    # force integer:
    num_plots = int(num_plots)
    if num_plots < 1:
        raise ValueError('num_plots was less than 0')

    if fewer_rows:
        nrows = int(np.floor(np.sqrt(num_plots)))
        ncols = int(np.ceil(num_plots / nrows))
    else:
        ncols = int(np.floor(np.sqrt(num_plots)))
        nrows = int(np.ceil(num_plots / ncols))

    return nrows, ncols


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
    assert isinstance(font_size, Number)
    font_size = max(1, int(font_size))

    def __set_axis_tick(axis):
        """
        Sets the font sizes to the x and y axis in the given axis object

        Parameters
        ----------
        axis : matplotlib.axes.Axes object
            axis to set font sizes
        """
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)

    mesg = 'axes must either be a matplotlib.axes.Axes object or an iterable containing such objects'

    if hasattr(axes, '__iter__'):
        for axis in axes:
            assert isinstance(axis, mpl.axes.Axes), mesg
            __set_axis_tick(axis)
    else:
        assert isinstance(axes, mpl.axes.Axes), mesg
        __set_axis_tick(axes)


def make_scalar_mappable(vmin, vmax, cmap=None):
    """
    Creates a scalar mappable object that can be used to create a colorbar for non-image (e.g. - line) plots

    Parameters
    ----------
    vmin : Number
        Minimum value for colorbar
    vmax : Number
        Maximum value for colorbar
    cmap : colormap object
        Colormap object to use

    Returns
    -------
    sm : matplotlib.pyplot.cm.ScalarMappable object
        The object that can used to create a colorbar via plt.colorbar(sm)

    Adapted from: https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    """
    assert isinstance(vmin, Number), 'vmin should be a number'
    assert isinstance(vmax, Number), 'vmax should be a number'
    assert vmin < vmax, 'vmin must be less than vmax'

    if cmap is None:
        cmap = default_cmap
    else:
        assert isinstance(cmap, (mpl.colors.Colormap, str, unicode))
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
    axis : matplotlib.axes.Axes
        Axis with multiple line objects
    num_steps : uint
        Number of steps in the colorbar
    discrete_ticks : (optional) bool
        Whether or not to have the ticks match the number of number of steps. Default = True
    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(num_steps, int) and num_steps > 0:
        raise TypeError('num_steps must be a whole number')
    assert isinstance(discrete_ticks, bool)

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
    elif not isinstance(cmap, mpl.colors.Colormap):
        raise TypeError('cmap should either be a matplotlib.colors.Colormap object or a string')
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
    if not isinstance(name, (str, unicode)):
        raise TypeError('name should be a string')
    if not isinstance(interp_vals, (list, tuple, np.array)):
        raise TypeError('interp_vals must be a list of tuples')
    if not isinstance(normalization_val, Number):
        raise TypeError('normalization_val must be a number')

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
    if not isinstance(name, (str, unicode)):
        raise TypeError('name should be a string')
    if not isinstance(solid_color, (list, tuple, np.ndarray)):
        raise TypeError('solid_color must be a list of numbers')
    if not len(solid_color) == 4:
        raise ValueError('solid-color should have fourth values')
    if not np.all([isinstance(x, Number) for x in solid_color]):
        raise TypeError('solid_color should have three numbers for red, green, blue')
    if not isinstance(normalization_val, Number):
        raise TypeError('normalization_val must be a number')
    if not isinstance(min_alpha, Number):
        raise TypeError('min_alpha should be a Number')
    if not isinstance(max_alpha, Number):
        raise TypeError('max_alpha should be a Number')
    if min_alpha >= max_alpha:
        raise ValueError('min_alpha must be less than max_alpha')

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

    elif isinstance(cmap, mpl.colors.Colormap):
        cmap = cmap.name
    elif not isinstance(cmap, (str, unicode)):
        raise TypeError('cmap should be a string or a matplotlib.colors.Colormap object')

    return plt.get_cmap(cmap, num_bins)


def rainbow_plot(axis, x_vec, y_vec, num_steps=32, **kwargs):
    """
    Plots the input against the output vector such that the color of the curve changes as a function of index

    Parameters
    ----------
    axis : matplotlib.axes.Axes object
        Axis to plot the curve
    x_vec : 1D float numpy array
        vector that forms the X axis
    y_vec : 1D float numpy array
        vector that forms the Y axis
    num_steps : unsigned int (Optional)
        Number of discrete color steps
    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(x_vec, (list, tuple, np.ndarray)):
        raise TypeError('x_vec must be array-like of numbers')
    if not isinstance(x_vec, (list, tuple, np.ndarray)):
        raise TypeError('x_vec must be array-like of numbers')
    x_vec = np.array(x_vec)
    y_vec = np.array(y_vec)
    assert x_vec.ndim == 1 and y_vec.ndim == 1, 'x_vec and y_vec must be 1D arrays'
    assert x_vec.shape == y_vec.shape, 'x_vec and y_vec must have the same shape'

    if not isinstance(num_steps, int):
        raise TypeError('num_steps must be an integer < size of x_vec')
    if num_steps < 2 or num_steps >= x_vec // 2:
        raise ValueError('num_steps should be a positive number. 1/4 to 1/16th of x_vec')
    assert num_steps < x_vec.size, 'num_steps must be an integer < size of x_vec'

    assert isinstance(kwargs, dict)
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
    axis : matplotlib.axes.Axes object
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
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(x_vec, (list, tuple, np.ndarray)):
        raise TypeError('x_vec must be array-like of numbers')
    x_vec = np.array(x_vec)
    assert x_vec.ndim == 1, 'x_vec must be a 1D array'
    if not isinstance(line_family, list):
        line_family = np.array(line_family)
    if not isinstance(line_family, np.ndarray):
        raise TypeError('line_family must be a 2d array of numbers')
    assert line_family.ndim == 2, 'line_family must be a 2D array'
    assert x_vec.size == line_family.shape[1], 'The size of the 2nd dimension of line_family must match with of x_vec'
    num_lines = line_family.shape[0]
    for var, var_name in zip([label_suffix, label_prefix], ['label_suffix', 'label_prefix']):
        if not isinstance(var, (str, unicode)):
            raise TypeError(var_name + ' needs to be a string')
    if not isinstance(y_offset, Number):
        raise TypeError('y_offset should be a Number')
    assert isinstance(show_cbar, bool)
    if line_names is not None:
        if not isinstance(line_names, (list, tuple)):
            raise TypeError('line_names should be a list of strings')
        if not np.all([isinstance(x, (str, unicode)) for x in line_names]):
            raise TypeError('line_names should be a list of strings')
        if len(line_names) != num_lines:
            raise ValueError('length of line_names not matching with that of line_family')

    cmap = get_cmap_object(kwargs.pop('cmap', None))

    if line_names is None:
        label_prefix = 'Line '
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


def plot_map(axis, img, show_xy_ticks=True, show_cbar=True, x_vec=None, y_vec=None,
             num_ticks=4, stdevs=None, cbar_label=None, tick_font_size=14, **kwargs):
    """
    Plots an image within the given axis with a color bar + label and appropriate X, Y tick labels.
    This is particularly useful to get readily interpretable plots for papers
    Parameters
    ----------
    axis : matplotlib.axes.Axes object
        Axis to plot this image onto
    img : 2D numpy array with real values
        Data for the image plot
    show_xy_ticks : bool, Optional, default = None, shown unedited
        Whether or not to show X, Y ticks
    show_cbar : bool, optional, default = True
        Whether or not to show the colorbar
    x_vec : array-like, 1D, optional
        The references values that will be used for tick values on the X axis
    y_vec : array-like, 1D, optional
        The references values that will be used for tick values on the Y axis
    num_ticks : unsigned int, optional, default = 4
        Number of tick marks on the X and Y axes
    stdevs : unsigned int (Optional. Default = None)
        Number of standard deviations to consider for plotting.  If None, full range is plotted.
    cbar_label : str, optional, default = None
        Labels for the colorbar. Use this for something like quantity (units)
    tick_font_size : unsigned int, optional, default = 14
        Font size to apply to x, y, colorbar ticks and colorbar label
    kwargs : dictionary
        Anything else that will be passed on to imshow

    Returns
    -------
    im_handle : handle to image plot
        handle to image plot
    cbar : handle to color bar
        handle to color bar

    Note
    ----
    The origin of the image will be set to the lower left corner. Use the kwarg 'origin' to change this
    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be a numpy array')
    if not img.ndim == 2:
        raise ValueError('img should be a 2D array')
    if not isinstance(show_xy_ticks, bool):
        raise TypeError('show_xy_ticks should be a boolean value')
    if not isinstance(show_cbar, bool):
        raise TypeError('show_cbar should be a boolean value')
    # checks for x_vec and y_vec are done below
    if num_ticks is not None:
        if not isinstance(num_ticks, int):
            raise TypeError('num_ticks should be a whole number')
        if num_ticks < 2:
            raise ValueError('num_ticks should be at least 2')
    if tick_font_size is not None:
        if not isinstance(tick_font_size, Number):
            raise TypeError('tick_font_size must be a whole number')
        if tick_font_size < 0:
            raise ValueError('tick_font_size must be a whole number')
    if stdevs is not None:
        if not isinstance(stdevs, Number):
            raise TypeError('stdevs should be a Number')
        data_mean = np.mean(img)
        data_std = np.std(img)
        kwargs.update({'clim': [data_mean - stdevs * data_std,
                                data_mean + stdevs * data_std]})

    kwargs.update({'origin': kwargs.pop('origin', 'lower')})

    im_handle = axis.imshow(img, **kwargs)
    assert isinstance(show_xy_ticks, bool)
    if show_xy_ticks is True:

        x_ticks = np.linspace(0, img.shape[1] - 1, num_ticks, dtype=int)
        if x_vec is not None:
            if not isinstance(x_vec, (np.ndarray, list, tuple, range)) or len(x_vec) != img.shape[1]:
                raise ValueError('x_vec should be array-like with shape equal to the second axis of img')
            x_tick_labs = [str(np.round(x_vec[ind], 2)) for ind in x_ticks]
        else:
            x_tick_labs = [str(ind) for ind in x_ticks]

        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_tick_labs)

        y_ticks = np.linspace(0, img.shape[0] - 1, num_ticks, dtype=int)
        if y_vec is not None:
            if not isinstance(y_vec, (np.ndarray, list, tuple, range)) or len(y_vec) != img.shape[0]:
                raise ValueError('y_vec should be array-like with shape equal to the first axis of img')
            y_tick_labs = [str(np.round(y_vec[ind], 2)) for ind in y_ticks]
        else:
            y_tick_labs = [str(ind) for ind in y_ticks]

        axis.set_yticks(y_ticks)
        axis.set_yticklabels(y_tick_labs)

        set_tick_font_size(axis, tick_font_size)
    else:
        axis.set_xticks([])
        axis.set_yticks([])

    cbar = None
    if not isinstance(show_cbar, bool):
        show_cbar = False

    if show_cbar:
        cbar = plt.colorbar(im_handle, ax=axis, orientation='vertical',
                            fraction=0.046, pad=0.04, use_gridspec=True)
        # cbar = axis.cbar_axes[count].colorbar(im_handle)

        if cbar_label is not None:
            if not isinstance(cbar_label, (str, unicode)):
                raise TypeError('cbar_label should be a string')
            cbar.set_label(cbar_label, fontsize=tick_font_size)
        cbar.ax.tick_params(labelsize=tick_font_size)
    return im_handle, cbar


def plot_curves(excit_wfms, datasets, line_colors=[], dataset_names=[], evenly_spaced=True,
                num_plots=25, x_label='', y_label='', subtitle_prefix='Position', title='',
                use_rainbow_plots=False, fig_title_yoffset=1.05, h5_pos=None, **kwargs):
    """
    Plots curves / spectras from multiple datasets from up to 25 evenly spaced positions
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
    num_plots : unsigned int
        Number of plots
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
    for var, var_name in zip([use_rainbow_plots, evenly_spaced], ['use_rainbow_plots', 'evenly_spaced']):
        if not isinstance(var, bool):
            raise TypeError(var_name + ' should be of type: bool')
    for var, var_name in zip([x_label, y_label, subtitle_prefix, title],
                             ['x_label', 'y_label', 'subtitle_prefix', 'title']):
        if not isinstance(var, (str, unicode)):
            raise TypeError(var_name + ' should be of type: str')
    if not isinstance(fig_title_yoffset, Number):
        raise TypeError('fig_title_yoffset should be a Number')
    if h5_pos is not None:
        if not isinstance(h5_pos, h5py.Dataset):
            raise TypeError('h5_pos should be a h5py.Dataset object')
    if not isinstance(num_plots, int) or num_plots < 1:
        raise TypeError('num_plots should be a number')

    for var, var_name, dim_size in zip([datasets, excit_wfms], ['datasets', 'excit_wfms'], [2, 1]):
        mesg = '{} should be {}D arrays or iterables (list or tuples) of {}D arrays' \
               '.'.format(var_name, dim_size, dim_size)
        if isinstance(var, (h5py.Dataset, np.ndarray)):
            if not len(var.shape) == dim_size:
                raise ValueError(mesg)
        elif isinstance(var, (list, tuple)):
            if not np.all([isinstance(dset, (h5py.Dataset, np.ndarray)) for dset in datasets]):
                raise TypeError(mesg)
        else:
            raise TypeError(mesg)

    # modes:
    # 0 = one excitation waveform and one dataset
    # 1 = one excitation waveform but many datasets
    # 2 = one excitation waveform for each of many dataset
    if isinstance(datasets, (h5py.Dataset, np.ndarray)):
        # can be numpy array or h5py.dataset
        num_pos = datasets.shape[0]
        num_points = datasets.shape[1]
        datasets = [datasets]
        if isinstance(excit_wfms, (np.ndarray, h5py.Dataset)):
            excit_wfms = [excit_wfms]
        elif isinstance(excit_wfms, list):
            if len(excit_wfms) == num_points:
                excit_wfms = [np.array(excit_wfms)]
            elif len(excit_wfms) == 1 and len(excit_wfms[0]) == num_points:
                excit_wfms = [np.array(excit_wfms[0])]
            else:
                raise ValueError('If only a single dataset is provided, excit_wfms should be a 1D array')
        line_colors = ['b']
        dataset_names = ['Default']
        mode = 0
    else:
        # dataset is a list of datasets
        # First check if the datasets are correctly shaped:
        num_pos_es = list()
        num_points_es = list()

        for dataset in datasets:
            if not isinstance(dataset, (h5py.Dataset, np.ndarray)):
                raise TypeError('datasets can be a list of 2D h5py.Dataset or numpy array objects')
            if len(dataset.shape) != 2:
                raise ValueError('Each datset should be a 2D array')
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

        for var, var_name in zip([dataset_names, line_colors], ['dataset_names', 'line_colors']):
            if not isinstance(var, (list, tuple)) or not np.all([isinstance(x, (str, unicode)) for x in var]):
                raise TypeError(var_name + ' should be a list of strings')
            if len(var) > 0 and len(var) != len(datasets):
                raise ValueError(var_name + ' is not of same length as datasets: ' + len(datasets))

        # Next the identification of datasets:
        if len(dataset_names) == 0:
            dataset_names = ['Dataset' + ' ' + str(x) for x in range(len(dataset_names), len(datasets))]

        if len(line_colors) == 0:
            # TODO: Generate colors from a user-specified colormap or consider using line family
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

    num_plots = min(min(num_plots, 49), num_pos)
    nrows, ncols = get_plot_grid_size(num_plots)

    if evenly_spaced:
        chosen_pos = np.linspace(0, num_pos - 1, nrows * ncols, dtype=int)
    else:
        chosen_pos = np.arange(nrows * ncols, dtype=int)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(12, 12))
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

        if count % ncols == 0:
            axes_lin[count].set_ylabel(y_label, fontsize=12)
        if count >= (nrows - 1) * ncols:
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


def plot_complex_spectra(map_stack, x_vec=None, num_comps=4, title=None, x_label='', y_label='', evenly_spaced=True,
                           subtitle_prefix='Component', amp_units=None, stdevs=2, **kwargs):
    """
    Plots the amplitude and phase components of the provided stack of complex valued spectrograms (2D images)

    Parameters
    -------------
    map_stack : 2D or 3D numpy complex matrices
        stack of complex valued 1D spectra arranged as [component, spectra] or
        2D images arranged as - [component, row, col]
    x_vec : 1D array-like, optional, default=None
        If the data are spectra (1D) instead of spectrograms (2D), x_vec is the reference array against which
    num_comps : int
        Number of images to plot
    title : str, optional
        Title to plot above everything else
    x_label : str, optional
        Label for x axis
    y_label : str, optional
        Label for y axis
    evenly_spaced : bool, optional. Default = True
        If True, images will be sampled evenly over the given dataset. Else, the first num_comps images will be plotted
    subtitle_prefix : str, optional
        Prefix for the title over each image
    amp_units : str, optional
        Units for amplitude
    stdevs : int
        Number of standard deviations to consider for plotting

    **kwargs will be passed on either to plot_map() or pyplot.plot()

    Returns
    ---------
    fig, axes
    """
    if not isinstance(map_stack, np.ndarray) or not map_stack.ndim in [2, 3]:
        raise TypeError('map_stack should be a 2/3 dimensional array arranged as [component, row, col] or '
                        '[component, spectra')
    if x_vec is not None:
        if not isinstance(x_vec, (list, tuple, np.ndarray)):
            raise TypeError('x_vec should be a 1D array')
        x_vec = np.array(x_vec)
        if x_vec.ndim != 1:
            raise ValueError('x_vec should be a 1D array')
        if x_vec.size != map_stack.shape[1]:
            raise ValueError('x_vec should be of the same size as the second dimension of map_stack')
    else:
        if map_stack.ndim == 2:
            x_vec = np.arange(map_stack.shape[1])

    if num_comps is None:
        num_comps = 4  # Default
    else:
        if not isinstance(num_comps, int) or not num_comps > 0:
            raise TypeError('num_comps should be a positive integer')
    for var, var_name in zip([title, x_label, y_label, subtitle_prefix, amp_units],
                             ['title', 'x_label', 'y_label', 'subtitle_prefix', 'amp_units']):
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be a string')
    if amp_units is None:
        amp_units = 'a.u.'
    if not isinstance(stdevs, Number) or stdevs <= 0:
        raise TypeError('stdevs should be a positive number')

    figsize = kwargs.pop('figsize', (4, 4))
    figsize = (figsize[0] * num_comps, 8)

    num_comps = min(24, min(num_comps, map_stack.shape[0]))

    if evenly_spaced:
        chosen_pos = np.linspace(0, map_stack.shape[0] - 1, num_comps, dtype=int)
    else:
        chosen_pos = np.arange(num_comps, dtype=int)

    nrows, ncols = get_plot_grid_size(num_comps)

    fig, axes = plt.subplots(nrows * 2, ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    if title is not None:
        fig.canvas.set_window_title(title)
        fig.suptitle(title, y=1.025)

    title_prefix = ''
    for comp_counter, comp_pos in enumerate(chosen_pos):
        ax_ind = (comp_counter // ncols) * (2 * ncols) + comp_counter % ncols
        cur_axes = [axes.flat[ax_ind], axes.flat[ax_ind + ncols]]
        funcs = [np.abs, np.angle]
        labels = ['Amplitude (' + amp_units + ')', 'Phase (rad)']
        for func, comp_name, axis, std_val in zip(funcs, labels, cur_axes, [stdevs, None]):
            y_vec = func(map_stack[comp_pos])
            if map_stack.ndim > 2:
                kwargs['stdevs'] = std_val
                _ = plot_map(axis, y_vec, **kwargs)
            else:
                axis.plot(x_vec, y_vec, **kwargs)

            if num_comps > 1:
                title_prefix = '%s %d - ' % (subtitle_prefix, comp_counter)
            axis.set_title('%s%s' % (title_prefix, comp_name))

            axis.set_aspect('auto')
            if ax_ind % ncols == 0:
                axis.set_ylabel(y_label)
            if np.ceil((ax_ind + ncols)/ncols) == nrows:
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
    if isinstance(scree, (list, tuple)):
        scree = np.array(scree)

    if not isinstance(scree, np.ndarray):
        raise TypeError('scree must be a 1D array')
    if not isinstance(title, (str, unicode)):
        raise TypeError('title must be a string')

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


def plot_map_stack(map_stack, num_comps=9, stdevs=2, color_bar_mode=None, evenly_spaced=False, reverse_dims=False,
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
    reverse_dims : Boolean (Optional), default = False
        Set this to True to accept data structured as [rows, cols, component]
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
    if not isinstance(map_stack, np.ndarray) or not map_stack.ndim == 3:
        raise TypeError('map_stack should be a 3 dimensional array arranged as [component, row, col]')
    if num_comps is None:
        num_comps = 4  # Default
    else:
        if not isinstance(num_comps, int) or num_comps < 1:
            raise TypeError('num_comps should be a positive integer')
    for var, var_name in zip([title, heading, colorbar_label, color_bar_mode],
                             ['title', 'heading', 'colorbar_label', 'color_bar_mode']):
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be a string')
    if color_bar_mode not in [None, 'single', 'each']:
        raise ValueError('color_bar_mode must be either None, "single", or "each"')
    for var, var_name in zip([stdevs, fig_title_yoffset, fig_title_size],
                             ['stdevs', 'fig_title_yoffset', 'fig_title_size']):
        if var is not None:
            if not isinstance(var, Number) or var <= 0:
                raise TypeError(var_name + ' of value: {} should be a number > 0'.format(var))
    for var, var_name in zip([evenly_spaced, reverse_dims], ['evenly_spaced', 'reverse_dims']):
        if not isinstance(var, bool):
            raise TypeError(var_name + ' should be a bool')
    for var, var_name in zip([fig_mult, pad_mult], ['fig_mult', 'pad_mult']):
        if not isinstance(var, (list, tuple, np.ndarray)) or len(var) != 2:
            raise TypeError(var_name + ' should be a tuple / list / numpy array of size 2')
        if not np.all([x > 0 and isinstance(x, Number) for x in var]):
            raise ValueError(var_name + ' should contain positive numbers')

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

    p_rows, p_cols = get_plot_grid_size(num_comps)

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
    axes : 1D array_like of matplotlib.axes.Axes objects
        Axes of the individual plots within `fig`
    """
    if not isinstance(h5_group, h5py.Group):
        raise TypeError('h5_group should be a h5py.Group')
    if not isinstance(centroids_together, bool):
        raise TypeError('centroids_together should be a bool')
    if not isinstance(cmap, (str, unicode, mpl.colors.Colormap)):
        raise TypeError('cmap should either be a string or a matplotlib.colors.Colormap object')

    h5_labels = h5_group['Labels']
    try:
        h5_mean_resp = h5_group['Mean_Response']
    except KeyError:
        # old PySPM format:
        h5_mean_resp = h5_group['Centroids']

    # Reshape the mean response to N dimensions
    mean_response, success = reshape_to_n_dims(h5_mean_resp)

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
    axes : 1D array_like of matplotlib.axes.Axes objects
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


def save_fig_filebox_button(fig, filename):
    """
    Create ipython widgets to allow the user to save a figure to the
    specified file.

    Parameters
    ----------
    fig : matplotlib.Figure
        The figure to be saved.
    filename : str
        The filename the figure should be saved to

    Returns
    -------
    widget_box : ipywidgets.HBox
        Widget box holding the text entry and save button

    """
    filename = os.path.abspath(filename)
    file_dir, filename = os.path.split(filename)

    name_box = widgets.Text(value=filename,
                            placeholder='Type something',
                            description='Output Filename:',
                            disabled=False,
                            layout={'width': '50%'})
    save_button = widgets.Button(description='Save figure')

    def _save_fig():
        save_path = os.path.join(file_dir, filename)
        fig.save_fig(save_path, dpi='figure')
        print('Figure saved to "{}".'.format(save_path))

    widget_box = widgets.HBox([name_box, save_button])

    save_button.on_click(_save_fig)

    return widget_box


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
                im_dict[im_lab] = im.get_array().data

                # X-Axis
                x_ax = ax.get_xaxis()
                x_lab = x_ax.label.get_label()
                if x_lab == '':
                    x_lab = 'X'

                im_dict[im_lab + x_lab] = x_ax.get_data_interval()

                # Y-Axis
                y_ax = ax.get_yaxis()
                y_lab = y_ax.label.get_label()
                if y_lab == '':
                    y_lab = 'Y'

                im_dict[im_lab + y_lab] = y_ax.get_data_interval()

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

    spacer = r'**********************************************\n'

    data_file = open(filename, 'w')

    data_file.write(fig.get_label() + '\n')
    data_file.write('\n')

    for ax_lab, ax in axes_dict.items():
        data_file.write('Axis: {} \n'.format(ax_lab))

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

