# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath, Chris R. Smith
"""
# TODO: All general plotting functions should support data with 1, 2, or 3 spatial dimensions.

from __future__ import division, print_function, absolute_import, unicode_literals

import inspect
import os
import sys
from numbers import Number
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid

if sys.version_info.major == 3:
    unicode = str

default_cmap = plt.cm.viridis


def reset_plot_params():
    """
    Resets the plot parameters to matplotlib default values
    Adapted from:
    https://stackoverflow.com/questions/26413185/how-to-recover-matplotlib-defaults-after-setting-stylesheet
    """

    mpl.rcParams.update(mpl.rcParamsDefault)
    # Also resetting ipython inline parameters
    inline_rc = dict(mpl.rcParams)
    mpl.rcParams.update(inline_rc)


def use_nice_plot_params():
    """
    Resets default plot parameters such as figure size, font sizes etc. to values better suited for scientific
    publications
    """
    # mpl.rcParams.keys()  # gets all allowable keys
    # mpl.rc('figure', figsize=(5.5, 5))
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
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be of type: str')
        else:
            var = ''

    if fig_title_yoffset is not None:
        if not isinstance(fig_title_yoffset, Number):
            raise TypeError('fig_title_yoffset should be a Number')
    else:
        fig_title_yoffset = 1.0

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
            raise ValueError('x_vec: {} should be of the same size as the second dimension of map_stack: '
                             '{}'.format(x_vec.shape, map_stack.shape))
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
    if stdevs is not None:
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
            if np.ceil((ax_ind + ncols) / ncols) == nrows:
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

    if not (isinstance(scree, np.ndarray) or isinstance(scree, h5py.Dataset)):
        raise TypeError('scree must be a 1D array or Dataset')
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
                   subtitle='Component', title='Map Stack', colorbar_label='', fig_mult=(5, 5), pad_mult=(0.1, 0.07),
                   x_label=None, y_label=None, title_yoffset=None, title_size=None, **kwargs):
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
    subtitle : String or list of strings
        The titles for each of the plots.
        If a single string is provided, the plot titles become ['title 01', title 02', ...].
        if a list of strings (equal to the number of components) are provided, these are used instead.
    title : String
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
    x_label : (optional) String
        X Label for all plots
    y_label : (optional) String
        Y label for all plots
    title_yoffset : float
        Offset to move the figure title vertically in the figure
    title_size : float
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

    for var, var_name in zip([title, colorbar_label, color_bar_mode, x_label, y_label],
                             ['title', 'colorbar_label', 'color_bar_mode', 'x_label', 'y_label']):
        if var is not None:
            if not isinstance(var, (str, unicode)):
                raise TypeError(var_name + ' should be a string')
        else:
            var = ''

    if color_bar_mode not in [None, 'single', 'each']:
        raise ValueError('color_bar_mode must be either None, "single", or "each"')
    for var, var_name in zip([stdevs, title_yoffset, title_size],
                             ['stdevs', 'title_yoffset', 'title_size']):
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

    if isinstance(subtitle, list):

        if len(subtitle) > num_comps:
            # remove additional subtitles
            subtitle = subtitle[:num_comps]
        elif len(subtitle) < num_comps:
            # add subtitles
            subtitle += ['Component' + ' ' + str(x) for x in range(len(subtitle), num_comps)]
    else:
        if not isinstance(subtitle, str):
            subtitle = 'Component'
        subtitle = [subtitle + ' ' + str(x) for x in chosen_pos]

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

    fig = plt.figure(figsize=(p_cols * fig_w, p_rows * fig_h), **figkwargs)

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

    axes = ImageGrid(fig, 111, nrows_ncols=(p_rows, p_cols),
                        cbar_mode=color_bar_mode,
                        axes_pad=(pad_w * fig_w, pad_h * fig_h),
                        **igkwargs)

    fig.canvas.set_window_title(title)
    # These parameters have not been easy to fix:
    if title_yoffset is None:
        title_yoffset = 0.9
    if title_size is None:
        title_size = 16 + (p_rows + p_cols)
    fig.suptitle(title, fontsize=title_size, y=title_yoffset)

    for count, index, curr_subtitle in zip(range(chosen_pos.size), chosen_pos, subtitle):
        im, im_cbar = plot_map(axes[count],
                               map_stack[index],
                               stdevs=stdevs, show_cbar=False, **kwargs)
        axes[count].set_title(curr_subtitle)

        if color_bar_mode is 'each':
            cb = axes.cbar_axes[count].colorbar(im)
            cb.set_label_text(colorbar_label)

        if count % p_cols == 0:
            axes[count].set_ylabel(y_label)
        if count >= (p_rows - 1) * p_cols:
            axes[count].set_xlabel(x_label)

    if color_bar_mode is 'single':
        cb = axes.cbar_axes[0].colorbar(im)
        cb.set_label_text(colorbar_label)

    return fig, axes


###############################################################################


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
