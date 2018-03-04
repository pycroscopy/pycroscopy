"""
Created on 11/11/16 10:08 AM
@author: Suhas Somnath, Chris Smith
"""

import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
import numpy as np
import sys

from .plot_utils import plot_map, set_tick_font_size

if sys.version_info.major == 3:
    unicode = str


class VizDimension(object):
    def __init__(self, name, units, values=None):
        """
        Simple object that describes a dimension in a dataset by its name, units, and values
        Parameters
        ----------
        name : str / unicode
            Name of the dimension. For example 'Bias'
        units : str / unicode
            Units for this dimension. For example: 'V'
        values : array-like, optional
            Values over which this dimension was varied. A linearly increasing set of values will be assumed as default.
        """
        if not isinstance(name, (str, unicode)):
            raise TypeError('name should be a string')
        if not isinstance(units, (str, unicode)):
            raise TypeError('units should be a string')
        if values is not None:
            if not isinstance(values, (np.ndarray, list, tuple)):
                raise TypeError('values should be array like')
        self.name = name
        self.units = units
        self.values = values


def simple_ndim_visualizer(data_mat, pos_dims, spec_dims, spec_xdim=None, pos_xdim=None, verbose=False):
    """
    Generates a simple visualizer for visualizing simple datasets (up to 4 dimensions). The visualizer will ONLY work
    within the context of a jupyter notebook!

    The visualizer consists of two panels - spatial map and spectrograms. slider widgets will be generated to slice
    dimensions. The data matrix can be real, complex or compound valued

    Parameters
    ----------
    data_mat : numpy.array object
        Data to be visualized
    pos_dims : list / tuple
        List of VizDimension objects specifying all position dimensions in the same order as in data_mat
    spec_dims : list / tuple
        List of VizDimension objects specifying all position dimensions in the same order as in data_mat
    spec_xdim : str, optional
        Name of dimension with respect to which the spectral data will be plotted for 1D plots
    pos_xdim : str, optional
        Name of dimension with respect to which the position data will be plotted for 1D plots
    verbose : bool, optional
        Whether or not to print log statements
    """
    pos_dim_names = [item.name for item in pos_dims]
    spec_dim_names = [item.name for item in spec_dims]

    def check_data_type(data_mat):
        if data_mat.dtype.names is not None:
            return 2, list(data_mat.dtype.names), None
        if data_mat.dtype in [np.complex64, np.complex128, np.complex]:
            return 1, ['Real', 'Imaginary', 'Amplitude', 'Phase'], [np.real, np.imag, np.abs, np.angle]
        else:
            return 0, None, None

    def get_clims(data, stdev=2):
        avg = np.mean(data)
        std = np.std(data)
        return avg - stdev * std, avg + stdev * std

    def get_slice_string(slice_dict, dim_list):
        slice_str = ''
        for dimension in dim_list:
            assert isinstance(dimension, VizDimension)
            if dimension.name in slice_dict.keys():
                slice_str += '{} = {} {}\n'.format(dimension.name,
                                                   dimension.values[slice_dict[dimension.name]],
                                                   dimension.units)
        slice_str = slice_str[:-1]
        return slice_str

    def get_slicing_tuple(slice_dict):
        slice_list = []
        for dim_name in pos_dim_names + spec_dim_names:
            cur_slice = slice(None)
            if slice_dict[dim_name] is not None:
                cur_slice = slice(slice_dict[dim_name], slice_dict[dim_name] + 1)
            slice_list.append(cur_slice)
        return tuple(slice_list)

    def naive_slice(data_mat, slice_dict):
        return np.squeeze(data_mat[get_slicing_tuple(slice_dict)])

    def get_spatmap_slice_dict(slice_dict={}):
        spatmap_slicing = {}
        for name in pos_dim_names:
            spatmap_slicing[name] = None
        for ind, name in enumerate(spec_dim_names):
            spatmap_slicing[name] = slice_dict.get(name, data_mat.shape[ind + len(pos_dim_names)] // 2)
        return spatmap_slicing

    def get_spgram_slice_dict(slice_dict={}):
        spgram_slicing = {}
        for ind, name in enumerate(pos_dim_names):
            spgram_slicing[name] = slice_dict.get(name, data_mat.shape[ind] // 2)
        for name in spec_dim_names:
            spgram_slicing[name] = None
        return spgram_slicing

    def plot_1d(axis, image_mat, dim_name, dim_dict, component_title):
        axis.set_xlabel(dim_name + ' (' + dim_dict[dim_name].units + ')', fontsize=16)
        axis.set_ylabel(component_title, fontsize=16)
        if image_mat.shape[0] != dim_dict[dim_name].values.size:
            image_mat = image_mat.T
        img_handle = axis.plot(dim_dict[dim_name].values, image_mat)
        if image_mat.ndim > 1:
            other_dims = list(dim_dict.keys()).copy()
            other_dims.remove(dim_name)
            other_dims = other_dims[0]
            axis.legend(dim_dict[other_dims].values, fontsize=14)
        set_tick_font_size(axis, 14)
        return img_handle

    def plot_2d(axis, image_mat, clims, dim_list):
        if verbose:
            print('image shape: {}, x_vec: {}, y_vec: {}'.format(image_mat.shape, dim_list[1].values.shape,
                                                                 dim_list[0].values.shape))
        img, cbar = plot_map(axis, image_mat, aspect='auto', clim=clims,
                             x_vec=dim_list[1].values, y_vec=dim_list[0].values)
        axis.set_xlabel(dim_list[1].name + ' (' + dim_list[1].units + ')', fontsize=16)
        axis.set_ylabel(dim_list[0].name + ' (' + dim_list[0].units + ')', fontsize=16)
        return img, cbar

    def update_image(axis, img_handle, data_mat, slice_dict, twoD=True):
        if twoD:
            img_handle.set_data(naive_slice(data_mat, slice_dict))
        else:
            y_mat = naive_slice(data_mat, slice_dict)
            if y_mat.ndim > 1:
                if y_mat.shape[0] != len(img_handle):
                    y_mat = y_mat.T
            for line_handle, y_vec in zip(img_handle, y_mat):
                line_handle.set_ydata(y_vec)
            axis.set_ylim([np.min(y_mat), np.max(y_mat)])

    # ###########################################################################

    for parm, parm_name in zip([pos_dims, spec_dims], ['pos_dims', 'spec_dims']):
        if not isinstance(parm, (list, tuple)):
            raise TypeError('Expected {} to be of type: Iterable - example list or tuple'.format(parm_name))
        for item in parm:
            if not isinstance(item, VizDimension):
                raise TypeError('Expected items in {} to be of type: VizDimension'.format(parm_name))
        if len(parm) > 2:
            raise NotImplementedError('Currently not able to handle more than 2 position or spectroscopic dimensions.'
                                      ' {} contains {} dimensions'.format(parm_name, len(parm)))

    if len(pos_dims) + len(spec_dims) != data_mat.ndim:
        raise ValueError('Lengths of pos_dims and spec_dims not matching with that of the dimensions of data_mat')

    # now check if the dimension matches with that of the N dimensional dataset
    for parm, dim_type in zip([pos_dims, spec_dims], ['Position', 'Spectroscopic']):
        offset = 0
        if dim_type == 'Spectroscopic':
            offset += len(pos_dims)
        for ind, item in enumerate(parm):
            actual_ind = ind + offset
            if item.values is None:
                # Let's take this oppurtunity to fill in the values:
                item.values = np.arange(data_mat.shape[actual_ind])
                if verbose:
                    print('automatically generated reference {} values for dimension: {}'.format(dim_type, item.name))
            else:
                if len(item.values) != data_mat.shape[actual_ind]:
                    raise ValueError(
                        '{} dimension {} of size {} in the dataset does not have values of the same length: {}'
                        '.'.format(dim_type, item.name, data_mat.shape[actual_ind], len(item.values)))

    # create a dictionary that will allow lookup of values and units by name:
    pos_dims_dict = {}
    for dimension in pos_dims:
        pos_dims_dict[dimension.name] = dimension

    spec_dims_dict = {}
    for dimension in spec_dims:
        spec_dims_dict[dimension.name] = dimension

    if spec_xdim is not None:
        if not isinstance(spec_xdim, (str, unicode)):
            raise TypeError('spec_xdim should have been a string')
        if spec_xdim not in spec_dims_dict.keys():
            raise KeyError('{} not among the provided spectroscopic dimensions'.format(spec_xdim))

    if pos_xdim is not None:
        if not isinstance(pos_xdim, (str, unicode)):
            raise TypeError('spec_xdim should have been a string')
        if pos_xdim not in pos_dims_dict.keys():
            raise KeyError('{} not among the provided position dimensions'.format(pos_xdim))

    pos_plot_2d = len(pos_dims) > 1 and pos_xdim is None
    spec_plot_2d = len(spec_dims) > 1 and spec_xdim is None
    if verbose:
        print('Plot 2D: Positions: {}, Spectroscopic: {}'.format(pos_plot_2d, spec_plot_2d))

    if not spec_plot_2d and spec_xdim is None:
        # Take the largest dimension you can find:
        max_ind = np.argmax([len(item.values) for item in spec_dims])
        spec_xdim = spec_dims[max_ind].name
        if verbose:
            print('automatically chose X axis for 1D Spectroscopic plot as {}'.format(spec_xdim))

    if not pos_plot_2d and pos_xdim is None:
        # Take the largest dimension you can find:
        max_ind = np.argmax([len(item.values) for item in pos_dims])
        pos_xdim = pos_dims[max_ind].name
        if verbose:
            print('automatically chose X axis for 1D Position plot as {}'.format(pos_xdim))

    data_type, data_names, data_funcs = check_data_type(data_mat)

    sub_data = data_mat
    component_name = 'Real'

    if data_type == 1:
        if verbose:
            print('Data found to be of type: complex')
        sub_data = data_funcs[0](data_mat)
        component_name = data_names[0]
    elif data_type == 2:
        if verbose:
            print('Data found to be of type: compound')
        component_name = data_names[0]
        sub_data = data_mat[component_name]
    else:
        if verbose:
            print('Data found to be of type: scalar / real')

    component_title = 'Component: ' + component_name
    if verbose:
        print('default component name: {}'.format(component_name))

    clims = get_clims(sub_data)
    if verbose:
        print('Default clims: {}'.format(clims))

    spatmap_slicing = get_spatmap_slice_dict()
    spgram_slicing = get_spgram_slice_dict()
    if verbose:
        print('Slicing: Spatial: {}, Spectrogram: {}'.format(spatmap_slicing, spgram_slicing))
    current_spatmap = naive_slice(sub_data, spatmap_slicing)
    current_spgram = naive_slice(sub_data, spgram_slicing)
    if verbose:
        print('Spatial map data shape: {}, Spectrogram data shape: {}'.format(current_spatmap.shape,
                                                                              current_spgram.shape))

    fig, axes = plt.subplots(ncols=2, figsize=(15.5, 7))
    # axes[0].hold(True)
    spec_titles = get_slice_string(spatmap_slicing, spec_dims)
    axes[0].set_title('Spatial Map for\n' + component_title + '\n' + spec_titles, fontsize=18)
    if pos_plot_2d:
        img_spat, cbar_spat = plot_2d(axes[0], current_spatmap, clims, pos_dims)
        main_vert_line = axes[0].axvline(x=spgram_slicing[pos_dims[1].name], color='k')
        main_hor_line = axes[0].axhline(y=spgram_slicing[pos_dims[0].name], color='k')
    else:
        img_spat = plot_1d(axes[0], current_spatmap, pos_xdim, pos_dims_dict, component_title)

    pos_titles = get_slice_string(spgram_slicing, pos_dims)
    axes[1].set_title('Spectrogram for\n' + component_title + '\n' + pos_titles, fontsize=18)
    if spec_plot_2d:
        img_spec, cbar_spec = plot_2d(axes[1], current_spgram, clims, spec_dims)
    else:
        img_spec = plot_1d(axes[1], current_spgram, spec_xdim, spec_dims_dict, component_title)

    fig.tight_layout()

    slice_dict = {}
    for dim_ind, dim_name in enumerate([item.name for item in pos_dims]):
        slice_dict[dim_name] = (0, sub_data.shape[dim_ind] - 1, 1)
    for dim_ind, dim_name in enumerate([item.name for item in spec_dims]):
        slice_dict[dim_name] = (0, sub_data.shape[dim_ind + len(pos_dims)] - 1, 1)
    if data_type > 0:
        slice_dict['component'] = data_names

    # stupid and hacky way of doing this:
    global_vars = {'sub_data': sub_data, 'component_title': component_title}

    def update_plots(**kwargs):
        component_name = kwargs.get('component', None)
        if component_name is not None:
            if component_name != slice_dict['component']:
                # update the data and title:
                if data_type == 1:
                    func_ind = data_names.index(component_name)
                    sub_data = data_funcs[func_ind](data_mat)
                elif data_type == 2:
                    sub_data = data_mat[component_name]
                component_title = 'Component: ' + component_name
                # sub data and component_title here are now local, update gobal vars!
                global_vars.update({'sub_data': sub_data, 'component_title': component_title})

                clims = get_clims(sub_data)
                update_image(axes[0], img_spat, sub_data, spatmap_slicing, twoD=pos_plot_2d)
                if pos_plot_2d:
                    img_spat.set_clim(clims)
                else:
                    axes[0].set_ylabel(component_title, fontsize=16)
                update_image(axes[1], img_spec, sub_data, spgram_slicing, twoD=spec_plot_2d)
                if spec_plot_2d:
                    img_spec.set_clim(clims)
                else:
                    axes[1].set_ylabel(component_title, fontsize=16)

                spec_titles = get_slice_string(spatmap_slicing, spec_dims)
                axes[0].set_title('Spatial Map for\n' + component_title + '\n' + spec_titles)
                pos_titles = get_slice_string(spgram_slicing, pos_dims)
                axes[1].set_title('Spectrogram for\n' + component_title + '\n' + pos_titles)
                # print('Updated component!')

        # Check to see if spectrogram needs to be updated:
        update_spgram = False
        for dim_name in [item.name for item in pos_dims]:
            if kwargs[dim_name] != slice_dict[dim_name]:
                update_spgram = True
                break
        if update_spgram:
            # print('updating spectrogam + crosshairs')
            spgram_slicing.update(get_spgram_slice_dict(slice_dict=kwargs))
            update_image(axes[1], img_spec, global_vars['sub_data'], spgram_slicing, twoD=spec_plot_2d)
            pos_titles = get_slice_string(spgram_slicing, pos_dims)
            axes[1].set_title('Spectrogram for\n' + global_vars['component_title'] + '\n' + pos_titles, fontsize=18)
            if pos_plot_2d:
                main_vert_line.set_xdata(spgram_slicing[pos_dims[1].name])
                main_hor_line.set_ydata(spgram_slicing[pos_dims[0].name])

        update_spatmap = False
        for dim_name in [item.name for item in spec_dims]:
            if kwargs[dim_name] != slice_dict[dim_name]:
                update_spatmap = True
                break
        if update_spatmap:
            # print('updating spatial map')
            spatmap_slicing.update(get_spatmap_slice_dict(slice_dict=kwargs))
            update_image(axes[0], img_spat, global_vars['sub_data'], spatmap_slicing, twoD=pos_plot_2d)
            spec_titles = get_slice_string(spatmap_slicing, spec_dims)
            axes[0].set_title('Spatial Map for\n' + global_vars['component_title'] + '\n' + spec_titles, fontsize=18)

        slice_dict.update(kwargs)
        display(fig)

    widgets.interact(update_plots, **slice_dict)
