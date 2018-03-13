"""
Created on 11/11/16 10:08 AM
@author: Suhas Somnath, Chris Smith
"""
import os

import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
import numpy as np

from .plot_utils import plot_map, export_fig_data


def simple_ndim_visualizer(data_mat, pos_dim_names, pos_dim_units_old, spec_dim_names, spec_dim_units_old,
                           pos_ref_vals=None, spec_ref_vals=None, pos_plot_2d=True, spec_plot_2d=True, spec_xdim=None,
                           pos_xdim=None):
    """
    Generates a simple visualizer for visualizing simple datasets (up to 4 dimensions). The visualizer will ONLY work
    within the context of a jupyter notebook!

    The visualizer consists of two panels - spatial map and spectrograms. slider widgets will be generated to slice
    dimensions. The data matrix can be real, complex or compound valued

    Parameters
    ----------
    data_mat : numpy.array object
        Data to be visualized
    pos_dim_names : list of strings
        Names of the position dimensions
    pos_dim_units_old : list of strings
        Units for the position dimension
    spec_dim_names : list of strings
        Names of the spectroscopic dimensions
    spec_dim_units_old : list of strings
        Units for the spectroscopic dimensions
    pos_ref_vals : dictionary, optional
        Dictionary of names and reference values for each of the position dimensions.
        Default - linear distribution for each dimension
    spec_ref_vals : dictionary, optional
        Dictionary of names and reference values for each of the spectroscopic dimensions.
        Default - linear distribution for each dimension
    pos_plot_2d : bool, optional
        Whether or not to plot spatial data as 2D images. Default = True
    spec_plot_2d : bool, optional
        Whether or not to plot spectral data as 2D images. Default = True
    spec_xdim : str, optional
        Name of dimension with respect to which the spectral data will be plotted for 1D plots
    pos_xdim : str, optional
        Name of dimension with respect to which the position data will be plotted for 1D plots
    """
    def check_data_type(data_mat):
        if data_mat.dtype.names is not None:
            return 2, list(data_mat.dtype.names), None
        if data_mat.dtype in [np.complex64, np.complex128, np.complex]:
            return 1, ['Real','Imaginary', 'Amplitude','Phase'], [np.real, np.imag, np.abs, np.angle]
        else:
            return 0, None, None

    def get_clims(data, stdev=2):
        avg = np.mean(data)
        std = np.std(data)
        return (avg -stdev*std, avg + stdev*std)

    def get_slice_string(slice_dict, dim_names, values_dict, units_dict):
        slice_str = ''
        for cur_name in dim_names:
            if cur_name in dim_names:
                slice_str += '{} = {} {}\n'.format(cur_name,
                                                 values_dict[cur_name][slice_dict[cur_name]],
                                                 units_dict[cur_name])
        slice_str = slice_str[:-1]
        return slice_str

    def get_slicing_tuple(slice_dict):
        slice_list = []
        for dim_name in pos_dim_names + spec_dim_names:
            cur_slice = slice(None)
            if slice_dict[dim_name] is not None:
                cur_slice = slice(slice_dict[dim_name], slice_dict[dim_name]+1)
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

    def update_image(img_handle, data_mat, slice_dict, twoD=True):
        if twoD:
            img_handle.set_data(naive_slice(data_mat, slice_dict))
        else:
            y_mat = naive_slice(data_mat, slice_dict)
            if y_mat.ndim > 1:
                if y_mat.shape[0] != len(img_handle):
                    y_mat = y_mat.T
            for line_handle, y_vec in zip(img_handle, y_mat):
                line_handle.set_ydata(y_vec)
            img_handle[0].get_axes().set_ylim([np.min(y_mat), np.max(y_mat)])

    # ###########################################################################

    pos_plot_2d = pos_plot_2d and len(pos_dim_names) > 1
    spec_plot_2d = spec_plot_2d and len(spec_dim_names) > 1

    if not spec_plot_2d and spec_xdim is None:
        # Take the largest dimension you can find:
        spec_xdim = spec_dim_names[np.argmax(data_mat.shape[len(pos_dim_names):])]

    if not pos_plot_2d and pos_xdim is None:
        # Take the largest dimension you can find:
        pos_xdim = pos_dim_names[np.argmax(data_mat.shape[:len(pos_dim_names)])]

    if pos_ref_vals is None:
        spec_ref_vals = {}
        for ind, name in enumerate(pos_dim_names):
            spec_ref_vals[name] = np.arange(data_mat.shape[ind + len(pos_dim_names)])

    if spec_ref_vals is None:
        pos_ref_vals = {}
        for ind, name in enumerate(pos_dim_names):
            pos_ref_vals[name] = np.arange(data_mat.shape[ind])

    pos_dim_units = {}
    spec_dim_units = {}
    for name, unit in zip(pos_dim_names, pos_dim_units_old):
        pos_dim_units[name] = unit
    for name, unit in zip(spec_dim_names, spec_dim_units_old):
        spec_dim_units[name] = unit

    data_type, data_names, data_funcs = check_data_type(data_mat)

    sub_data = data_mat
    component_name = 'Real'

    if data_type == 1:
        sub_data = data_funcs[0](data_mat)
        component_name = data_names[0]
    elif data_type == 2:
        component_name = data_names[0]
        sub_data = data_mat[component_name]

    component_title = 'Component: ' + component_name

    clims = get_clims(sub_data)

    spatmap_slicing = get_spatmap_slice_dict()
    current_spatmap = naive_slice(sub_data, spatmap_slicing)
    spgram_slicing = get_spgram_slice_dict()
    current_spgram = naive_slice(sub_data, spgram_slicing)

    # print(current_spatmap.shape, current_spgram.shape)

    fig, axes = plt.subplots(ncols=2, figsize=(14,7))
    # axes[0].hold(True)
    spec_titles = get_slice_string(spatmap_slicing, spec_dim_names, spec_ref_vals, spec_dim_units)
    axes[0].set_title('Spatial Map for\n' + component_title + '\n' + spec_titles)
    if pos_plot_2d:
        img_spat, cbar_spat = plot_map(axes[0], current_spatmap, x_size=data_mat.shape[1], y_size=data_mat.shape[0],
                                       clim=clims)
        axes[0].set_xlabel(pos_dim_names[1] + ' (' + pos_dim_units_old[1] + ')')
        axes[0].set_ylabel(pos_dim_names[0] + ' (' + pos_dim_units_old[0] + ')')
        main_vert_line = axes[0].axvline(x=spgram_slicing[pos_dim_names[1]], color='k')
        main_hor_line = axes[0].axhline(y=spgram_slicing[pos_dim_names[0]], color='k')
    else:
        axes[0].set_xlabel(pos_xdim + ' (' + pos_dim_units[pos_xdim] + ')')
        if current_spatmap.shape[0] != pos_ref_vals[pos_xdim].size:
            current_spatmap = current_spatmap.T
        img_spat = axes[0].plot(pos_ref_vals[pos_xdim], current_spatmap)
        if current_spatmap.ndim > 1:
            other_pos_dim = pos_dim_names.copy()
            other_pos_dim.remove(pos_xdim)
            other_pos_dim = other_pos_dim[0]
            axes[0].legend(pos_ref_vals[other_pos_dim])

    pos_titles = get_slice_string(spgram_slicing, pos_dim_names, pos_ref_vals, pos_dim_units)
    axes[1].set_title('Spectrogram for\n' + component_title + '\n' + pos_titles)
    if spec_plot_2d:
        axes[1].set_xlabel(spec_dim_names[1] + ' (' + spec_dim_units_old[1] + ')')
        axes[1].set_ylabel(spec_dim_names[0] + ' (' + spec_dim_units_old[0] + ')')
        img_spec, cbar_spec = plot_map(axes[1], current_spgram,
                                       x_size=data_mat.shape[len(pos_dim_names) + 1],
                                       y_size=data_mat.shape[len(pos_dim_names)],
                                        cbar_label=component_name, clim=clims)
    else:
        axes[1].set_xlabel(spec_xdim + ' (' + spec_dim_units[spec_xdim] + ')')
        if current_spgram.shape[0] != spec_ref_vals[spec_xdim].size:
            current_spgram = current_spgram.T
        img_spec = axes[1].plot(spec_ref_vals[spec_xdim], current_spgram)
        if current_spgram.ndim > 1:
            other_spec_dim = spec_dim_names.copy()
            other_spec_dim.remove(spec_xdim)
            other_spec_dim = other_spec_dim[0]
            axes[1].legend(spec_ref_vals[other_spec_dim])

    fig.tight_layout()

    slice_dict = {}
    for dim_ind, dim_name in enumerate(pos_dim_names):
        slice_dict[dim_name] = (0, sub_data.shape[dim_ind] -1, 1)
    for dim_ind, dim_name in enumerate(spec_dim_names):
        slice_dict[dim_name] = (0, sub_data.shape[dim_ind + len(pos_dim_names)] - 1, 1)
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
                update_image(img_spat, sub_data, spatmap_slicing, twoD=pos_plot_2d)
                if pos_plot_2d:
                    img_spat.set_clim(clims)
                update_image(img_spec, sub_data, spgram_slicing, twoD=spec_plot_2d)
                if spec_plot_2d:
                    img_spec.set_clim(clims)

                spec_titles = get_slice_string(spatmap_slicing, spec_dim_names, spec_ref_vals, spec_dim_units)
                axes[0].set_title('Spatial Map for\n' + component_title + '\n' + spec_titles)
                pos_titles = get_slice_string(spgram_slicing, pos_dim_names, pos_ref_vals, pos_dim_units)
                axes[1].set_title('Spectrogram for\n' + component_title + '\n' + pos_titles)
                # print('Updated component!')

        # Check to see if spectrogram needs to be updated:
        update_spgram = False
        for dim_name in pos_dim_names:
            if kwargs[dim_name] != slice_dict[dim_name]:
                update_spgram = True
                break
        if update_spgram:
            # print('updating spectrogam + crosshairs')
            spgram_slicing.update(get_spgram_slice_dict(slice_dict=kwargs))
            update_image(img_spec, global_vars['sub_data'], spgram_slicing, twoD=spec_plot_2d)
            pos_titles = get_slice_string(spgram_slicing, pos_dim_names, pos_ref_vals, pos_dim_units)
            axes[1].set_title('Spectrogram for\n' + global_vars['component_title'] + '\n' + pos_titles)
            if pos_plot_2d:
                main_vert_line.set_xdata(spgram_slicing[pos_dim_names[1]])
                main_hor_line.set_ydata(spgram_slicing[pos_dim_names[0]])

        update_spatmap = False
        for dim_name in spec_dim_names:
            if kwargs[dim_name] != slice_dict[dim_name]:
                update_spatmap = True
                break
        if update_spatmap:
            # print('updating spatial map')
            spatmap_slicing.update(get_spatmap_slice_dict(slice_dict=kwargs))
            update_image(img_spat, global_vars['sub_data'], spatmap_slicing, twoD=pos_plot_2d)
            spec_titles = get_slice_string(spatmap_slicing, spec_dim_names, spec_ref_vals, spec_dim_units)
            axes[0].set_title('Spatial Map for\n' + global_vars['component_title'] + '\n' + spec_titles)

        slice_dict.update(kwargs)
        display(fig)

    widgets.interact(update_plots, **slice_dict);


def save_fig_filebox_button(fig, filename):
    """
    Create ipython widgets to allow the user to save a figure to the
    specified file.  If a .txt file is specified, the data within the
    figure will be exported instead.

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

    def _save_fig(junk):
        filename = name_box.value
        save_path = os.path.join(file_dir, filename)
        _, ext = os.path.splitext(filename)
        if ext == '.txt':
            export_fig_data(fig, save_path, include_images=True)
            print('Figure data exported to "{}".'.format(save_path))
        else:
            fig.savefig(save_path, dpi='figure')
            print('Figure saved to "{}".'.format(save_path))

    widget_box = widgets.HBox([name_box, save_button])

    save_button.on_click(_save_fig)

    return widget_box
