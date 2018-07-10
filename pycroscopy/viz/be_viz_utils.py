# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 13:29:12 2017

@author: Chris R. Smith, Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import os
from warnings import warn

import ipywidgets as widgets
import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from functools import partial

from pyUSID.viz.plot_utils import plot_curves, plot_map_stack, get_cmap_object, plot_map, set_tick_font_size, \
    plot_complex_spectra
from pyUSID.viz.jupyter_utils import save_fig_filebox_button
from ..analysis.utils.be_loop import loop_fit_function
from ..analysis.utils.be_sho import SHOfunc
from pyUSID.io.hdf_utils import reshape_to_n_dims, get_auxiliary_datasets, get_sort_order, get_dimensionality, \
    get_attr, get_source_dataset
from pyUSID import USIDataset


def visualize_sho_results(h5_main, save_plots=True, show_plots=True, cmap=None):
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
    cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
        Requested color map

    Returns
    -------
    None
    """
    cmap = get_cmap_object(cmap)

    def __plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, spec_var_title, meas_var_title, save_plots,
                          folder_path, basename, num_rows, num_cols):
        plt_title = grp_name + '_' + win_title + '_Loops'
        fig, ax = plot_curves(ac_vec, resp_mat, evenly_spaced=True, num_plots=25, x_label=spec_var_title,
                              y_label=meas_var_title, subtitle_prefix='Position', title=plt_title)
        if save_plots:
            fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

        plt_title = grp_name + '_' + win_title + '_Snaps'
        fig, axes = plot_map_stack(resp_mat.reshape(num_rows, num_cols, resp_mat.shape[1]),
                                   color_bar_mode="each", evenly_spaced=True, subtitle='UDVS Step #',
                                   title=plt_title, cmap=cmap)
        if save_plots:
            fig.savefig(os.path.join(folder_path, basename + '_' + plt_title + '.png'), format='png', dpi=300)

        return fig

    print('Creating plots of SHO Results from {}.'.format(h5_main.name))

    h5_file = h5_main.file

    expt_type = get_attr(h5_file, 'data_type')
    if expt_type not in ['BEPSData', 'BELineData']:
        warn('Unsupported data format')
        return
    isBEPS = expt_type == 'BEPSData'

    (folder_path, basename) = os.path.split(h5_file.filename)
    basename, _ = os.path.splitext(basename)

    sho_grp = h5_main.parent

    chan_grp = h5_file['/'.join(sho_grp.name[1:].split('/')[:2])]

    grp_name = '_'.join(chan_grp.name[1:].split('/'))
    grp_name = '_'.join([grp_name, sho_grp.name.split('/')[-1].split('-')[0], h5_main.name.split('/')[-1]])

    try:
        h5_pos = h5_main.h5_pos_inds
    except KeyError:
        print('No Position_Indices found as attribute of {}'.format(h5_main.name))
        print('Rows and columns will be calculated from dataset shape.')
        num_rows = int(np.floor((np.sqrt(h5_main.shape[0]))))
        num_cols = int(np.reshape(h5_main, [num_rows, -1, h5_main.shape[1]]).shape[1])
    else:
        num_rows, num_cols = h5_main.pos_dim_sizes

    try:
        h5_spec_vals = h5_file[get_attr(h5_main, 'Spectroscopic_Values')]
    except Exception:
        raise

    # Assume that there's enough memory to load all the guesses into memory
    amp_mat = h5_main['Amplitude [V]'] * 1000  # convert to mV ahead of time
    freq_mat = h5_main['Frequency [Hz]'] / 1000
    q_mat = h5_main['Quality Factor']
    phase_mat = h5_main['Phase [rad]']
    rsqr_mat = h5_main['R2 Criterion']

    fig_list = list()
    if isBEPS:
        meas_type = chan_grp.parent.attrs['VS_mode']
        # basically 3 kinds for now - DC/current, AC, UDVS - lets ignore this
        if meas_type == 'load user defined VS Wave from file':
            warn('Not handling custom experiments for now')
            # h5_file.close()
            return

        # Plot amplitude and phase maps at one or more UDVS steps
        if meas_type == 'AC modulation mode with time reversal':
            center = int(h5_spec_vals.shape[1] * 0.5)
            ac_vec = np.squeeze(h5_spec_vals[h5_spec_vals.attrs['AC_Amplitude']][:, 0:center])

            forw_resp = np.squeeze(amp_mat[:, slice(0, center)])
            rev_resp = np.squeeze(amp_mat[:, slice(center, None)])

            for win_title, resp_mat in zip(['Forward', 'Reverse'], [forw_resp, rev_resp]):
                fig_list.append(__plot_loops_maps(ac_vec, resp_mat, grp_name, win_title, 'AC Amplitude', 'Amplitude',
                                                  save_plots, folder_path, basename, num_rows, num_cols))
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
                    fig_list.append(__plot_loops_maps(dc_vec, resp_mat, grp_name, win_title, 'DC Bias',
                                                      'Piezoresponse (a.u.)', save_plots, folder_path,
                                                      basename, num_rows, num_cols))
            else:
                fig_list.append(__plot_loops_maps(dc_vec, phase_mat * amp_mat, grp_name, '', 'DC Bias',
                                                  'Piezoresponse (a.u.)', save_plots, folder_path, basename,
                                                  num_rows, num_cols))

    else:  # BE-Line can only visualize the amplitude and phase maps:
        amp_mat = amp_mat.reshape(num_rows, num_cols)
        freq_mat = freq_mat.reshape(num_rows, num_cols)
        q_mat = q_mat.reshape(num_rows, num_cols)
        phase_mat = phase_mat.reshape(num_rows, num_cols)
        rsqr_mat = rsqr_mat.reshape(num_rows, num_cols)

        fig_ms, ax_ms = plot_map_stack(np.dstack((amp_mat, freq_mat, q_mat, phase_mat, rsqr_mat)).T,
                                       num_comps=5, color_bar_mode='each', title=grp_name,
                                       subtitle=['Amplitude (mV)', 'Frequency (kHz)', 'Quality Factor', 'Phase (deg)',
                                                 'R^2 Criterion'], cmap=cmap)

        fig_list.append(fig_ms)
        if save_plots:
            plt_path = os.path.join(folder_path, basename + '_' + grp_name + 'Maps.png')
            fig_ms.savefig(plt_path, format='png', dpi=300)

    if show_plots:
        plt.show()

    return fig_list


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
    loops_shifted = np.roll(ds_proj_loops, shift_ind, axis=1)

    num_plots = np.min([5, int(np.sqrt(ds_proj_loops.shape[0]))])
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots, figsize=(18, 18))
    positions = np.linspace(0, ds_proj_loops.shape[0] - 1, num_plots ** 2, dtype=np.int)
    for ax, pos in zip(axes.flat, positions):
        ax.plot(vdc_shifted, loops_shifted[pos, :], 'k', label='Raw')
        ax.plot(vdc_shifted, loop_fit_function(vdc_shifted, np.array(list(ds_guess[pos]))), 'g', label='guess')
        ax.plot(vdc_shifted, loop_fit_function(vdc_shifted, np.array(list(ds_fit[pos]))), 'r--', label='Fit')
        ax.set_xlabel('V_DC (V)')
        ax.set_ylabel('PR (a.u.)')
        ax.set_title('Position ' + str(pos))
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def jupyter_visualize_beps_sho(pc_sho_dset, step_chan, resp_func=None, resp_label='Response', cmap=None):
    """
    Jupyer notebook ONLY function. Sets up an interactive visualizer for viewing SHO fitted BEPS data.
    Currently, this is limited to DC and AC spectroscopy datasets.

    Parameters
    ----------
    pc_sho_dset : USIDataset
        dataset to be plotted
    step_chan : string / unicode
        Name of the channel that forms the primary spectroscopic axis (eg - DC offset)
    resp_func : function (optional)
        Function to apply to the spectroscopic data. Currently, DC spectroscopy uses A*cos(phi) and AC spectroscopy
        uses A
    resp_label : string / unicode (optional)
        Label for the response (y) axis.
    cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
        Requested color map
    """
    cmap = get_cmap_object(cmap)

    h5_sho_spec_inds = pc_sho_dset.h5_spec_inds
    h5_sho_spec_vals = pc_sho_dset.h5_spec_vals
    spec_nd, _ = reshape_to_n_dims(h5_sho_spec_inds, h5_spec=h5_sho_spec_inds)
    sho_spec_dims = pc_sho_dset.spec_dim_sizes
    sho_spec_labels = pc_sho_dset.spec_dim_labels

    h5_pos_inds = pc_sho_dset.h5_pos_inds
    pos_nd, _ = reshape_to_n_dims(h5_pos_inds, h5_pos=h5_pos_inds)
    pos_dims = pc_sho_dset.pos_dim_sizes
    pos_labels = pc_sho_dset.pos_dim_labels

    # reshape to X, Y, step, all others
    spec_step_dim_ind = sho_spec_labels.index(step_chan)
    step_dim_ind = len(pos_dims) + spec_step_dim_ind

    # move the step dimension to be the first after all position dimensions
    rest_sho_dim_order = list(range(len(pos_dims), len(pc_sho_dset.n_dim_sizes)))
    rest_sho_dim_order.remove(step_dim_ind)
    new_order = list(range(len(pos_dims))) + [step_dim_ind] + rest_sho_dim_order

    # Transpose the 3D dataset to this shape:
    guess_nd_data = pc_sho_dset.get_n_dim_form()
    guess_nd_data = np.transpose(guess_nd_data, new_order)

    # Now move the step dimension to the front for the spec labels as well
    new_spec_order = list(range(len(sho_spec_labels)))
    new_spec_order.remove(spec_step_dim_ind)
    new_spec_order = [spec_step_dim_ind] + new_spec_order

    # new_spec_labels = sho_spec_labels[new_spec_order]
    new_spec_dims = np.array(sho_spec_dims)[new_spec_order]

    # Now collapse all additional dimensions
    final_guess_shape = pos_dims + [new_spec_dims[0]] + [-1]
    sho_dset_collapsed = np.reshape(guess_nd_data, final_guess_shape).squeeze()

    # Get the bias matrix:
    bias_mat, _ = reshape_to_n_dims(h5_sho_spec_vals, h5_spec=h5_sho_spec_inds)
    bias_mat = np.transpose(bias_mat[spec_step_dim_ind],
                            new_spec_order).reshape(sho_dset_collapsed.shape[len(pos_dims):])
    if bias_mat.ndim == 1:
        bias_mat = np.atleast_2d(bias_mat).T

    # This is just the visualizer:
    sho_quantity = 'Amplitude [V]'
    step_ind = 0
    row_ind = 1
    col_ind = 1

    def dc_spectroscopy_func(resp_vec):
        return resp_vec['Amplitude [V]'] * np.cos(resp_vec['Phase [rad]']) * 1E+3

    def ac_spectroscopy_func(resp_vec):
        return resp_vec['Amplitude [V]']

    if resp_func is None:
        if step_chan == 'DC_Offset':
            resp_func = dc_spectroscopy_func
            resp_label = 'A cos($\phi$) (a. u.)'
        else:
            resp_func = ac_spectroscopy_func
            resp_label = 'Amplitude (a. u.)'

    not_step_chan = sho_spec_labels.copy()
    not_step_chan.remove(step_chan)
    spatial_dict = {step_chan: [step_ind]}
    resp_dict = {pos_labels[-1]: [row_ind],
                 pos_labels[-2]: [col_ind]}
    for key in pos_labels[:-2]:
        spatial_dict[key] = [0]
        resp_dict[key] = [0]
    if not_step_chan is not None:
        for key in not_step_chan:
            spatial_dict[key] = [0]

    spatial_map = pc_sho_dset.slice(spatial_dict, as_scalar=False)[0][sho_quantity].squeeze()
    resp_vec = resp_func(pc_sho_dset.slice(resp_dict, as_scalar=False)[0].reshape(bias_mat.shape))

    fig = plt.figure(figsize=(12, 8))
    ax_bias = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=1)
    ax_map = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    ax_loop = plt.subplot2grid((3, 2), (0, 1), colspan=1, rowspan=3)

    ax_bias.plot(bias_mat[:, 0])
    ax_bias.set_xlabel('Bias Step')
    ax_bias.set_ylabel(step_chan.replace('_', ' ') + ' (V)')
    bias_slider = ax_bias.axvline(x=step_ind, color='r')

    img_map, img_cmap = plot_map(ax_map, spatial_map.T, show_xy_ticks=True)

    map_title = '{} - {}={}'.format(sho_quantity, step_chan, bias_mat[step_ind][0])
    ax_map.set_xlabel(pos_labels[-1])
    ax_map.set_ylabel(pos_labels[-2])
    ax_map.set_title(map_title)
    crosshair = ax_map.plot(row_ind, col_ind, 'k+')[0]

    ax_loop.axvline(x=0, color='gray', linestyle='--')
    ax_loop.axhline(y=0, color='gray', linestyle='--')
    line_handles = ax_loop.plot(bias_mat, resp_vec)
    ax_loop.set_xlabel(step_chan.replace('_', ' ') + ' (V)')
    ax_loop.set_ylabel(resp_label)
    fig.tight_layout()

    plt.show()

    # Build sliders for any extra Position Dimensions
    pos_sliders = dict()
    for ikey, key in enumerate(pos_labels[:-2]):
        pos_sliders[key] = widgets.IntSlider(value=0, min=0, max=pos_dims[ikey] - 1,
                                             step=1, description='{} Step:'.format(key),
                                             continuous_update=False)

    def update_sho_plots(sho_quantity, step_ind):
        bias_slider.set_xdata((step_ind, step_ind))
        spatial_dict[step_chan] = [step_ind]
        spatial_map = pc_sho_dset.slice(spatial_dict, as_scalar=False)[0][sho_quantity].squeeze()
        map_title = '{} - {}={}'.format(sho_quantity, step_chan, bias_mat[step_ind][0])
        ax_map.set_title(map_title)
        img_map.set_data(spatial_map.T)
        spat_mean = np.mean(spatial_map)
        spat_std = np.std(spatial_map)
        img_map.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)

    def update_resp_plot(resp_dict):
        resp_vec = resp_func(pc_sho_dset.slice(resp_dict, as_scalar=False)[0].reshape(bias_mat.shape)).T
        for line_handle, data in zip(line_handles, resp_vec):
            line_handle.set_ydata(data)

        ax_loop.relim()
        ax_loop.autoscale_view()

    def pos_picker(event):
        if not img_map.axes.in_axes(event):
            return

        xdata = int(round(event.xdata))
        ydata = int(round(event.ydata))

        resp_dict[pos_labels[-1]] = [xdata]
        resp_dict[pos_labels[-2]] = [ydata]

        crosshair.set_xdata(xdata)
        crosshair.set_ydata(ydata)

        update_resp_plot(resp_dict)

        fig.canvas.draw()

    def pos_slider_update(slider):
        for key in pos_labels[:-2]:
            spatial_dict[key] = [pos_sliders[key].value]
            resp_dict[key] = [pos_sliders[key].value]
        step = bias_step_picker.value
        sho_quantity = sho_quantity_picker.value

        update_resp_plot(resp_dict)
        update_sho_plots(sho_quantity, step)

        fig.canvas.draw()

    slider_dict = dict()
    slider_dict['Bias Step'] = (0, bias_mat.shape[0] - 1, 1)

    sho_quantity_picker = widgets.Dropdown(options=list(sho_dset_collapsed.dtype.names[:-1]),
                                           description='SHO Quantity')
    bias_step_picker = widgets.IntSlider(min=0, max=bias_mat.shape[0] - 1, step=1,
                                         description='Bias Step')

    fig_filename, _ = os.path.splitext(pc_sho_dset.file.filename)
    display(save_fig_filebox_button(fig, fig_filename + '.png'))

    for key, slider in pos_sliders.items():
        widgets.interact(pos_slider_update, slider=slider)

    cid = img_map.figure.canvas.mpl_connect('button_press_event', pos_picker)
    widgets.interact(update_sho_plots, sho_quantity=sho_quantity_picker, step_ind=bias_step_picker)

    return fig


def jupyter_visualize_be_spectrograms(pc_main, cmap=None):
    """
    Jupyer notebook ONLY function. Sets up a simple visualzier for visualizing raw BE data.
    Sliders for position indices can be used to visualize BE spectrograms (frequency, UDVS step).
    In the case of 2 spatial dimensions, a spatial map will be provided as well

    Parameters
    ----------
    pc_main : USIDataset
        Raw Band Excitation dataset
    cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
        Requested color map
    """
    cmap = get_cmap_object(cmap)

    h5_pos_inds = pc_main.h5_pos_inds
    pos_dims = pc_main.pos_dim_sizes
    pos_labels = pc_main.pos_dim_labels

    h5_spec_vals = pc_main.h5_spec_vals
    h5_spec_inds = pc_main.h5_spec_inds
    spec_dims = pc_main.spec_dim_sizes
    spec_labels = pc_main.spec_dim_labels

    ifreq = spec_labels.index('Frequency')
    freqs_nd = reshape_to_n_dims(h5_spec_vals, h5_spec=h5_spec_inds)[0][ifreq].squeeze()
    freqs_2d = freqs_nd.reshape(freqs_nd.shape[0], -1) / 1000  # Convert to kHz

    num_udvs_steps = int(np.prod([spec_dims[idim] for idim in range(len(spec_dims)) if idim != ifreq]))

    if len(pos_dims) >= 2:
        # Build initial slice dictionaries
        spatial_slice_dict = {'X': slice(None), 'Y': slice(None)}
        for key in pos_labels:
            if key in spatial_slice_dict.keys():
                continue
            else:
                spatial_slice_dict[key] = [0]

        spectrogram_slice_dict = {key: [0] for key in pos_labels}

        spatial_slice, _ = pc_main._get_pos_spec_slices(slice_dict=spatial_slice_dict)

        x_size = pos_dims[-1]
        y_size = pos_dims[-2]

        spatial_map = np.abs(np.reshape(pc_main[spatial_slice, 0], (y_size, x_size)))
        spectrogram = np.reshape(pc_main[0], (num_udvs_steps, -1))
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4), subplot_kw={'adjustable': 'box'})
        spatial_img, spatial_cbar = plot_map(axes[0], np.abs(spatial_map), cmap=cmap)
        axes[0].set_aspect('equal')
        axes[0].set_xlabel(pos_labels[-1])
        axes[0].set_ylabel(pos_labels[-2])

        xdata = int(0.5 * x_size)
        ydata = int(0.5 * y_size)
        crosshair = axes[0].plot(xdata, ydata, 'k+')[0]

        if len(spec_dims) > 1:
            amp_img, amp_cbar = plot_map(axes[1], np.abs(spectrogram), show_xy_ticks=True, cmap=cmap,
                                         extent=[freqs_2d[0, 0], freqs_2d[-1, 0], 0, num_udvs_steps])

            phase_img, phase_cbar = plot_map(axes[2], np.angle(spectrogram), show_xy_ticks=True, cmap=cmap,
                                             extent=[freqs_2d[0, 0], freqs_2d[-1, 0], 0, num_udvs_steps])
            phase_img.set_clim(vmin=-np.pi, vmax=np.pi)

            for axis in axes[1:3]:
                axis.set_ylabel('BE step')
                axis.axis('tight')
                x0, x1 = (freqs_2d[0, 0], freqs_2d[-1, 0])
                y0, y1 = (0, num_udvs_steps)
                axis.set_aspect(np.abs(x1 - x0) / np.abs(y1 - y0))

        else:
            # BE-Line
            axes[1].set_ylabel('Amplitude (a. u.)')
            axes[2].set_ylabel('Phase (rad)')
            spectrogram = np.squeeze(spectrogram)
            amp_img = axes[1].plot(np.abs(spectrogram))[0]
            phase_img = axes[2].plot(np.angle(spectrogram))[0]
            amp_full = np.abs(pc_main[()])
            amp_mean = np.mean(amp_full)
            amp_std = np.std(amp_full)
            st_devs = 4

            axes[1].set_ylim([0, amp_mean + st_devs * amp_std])
            axes[2].set_ylim([-np.pi, np.pi])

        pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                      pos_labels[-2] + ': ' + str(ydata) + ', '
        for dim_name in pos_labels[-3::-1]:
            pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '

        axes[1].set_title('Amplitude \n' + pos_heading)
        axes[1].set_xlabel('Frequency (kHz)')

        axes[2].set_title('Phase \n' + pos_heading)
        axes[2].set_xlabel('Frequency (kHz)')

        fig.tight_layout()

        fig_filename, _ = os.path.splitext(pc_main.file.filename)
        display(save_fig_filebox_button(fig, fig_filename + '.png'))

        # Build sliders for any extra Position Dimensions
        pos_sliders = dict()
        for ikey, key in enumerate(pos_labels[:-2]):
            pos_sliders[key] = widgets.IntSlider(value=0, min=0, max=pos_dims[ikey] - 1,
                                                 step=1, description='{} Step:'.format(key),
                                                 continuous_update=False)

        def get_spatial_slice():
            xdata, ydata = crosshair.get_xydata().squeeze()
            spatial_slice_dict[pos_labels[-1]] = [int(xdata)]
            spatial_slice_dict[pos_labels[-2]] = [int(ydata)]
            for key in pos_labels[:-2]:
                spatial_slice_dict[key] = [pos_sliders[key].value]

            spatial_slice, _ = pc_main._get_pos_spec_slices(slice_dict=spatial_slice_dict)

            return spatial_slice

        def spec_index_unpacker(step):
            spatial_slice_dict[pos_labels[-1]] = slice(None)
            spatial_slice_dict[pos_labels[-2]] = slice(None)
            for key in pos_labels[:-2]:
                spatial_slice_dict[key] = [pos_sliders[key].value]

            spatial_slice, _ = pc_main._get_pos_spec_slices(slice_dict=spatial_slice_dict)

            spatial_map = np.abs(np.reshape(pc_main[spatial_slice, step], (x_size, y_size)))
            spatial_img.set_data(spatial_map)
            spat_mean = np.mean(spatial_map)
            spat_std = np.std(spatial_map)
            spatial_img.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)

            spec_heading = ''
            for dim_ind, dim_name in enumerate(spec_labels):
                spec_heading += dim_name + ': ' + str(h5_spec_vals[dim_ind, step]) + ', '
            axes[0].set_title(spec_heading[:-2])
            fig.canvas.draw()

        def pos_picker(event):
            if not spatial_img.axes.in_axes(event):
                return

            xdata = int(round(event.xdata))
            ydata = int(round(event.ydata))

            crosshair.set_xdata(xdata)
            crosshair.set_ydata(ydata)

            spatial_slice = get_spatial_slice()

            pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                          pos_labels[-2] + ': ' + str(ydata) + ', '
            for dim_name in pos_labels[-3::-1]:
                pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '
            axes[1].set_title('Amplitude \n' + pos_heading)
            axes[2].set_title('Phase \n' + pos_heading)

            spectrogram = np.reshape(pc_main[spatial_slice, :], (num_udvs_steps, -1))

            if len(spec_dims) > 1:
                amp_map = np.abs(spectrogram)
                amp_img.set_data(np.abs(spectrogram))
                phase_img.set_data(np.angle(spectrogram))
                amp_mean = np.mean(amp_map)
                amp_std = np.std(amp_map)
                amp_img.set_clim(vmin=amp_mean - 3 * amp_std, vmax=amp_mean + 3 * amp_std)
            else:
                amp_img.set_ydata(np.abs(spectrogram))
                phase_img.set_ydata(np.angle(spectrogram))
            amp_cbar.changed()
            phase_cbar.changed()

            fig.canvas.draw()

        def pos_slider_update(slider):
            spatial_slice = get_spatial_slice()
            step = spec_index_slider.value

            spec_index_unpacker(step)

            pos_heading = pos_labels[-1] + ': ' + str(xdata) + ', ' + \
                          pos_labels[-2] + ': ' + str(ydata) + ', '
            for dim_name in pos_labels[-3::-1]:
                pos_heading += dim_name + ': ' + str(spatial_slice_dict[dim_name]) + ', '
            axes[1].set_title('Amplitude \n' + pos_heading)
            axes[2].set_title('Phase \n' + pos_heading)

            spectrogram = np.reshape(pc_main[spatial_slice, :], (num_udvs_steps, -1))

            if len(spec_dims) > 1:
                amp_img.set_data(np.abs(spectrogram))
                phase_img.set_data(np.angle(spectrogram))
            else:
                amp_img.set_ydata(np.abs(spectrogram))
                phase_img.set_ydata(np.angle(spectrogram))
            amp_cbar.changed()
            phase_cbar.changed()

            fig.canvas.draw()

        spec_index_slider = widgets.IntSlider(value=0, min=0, max=pc_main.shape[1], step=1,
                                              description='Step')
        cid = spatial_img.figure.canvas.mpl_connect('button_press_event', pos_picker)
        widgets.interact(spec_index_unpacker, step=spec_index_slider)
        for key, slider in pos_sliders.items():
            widgets.interact(pos_slider_update, slider=slider)
        # plt.show()

    else:
        def plot_spectrogram(data, freq_vals):
            fig, axes = plt.subplots(ncols=2, figsize=(9, 5), sharey=True)
            im_handles = list()
            im_handles.append(axes[0].imshow(np.abs(data), cmap=cmap,
                                             extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                     data.shape[0], 0],
                                             interpolation='none'))
            axes[0].set_title('Amplitude')
            axes[0].set_ylabel('BE step')
            im_handles.append(axes[1].imshow(np.angle(data), cmap=cmap,
                                             extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                     data.shape[0], 0],
                                             interpolation='none'))
            axes[1].set_title('Phase')
            axes[0].set_xlabel('Frequency index')
            axes[1].set_xlabel('Frequency index')
            for axis in axes:
                axis.axis('tight')
                axis.set_ylim(0, data.shape[0])
            fig.tight_layout()
            return fig, axes, im_handles

        fig, axes, im_handles = plot_spectrogram(np.reshape(pc_main[0], (num_udvs_steps, -1)), freqs_2d)

        def position_unpacker(**kwargs):
            pos_dim_vals = range(len(pos_labels))
            for pos_dim_ind, pos_dim_name in enumerate(pos_labels):
                pos_dim_vals[pos_dim_ind] = kwargs[pos_dim_name]
            pix_ind = pos_dim_vals[0]
            for pos_dim_ind in range(1, len(pos_labels)):
                pix_ind += pos_dim_vals[pos_dim_ind] * pos_dims[pos_dim_ind - 1]
            spectrogram = np.reshape(pc_main[pix_ind], (num_udvs_steps, -1))
            im_handles[0].set_data(np.abs(spectrogram))
            im_handles[1].set_data(np.angle(spectrogram))
            display(fig)

        pos_dict = dict()
        for pos_dim_ind, dim_name in enumerate(pos_labels):
            pos_dict[dim_name] = (0, pos_dims[pos_dim_ind] - 1, 1)

        widgets.interact(position_unpacker, **pos_dict)
        display(fig)

    return fig


def jupyter_visualize_beps_loops(h5_projected_loops, h5_loop_guess, h5_loop_fit, step_chan='DC_Offset', cmap=None):
    """
    Interactive plotting of the BE Loops

    Parameters
    ----------
    h5_projected_loops : h5py.Dataset
        Dataset holding the loop projections
    h5_loop_guess : h5py.Dataset
        Dataset holding the loop guesses
    h5_loop_fit : h5py.Dataset
        Dataset holding the loop fits
    step_chan : str, optional
        The name of the Spectroscopic dimension to plot versus.  Needs testing.
        Default 'DC_Offset'
    cmap : String, or matplotlib.colors.LinearSegmentedColormap object (Optional)
        Requested color map

    Returns
    -------
    None
    """
    cmap = get_cmap_object(cmap)

    # Prepare some variables for plotting loops fits and guesses
    # Plot the Loop Guess and Fit Results
    proj_nd, _ = reshape_to_n_dims(h5_projected_loops)
    guess_nd, _ = reshape_to_n_dims(h5_loop_guess)
    fit_nd, _ = reshape_to_n_dims(h5_loop_fit)

    h5_projected_loops = h5_loop_guess.parent['Projected_Loops']
    h5_proj_spec_inds = get_auxiliary_datasets(h5_projected_loops,
                                               aux_dset_name='Spectroscopic_Indices')[-1]
    h5_proj_spec_vals = get_auxiliary_datasets(h5_projected_loops,
                                               aux_dset_name='Spectroscopic_Values')[-1]
    h5_pos_inds = get_auxiliary_datasets(h5_projected_loops,
                                         aux_dset_name='Position_Indices')[-1]
    pos_nd, _ = reshape_to_n_dims(h5_pos_inds, h5_pos=h5_pos_inds)
    pos_dims = list(pos_nd.shape[:h5_pos_inds.shape[1]])
    pos_labels = get_attr(h5_pos_inds, 'labels')

    # reshape the vdc_vec into DC_step by Loop
    spec_nd, _ = reshape_to_n_dims(h5_proj_spec_vals, h5_spec=h5_proj_spec_inds)
    loop_spec_dims = np.array(spec_nd.shape[1:])
    loop_spec_labels = get_attr(h5_proj_spec_vals, 'labels')

    spec_step_dim_ind = np.where(loop_spec_labels == step_chan)[0][0]

    # # move the step dimension to be the first after all position dimensions
    rest_loop_dim_order = list(range(len(pos_dims), len(proj_nd.shape)))
    rest_loop_dim_order.pop(spec_step_dim_ind)
    new_order = list(range(len(pos_dims))) + [len(pos_dims) + spec_step_dim_ind] + rest_loop_dim_order

    new_spec_order = np.array(new_order[len(pos_dims):], dtype=np.uint32) - len(pos_dims)

    # Also reshape the projected loops to Positions-DC_Step-Loop
    final_loop_shape = pos_dims + [loop_spec_dims[spec_step_dim_ind]] + [-1]
    proj_nd2 = np.moveaxis(proj_nd, spec_step_dim_ind + len(pos_dims), len(pos_dims))
    proj_nd_3 = np.reshape(proj_nd2, final_loop_shape)

    # Do the same for the guess and fit datasets
    guess_3d = np.reshape(guess_nd, pos_dims + [-1])
    fit_3d = np.reshape(fit_nd, pos_dims + [-1])

    # Get the bias vector:
    spec_nd2 = np.moveaxis(spec_nd[spec_step_dim_ind], spec_step_dim_ind, 0)
    bias_vec = np.reshape(spec_nd2, final_loop_shape[len(pos_dims):])

    # Shift the bias vector and the loops by a quarter cycle
    shift_ind = int(-1 * bias_vec.shape[0] / 4)
    bias_shifted = np.roll(bias_vec, shift_ind, axis=0)
    proj_nd_shifted = np.roll(proj_nd_3, shift_ind, axis=len(pos_dims))

    # This is just the visualizer:
    loop_field_names = fit_nd.dtype.names
    loop_field = loop_field_names[0]
    loop_ind = 0
    row_ind = 0
    col_ind = 0

    # Initial plot data
    spatial_map = fit_3d[:, :, loop_ind][loop_field]
    proj_data = proj_nd_shifted[col_ind, row_ind, :, loop_ind]
    bias_data = bias_shifted[:, loop_ind]
    guess_data = loop_fit_function(bias_data, np.array(list(guess_3d[col_ind, row_ind, loop_ind])))
    fit_data = loop_fit_function(bias_data, np.array(list(fit_3d[col_ind, row_ind, loop_ind])))

    fig = plt.figure(figsize=(12, 8))
    ax_map = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    ax_loop = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)

    im_map, im_cbar = plot_map(ax_map, spatial_map.T,
                               x_vec=np.arange(spatial_map.shape[0]),
                               y_vec=np.arange(spatial_map.shape[1]),
                               cmap=cmap)

    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.set_title('{} - Loop {}'.format(loop_field, loop_ind))
    crosshair = ax_map.plot(row_ind, col_ind, 'k+')[0]

    ax_loop.plot(bias_data, proj_data, 'k', label='Projection')
    ax_loop.plot(bias_data, guess_data, 'g', label='Guess')
    ax_loop.plot(bias_data, fit_data, 'r--', label='Fit')
    line_handles = ax_loop.get_lines()

    ax_loop.tick_params(labelleft=False, labelright=True)
    ax_loop.yaxis.set_label_position('right')
    ax_loop.set_ylabel('PR (a.u.)')
    ax_loop.set_xlabel('V_DC (V)')
    ax_loop.set_title('Position ({},{})'.format(col_ind, row_ind))
    ax_loop.legend()
    fig.tight_layout()

    plt.show()
    fig_filename, _ = os.path.splitext(h5_projected_loops.file.filename)
    display(save_fig_filebox_button(fig, fig_filename + '.png'))

    loop_slider = widgets.IntSlider(min=0, max=bias_vec.shape[1] - 1, description='Loop Number')

    def update_loop_plots(loop_field, loop_ind):
        spatial_map = fit_3d[:, :, loop_ind][loop_field]
        im_map.set_data(spatial_map.T)
        spat_mean = np.mean(spatial_map)
        spat_std = np.std(spatial_map)
        im_map.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)
        ax_map.set_title('{} - Loop {}'.format(loop_field, loop_ind))

        xdata, ydata = crosshair.get_xydata().flatten().astype('int')

        proj_data = proj_nd_shifted[xdata, ydata, :, loop_ind]
        bias_data = bias_shifted[:, loop_ind]
        guess_data = loop_fit_function(bias_data, np.array(list(guess_3d[xdata, ydata, loop_ind])))
        fit_data = loop_fit_function(bias_data, np.array(list(fit_3d[xdata, ydata, loop_ind])))

        for line_handle, data in zip(line_handles, [proj_data, guess_data, fit_data]):
            line_handle.set_ydata(data)

        ax_loop.set_title('Position ({},{})'.format(xdata, ydata))

        ax_loop.relim()
        ax_loop.autoscale_view()

        fig.canvas.draw()

    def pos_picker(event):
        if not im_map.axes.in_axes(event):
            return

        xdata = int(round(event.xdata))
        ydata = int(round(event.ydata))
        current_pos = {pos_labels[0]: xdata, pos_labels[1]: ydata}

        pos_dim_vals = list(range(len(pos_labels)))

        for pos_dim_ind, pos_dim_name in enumerate(pos_labels):
            pos_dim_vals[pos_dim_ind] = current_pos[pos_dim_name]

        crosshair.set_xdata(xdata)
        crosshair.set_ydata(ydata)

        loop_ind = loop_slider.value

        proj_data = proj_nd_shifted[xdata, ydata, :, loop_ind]
        bias_data = bias_shifted[:, loop_ind]
        guess_data = loop_fit_function(bias_data, np.array(list(guess_3d[xdata, ydata, loop_ind])))
        fit_data = loop_fit_function(bias_data, np.array(list(fit_3d[xdata, ydata, loop_ind])))
        for line_handle, data in zip(line_handles, [proj_data, guess_data, fit_data]):
            line_handle.set_ydata(data)
        ax_loop.set_title('Position ({},{})'.format(xdata, ydata))

        ax_loop.relim()
        ax_loop.autoscale_view()

        fig.canvas.draw()

    cid = im_map.figure.canvas.mpl_connect('button_press_event', pos_picker)

    widgets.interact(update_loop_plots, loop_field=list(fit_nd.dtype.names), loop_ind=loop_slider)

    return fig


def jupyter_visualize_parameter_maps(h5_loop_parameters, cmap=None, **kwargs):
    """
    Interactive plot of the spatial maps of the loop parameters for all cycles.

    Parameters
    ----------
    h5_loop_parameters : h5py.Dataset
        The dataset containing the loop parameters to be visualized
    cmap : str or matplotlib.colors.Colormap

    Returns
    -------
    None

    """
    if not isinstance(h5_loop_parameters, USIDataset):
        h5_loop_parameters = USIDataset(h5_loop_parameters)

    # Get the position and spectroscopic datasets
    pos_dims = h5_loop_parameters.pos_dim_sizes
    num_cycles = h5_loop_parameters.shape[1]

    parameter_names = h5_loop_parameters.dtype.names

    parameter_map_stack = np.reshape(h5_loop_parameters[parameter_names[0]],
                                     [pos_dims[0], pos_dims[1], -1])

    parameter_map_stack = np.moveaxis(parameter_map_stack, -1, 0)

    loop_spec_labs = h5_loop_parameters.spec_dim_labels

    kwargs.update({'cmap': get_cmap_object(cmap)})

    map_titles = list()
    for icycle in range(num_cycles):
        title_list = list()
        for label in loop_spec_labs:
            val = h5_loop_parameters.get_spec_values(label)
            title_list.append('{}: {}'.format(label, val))
        map_titles.append(' - '.join(title_list))

    fig, axes = plot_map_stack(parameter_map_stack, num_comps=num_cycles, color_bar_mode='each',
                               subtitle=map_titles, title='Maps of Loop Parameter {}'.format(parameter_names[0]),
                               **kwargs)

    def update_loop_maps(parameter_name):
        parameter_map_stack = np.reshape(h5_loop_parameters[parameter_name],
                                         [pos_dims[0], pos_dims[1], -1])
        parameter_map_stack = np.moveaxis(parameter_map_stack, -1, 0)

        fig.suptitle('Maps of Loop Parameter {}'.format(parameter_name))
        # Loop over all axes
        for icycle, ax_cycle in enumerate(axes[:num_cycles]):
            image = ax_cycle.get_images()[0]
            image.set_data(parameter_map_stack[:, :, icycle])
            image.set_clim(vmin=np.min(parameter_map_stack[:, :, icycle]),
                           vmax=np.max(parameter_map_stack[:, :, icycle]))

        fig.canvas.draw()

    plt.show()
    fig_filename, _ = os.path.splitext(h5_loop_parameters.file.filename)
    display(save_fig_filebox_button(fig, fig_filename + '.png'))
    widgets.interact(update_loop_maps, parameter_name=list(parameter_names))

    return fig


def jupyter_visualize_loop_sho_raw_comparison(h5_loop_parameters, cmap=None):
    """

    Parameters
    ----------
    h5_loop_parameters
    cmap : str or matplotlib.colors.Colormap

    Returns
    -------

    """

    # Find the precursor datasets used to calculate these parameters
    h5_loop_grp = h5_loop_parameters.parent
    h5_loop_projections = h5_loop_grp['Projected_Loops']
    h5_loop_fit = h5_loop_grp['Fit']
    h5_loop_guess = h5_loop_grp['Guess']

    h5_sho_grp = h5_loop_grp.parent
    h5_sho_fit = h5_sho_grp['Fit']
    h5_sho_guess = h5_sho_grp['Guess']

    h5_main = get_source_dataset(h5_sho_grp)

    # Now get the needed ancillary datasets for each main dataset
    h5_pos_inds = get_auxiliary_datasets(h5_loop_parameters, 'Position_Indices')[0]
    h5_pos_vals = get_auxiliary_datasets(h5_loop_parameters, 'Position_Values')[0]
    pos_order = get_sort_order(np.transpose(h5_pos_inds))
    pos_dims = get_dimensionality(np.transpose(h5_pos_inds), pos_order)
    pos_labs = get_attr(h5_pos_inds, 'labels')

    h5_loop_spec_inds = get_auxiliary_datasets(h5_loop_parameters, 'Spectroscopic_Indices')[0]
    h5_loop_spec_vals = get_auxiliary_datasets(h5_loop_parameters, 'Spectroscopic_Values')[0]
    loop_spec_order = get_sort_order(h5_loop_spec_inds)
    loop_spec_dims = get_dimensionality(h5_loop_spec_inds, loop_spec_order)
    loop_spec_labs = get_attr(h5_loop_spec_inds, 'labels')

    h5_sho_spec_inds = h5_sho_fit.h5_spec_inds
    h5_sho_spec_vals = h5_sho_fit.h5_spec_vals
    sho_spec_order = get_sort_order(h5_sho_spec_inds)
    sho_spec_dims = get_dimensionality(h5_sho_spec_inds, sho_spec_order)
    sho_spec_labs = get_attr(h5_sho_spec_inds, 'labels')

    h5_main_spec_inds = h5_main.h5_spec_inds
    h5_main_spec_vals = h5_main.h5_spec_vals
    main_spec_order = get_sort_order(h5_main_spec_inds)
    main_spec_dims = get_dimensionality(h5_main_spec_inds, main_spec_order)
    main_spec_labs = get_attr(h5_main_spec_inds, 'labels')

    '''
    Select the initial plotting slices
    '''
    loop_parameter_names = h5_loop_parameters.dtype.names
    loop_num_cycles = h5_loop_parameters.shape[1]
    loop_parameter_spec_labs = get_attr(h5_loop_spec_vals, 'labels')
    sho_bias_dim = np.argwhere(sho_spec_labs[sho_spec_order] == 'DC_Offset').squeeze()
    steps_per_loop = sho_spec_dims[sho_bias_dim]
    main_bias_dim = np.argwhere(main_spec_labs[main_spec_order] == 'DC_Offset').squeeze()

    selected_loop_parm = loop_parameter_names[0]
    selected_loop_cycle = 0
    selected_loop_ndims = np.unravel_index(selected_loop_cycle, loop_spec_dims, order='F')
    selected_loop_pos = int(pos_dims[0] / 2), int(pos_dims[1] / 2)
    selected_step = int(steps_per_loop / 2)

    '''
    Get the bias vector to be plotted against
    '''
    loop_bias_vec = h5_sho_spec_vals[get_attr(h5_sho_spec_vals, 'DC_Offset')].squeeze()
    shift_ind = int(-1 * steps_per_loop / 4)
    loop_bias_vec = loop_bias_vec.reshape(sho_spec_dims)
    loop_bias_vec = np.moveaxis(loop_bias_vec, sho_bias_dim, 0).reshape(sho_spec_dims[sho_bias_dim], -1)
    loop_bias_vec = np.roll(loop_bias_vec.reshape(steps_per_loop, -1), shift_ind, axis=0)

    '''
    Get the frequency vector to be plotted against
    '''
    full_w_vec = h5_main_spec_vals[h5_main_spec_vals.attrs['Frequency']]
    full_w_vec, _ = reshape_to_n_dims(full_w_vec, h5_spec=h5_main_spec_inds)
    full_w_vec = full_w_vec.squeeze()

    '''
    Define functions to get the data
    '''

    def _get_loop_map(selected_loop_parm, selected_loop_cycle):
        # Build the map of the chosen loop parameter
        loop_parameter_map = np.reshape(h5_loop_parameters[selected_loop_parm, :, selected_loop_cycle],
                                        [pos_dims[0], pos_dims[1]])

        # Also create the title string for the map
        loop_map_title = list()
        for label in loop_parameter_spec_labs:
            val = h5_loop_spec_vals[get_attr(h5_loop_spec_vals, label)].squeeze()[selected_loop_cycle]
            loop_map_title.append('{}: {}'.format(label, val))

        return loop_parameter_map, loop_map_title

    def _get_loops(selected_loop_cycle, selected_loop_pos):
        selected_loop_ndims = np.unravel_index(selected_loop_cycle, loop_spec_dims, order='F')
        # Now build the loop plot for the selected position in the loop map
        selected_loop_bias_vec = loop_bias_vec[:, selected_loop_cycle]

        pos_ind = np.ravel_multi_index(selected_loop_pos, pos_dims)

        loop_proj_vec = h5_loop_projections[pos_ind].reshape(sho_spec_dims[::-1])
        loop_proj_vec = np.moveaxis(loop_proj_vec, sho_bias_dim, -1)[selected_loop_ndims]
        loop_proj_vec = np.roll(loop_proj_vec, shift_ind)
        loop_guess_vec = loop_fit_function(selected_loop_bias_vec,
                                           h5_loop_guess[pos_ind, selected_loop_cycle].tolist())
        loop_fit_vec = loop_fit_function(selected_loop_bias_vec,
                                         h5_loop_fit[pos_ind, selected_loop_cycle].tolist())

        return selected_loop_bias_vec, loop_proj_vec, loop_guess_vec, loop_fit_vec

    def _get_sho(selected_loop_pos, selected_step, selected_loop_cycle):
        selected_loop_ndims = np.unravel_index(selected_loop_cycle, loop_spec_dims, order='F')
        # get the SHO Guess and Fit and Raw Data for the selected position, cycle, and step
        pos_ind = np.ravel_multi_index(selected_loop_pos, pos_dims)

        # Get the frequency vector for the selected step
        w_vec = np.moveaxis(full_w_vec, main_bias_dim - len(pos_dims), -1)[selected_step][selected_loop_ndims]

        # Get the slice of the sho guess and fit
        sho_guess = h5_sho_guess[pos_ind].reshape(sho_spec_dims[::-1])
        sho_guess = np.moveaxis(sho_guess, sho_bias_dim, -1)[selected_loop_ndims][selected_step]
        sho_guess = SHOfunc(sho_guess, w_vec)
        sho_fit = h5_sho_fit[pos_ind].reshape(sho_spec_dims[::-1])
        sho_fit = np.moveaxis(sho_fit, sho_bias_dim, -1)[selected_loop_ndims][selected_step]
        sho_fit = SHOfunc(sho_fit, w_vec)

        # Get the slice of the Raw Data
        raw_data_vec, _ = reshape_to_n_dims(np.atleast_2d(h5_main[pos_ind]), h5_spec=h5_main_spec_inds)
        raw_data_vec = np.moveaxis(raw_data_vec.squeeze(),
                                   main_bias_dim - len(pos_dims), -1)[selected_step][selected_loop_ndims]

        return w_vec, sho_guess, sho_fit, raw_data_vec

    '''
    Get the starting values
    '''
    loop_parm_map, loop_map_title = _get_loop_map(selected_loop_parm, selected_loop_cycle)
    current_loop_bias_vec, current_loop_proj_vec, current_loop_guess_vec, current_loop_fit_vec = \
        _get_loops(selected_loop_cycle, selected_loop_pos)
    current_w_vec, current_sho_guess, current_sho_fit, current_raw_data = _get_sho(selected_loop_pos,
                                                                                   selected_step,
                                                                                   selected_loop_cycle)

    cmap = get_cmap_object(cmap)

    '''
    Build the figure
    '''
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(17, 17))

    ax_loop_map = axes.flatten()[0]
    ax_loop_pgf = axes.flatten()[1]
    ax_sho_amp = axes.flatten()[2]
    ax_sho_phase = axes.flatten()[3]

    # Plot the map of the loop parameters
    plt_loop_map, loop_cbar = plot_map(ax_loop_map, loop_parm_map, cmap=cmap)
    loop_vert_line = ax_loop_map.axvline(x=pos_dims[0], color='k')
    loop_horz_line = ax_loop_map.axhline(y=pos_dims[1], color='k')
    ax_loop_map.set_title(loop_map_title)

    # Plot the Loops
    plt_loop_proj = ax_loop_pgf.plot(current_loop_bias_vec, current_loop_proj_vec, 'k', label='Loop Projection')
    plt_loop_guess = ax_loop_pgf.plot(current_loop_bias_vec, current_loop_guess_vec, 'g', label='Loop Guess')
    plt_loop_fit = ax_loop_pgf.plot(current_loop_bias_vec, current_loop_fit_vec, 'r--', label='Loop Fit')
    ax_loop_pgf.legend()

    # Plot the amplitude of the SHO Guess, SHO Fit, and Raw Data
    plt_raw_data_amp = ax_sho_amp.plot(current_w_vec, np.abs(current_raw_data), 'k', label='Raw Data')
    plt_sho_guess_amp = ax_sho_amp.plot(current_w_vec, np.abs(current_sho_guess), 'g', label='SHO Guess')
    plt_sho_fit_amp = ax_sho_amp.plot(current_w_vec, np.abs(current_sho_fit), 'r--', label='SHO Fit')
    ax_sho_amp.legend()

    # Plot the phase of the SHO Guess, SHO Fit, and Raw Data
    plt_raw_data_phase = ax_sho_phase.plot(current_w_vec, np.angle(current_raw_data), 'k', label='Raw Data')
    plt_sho_guess_phase = ax_sho_phase.plot(current_w_vec, np.angle(current_sho_guess), 'g', label='SHO Guess')
    plt_sho_fit_phase = ax_sho_phase.plot(current_w_vec, np.angle(current_sho_fit), 'r--', label='SHO Fit')
    ax_sho_phase.legend()

    fig.tight_layout()

    def _update_loop_map(parameter_name, selected_cycle):
        loop_parm_map, loop_map_title = _get_loop_map(parameter_name, selected_cycle)
        plt_loop_map.set_data(loop_parm_map)
        ax_loop_map.set_title(loop_map_title)

    def _update_crosshairs(x_pos, y_pos):
        loop_vert_line.set_xdata(x_pos)
        loop_horz_line.set_ydata(y_pos)

    def _update_loop_plots(selected_loop_cycle, x_pos, y_pos):
        current_loop_bias_vec, current_loop_proj_vec, current_loop_guess_vec, current_loop_fit_vec = \
            _get_loops(selected_loop_cycle, [x_pos, y_pos])

        # Update the bias vectors
        plt_loop_proj.set_xdata(current_loop_bias_vec)
        plt_loop_guess.set_xdata(current_loop_bias_vec)
        plt_loop_fit.set_xdata(current_loop_bias_vec)

        # Update the ploted data
        plt_loop_proj.set_xdata(current_loop_proj_vec)
        plt_loop_guess.set_xdata(current_loop_guess_vec)
        plt_loop_fit.set_xdata(current_loop_fit_vec)

    def _update_sho_plots(selected_loop_cycle, x_pos, y_pos, selected_step):
        current_w_vec, current_sho_guess, current_sho_fit, current_raw_data = _get_sho([x_pos, y_pos],
                                                                                       selected_step,
                                                                                       selected_loop_cycle)
        # Update x-date with new w_vec
        plt_raw_data_amp.set_xdata(current_w_vec)
        plt_raw_data_phase.set_xdata(current_w_vec)
        plt_sho_guess_amp.set_xdata(current_w_vec)
        plt_sho_guess_phase.set_xdata(current_w_vec)
        plt_sho_fit_amp.set_xdata(current_w_vec)
        plt_sho_fit_phase.set_xdata(current_w_vec)

        # Update the y-data
        plt_raw_data_amp.set_ydata(np.amp(current_raw_data))
        plt_raw_data_phase.set_ydata(np.angle(current_raw_data))
        plt_sho_guess_amp.set_ydata(np.amp(current_sho_guess))
        plt_sho_guess_phase.set_ydata(np.angle(current_sho_guess))
        plt_sho_fit_amp.set_ydata(np.amp(current_sho_fit))
        plt_sho_fit_phase.set_ydata(np.angle(current_sho_fit))

    fig_filename, _ = os.path.splitext(h5_loop_parameters.file.filename)
    display(save_fig_filebox_button(fig, fig_filename + '.png'))
    display(fig)
    x_pos_widget = widgets.FloatSlider(min=0.0, max=float(pos_dims[0]), step=pos_dims[0] / 100,
                                       value=selected_loop_pos[0])
    y_pos_widget = widgets.FloatSlider(min=0.0, max=float(pos_dims[1]), step=pos_dims[1] / 100,
                                       value=selected_loop_pos[1])
    loop_cycle_widget = widgets.IntSlider(min=0, max=loop_num_cycles, step=1, value=selected_loop_cycle)
    spec_step_widget = widgets.IntSlider(min=0, max=steps_per_loop, step=1, value=selected_step)
    widgets.interact(_update_loop_map,
                     parameter_name=list(loop_parameter_names),
                     selected_cycle=loop_cycle_widget)
    widgets.interact(_update_crosshairs,
                     x_pos=x_pos_widget, y_pos=y_pos_widget)
    widgets.interact(_update_loop_plots,
                     selected_loop_cycle=loop_cycle_widget,
                     x_pos=x_pos_widget, y_pos=y_pos_widget)
    widgets.interact(_update_sho_plots,
                     selected_loop_cycle=loop_cycle_widget,
                     x_pos=x_pos_widget, y_pos=y_pos_widget,
                     selected_step=spec_step_widget)

    return fig


def plot_loop_sho_raw_comparison(h5_loop_parameters, selected_loop_parm=None, selected_loop_cycle=0,
                                 selected_loop_pos=[0, 0], selected_step=0, tick_font_size=14, cmap='viridis',
                                 step_chan='DC_Offset'):
    """

    Parameters
    ----------
    h5_loop_parameters : h5py.Dataset
        Dataset containing the loop parameters
    selected_loop_parm : str
        The initial loop parameter to be plotted
    selected_loop_cycle : int
        The initial loop cycle to be plotted
    selected_loop_pos : array-like of two ints
        The initial position to be plotted
    selected_step : int
        The initial bias step to be plotted
    tick_font_size : int
        Font size for the axes tick labels
    cmap : str or matplotlib.colors.Colormap
        Colormap to be used in plotting the parameter map
    step_chan : str
        Name of spectral dimension loops were fit over

    Returns
    -------
    None

    """
    if not isinstance(h5_loop_parameters, USIDataset):
        h5_loop_parameters = USIDataset(h5_loop_parameters)

    # Find the precursor datasets used to calculate these parameters
    h5_loop_grp = h5_loop_parameters.parent
    h5_loop_projections = USIDataset(h5_loop_grp['Projected_Loops'], sort_dims=False)
    h5_loop_fit = USIDataset(h5_loop_grp['Fit'], sort_dims=False)
    h5_loop_guess = USIDataset(h5_loop_grp['Guess'], sort_dims=False)

    h5_sho_grp = h5_loop_grp.parent
    h5_sho_fit = USIDataset(h5_sho_grp['Fit'], sort_dims=False)
    h5_sho_guess = USIDataset(h5_sho_grp['Guess'], sort_dims=False)

    h5_main = get_source_dataset(h5_sho_grp)
    # h5_main.toggle_sorting()

    # Now get the needed ancillary datasets for each main dataset
    pos_dims = h5_loop_parameters.pos_dim_sizes
    pos_labs = h5_loop_parameters.pos_dim_labels

    # h5_loop_spec_inds = h5_loop_parameters.h5_spec_inds
    h5_loop_spec_vals = h5_loop_parameters.h5_spec_vals
    # loop_spec_order = get_sort_order(h5_loop_spec_inds)
    loop_spec_dims = h5_loop_parameters.spec_dim_sizes
    loop_spec_labs = h5_loop_parameters.spec_dim_labels

    h5_sho_spec_inds = h5_sho_fit.h5_spec_inds
    h5_sho_spec_vals = h5_sho_fit.h5_spec_vals
    # sho_spec_order = get_sort_order(h5_sho_spec_inds)
    sho_spec_dims = h5_sho_fit.spec_dim_sizes
    sho_spec_labs = h5_sho_fit.spec_dim_labels

    h5_main_spec_inds = h5_main.h5_spec_inds
    h5_main_spec_vals = h5_main.h5_spec_vals
    # main_spec_order = get_sort_order(h5_main_spec_inds)
    main_spec_dims = h5_main.spec_dim_sizes
    main_spec_labs = h5_main.spec_dim_labels

    '''
    Select the initial plotting slices
    '''
    loop_parameter_names = h5_loop_parameters.dtype.names
    loop_num_cycles = h5_loop_parameters.shape[1]
    # loop_parameter_spec_labs = h5_loop_parameters.spec_dim_labels
    sho_bias_dim = sho_spec_labs.index(step_chan)
    steps_per_loop = sho_spec_dims[sho_bias_dim]
    main_bias_dim = main_spec_labs.index(step_chan)
    main_freq_dim = main_spec_labs.index('Frequency')

    if selected_loop_parm is None:
        selected_loop_parm = loop_parameter_names[0]
    selected_loop_ndims = np.unravel_index(selected_loop_cycle, loop_spec_dims, order='F')

    '''
    Get the bias vector to be plotted against
    '''
    loop_bias_vec = h5_sho_spec_vals[get_attr(h5_sho_spec_vals, step_chan)].squeeze()
    shift_ind = int(-1 * steps_per_loop / 4)
    loop_bias_vec = loop_bias_vec.reshape(sho_spec_dims[::-1])
    loop_bias_vec = np.moveaxis(loop_bias_vec, len(loop_bias_vec.shape) - sho_bias_dim - 1, 0)
    loop_bias_vec = np.reshape(loop_bias_vec, [sho_spec_dims[sho_bias_dim], -1])
    loop_bias_vec = np.roll(loop_bias_vec.reshape(steps_per_loop, -1), shift_ind, axis=0)

    '''
    Get the frequency vector to be plotted against
    '''
    full_w_vec = h5_main_spec_vals[h5_main_spec_vals.attrs['Frequency']]
    full_w_vec, _ = reshape_to_n_dims(full_w_vec, h5_spec=h5_main_spec_inds)
    full_w_vec = full_w_vec.squeeze()

    '''
    Define functions to get the data
    '''

    def _get_loop_map(selected_loop_parm, selected_loop_cycle):
        # Build the map of the chosen loop parameter
        loop_parameter_map = np.reshape(h5_loop_parameters[selected_loop_parm, :, selected_loop_cycle],
                                        [pos_dims[0], pos_dims[1]])

        # Also create the title string for the map
        loop_map_title = list()
        for label in loop_spec_labs:
            val = h5_loop_spec_vals[get_attr(h5_loop_spec_vals, label)].squeeze()[selected_loop_cycle]
        loop_map_title.append('{}: {}'.format(label, val))

        loop_map_title = ' - '.join(loop_map_title)

        return loop_parameter_map, loop_map_title

    def _get_loops(selected_loop_cycle, selected_loop_pos):
        selected_loop_ndims = np.unravel_index(selected_loop_cycle, loop_spec_dims, order='F')
        # Now build the loop plot for the selected position in the loop map
        selected_loop_bias_vec = loop_bias_vec[:, selected_loop_cycle]

        pos_ind = np.ravel_multi_index(selected_loop_pos, pos_dims)

        slice_dict = dict()
        for pos_dim, dim_ind in zip(pos_labs, selected_loop_pos):
            slice_dict[pos_dim] = [dim_ind]

        for spec_dim, dim_ind in zip(loop_spec_labs, selected_loop_ndims):
            slice_dict[spec_dim] = [dim_ind]

        loop_proj_vec, _ = h5_loop_projections.slice(slice_dict, as_scalar=False)
        loop_proj_vec2 = np.roll(loop_proj_vec.squeeze(), shift_ind)

        loop_guess_slice, _ = h5_loop_guess.slice(slice_dict, as_scalar=True)
        loop_fit_slice, _ = h5_loop_fit.slice(slice_dict, as_scalar=True)

        loop_guess_vec = loop_fit_function(selected_loop_bias_vec,
                                           loop_guess_slice.squeeze().tolist())
        loop_fit_vec = loop_fit_function(selected_loop_bias_vec,
                                         loop_fit_slice.squeeze().tolist())

        return selected_loop_bias_vec, loop_proj_vec2, loop_guess_vec, loop_fit_vec

    def _get_sho(selected_loop_pos, selected_step, selected_loop_cycle):
        selected_loop_ndims = np.unravel_index(selected_loop_cycle, loop_spec_dims, order='F')
        # get the SHO Guess and Fit and Raw Data for the selected position, cycle, and step
        pos_ind = np.ravel_multi_index(selected_loop_pos, pos_dims)

        # Get the slice of the sho guess and fit
        sho_slice = {key: int(val) for key, val in zip(pos_labs, selected_loop_pos)}
        sho_slice[sho_spec_labs[sho_bias_dim]] = selected_step
        for key, val in zip(loop_spec_labs, selected_loop_ndims):
            sho_slice[key] = int(val)

        # Get the slice of the Raw Data
        raw_data_vec = h5_main.slice(sho_slice)[0]

        # Get the frequency vector for the selected step
        w_vec = np.moveaxis(full_w_vec, main_freq_dim, len(main_spec_dims) - 1)  # Move frequency to the end
        w_vec2 = np.rollaxis(w_vec, main_bias_dim - 1, 0)[selected_step][selected_loop_ndims]

        sho_guess = h5_sho_guess.slice(sho_slice)[0].tolist()

        sho_guess = SHOfunc(sho_guess, w_vec2)
        sho_fit = h5_sho_fit.slice(sho_slice)[0].tolist()
        sho_fit = SHOfunc(sho_fit, w_vec2)

        return w_vec2, sho_guess, sho_fit, raw_data_vec

    '''
    Get the starting values
    '''
    loop_parm_map, loop_map_title = _get_loop_map(selected_loop_parm, selected_loop_cycle)
    current_loop_bias_vec, current_loop_proj_vec, current_loop_guess_vec, current_loop_fit_vec = \
        _get_loops(selected_loop_cycle, selected_loop_pos)
    current_w_vec, current_sho_guess, current_sho_fit, current_raw_data = _get_sho(selected_loop_pos,
                                                                                   selected_step,
                                                                                   selected_loop_cycle)

    '''
    Build the figure
    '''
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(17, 17))

    ax_loop_map = axes.flatten()[0]
    ax_loop_pgf = axes.flatten()[1]
    ax_sho_amp = axes.flatten()[2]
    ax_sho_phase = axes.flatten()[3]

    # Plot the map of the loop parameters
    loop_map, loop_map_cbar = plot_map(ax_loop_map, loop_parm_map)
    crosshair = ax_loop_map.plot(selected_loop_pos[0], selected_loop_pos[1], 'k+')[0]
    ax_loop_map.set_title(loop_map_title)
    ax_loop_map.set_xlabel('X Position')
    ax_loop_map.set_ylabel('Y Position')

    # Plot the Loops
    loop_proj_plot = ax_loop_pgf.plot(current_loop_bias_vec, current_loop_proj_vec, 'k', label='Loop Projection')[0]
    loop_guess_plot = ax_loop_pgf.plot(current_loop_bias_vec, current_loop_guess_vec, 'g', label='Loop Guess')[0]
    loop_fit_plot = ax_loop_pgf.plot(current_loop_bias_vec, current_loop_fit_vec, 'r--', label='Loop Fit')[0]
    ax_loop_pgf.legend()
    loop_pgf_title = ' - '.join(['Position: ({}, {})'.format(selected_loop_pos[0], selected_loop_pos[1]),
                                 loop_map_title])
    ax_loop_pgf.set_title(loop_pgf_title)
    ax_loop_pgf.set_ylabel('PR (a.u.)')
    ax_loop_pgf.set_xlabel('V_DC (V)')
    ax_loop_pgf.axis('tight')

    # Plot the amplitude of the SHO Guess, SHO Fit, and Raw Data
    raw_amp_plot = ax_sho_amp.plot(current_w_vec, np.abs(current_raw_data), 'k*', label='Raw Data Amplitude')[0]
    guess_amp_plot = ax_sho_amp.plot(current_w_vec, np.abs(current_sho_guess), 'g', label='SHO Guess Amplitude')[0]
    fit_amp_plot = ax_sho_amp.plot(current_w_vec, np.abs(current_sho_fit), 'r:', label='SHO Fit Amplitude')[0]
    ax_sho_amp.legend()
    sho_title = ' - '.join(['Bias Step: {}'.format(selected_step), loop_pgf_title])
    ax_sho_amp.set_title(sho_title)
    ax_sho_amp.set_ylabel('Amplitude (V)')
    ax_sho_amp.set_xlabel('Frequency (Hz)')
    ax_sho_amp.axis('tight')

    # Plot the phase of the SHO Guess, SHO Fit, and Raw Data
    raw_phase_plot = ax_sho_phase.plot(current_w_vec, np.angle(current_raw_data), 'k*', label='Raw Data Phase')[0]
    guess_phase_plot = ax_sho_phase.plot(current_w_vec, np.angle(current_sho_guess), 'g', label='SHO Guess Phase')[0]
    fit_phase_plot = ax_sho_phase.plot(current_w_vec, np.angle(current_sho_fit), 'r:', label='SHO Fit Phase')[0]
    ax_sho_phase.legend()
    ax_sho_phase.set_title(sho_title)
    ax_sho_phase.set_ylabel('Amplitude (V)')
    ax_sho_phase.set_xlabel('Frequency (Hz)')
    ax_sho_phase.axis('tight')

    for axis in [ax_loop_pgf, ax_sho_amp, ax_sho_phase]:
        set_tick_font_size(axis, tick_font_size)
        axis.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

    plt.show()

    # Make some widgets now so we can reference them in updates
    loop_parm_widget = widgets.Dropdown(options=loop_parameter_names, value=selected_loop_parm)
    loop_cycle_widget = widgets.IntSlider(min=0, max=loop_num_cycles - 1, step=1, value=selected_loop_cycle)
    spec_step_widget = widgets.IntSlider(min=0, max=steps_per_loop - 1, step=1, value=selected_step)

    def _update_loop_plots(selected_loop_cycle, xdata, ydata):
        # Get new data
        current_loop_bias_vec, current_loop_proj_vec, current_loop_guess_vec, current_loop_fit_vec = \
            _get_loops(selected_loop_cycle, (xdata, ydata))

        # Update loop plots
        loop_proj_plot.set_xdata(current_loop_bias_vec)
        loop_proj_plot.set_ydata(current_loop_proj_vec)
        loop_guess_plot.set_xdata(current_loop_bias_vec)
        loop_guess_plot.set_ydata(current_loop_guess_vec)
        loop_fit_plot.set_xdata(current_loop_bias_vec)
        loop_fit_plot.set_ydata(current_loop_fit_vec)
        loop_map_title = ax_loop_map.get_title()
        loop_pgf_title = ' - '.join(['Position: ({}, {})'.format(xdata, ydata),
                                     loop_map_title])
        ax_loop_pgf.set_title(loop_pgf_title)
        loop_min = np.min([current_loop_proj_vec, current_loop_guess_vec, current_loop_fit_vec])
        loop_max = np.max([current_loop_proj_vec, current_loop_guess_vec, current_loop_fit_vec])
        loop_min *= (1 - np.sign(loop_min) * 0.5)
        loop_max *= (1 + np.sign(loop_max) * 0.5)

        ax_loop_pgf.set_ylim([loop_min, loop_max])

    def _update_sho_plots(selected_loop_cycle, selected_step, xdata, ydata):
        # Get the updated data
        current_w_vec, current_sho_guess, current_sho_fit, current_raw_data = _get_sho((xdata, ydata),
                                                                                       selected_step,
                                                                                       selected_loop_cycle)
        # Update SHO/Raw plots
        loop_pgf_title = ax_loop_pgf.get_title()
        sho_title = ' - '.join(['Bias Step: {}'.format(selected_step), loop_pgf_title])
        raw_amp_plot.set_xdata(current_w_vec)
        raw_amp_plot.set_ydata(np.abs(current_raw_data))
        guess_amp_plot.set_xdata(current_w_vec)
        guess_amp_plot.set_ydata(np.abs(current_sho_guess))
        fit_amp_plot.set_xdata(current_w_vec)
        fit_amp_plot.set_ydata(np.abs(current_sho_fit))
        ax_sho_amp.set_title(sho_title)
        ax_sho_amp.set_ylim([np.min(np.abs([current_raw_data, current_sho_guess, current_sho_fit])) * 0.95,
                             np.max(np.abs([current_raw_data, current_sho_guess, current_sho_fit])) * 1.05])

        raw_phase_plot.set_xdata(current_w_vec)
        raw_phase_plot.set_ydata(np.angle(current_raw_data))
        guess_phase_plot.set_xdata(current_w_vec)
        guess_phase_plot.set_ydata(np.angle(current_sho_guess))
        fit_phase_plot.set_xdata(current_w_vec)
        fit_phase_plot.set_ydata(np.angle(current_sho_fit))
        ax_sho_phase.set_title(sho_title)

    def pos_picker(event):
        # Allow for picking a new position.
        if not ax_loop_map.axes.in_axes(event):
            return

        selected_loop_cycle = loop_cycle_widget.value
        selected_step = spec_step_widget.value

        xdata = int(round(event.xdata))
        ydata = int(round(event.ydata))
        current_pos = {pos_labs[0]: xdata, pos_labs[1]: ydata}

        pos_dim_vals = list(range(len(pos_labs)))

        for pos_dim_ind, pos_dim_name in enumerate(pos_labs):
            pos_dim_vals[pos_dim_ind] = current_pos[pos_dim_name]

        crosshair.set_xdata(xdata)
        crosshair.set_ydata(ydata)

        _update_loop_plots(selected_loop_cycle, xdata, ydata)
        _update_sho_plots(selected_loop_cycle, selected_step, xdata, ydata)

        fig.canvas.draw()

    def _update_loop_parm(selected_loop_parm):
        loop_parm_map, loop_map_title = _get_loop_map(selected_loop_parm, loop_cycle_widget.value)
        loop_map.set_data(loop_parm_map)
        loop_map.set_clim(vmin=np.min(loop_parm_map), vmax=np.max(loop_parm_map))
        ax_loop_map.set_title(loop_map_title)

        fig.canvas.draw()

    def _update_loop_cycle(selected_loop_cycle):
        selected_step = spec_step_widget.value
        xdata, ydata = crosshair.get_xydata().flatten().astype('int')

        # Update the loop map
        loop_parm_map, loop_map_title = _get_loop_map(loop_parm_widget.value, selected_loop_cycle)
        loop_map.set_data(loop_parm_map)
        loop_map.set_clim(vmin=np.min(loop_parm_map), vmax=np.max(loop_parm_map))
        ax_loop_map.set_title(loop_map_title)

        # Update the linet plots
        _update_loop_plots(selected_loop_cycle, xdata, ydata)
        _update_sho_plots(selected_loop_cycle, selected_step, xdata, ydata)

        fig.canvas.draw()

    def _update_spec_step(selected_step):
        selected_loop_cycle = loop_cycle_widget.value
        xdata, ydata = crosshair.get_xydata().flatten().astype('int')

        _update_sho_plots(selected_loop_cycle, selected_step, xdata, ydata)

        fig.canvas.draw()

    cid = loop_map.figure.canvas.mpl_connect('button_press_event', pos_picker)

    fig_filename, _ = os.path.splitext(h5_loop_projections.file.filename)
    display(save_fig_filebox_button(fig, fig_filename + '.png'))
    widgets.interact(_update_loop_parm, selected_loop_parm=loop_parm_widget)
    widgets.interact(_update_loop_cycle, selected_loop_cycle=loop_cycle_widget)
    widgets.interact(_update_spec_step, selected_step=spec_step_widget)

    return fig


def _add_loop_parameters(axes, switching_coef_vec):
    """
    Add the loop parameters for the given loop to a list of axes

    Parameters
    ----------
    axes : list of matplotlib.pyplo.axes
        Plot axes to add the coeffients to
    switching_coef_vec : 1D numpy.ndarray
        Array of loop parameters arranged by position

    Returns
    -------
    axes : list of matplotlib.pyplo.axes
    """
    positions = np.linspace(0, switching_coef_vec.shape[0] - 1, len(axes.flat), dtype=np.int)

    for ax, pos in zip(axes.flat, positions):
        ax.axvline(switching_coef_vec[pos]['V+'], c='k', label='V+')
        ax.axvline(switching_coef_vec[pos]['V-'], c='r', label='V-')
        ax.axvline(switching_coef_vec[pos]['Nucleation Bias 1'], c='k', ls=':', label='Nucleation Bias 1')
        ax.axvline(switching_coef_vec[pos]['Nucleation Bias 2'], c='r', ls=':', label='Nucleation Bias 2')
        ax.axhline(switching_coef_vec[pos]['R+'], c='k', ls='-.', label='R+')
        ax.axhline(switching_coef_vec[pos]['R-'], c='r', ls='-.', label='R-')

    return axes


def plot_1d_spectrum(data_vec, freq, title, **kwargs):
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

    Returns
    ---------
    fig : Matplotlib.pyplot figure
        Figure handle
    axes : Matplotlib.pyplot axis
        Axis handle
    """
    if len(data_vec) != len(freq):
        raise ValueError('Incompatible data sizes! spectrum: '
                         + str(len(data_vec)) + ', frequency: ' + str(freq.shape))
    freq *= 1E-3  # to kHz

    title = title + ': mean UDVS, mean spatial response'
    fig, axes = plot_complex_spectra(np.expand_dims(data_vec, axis=0), freq, title=title,
                                     subtitle_prefix='', num_comps=1, x_label='Frequency (kHz)',
                                     figsize=(5, 3), amp_units='V', **kwargs)
    return fig, axes


def plot_2d_spectrogram(mean_spectrogram, freq, title=None, **kwargs):
    """
    Plots the position averaged spectrogram

    Parameters
    ------------
    mean_spectrogram : 2D numpy complex array
        Means spectrogram arranged as [frequency, UDVS step]
    freq : 1D numpy float array
        BE frequency that serves as the X axes of the plot
    title : str, optional
        Plot group name

    Returns
    ---------
    fig : Matplotlib.pyplot figure
        Figure handle
    axes : Matplotlib.pyplot axes
        Axis handle
    """
    if mean_spectrogram.shape[1] != freq.size:
        if mean_spectrogram.shape[0] == freq.size:
            mean_spectrogram = mean_spectrogram.T
        else:
            raise ValueError('plot_2d_spectrogram: Incompatible data sizes!!!! spectrogram: '
                             + str(mean_spectrogram.shape) + ', frequency: ' + str(freq.shape))

    fig, axes = plot_complex_spectra(np.expand_dims(mean_spectrogram, axis=0), num_comps=1, title=title,
                                     x_label='Frequency (kHz)', y_label='UDVS step', subtitle_prefix='',
                                     figsize=(5, 3), origin='lower', stdevs=None, amp_units='V', **kwargs)

    # Changing the X axis labels
    x_ticks = np.linspace(0, mean_spectrogram.shape[1] - 1, 5, dtype=int)
    x_tick_labs = [str(np.round(freq[ind], 2)) for ind in x_ticks]

    for axis in axes:
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_tick_labs)

    return fig, axes


def plot_histograms(p_hist, p_hbins, title, figure_path=None):
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
