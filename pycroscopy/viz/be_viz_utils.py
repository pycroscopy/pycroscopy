# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 13:29:12 2017

@author: Suhas Somnath, Chris R. Smith
"""
from __future__ import division, print_function, absolute_import
import os
from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display

from .plot_utils import plot_loops, plot_map_stack, cmap_jet_white_center, plot_map
from ..io.hdf_utils import reshape_to_Ndims, getAuxData, get_sort_order, get_dimensionality
from ..analysis.utils.be_loop import loop_fit_function


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
                             x_label=spec_var_title, y_label=meas_var_title, subtitles='Position', title=plt_title)
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

    chan_grp = h5_file['/'.join(sho_grp.name[1:].split('/')[:2])]

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

        fig_ms, ax_ms = plot_map_stack(np.dstack((amp_mat, freq_mat, q_mat, phase_mat, rsqr_mat)),
                                       num_comps=5, color_bar_mode='each', heading=grp_name,
                                       title=['Amplitude (mV)', 'Frequency (kHz)', 'Quality Factor', 'Phase (deg)',
                                              'R^2 Criterion'], cmap=cmap_jet_white_center())

        if save_plots:
            plt_path = os.path.join(folder_path, basename + '_' + grp_name + 'Maps.png')
            fig_ms.savefig(plt_path, format='png', dpi=300)

    if show_plots:
        plt.show()

    plt.close('all')


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


def jupyter_visualize_beps_sho(h5_sho_dset, step_chan, resp_func=None, resp_label='Response'):
    """
    Jupyer notebook ONLY function. Sets up an interactive visualizer for viewing SHO fitted BEPS data.
    Currently, this is limited to DC and AC spectroscopy datasets.

    Parameters
    ----------
    h5_sho_dset : h5py.Dataset
        dataset to be plotted
    step_chan : string / unicode
        Name of the channel that forms the primary spectroscopic axis (eg - DC offset)
    resp_func : function (optional)
        Function to apply to the spectroscopic data. Currently, DC spectroscopy uses A*cos(phi) and AC spectroscopy
        uses A
    resp_label : string / unicode (optional)
        Label for the response (y) axis.
    """
    guess_3d_data, success = reshape_to_Ndims(h5_sho_dset)

    h5_sho_spec_inds = getAuxData(h5_sho_dset, 'Spectroscopic_Indices')[0]
    h5_sho_spec_vals = getAuxData(h5_sho_dset, 'Spectroscopic_Values')[0]
    spec_nd, _ = reshape_to_Ndims(h5_sho_spec_inds, h5_spec=h5_sho_spec_inds)
    # sho_spec_sort = get_sort_order(h5_sho_spec_inds)
    sho_spec_dims = np.array(spec_nd.shape[1:])
    sho_spec_labels = h5_sho_spec_inds.attrs['labels']

    h5_pos_inds = getAuxData(h5_sho_dset, auxDataName='Position_Indices')[-1]
    pos_nd, _ = reshape_to_Ndims(h5_pos_inds, h5_pos=h5_pos_inds)
    pos_dims = list(pos_nd.shape[:h5_pos_inds.shape[1]])
    pos_labels = h5_pos_inds.attrs['labels']

    # reshape to X, Y, step, all others
    spec_step_dim_ind = np.where(sho_spec_labels == step_chan)[0][0]
    step_dim_ind = len(pos_dims) + spec_step_dim_ind

    # move the step dimension to be the first after all position dimensions
    rest_sho_dim_order = range(len(pos_dims), len(guess_3d_data.shape))
    rest_sho_dim_order.remove(step_dim_ind)
    new_order = range(len(pos_dims)) + [step_dim_ind] + rest_sho_dim_order

    # Transpose the 3D dataset to this shape:
    sho_guess_Nd_1 = np.transpose(guess_3d_data, new_order)

    # Now move the step dimension to the front for the spec labels as well
    new_spec_order = range(len(sho_spec_labels))
    new_spec_order.remove(spec_step_dim_ind)
    new_spec_order = [spec_step_dim_ind] + new_spec_order

    # new_spec_labels = sho_spec_labels[new_spec_order]
    new_spec_dims = np.array(sho_spec_dims)[new_spec_order]

    # Now collapse all additional dimensions
    final_guess_shape = pos_dims + [new_spec_dims[0]] + [-1]
    sho_dset_collapsed = np.reshape(sho_guess_Nd_1, final_guess_shape)

    # Get the bias matrix:
    bias_mat, _ = reshape_to_Ndims(h5_sho_spec_vals, h5_spec=h5_sho_spec_inds)
    bias_mat = np.transpose(bias_mat[spec_step_dim_ind], new_spec_order).reshape(sho_dset_collapsed.shape[len(pos_dims):])

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

    spatial_map = sho_dset_collapsed[:, :, step_ind, 0][sho_quantity]
    resp_vec = sho_dset_collapsed[row_ind, col_ind, :, :]
    resp_vec = resp_func(resp_vec)

    # bias_vec = bias_mat[:, loop_ind]

    fig = plt.figure(figsize=(12, 8))
    ax_bias = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=1)
    ax_map = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    ax_loop = plt.subplot2grid((3, 2), (0, 1), colspan=1, rowspan=3)

    ax_bias.plot(bias_mat[:, 0])
    ax_bias.set_xlabel('Bias Step')
    ax_bias.set_ylabel(step_chan.replace('_', ' ') + ' (V)')
    bias_slider = ax_bias.axvline(x=step_ind, color='r')

    img_map = ax_map.imshow(spatial_map, cmap=cmap_jet_white_center(), origin='lower',
                            interpolation='none')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    main_vert_line = ax_map.axvline(x=row_ind, color='k')
    main_hor_line = ax_map.axhline(y=col_ind, color='k')

    ax_loop.axvline(x=0, color='gray', linestyle='--')
    ax_loop.axhline(y=0, color='gray', linestyle='--')
    line_handles = ax_loop.plot(bias_mat, resp_vec)
    ax_loop.set_xlabel(step_chan.replace('_', ' ') + ' (V)')
    ax_loop.set_ylabel(resp_label)
    fig.tight_layout()

    def update_sho_plots(sho_quantity, **kwargs):
        step_ind = kwargs['Bias Step']
        bias_slider.set_xdata((step_ind, step_ind))
        spatial_map = sho_dset_collapsed[:, :, step_ind, 0][sho_quantity]
        img_map.set_data(spatial_map)
        spat_mean = np.mean(spatial_map)
        spat_std = np.std(spatial_map)
        img_map.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)

        row_ind = kwargs['Y']
        col_ind = kwargs['X']
        main_vert_line.set_xdata((kwargs['X'], kwargs['X']))
        main_hor_line.set_ydata((kwargs['Y'], kwargs['Y']))

        resp_vec = sho_dset_collapsed[row_ind, col_ind, :, :]
        resp_vec = resp_func(resp_vec)
        for line_handle, data in zip(line_handles, np.transpose(resp_vec)):
            line_handle.set_ydata(data)
        ax_loop.set_ylim([np.min(resp_vec), np.max(resp_vec)])
        display(fig)

    slider_dict = dict()
    for pos_dim_ind, dim_name in enumerate(pos_labels):
        slider_dict[dim_name] = (0, pos_dims[pos_dim_ind] - 1, 1)
    slider_dict['Bias Step'] = (0, bias_mat.shape[0] - 1, 1)

    widgets.interact(update_sho_plots, sho_quantity=sho_dset_collapsed.dtype.names[:-1], **slider_dict)


def jupyter_visualize_be_spectrograms(h5_main):
    """
    Jupyer notebook ONLY function. Sets up a simple visualzier for visualizing raw BE data.
    Sliders for position indices can be used to visualize BE spectrograms (frequency, UDVS step).
    In the case of 2 spatial dimensions, a spatial map will be provided as well

    Parameters
    ----------
    h5_main : h5py.Dataset
        Raw dataset
    """
    h5_pos_inds = getAuxData(h5_main, auxDataName='Position_Indices')[-1]
    pos_sort = get_sort_order(np.transpose(h5_pos_inds))
    pos_dims = get_dimensionality(np.transpose(h5_pos_inds), pos_sort)
    pos_labels = np.array(h5_pos_inds.attrs['labels'])[pos_sort]

    h5_spec_vals = getAuxData(h5_main, auxDataName='Spectroscopic_Values')[-1]
    h5_spec_inds = getAuxData(h5_main, auxDataName='Spectroscopic_Indices')[-1]
    spec_sort = get_sort_order(h5_spec_inds)
    spec_dims = get_dimensionality(h5_spec_inds, spec_sort)
    spec_labels = np.array(h5_spec_inds.attrs['labels'])[spec_sort]

    ifreq = int(np.argwhere(spec_labels == 'Frequency'))
    freqs_nd = reshape_to_Ndims(h5_spec_vals, h5_spec=h5_spec_inds)[0][ifreq].squeeze()
    freqs_2d = freqs_nd.reshape(freqs_nd.shape[0], -1) / 1000  # Convert to kHz
    try:
        num_udvs_steps = h5_main.parent.parent.attrs['num_udvs_steps']
    except KeyError:
        num_udvs_steps = h5_main.parent.parent.attrs['num_UDVS_steps']
    # h5_udvs_inds = getAuxData(h5_main, auxDataName='UDVS_Indices')[-1]
    h5_freqs = getAuxData(h5_main, auxDataName='Bin_Frequencies')[-1]
    wfm_type_vec = getAuxData(h5_main, auxDataName='Bin_Wfm_Type')[-1][()]
    freq_inds = wfm_type_vec == np.unique(wfm_type_vec)[-1]
    freq_vals = h5_freqs[freq_inds] / 1000  # convert to kHz
    pos_labels = np.array(h5_pos_inds.attrs['labels'])
    pos_labels = pos_labels[pos_sort]

    if len(pos_dims) == 2:
        spatial_map = np.abs(np.reshape(h5_main[:, 0], pos_dims))
        spectrogram = np.reshape(h5_main[0], (num_udvs_steps, -1))
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
        # spatial_img = axes[0].imshow(np.abs(spatial_map), cmap=plt.cm.jet)
        spatial_img = plot_map(axes[0], np.abs(spatial_map), origin='lower',
                               cmap=cmap_jet_white_center())
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        main_vert_line = axes[0].axvline(x=int(0.5 * spatial_map.shape[1]), color='k')
        main_hor_line = axes[0].axhline(y=int(0.5 * spatial_map.shape[0]), color='k')
        amp_img = axes[1].imshow(np.abs(spectrogram), cmap=plt.cm.jet,
                                 extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                         spectrogram.shape[0], 0],
                                 interpolation='none')
        axes[1].set_title('Amplitude')
        axes[1].set_xlabel('Frequency (kHz)')
        axes[1].set_ylabel('BE step')
        phase_img = axes[2].imshow(np.angle(spectrogram), cmap=plt.cm.jet,
                                   extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                           spectrogram.shape[0], 0],
                                   interpolation='none')
        axes[2].set_title('Phase')
        axes[2].set_xlabel('Frequency (kHz)')
        for axis in axes[1:3]:
            axis.axis('tight')
            axis.set_ylim(0, spectrogram.shape[0])
        fig.tight_layout()

        def index_unpacker(**kwargs):
            spatial_map = np.abs(np.reshape(h5_main[:, kwargs['spectroscopic']], pos_dims))
            spatial_img.set_data(spatial_map)
            spat_mean = np.mean(spatial_map)
            spat_std = np.std(spatial_map)
            spatial_img.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)

            spec_heading = ''
            for dim_ind, dim_name in enumerate(h5_spec_vals.attrs['labels']):
                spec_heading += dim_name + ': ' + str(h5_spec_vals[dim_ind, kwargs['spectroscopic']]) + ', '
            axes[0].set_title(spec_heading[:-2])

            pos_dim_vals = range(len(pos_labels))
            for pos_dim_ind, pos_dim_name in enumerate(pos_labels):
                pos_dim_vals[pos_dim_ind] = kwargs[pos_dim_name]

            main_vert_line.set_xdata((kwargs['X'], kwargs['X']))
            main_hor_line.set_ydata((kwargs['Y'], kwargs['Y']))

            pix_ind = pos_dim_vals[0]
            for pos_dim_ind in range(1, len(pos_labels)):
                pix_ind += pos_dim_vals[pos_dim_ind] * pos_dims[pos_dim_ind - 1]
            spectrogram = np.reshape(h5_main[pix_ind], (num_udvs_steps, -1))
            amp_img.set_data(np.abs(spectrogram))
            phase_img.set_data(np.angle(spectrogram))
            display(fig)

        pos_dict = dict()
        for pos_dim_ind, dim_name in enumerate(pos_labels):
            pos_dict[dim_name] = (0, pos_dims[pos_dim_ind] - 1, 1)
        pos_dict['spectroscopic'] = (0, h5_main.shape[1] - 1, 1)

        widgets.interact(index_unpacker, **pos_dict);
    else:
        def plot_spectrogram(data, freq_vals):
            fig, axes = plt.subplots(ncols=2, figsize=(9, 5), sharey=True)
            im_handles = list()
            im_handles.append(axes[0].imshow(np.abs(data), cmap=plt.cm.jet,
                                             extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                     data.shape[0], 0],
                                             interpolation='none'))
            axes[0].set_title('Amplitude')
            axes[0].set_ylabel('BE step')
            im_handles.append(axes[1].imshow(np.angle(data), cmap=plt.cm.jet,
                                             extent=[freqs_2d[0, 0], freqs_2d[-1, 0],
                                                     data.shape[0], 0],
                                             interpolation='none'))
            axes[1].set_title('Phase');
            axes[0].set_xlabel('Frequency index')
            axes[1].set_xlabel('Frequency index')
            for axis in axes:
                axis.axis('tight')
                axis.set_ylim(0, data.shape[0])
            fig.tight_layout()
            return fig, axes, im_handles

        fig, axes, im_handles = plot_spectrogram(np.reshape(h5_main[0], (num_udvs_steps, -1)), freq_vals)

        def position_unpacker(**kwargs):
            pos_dim_vals = range(len(pos_labels))
            for pos_dim_ind, pos_dim_name in enumerate(pos_labels):
                pos_dim_vals[pos_dim_ind] = kwargs[pos_dim_name]
            pix_ind = pos_dim_vals[0]
            for pos_dim_ind in range(1, len(pos_labels)):
                pix_ind += pos_dim_vals[pos_dim_ind] * pos_dims[pos_dim_ind - 1]
            spectrogram = np.reshape(h5_main[pix_ind], (num_udvs_steps, -1))
            im_handles[0].set_data(np.abs(spectrogram))
            im_handles[1].set_data(np.angle(spectrogram))
            display(fig)

        pos_dict = dict()
        for pos_dim_ind, dim_name in enumerate(pos_labels):
            pos_dict[dim_name] = (0, pos_dims[pos_dim_ind] - 1, 1)

        widgets.interact(position_unpacker, **pos_dict)


def jupyter_visualize_beps_loops(h5_projected_loops, h5_loop_guess, h5_loop_fit, step_chan='DC_Offset'):
    """
    Interactive plotting of the BE Loops
    
    Parameters
    ----------
    h5_projected_loops : h5py.Dataset
        Dataset holding the loop projections
    h5_loop_guess : h5py.Dataset
        Dataset holding the loop guesses
    h5_loop_fit : h5py.Dataset
        Dataset holding the loop gits
    step_chan : str, optional
        The name of the Spectroscopic dimension to plot versus.  Needs testing.
        Default 'DC_Offset'

    Returns
    -------
    None
    
    """
    # Prepare some variables for plotting loops fits and guesses
    # Plot the Loop Guess and Fit Results
    proj_nd, _ = reshape_to_Ndims(h5_projected_loops)
    guess_nd, _ = reshape_to_Ndims(h5_loop_guess)
    fit_nd, _ = reshape_to_Ndims(h5_loop_fit)

    h5_projected_loops = h5_loop_guess.parent['Projected_Loops']
    h5_proj_spec_inds = getAuxData(h5_projected_loops,
                                                auxDataName='Spectroscopic_Indices')[-1]
    h5_proj_spec_vals = getAuxData(h5_projected_loops,
                                                auxDataName='Spectroscopic_Values')[-1]
    h5_pos_inds = getAuxData(h5_projected_loops,
                             auxDataName='Position_Indices')[-1]
    pos_nd, _ = reshape_to_Ndims(h5_pos_inds, h5_pos=h5_pos_inds)
    pos_dims = list(pos_nd.shape[:h5_pos_inds.shape[1]])
    pos_labels = h5_pos_inds.attrs['labels']

    # reshape the vdc_vec into DC_step by Loop
    spec_nd, _ = reshape_to_Ndims(h5_proj_spec_vals, h5_spec=h5_proj_spec_inds)
    loop_spec_dims = np.array(spec_nd.shape[1:])
    loop_spec_labels = h5_proj_spec_vals.attrs['labels']

    spec_step_dim_ind = np.where(loop_spec_labels == step_chan)[0][0]

    # # move the step dimension to be the first after all position dimensions
    rest_loop_dim_order = range(len(pos_dims), len(proj_nd.shape))
    rest_loop_dim_order.pop(spec_step_dim_ind)
    new_order = range(len(pos_dims)) + [len(pos_dims)+spec_step_dim_ind] + rest_loop_dim_order

    new_spec_order = np.array(new_order[len(pos_dims):], dtype=np.uint32)-len(pos_dims)
    # new_spec_dims = loop_spec_dims[new_spec_order]
    # new_spec_labels = loop_spec_labels[new_spec_order]

    #Also reshape the projected loops to Positions-DC_Step-Loop
    final_loop_shape = pos_dims+[loop_spec_dims[spec_step_dim_ind]]+[-1]
    proj_nd_3 = np.reshape(proj_nd, final_loop_shape)

    # Do the same for the guess and fit datasets
    guess_3d = np.reshape(guess_nd, pos_dims+[-1])
    fit_3d = np.reshape(fit_nd, pos_dims + [-1])

    # Get the bias vector:
    bias_vec = np.reshape(spec_nd[spec_step_dim_ind], final_loop_shape[len(pos_dims):])

    # Shift the bias vector and the loops by a quarter cycle
    shift_ind = int(-1*bias_vec.shape[0]/4)
    bias_shifted = np.roll(bias_vec, shift_ind, axis=0)
    proj_nd_shifted = np.roll(proj_nd_3, shift_ind, axis=len(pos_dims))

    # This is just the visualizer:
    loop_field_names = fit_nd.dtype.names
    loop_field = loop_field_names[0]
    loop_ind = 0
    row_ind = 0
    col_ind = 0

    # Initial plot data
    spatial_map = fit_3d[:,:, loop_ind][loop_field]
    proj_data = proj_nd_shifted[row_ind, col_ind, :, loop_ind]
    bias_data = bias_shifted[:, loop_ind]
    guess_data = loop_fit_function(bias_data, np.array(list(guess_3d[row_ind, col_ind, loop_ind])))
    fit_data = loop_fit_function(bias_data, np.array(list(fit_3d[row_ind, col_ind, loop_ind])))

    fig = plt.figure(figsize=(12, 8))
    ax_map = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    ax_loop = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)

    im_map = ax_map.imshow(spatial_map, cmap=cmap_jet_white_center(),
                           origin='lower', interpolation='none')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    main_vert_line = ax_map.axvline(x=row_ind, color='k')
    main_hor_line = ax_map.axhline(y=col_ind, color='k')

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

    def update_loop_plots(loop_field, **kwargs):
        loop_ind = kwargs['Loop Number']
        spatial_map = fit_3d[:,:, loop_ind][loop_field]
        im_map.set_data(spatial_map)
        spat_mean = np.mean(spatial_map)
        spat_std = np.std(spatial_map)
        im_map.set_clim(vmin=spat_mean - 3 * spat_std, vmax=spat_mean + 3 * spat_std)

        row_ind = kwargs['Y']
        col_ind = kwargs['X']
        main_vert_line.set_xdata((kwargs['X'], kwargs['X']))
        main_hor_line.set_ydata((kwargs['Y'], kwargs['Y']))

        proj_data = proj_nd_shifted[row_ind, col_ind, :, loop_ind]
        bias_data = bias_shifted[:, loop_ind]
        guess_data = loop_fit_function(bias_data, np.array(list(guess_3d[row_ind, col_ind, loop_ind])))
        fit_data = loop_fit_function(bias_data, np.array(list(fit_3d[row_ind, col_ind, loop_ind])))
        for line_handle, data in zip(line_handles, [proj_data, guess_data, fit_data]):
            line_handle.set_ydata(data)
        ax_loop.set_ylim([np.min([proj_data, guess_data, fit_data]), np.max([proj_data, guess_data, fit_data])])
        ax_loop.set_title('Position ({},{})'.format(col_ind, row_ind))
        display(fig)

    slider_dict = dict()
    for pos_dim_ind, dim_name in enumerate(pos_labels):
        slider_dict[dim_name] = (0, pos_dims[pos_dim_ind] - 1, 1)
    slider_dict['Loop Number'] = (0, bias_vec.shape[1] - 1, 1)
    widgets.interact(update_loop_plots, loop_field=fit_nd.dtype.names, **slider_dict)
