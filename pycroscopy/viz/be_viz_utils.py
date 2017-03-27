import os
from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display

from .plot_utils import plot_loops, plot_map_stack, cmap_jet_white_center, plot_map
from ..io.hdf_utils import reshape_to_Ndims, getAuxData, get_sort_order, get_dimensionality


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


def jupyter_visualize_beps_sho(h5_sho_dset, step_chan, resp_func=None, resp_label='Response'):

    guess_3d_data, success = reshape_to_Ndims(h5_sho_dset)

    h5_sho_spec_inds = getAuxData(h5_sho_dset, 'Spectroscopic_Indices')[0]
    h5_sho_spec_vals = getAuxData(h5_sho_dset, 'Spectroscopic_Values')[0]
    sho_spec_sort = get_sort_order(h5_sho_spec_inds)
    sho_spec_dims = get_dimensionality(h5_sho_spec_inds[()], sho_spec_sort)
    sho_spec_labels = np.array(h5_sho_spec_vals.attrs['labels'])
    sho_spec_labels = sho_spec_labels[sho_spec_sort]

    h5_pos_inds = getAuxData(h5_sho_dset, auxDataName='Position_Indices')[-1]
    pos_sort = get_sort_order(np.transpose(h5_pos_inds))
    pos_dims = get_dimensionality(np.transpose(h5_pos_inds), pos_sort)
    pos_labels = np.array(h5_pos_inds.attrs['labels'])
    pos_labels = pos_labels[pos_sort]

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

    # Get the bias vector:
    bias_vec = []
    bias_ind_vec = np.unique(h5_sho_spec_inds[spec_step_dim_ind])
    for step_ind in np.unique(bias_ind_vec):
        temp = np.where(h5_sho_spec_inds[spec_step_dim_ind] == step_ind)[0][0]
        bias_vec.append(h5_sho_spec_vals[spec_step_dim_ind, temp])
    bias_vec = np.array(bias_vec)

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

    fig = plt.figure(figsize=(12, 8))
    ax_bias = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=1)
    ax_map = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
    ax_loop = plt.subplot2grid((3, 2), (0, 1), colspan=1, rowspan=3)

    ax_bias.plot(bias_vec)
    ax_bias.set_xlabel('Bias Step')
    ax_bias.set_ylabel(step_chan.replace('_', ' ') + ' (V)')
    bias_slider = ax_bias.axvline(x=step_ind, color='r')

    img_map = ax_map.imshow(spatial_map, cmap=cmap_jet_white_center(), origin='lower')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    main_vert_line = ax_map.axvline(x=row_ind, color='k')
    main_hor_line = ax_map.axhline(y=col_ind, color='k')

    ax_loop.axvline(x=0, color='gray', linestyle='--')
    ax_loop.axhline(y=0, color='gray', linestyle='--')
    line_handles = ax_loop.plot(bias_vec, resp_vec)
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
    slider_dict['Bias Step'] = (0, bias_vec.size - 1, 1)

    widgets.interact(update_sho_plots, sho_quantity=sho_dset_collapsed.dtype.names[:-1], **slider_dict)


def jupyter_visualize_be_spectrograms(h5_main):

    h5_spec_vals = getAuxData(h5_main, auxDataName='Spectroscopic_Values')[-1]
    h5_pos_inds = getAuxData(h5_main, auxDataName='Position_Indices')[-1]
    pos_sort = get_sort_order(np.transpose(h5_pos_inds))
    pos_dims = get_dimensionality(np.transpose(h5_pos_inds), pos_sort)
    num_udvs_steps = h5_main.parent.parent.attrs['num_udvs_steps']
    h5_udvs_inds = getAuxData(h5_main, auxDataName='UDVS_Indices')[-1]
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
                                 extent=[freq_vals[0], freq_vals[-1], spectrogram.shape[0], 0])
        axes[1].set_title('Amplitude')
        axes[1].set_xlabel('Frequency (kHz)')
        axes[1].set_ylabel('BE step')
        phase_img = axes[2].imshow(np.angle(spectrogram), cmap=plt.cm.jet,
                                   extent=[freq_vals[0], freq_vals[-1], spectrogram.shape[0], 0])
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
                                             extent=[freq_vals[0], freq_vals[-1], data.shape[0], 0]))
            axes[0].set_title('Amplitude')
            axes[0].set_ylabel('BE step')
            im_handles.append(axes[1].imshow(np.angle(data), cmap=plt.cm.jet,
                                             extent=[freq_vals[0], freq_vals[-1], data.shape[0], 0]))
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
