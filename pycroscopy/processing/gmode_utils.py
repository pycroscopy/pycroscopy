# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import
import sys
from collections import Iterable
from warnings import warn
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np
from .fft import get_noise_floor, are_compatible_filters, build_composite_freq_filter
from pyUSID import USIDataset
from pyUSID.io.hdf_utils import check_if_main, get_attr, write_main_dataset, create_results_group
from pyUSID.viz.plot_utils import set_tick_font_size, plot_curves
from pyUSID.io.write_utils import Dimension

if sys.version_info.major == 3:
    unicode = str

# TODO: Phase rotation not implemented correctly. Find and use excitation frequency


###############################################################################

def test_filter(resp_wfm, frequency_filters=None, noise_threshold=None, excit_wfm=None, show_plots=True,
                plot_title=None, verbose=False):
    """
    Filters the provided response with the provided filters.

    Parameters
    ----------
    resp_wfm : array-like, 1D
        Raw response waveform in the time domain
    frequency_filters : (Optional) FrequencyFilter object or list of
        Frequency filters to apply to signal
    noise_threshold : (Optional) float
        Noise threshold to apply to signal
    excit_wfm : (Optional) 1D array-like
        Excitation waveform in the time domain. This waveform is necessary for plotting loops. If the length of
        resp_wfm matches excit_wfm, a single plot will be returned with the raw and filtered signals plotted against the
        excit_wfm. Else, resp_wfm and the filtered (filt_data) signal will be broken into chunks matching the length of
        excit_wfm and a figure with multiple plots (one for each chunk) with the raw and filtered signal chunks plotted
        against excit_wfm will be returned for fig_loops
    show_plots : (Optional) Boolean
        Whether or not to plot FFTs before and after filtering
    plot_title : str / unicode (Optional)
        Title for the raw vs filtered plots if requested. For example - 'Row 15'
    verbose : (Optional) Boolean
        Prints extra debugging information if True.  Default False

    Returns
    -------
    filt_data : 1D numpy float array
        Filtered signal in the time domain
    fig_fft : matplotlib.pyplot.figure object
        handle to the plotted figure if requested, else None
    fig_loops : matplotlib.pyplot.figure object
        handle to figure with the filtered signal and raw signal plotted against the excitation waveform
    """
    if not isinstance(resp_wfm, (np.ndarray, list)):
        raise TypeError('resp_wfm should be array-like')
    resp_wfm = np.array(resp_wfm)

    show_loops = False
    if excit_wfm is not None and show_plots:
        if len(resp_wfm) % len(excit_wfm) == 0:
            show_loops = True
        else:
            raise ValueError('Length of resp_wfm should be divisibe by length of excit_wfm')
    if show_loops:
        if plot_title is None:
            plot_title = 'FFT Filtering'
        else:
            assert isinstance(plot_title, (str, unicode))

    if frequency_filters is None and noise_threshold is None:
        raise ValueError('Need to specify at least some noise thresholding / frequency filter')

    if noise_threshold is not None:
        if noise_threshold >= 1 or noise_threshold <= 0:
            raise ValueError('Noise threshold must be within (0 1)')

    samp_rate = 1
    composite_filter = 1
    if frequency_filters is not None:
        if not isinstance(frequency_filters, Iterable):
            frequency_filters = [frequency_filters]
        if not are_compatible_filters(frequency_filters):
            raise ValueError('frequency filters must be a single or list of FrequencyFilter objects')
        composite_filter = build_composite_freq_filter(frequency_filters)
        samp_rate = frequency_filters[0].samp_rate

    resp_wfm = np.array(resp_wfm)
    num_pts = resp_wfm.size

    fft_pix_data = np.fft.fftshift(np.fft.fft(resp_wfm))

    if noise_threshold is not None:
        noise_floor = get_noise_floor(fft_pix_data, noise_threshold)[0]
        if verbose:
            print('The noise_floor is', noise_floor)

    fig_fft = None
    if show_plots:
        w_vec = np.linspace(-0.5 * samp_rate, 0.5 * samp_rate, num_pts) * 1E-3

        fig_fft, [ax_raw, ax_filt] = plt.subplots(figsize=(12, 8), nrows=2)
        axes_fft = [ax_raw, ax_filt]
        set_tick_font_size(axes_fft, 14)

        r_ind = num_pts
        if isinstance(composite_filter, np.ndarray):
            r_ind = np.max(np.where(composite_filter > 0)[0])

        x_lims = slice(len(w_vec) // 2, r_ind)
        amp = np.abs(fft_pix_data)
        ax_raw.semilogy(w_vec[x_lims], amp[x_lims], label='Raw')
        if frequency_filters is not None:
            ax_raw.semilogy(w_vec[x_lims], (composite_filter[x_lims] + np.min(amp)) * (np.max(amp) - np.min(amp)),
                            linewidth=3, color='orange', label='Composite Filter')
        if noise_threshold is not None:
            ax_raw.axhline(noise_floor,
                           # ax_raw.semilogy(w_vec, np.ones(r_ind - l_ind) * noise_floor,
                           linewidth=2, color='r', label='Noise Threshold')
        ax_raw.legend(loc='best', fontsize=14)
        ax_raw.set_title('Raw Signal', fontsize=16)
        ax_raw.set_ylabel('Magnitude (a. u.)', fontsize=14)

    fft_pix_data *= composite_filter

    if noise_threshold is not None:
        fft_pix_data[np.abs(fft_pix_data) < noise_floor] = 1E-16  # DON'T use 0 here. ipython kernel dies

    if show_plots:
        ax_filt.semilogy(w_vec[x_lims], np.abs(fft_pix_data)[x_lims])
        ax_filt.set_title('Filtered Signal', fontsize=16)
        ax_filt.set_xlabel('Frequency(kHz)', fontsize=14)
        ax_filt.set_ylabel('Magnitude (a. u.)', fontsize=14)
        if noise_threshold is not None:
            ax_filt.set_ylim(bottom=noise_floor)  # prevents the noise threshold from messing up plots
        fig_fft.tight_layout()

    filt_data = np.real(np.fft.ifft(np.fft.ifftshift(fft_pix_data)))

    if verbose:
        print('The shape of the filtered data is {}'.format(filt_data.shape))
        print('The shape of the excitation waveform is {}'.format(excit_wfm.shape))

    fig_loops = None
    if show_loops:
        if len(resp_wfm) == len(excit_wfm):
            # single plot:
            fig_loops, axis = plt.subplots(figsize=(5.5, 5))
            axis.plot(excit_wfm, resp_wfm, 'r', label='Raw')
            axis.plot(excit_wfm, filt_data, 'k', label='Filtered')
            axis.legend(fontsize=14)
            set_tick_font_size(axis, 14)
            axis.set_xlabel('Excitation', fontsize=16)
            axis.set_ylabel('Signal', fontsize=16)
            axis.set_title(plot_title, fontsize=16)
            fig_loops.tight_layout()
        else:
            # N loops:
            raw_pixels = np.reshape(resp_wfm, (-1, len(excit_wfm)))
            filt_pixels = np.reshape(filt_data, (-1, len(excit_wfm)))
            print(raw_pixels.shape, filt_pixels.shape)

            fig_loops, axes_loops = plot_curves(excit_wfm, [raw_pixels, filt_pixels], line_colors=['r', 'k'],
                                                dataset_names=['Raw', 'Filtered'], x_label='Excitation',
                                                y_label='Signal', subtitle_prefix='Col ', num_plots=16,
                                                title=plot_title)

    return filt_data, fig_fft, fig_loops


###############################################################################


def decompress_response(f_condensed_mat, num_pts, hot_inds):
    """
    Returns the time domain representation of waveform(s) that are compressed in the frequency space
    
    Parameters
    ----------
    f_condensed_mat : 1D or 2D complex numpy arrays
        Frequency domain signals arranged as [position, frequency]. 
        Only the positive frequncy bins must be in the compressed dataset. 
        The dataset is assumed to have been FFT shifted (such that 0 Hz is at the center).
    num_pts : unsigned int
        Number of points in the time domain signal
    hot_inds : 1D unsigned int numpy array
        Indices of the frequency bins in the compressed data. 
        This index array will be necessary to reverse map the condensed 
        FFT into its original form
        
    Returns
    -------
    time_resp : 2D numpy array
        Time domain response arranged as [position, time]
        
    Notes
    -----
    Memory is given higher priority here, so this function loops over the position
    instead of doing the inverse FFT on the complete data.

    """
    if num_pts % 1 != 0:
        raise ValueError('num_pts should be an integer')
    if not isinstance(f_condensed_mat, (np.ndarray, list)):
        raise TypeError('f_condensed_mat should be array-like')
    if f_condensed_mat.dtype not in [np.complex, np.complex64, np.complex128]:
        raise TypeError('f_condensed_mat should be a complex array')
    if not isinstance(hot_inds, (np.ndarray, list)):
        raise TypeError('hot_inds should be array-like')
    hot_inds = np.array(hot_inds)
    if hot_inds.ndim > 1:
        raise ValueError('hot_inds should be a 1D array')

    f_condensed_mat = np.array(f_condensed_mat)
    f_condensed_mat = np.atleast_2d(f_condensed_mat)
    hot_inds_mirror = np.flipud(num_pts - hot_inds)
    time_resp = np.zeros(shape=(f_condensed_mat.shape[0], num_pts), dtype=np.float32)
    for pos in range(f_condensed_mat.shape[0]):
        f_complete = np.zeros(shape=num_pts, dtype=np.complex)
        f_complete[hot_inds] = f_condensed_mat[pos, :]
        # Now add the mirror (FFT in negative X axis that was removed)
        f_complete[hot_inds_mirror] = np.flipud(f_condensed_mat[pos, :])
        time_resp[pos, :] = np.real(np.fft.ifft(np.fft.ifftshift(f_complete)))

    return np.squeeze(time_resp)


def reshape_from_lines_to_pixels(h5_main, pts_per_cycle, scan_step_x_m=None):
    """
    Breaks up the provided raw G-mode dataset into lines and pixels (from just lines)

    Parameters
    ----------
    h5_main : h5py.Dataset object
        Reference to the main dataset that contains the raw data that is only broken up by lines
    pts_per_cycle : unsigned int
        Number of points in a single pixel
    scan_step_x_m : float
        Step in meters for pixels

    Returns
    -------
    h5_resh : h5py.Dataset object
        Reference to the main dataset that contains the reshaped data
    """
    if not check_if_main(h5_main):
        raise TypeError('h5_main is not a Main dataset')
    h5_main = USIDataset(h5_main)
    if pts_per_cycle % 1 != 0 or pts_per_cycle < 1:
        raise TypeError('pts_per_cycle should be a positive integer')
    if scan_step_x_m is not None:
        if not isinstance(scan_step_x_m, Number):
            raise TypeError('scan_step_x_m should be a real number')
    else:
        scan_step_x_m = 1

    if h5_main.shape[1] % pts_per_cycle != 0:
        warn('Error in reshaping the provided dataset to pixels. Check points per pixel')
        raise ValueError

    num_cols = int(h5_main.shape[1] / pts_per_cycle)

    # TODO: DO NOT assume simple 1 spectral dimension!
    single_ao = np.squeeze(h5_main.h5_spec_vals[:, :pts_per_cycle])

    spec_dims = Dimension(get_attr(h5_main.h5_spec_vals, 'labels')[0],
                          get_attr(h5_main.h5_spec_vals, 'units')[0], single_ao)

    # TODO: DO NOT assume simple 1D in positions!
    pos_dims = [Dimension('X', 'm', np.linspace(0, scan_step_x_m, num_cols)),
                Dimension('Y', 'm', np.linspace(0, h5_main.h5_pos_vals[1, 0], h5_main.shape[0]))]

    h5_group = create_results_group(h5_main, 'Reshape')
    # TODO: Create empty datasets and then write for very large datasets
    h5_resh = write_main_dataset(h5_group, (num_cols * h5_main.shape[0], pts_per_cycle), 'Reshaped_Data',
                                 get_attr(h5_main, 'quantity')[0], get_attr(h5_main, 'units')[0], pos_dims, spec_dims,
                                 chunks=(10, pts_per_cycle), dtype=h5_main.dtype, compression=h5_main.compression)

    # TODO: DON'T write in one shot assuming small datasets fit in memory!
    print('Starting to reshape G-mode line data. Please be patient')
    h5_resh[()] = np.reshape(h5_main[()], (-1, pts_per_cycle))

    print('Finished reshaping G-mode line data to rows and columns')

    return USIDataset(h5_resh)
