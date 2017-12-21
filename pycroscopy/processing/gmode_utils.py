# -*- coding: utf-8 -*-
"""
Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import
from collections import Iterable
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from .fft import get_noise_floor, are_compatible_filters, build_composite_freq_filter
from ..io.io_hdf5 import ioHDF5
from ..io.pycro_data import PycroDataset
from ..io.hdf_utils import getH5DsetRefs, linkRefs, getAuxData, copy_main_attributes
from ..io.microdata import MicroDataGroup, MicroDataset
from ..viz.plot_utils import rainbow_plot, set_tick_font_size
from ..io.translators.utils import build_ind_val_dsets

# TODO: Phase rotation not implemented correctly. Find and use excitation frequency


###############################################################################

def test_filter(resp_wfm, frequency_filters=None, noise_threshold=None,
                excit_wfm=None, central_resp_size=None, show_plots=True, use_rainbow_plots=True,
                verbose=False):
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
        Excitation waveform in the time domain. This waveform is necessary for plotting loops.
    central_resp_size : (Optional) unsigned int
        Number of response sample points from the center of the waveform to show in plots. Useful for SPORC
    show_plots : (Optional) Boolean
        Whether or not to plot FFTs before and after filtering
    use_rainbow_plots : (Optional) Boolean
        Whether or not to plot loops whose color varied as a function of time
    verbose : (Optional) Boolean
        Prints extra debugging information if True.  Default False

    Returns
    -------
    filt_data : 1D numpy float array
        Filtered signal in the time domain
    fig : matplotlib.pyplot.figure object
        handle to the plotted figure if requested, else None
    axes : 1D list of matplotlib.pyplot axis objects
        handles to the axes in the plotted figure if requested, else None
    """

    show_loops = excit_wfm is not None and show_plots

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


    if show_plots:
        l_ind = int(0.5 * num_pts)
        if isinstance(composite_filter, np.ndarray):
            r_ind = np.max(np.where(composite_filter > 0)[0])
        else:
            r_ind = num_pts
        if verbose:
            print('The left index is {} and the right index is {}.'.format(l_ind, r_ind))

        w_vec = np.linspace(-0.5 * samp_rate, 0.5 * samp_rate, num_pts) * 1E-3
        if central_resp_size:
            sz = int(0.5 * central_resp_size)
            l_resp_ind = -sz + l_ind
            r_resp_ind = l_ind + sz
        else:
            l_resp_ind = l_ind
            r_resp_ind = num_pts
        if verbose:
            print('The left response index is {} and the right response index is {}.'.format(l_resp_ind, r_resp_ind))

        fig = plt.figure(figsize=(12, 8))
        lhs_colspan = 2
        if show_loops is False:
            lhs_colspan = 4
        else:
            ax_loops = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
        ax_raw = plt.subplot2grid((2, 4), (0, 0), colspan=lhs_colspan)
        ax_filt = plt.subplot2grid((2, 4), (1, 0), colspan=lhs_colspan)
        axes = [ax_raw, ax_filt]
        set_tick_font_size(axes, 14)
    else:
        fig = None
        axes = None

    if show_plots:
        amp = np.abs(fft_pix_data)
        ax_raw.semilogy(w_vec[l_ind:r_ind], amp[l_ind:r_ind], label='Raw')
        if frequency_filters is not None:
            ax_raw.semilogy(w_vec[l_ind:r_ind],
                            (composite_filter[l_ind:r_ind] + np.min(amp)) * (np.max(amp) - np.min(amp)),
                            linewidth=3, color='orange', label='Composite Filter')
        if noise_threshold is not None:
            ax_raw.semilogy(w_vec[l_ind:r_ind], np.ones(r_ind - l_ind) * noise_floor,
                            linewidth=2, color='r', label='Noise Threshold')
        ax_raw.legend(loc='best', fontsize=14)
        ax_raw.set_title('Raw Signal', fontsize=16)
        ax_raw.set_ylabel('Magnitude (a. u.)', fontsize=14)

    fft_pix_data *= composite_filter

    if noise_threshold is not None:
        fft_pix_data[np.abs(fft_pix_data) < noise_floor] = 1E-16  # DON'T use 0 here. ipython kernel dies

    if show_plots:
        ax_filt.semilogy(w_vec[l_ind:r_ind], np.abs(fft_pix_data[l_ind:r_ind]))
        ax_filt.set_title('Filtered Signal', fontsize=16)
        ax_filt.set_xlabel('Frequency(kHz)', fontsize=14)
        ax_filt.set_ylabel('Magnitude (a. u.)', fontsize=14)
        if noise_threshold is not None:
            ax_filt.set_ylim(bottom=noise_floor)  # prevents the noise threshold from messing up plots

    filt_data = np.real(np.fft.ifft(np.fft.ifftshift(fft_pix_data)))

    if verbose:
        print('The shape of the filtered data is {}'.format(filt_data.shape))
        print('The shape of the excitation waveform is {}'.format(excit_wfm.shape))

    if show_loops:
        if use_rainbow_plots:
            rainbow_plot(ax_loops, excit_wfm[l_resp_ind:r_resp_ind], filt_data[l_resp_ind:r_resp_ind] * 1E+3)
        else:
            ax_loops.plot(excit_wfm[l_resp_ind:r_resp_ind], filt_data[l_resp_ind:r_resp_ind] * 1E+3)
        ax_loops.set_title('AI vs AO', fontsize=16)
        ax_loops.set_xlabel('Input Bias (V)', fontsize=14)
        ax_loops.set_ylabel('Deflection (mV)', fontsize=14)
        set_tick_font_size(ax_loops, 14)
        axes.append(ax_loops)
        fig.tight_layout()
    return filt_data, fig, axes


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


def reshape_from_lines_to_pixels(h5_main, pts_per_cycle, scan_step_x_m=1):
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
    if h5_main.shape[1] % pts_per_cycle != 0:
        warn('Error in reshaping the provided dataset to pixels. Check points per pixel')
        raise ValueError

    num_cols = int(h5_main.shape[1] / pts_per_cycle)

    h5_spec_vals = getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0]
    h5_pos_vals = getAuxData(h5_main, auxDataName=['Position_Values'])[0]
    single_AO = h5_spec_vals[:, :pts_per_cycle]

    ds_spec_inds, ds_spec_vals = build_ind_val_dsets([single_AO.size], is_spectral=True,
                                                     labels=h5_spec_vals.attrs['labels'],
                                                     units=h5_spec_vals.attrs['units'], verbose=False)
    ds_spec_vals.data = np.atleast_2d(single_AO)  # The data generated above varies linearly. Override.

    ds_pos_inds, ds_pos_vals = build_ind_val_dsets([num_cols, h5_main.shape[0]], is_spectral=False,
                                                   steps=[scan_step_x_m, h5_pos_vals[1, 0]],
                                                   labels=['X', 'Y'], units=['m', 'm'], verbose=False)

    ds_reshaped_data = MicroDataset('Reshaped_Data', data=np.reshape(h5_main.value, (-1, pts_per_cycle)),
                                    compression='gzip', chunking=(10, pts_per_cycle))

    # write this to H5 as some form of filtered data.
    resh_grp = MicroDataGroup(h5_main.name.split('/')[-1] + '-Reshape_', parent=h5_main.parent.name)
    resh_grp.addChildren([ds_reshaped_data, ds_pos_inds, ds_pos_vals, ds_spec_inds, ds_spec_vals])

    hdf = ioHDF5(h5_main.file)
    print('Starting to reshape G-mode line data. Please be patient')
    h5_refs = hdf.writeData(resh_grp)

    h5_resh = getH5DsetRefs(['Reshaped_Data'], h5_refs)[0]
    # Link everything:
    linkRefs(h5_resh,
             getH5DsetRefs(['Position_Indices', 'Position_Values', 'Spectroscopic_Indices', 'Spectroscopic_Values'],
                           h5_refs))

    # Copy the two attributes that are really important but ignored:
    copy_main_attributes(h5_main, h5_resh)

    print('Finished reshaping G-mode line data to rows and columns')

    return PycroDataset(h5_resh)
