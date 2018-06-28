import numpy as np
import scipy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from pyUSID.viz.plot_utils import plot_map


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
