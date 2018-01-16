# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:42:41 2015

@author: Suhas Somnath
"""
###############################################################################

from __future__ import division, print_function, absolute_import
import numpy as np  # for all array, data operations
import matplotlib.pyplot as plt  # for all plots
from scipy.special import erf
from collections import Iterable
from warnings import warn


def get_fft_stack(image_stack):
    """
    Gets the 2D FFT for a single or stack of images by applying a blackman window

    Parameters
    ----------
    image_stack : 2D or 3D real numpy array
        Either a 2D matrix [x, y] or a stack of 2D images arranged as [z or spectral, x, y]

    Returns
    -------
    fft_stack : 2D or 3D real numpy array
        2 or 3 dimensional matrix arranged as [z or spectral, x, y]

    """
    if image_stack.ndim == 2:
        # single image
        image_stack = np.expand_dims(image_stack, axis=0)
    blackman_2d = np.atleast_2d(np.blackman(image_stack.shape[2])) * np.atleast_2d(np.blackman(image_stack.shape[1])).T
    blackman_3d = np.expand_dims(blackman_2d, axis=0)
    fft_stack = blackman_3d * image_stack
    fft_stack = np.abs(np.fft.fftshift(np.fft.fft2(fft_stack, axes=(1, 2)), axes=(1, 2)))
    return np.squeeze(fft_stack)


def build_radius_matrix(image_shape):
    """
    Builds a matrix where the value of a given pixel is its L2 distance from the origin, which is located at the
    center of the provided image rather one of the corners of the image. The result from this function is required
    by get_2d_gauss_lpf

    Parameters
    ----------
    image_shape: list or tuple
        Number of rows and columns in the image

    Returns
    -------
    radius_mat: 2d numpy float array
        Radius matrix
    """
    (u_mat, v_mat) = np.meshgrid(range(-image_shape[0] // 2, image_shape[0] // 2, 1),
                                 range(-image_shape[1] // 2, image_shape[1] // 2, 1))
    return np.sqrt(u_mat ** 2 + v_mat ** 2)


def get_2d_gauss_lpf(radius_mat, filter_width):
    """
    Builds a 2D, radially symmetric, low-pass Gaussian filter based on the provided radius matrix. The corresponding
    high pass filter can be built simply by subtracting the resulting low-pass filter from 1.

    Multiply the output of this function with the (shifted) fft of an image to apply the filter.

    Parameters
    ----------
    radius_mat: 2d numpy float array
        A [NxM] matrix of the same size as the image that this filter will be applied to
    filter_width: float
        Size of the filter

    Returns
    -------
    gauss_filt: 2D numpy float array
        matrix with a single gaussian peak at the center of the matrix.
    """
    return np.e ** (-(radius_mat * filter_width) ** 2)


def fft_to_real(image):
    """
    Provides the real-space equivalent of the provided image in Fourier space

    Parameters
    ----------
    image: 2D numpy float array
        FFT of image that has been fft shifted.

    Returns
    -------
    image : 2D numpy float array
        Image in real space
    """
    return np.real(np.fft.ifft2(np.fft.ifftshift(image)))


def get_noise_floor(fft_data, tolerance):
    """
    Calculate the noise floor from the FFT data. Algorithm originally written by Mahmut Okatan Baris

    Parameters
    ----------
    fft_data : 1D or 2D complex numpy array
        Signal in frequency space (ie - after FFT shifting) arranged as (channel or repetition, signal)
    tolerance : unsigned float
        Tolerance to noise. A smaller value gets rid of more noise.
        
    Returns
    -------
    noise_floor : 1D array-like
        One value per channel / repetition

    """

    fft_data = np.atleast_2d(fft_data)
    # Noise calculated on the second axis

    noise_floor = []

    fft_data = np.abs(fft_data)
    num_pts = fft_data.shape[1]

    for amp in fft_data:

        prev_val = np.sqrt(np.sum(amp ** 2) / (2 * num_pts))
        threshold = np.sqrt((2 * prev_val ** 2) * (-np.log(tolerance)))

        residual = 1
        iterations = 1

        while (residual > 10 ** -2) and iterations < 50:
            amp[amp > threshold] = 0
            new_val = np.sqrt(np.sum(amp ** 2) / (2 * num_pts))
            residual = np.abs(new_val - prev_val)
            threshold = np.sqrt((2 * new_val ** 2) * (-np.log(tolerance)))
            prev_val = new_val
            iterations += 1

        noise_floor.append(threshold)

    return noise_floor


###############################################################################

def down_sample(fft_vec, freq_ratio):
    """
    Downsamples the provided data vector
    
    Parameters
    ----------
    fft_vec : 1D complex numpy array
        Waveform that is already FFT shifted
    freq_ratio : float
        new sampling rate / old sampling rate (less than 1)
    
    Returns
    -------
    fft_vec : 1D numpy array
        downsampled waveform

    """
    if freq_ratio >= 1:
        warn('Error at downSample: New sampling rate > old sampling rate')
        return fft_vec

    vec_len = len(fft_vec)
    ind = np.round(float(vec_len) * (0.5 * float(freq_ratio)))
    fft_vec = fft_vec[max(0.5 * vec_len - ind, 0):min(0.5 * vec_len + ind, vec_len)]
    fft_vec = fft_vec * freq_ratio * 2

    return np.fft.ifft(np.fft.ifftshift(fft_vec))


###############################################################################


class FrequencyFilter(object):
    def __init__(self, signal_length, samp_rate, *args, **kwargs):
        for val, name in zip([signal_length, samp_rate], ['Signal length', 'Sampling rate']):
            if val % 1 != 0 or val < 0:
                raise ValueError(name + ' must be an unsigned integer')
        self.signal_length = abs(int(signal_length))
        self.samp_rate = samp_rate
        self.value = None

    def get_parms(self):
        return {'samp_rate': self.samp_rate, 'signal_length': self.signal_length}

    def is_compatible(self, other):
        assert isinstance(other, FrequencyFilter), "Other object must be a FrequencyFilter object"
        return self.signal_length == other.signal_length and self.samp_rate == other.samp_rate


def are_compatible_filters(frequency_filters):
    if isinstance(frequency_filters, FrequencyFilter):
        return True
    if not isinstance(frequency_filters, Iterable):
        raise TypeError('frequency filters must be a single or list of FrequencyFilter objects')
    tests = [isinstance(obj, FrequencyFilter) for obj in frequency_filters]
    if not np.all(np.array(tests)):
        raise TypeError('frequency filters must be a list of FrequencyFilter objects')
    ref_filter = frequency_filters[0]
    for ind in range(1, len(frequency_filters)):
        if not ref_filter.is_compatible(frequency_filters[ind]):
            return False
    return True


def build_composite_freq_filter(frequency_filters):
    if not are_compatible_filters(frequency_filters):
        raise ValueError('frequency filters must be a single or list of FrequencyFilter objects')

    if not isinstance(frequency_filters, Iterable):
        frequency_filters = [frequency_filters]

    comp_filter = np.float32(frequency_filters[0].value)

    for ind in range(1, len(frequency_filters)):
        comp_filter *= frequency_filters[ind].value

    return comp_filter


class NoiseBandFilter(FrequencyFilter):
    def __init__(self, signal_length, samp_rate, freqs, freq_widths, show_plots=False):
        """
        Builds a filter that removes specified noise frequencies

        Parameters
        ----------
        signal_length : unsigned int
            Number of points in the FFT signal
        samp_rate : unsigned int
            sampling rate in Hz
        freqs : 1D array or list
            Target frequencies as unsigned ints
        freq_widths : 1D array or list
            Width around the target frequency that should be set to 0\n
        show_plots : bool
            If True, plots will be displayed during calculation.  Default False

        Note
        ----
        sampRate, freqs, freq_widths have same units - eg MHz

        Returns
        -------
        noise_filter : 1D numpy array
            Array of ones set to 0 at noise bands

        """
        super(NoiseBandFilter, self).__init__(signal_length, samp_rate)

        w_vec = 1

        # Making code a little more robust with handling different inputs:
        samp_rate = float(samp_rate)
        freqs = np.array(freqs)
        freq_widths = np.array(freq_widths)
        if freqs.ndim != freq_widths.ndim:
            raise ValueError('Error in noiseBandFilter: dimensionality of frequencies and frequency widths do not match!')
        if freqs.shape != freq_widths.shape:
            raise ValueError('Error in noiseBandFilter: shape of frequencies and frequency widths do not match!')

        self.freqs = freqs
        self.freq_widths = freq_widths

        cent = int(round(0.5 * signal_length))

        noise_filter = np.ones(signal_length, dtype=np.int16)

        if show_plots:
            w_vec = np.arange(-0.5 * samp_rate, 0.5 * samp_rate, samp_rate / signal_length)
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(w_vec, noise_filter)
            ax[0].set_yscale('log')
            ax[0].axis('tight')
            ax[0].set_xlabel('Freq')
            ax[0].set_title('Before clean up')

        # Setting noise freq bands to 0
        for cur_freq, d_freq in zip(freqs, freq_widths):
            ind = int(round(signal_length * (cur_freq / samp_rate)))
            sz = int(round(cent * d_freq / samp_rate))
            noise_filter[cent - ind - sz:cent - ind + sz + 1] = 0
            noise_filter[cent + ind - sz:cent + ind + sz + 1] = 0

        if show_plots:
            ax[1].plot(w_vec, noise_filter)
            ax[1].set_yscale('log')
            ax[1].axis('tight')
            ax[1].set_xlabel('Freq')
            ax[1].set_title('After clean up')
            plt.show()

        self.value = noise_filter

    def get_parms(self):
        basic_parms = super(NoiseBandFilter, self).get_parms()
        prefix = 'noise_band_'
        this_parms = {prefix+'freqs': self.freqs, prefix+'widths': self.freq_widths}
        this_parms.update(basic_parms)
        return this_parms


class LowPassFilter(FrequencyFilter):
    def __init__(self, signal_length, samp_rate, f_cutoff, roll_off=0.05):
        """
        Builds a low pass filter

        Parameters
        ----------
        signal_length : unsigned int
            Points in the FFT. Assuming Signal in frequency space (ie - after FFT shifting)
        samp_rate : unsigned integer
            Sampling rate
        f_cutoff : unsigned integer
            Cutoff frequency for filter
        roll_off : 0 < float < 1
            Frequency band over which the filter rolls off. rol off = 0.05 on a
            100 kHz low pass filter -> roll off from 95 kHz (1) to 100 kHz (0)

        Returns
        -------
        LPF : 1D numpy array describing the low pass filter

        """

        if f_cutoff >= 0.5 * samp_rate:
            raise ValueError('Error in LPFClip --> LPF too high! Skipping')

        self.f_cutoff = f_cutoff
        self.roll_off = roll_off

        super(LowPassFilter, self).__init__(signal_length, samp_rate)

        cent = int(round(0.5 * signal_length))

        # BW = 0.1; %MHz - Nothing beyond BW.
        roll_off *= f_cutoff  # MHz

        sz = int(np.round(signal_length * (roll_off / samp_rate)))
        ind = int(np.round(signal_length * (f_cutoff / samp_rate)))

        lpf = np.zeros(signal_length, dtype=np.float32)

        extent = 5.0
        t2 = np.linspace(-extent / 2, extent / 2, num=sz)
        smoothing = 0.5 * (1 + erf(t2))

        lpf[cent - ind:cent - ind + sz] = smoothing
        lpf[cent - ind + sz:cent + ind - sz + 1] = 1
        lpf[cent + ind - sz + 1:cent + ind + 1] = 1 - smoothing

        self.value = lpf

    def get_parms(self):
        basic_parms = super(LowPassFilter, self).get_parms()
        prefix = 'low_pass_'
        this_parms = {prefix+'cut_off': self.f_cutoff, prefix+'widths': self.roll_off}
        this_parms.update(basic_parms)
        return this_parms


class HarmonicPassFilter(FrequencyFilter):
    def __init__(self, signal_length, samp_rate, first_freq, band_width, num_harm, do_plots=False):
        """
        Builds a filter that only keeps N harmonics

        Parameters
        ----------
        signal_length : unsigned int
            Number of points in the FFt signal
        samp_rate : unsigned int
            Sampling rate
        first_freq : unsigned int
            Frequency of the first harmonic
        band_width : unsigned int
            Frequency band around each harmonic that needs to be preserved
        num_harm : unsigned int
            Number of harmonics to preserve
        do_plots : Boolean (optional)
            Whether or not to generate plots. Not necessary after debugging

        Note that the frequency values must all have the same units

        Returns
        -------
        harm_filter : 1D numpy array
            0s where the signal is to be rejected and 1s at harmonics

        """

        super(HarmonicPassFilter, self).__init__(signal_length, samp_rate)

        signal_length = abs(int(signal_length))

        harm_filter = np.ones(signal_length, dtype=np.int16)

        cent = int(round(0.5 * signal_length))

        self.first_freq = first_freq
        self.band_width = band_width
        self.num_harm = num_harm

        w_vec = 1

        if do_plots:
            print(
                'OnlyKeepHarmonics: samp_rate = %2.1e Hz, first harmonic = %3.2f Hz, %d harmonics w/- %3.2f Hz bands\n' % (
                    samp_rate, first_freq, num_harm, band_width))
            w_vec = np.arange(-samp_rate / 2.0, samp_rate / 2.0, samp_rate / signal_length)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(w_vec, harm_filter)
            ax.set_title('Raw')

        sz = int(round(cent * band_width / samp_rate))

        # First harmonic
        ind = int(round(signal_length * (first_freq / samp_rate)))
        if ind >= signal_length:
            warn()
            raise ValueError('Invalid harmonic frequency')

        harm_filter[max(cent - ind + sz + 1, 0):min(signal_length, cent + ind - sz)] = 0

        if do_plots:
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.plot(w_vec, harm_filter)
            ax2.set_title('Step 1')

        # Last harmonic
        ind = int(round(signal_length * (num_harm * first_freq / samp_rate)))
        harm_filter[:cent - ind - sz] = 0
        harm_filter[cent + ind + sz + 1:] = 0

        if do_plots:
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            ax3.plot(w_vec, harm_filter)
            ax3.set_title('Step 2')

        if num_harm > 1:
            for harm_ind in range(1, num_harm):
                ind = int(round(signal_length * (harm_ind * first_freq / samp_rate)))
                ind2 = int(round(signal_length * ((harm_ind + 1) * first_freq / samp_rate)))
                harm_filter[cent - ind2 + sz + 1:cent - ind - sz] = 0
                harm_filter[cent + ind + sz + 1:cent + ind2 - sz] = 0
                if do_plots:
                    fig4, ax4 = plt.subplots(figsize=(5, 5))
                    ax4.plot(w_vec, harm_filter)
                    ax4.set_title('Step %d' % (harm_ind + 2))

        self.value = harm_filter

    def get_parms(self):
        basic_parms = super(HarmonicPassFilter, self).get_parms()
        prefix = 'harmonic_pass_'
        this_parms = {prefix+'start_freq': self.first_freq, prefix+'band_width': self.band_width,
                      prefix+'bands': self.num_harm}
        this_parms.update(basic_parms)
        return this_parms

    # def remove_noise_harmonics(F_AI_vec,samp_rate,noise_combs):
    #     """
    #     Removes specified noise frequencies from the signal
    #
    #     Parameters
    #     ---------------------
    #     * F_AI_vec -- matrix (chan x pts) already FFT + FFT shifted
    #
    #     * sampRate -- sampling rate
    #
    #     * freqs -- 1D array of target frequencies
    #
    #     * freqWidths -- 1D array of frequency windows that correspond to freqs that should be set to 0
    #
    #     * Note: sampRate, freqs, freqWidths have same units - eg MHz
    #     """
    #     numpts = size(F_AI_vec,2);
    #     freqs = []; freqWidths = [];
    #     for k1 = 1:length(noiseCombs):
    #         stFreq = noiseCombs(k1).first_harm_freq;
    #         nBands = noiseCombs(k1).num_harmonics;
    #         if nBands < 0
    #             nBands =
    #         end
    #     end
    #     F_AI_vec = removeNoiseFreqs(F_AI_vec,sampRate,freqs,freqWidths);
