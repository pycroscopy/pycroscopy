"""
Created on 10/5/16 3:44 PM
@author: Numan Laanait -- nlaanait@gmail.com
"""

from warnings import warn

import numpy as np
from scipy.signal import find_peaks_cwt


class GuessMethods(object):
    """
    This is a container class for the different strategies used to find guesses by which an optimization routine
    is initialized.
    To implement a new guess generation strategy, add it following exactly how it's done below.
    """
    def __init__(self):
        self.methods = ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']

    @classmethod
    def wavelet_peaks(self, *args, **kwargs):
        """
        This is a wrapper over scipy.signal.find_peaks_cwt() that finds peaks in the data using wavelet convolution.

        Parameters
        ----------
        args: numpy arrays.
            The first argument is the data vector.
            The second argument are the peak width bounds.

        kwargs: Passed to find_peaks_cwt().

        Returns
        -------
        wpeaks: callable function.

        """
        # vector = args
        try:
            peakwidth_bounds = kwargs.get('peak_widths')
            kwargs.pop('peak_widths')
            peak_width_step = kwargs.get('peak_step', 20)
            kwargs.pop('peak_step')
            wavelet_widths = np.linspace(peakwidth_bounds[0],peakwidth_bounds[1],peak_width_step)
            def wpeaks(vector):
                peakIndices = find_peaks_cwt(np.abs(vector), wavelet_widths,**kwargs)
                return peakIndices
            return wpeaks
        except KeyError:
            warn('Error: Please specify "peak_widths" kwarg to use this method')


    @classmethod
    def absolute_maximum(self, *args, **kwargs):
        """
        Finds maximum in 1d-array
        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        fastpeak: callable function
        """
        def fastpeak(vector):
            vec_max = np.argmax(vector)
            return vec_max
        return fastpeak

    @classmethod
    def relative_maximum(self, *args, **kwargs):
        pass

    @classmethod
    def gaussianProcesses(self, *args, **kwargs):
        pass
