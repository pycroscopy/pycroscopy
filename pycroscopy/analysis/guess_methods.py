"""
Created on 10/5/16 3:44 PM
@author: Numan Laanait -- nlaanait@gmail.com
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn

import numpy as np
from scipy.signal import find_peaks_cwt
from .utils.be_sho import SHOestimateGuess, SHOfunc


class GuessMethods(object):
    """
    This is a container class for the different strategies used to find guesses by which an optimization routine
    is initialized.
    To implement a new guess generation strategy, add it following exactly how it's done below.

    In essence, the guess methods here need to return a callable function that will take a feature vector as the sole
    input and return the guess parameters. The guess methods here use the keyword arguments to configure the returned
    function.
    """

    def __init__(self):
        self.methods = ['wavelet_peaks', 'relative_maximum', 'gaussian_processes', 'complex_gaussian']

    @staticmethod
    def wavelet_peaks(*args, **kwargs):
        """
        This is a wrapper over scipy.signal.find_peaks_cwt() that finds peaks in the data using wavelet convolution.

        Parameters
        ----------
        args: dictionary
            List of optional parameters for this function - not used.

        kwargs: dictionary
            Passed to find_peaks_cwt().

        Returns
        -------
        wpeaks: callable function.
        """
        try:
            peak_width_bounds = kwargs.get('peak_widths')
            kwargs.pop('peak_widths')
            peak_width_step = kwargs.get('peak_step', 20)
            kwargs.pop('peak_step')
            # The below numpy array is used to configure the returned function wpeaks
            wavelet_widths = np.linspace(peak_width_bounds[0], peak_width_bounds[1], peak_width_step)

            def wpeaks(vector):
                """
                This is the function that will be mapped by multiprocess. This is a wrapper around the scipy function.
                It uses a parameter - wavelet_widths that is configured outside this function.

                Parameters
                ----------
                vector : 1D numpy array
                    Feature vector containing peaks

                Returns
                -------
                peak_indices : list
                    List of indices of peaks within the prescribed peak widths
                """
                peak_indices = find_peaks_cwt(np.abs(vector), wavelet_widths, **kwargs)
                return peak_indices

            return wpeaks
        except KeyError:
            warn('Error: Please specify "peak_widths" kwarg to use this method')

    @staticmethod
    def absolute_maximum(*args, **kwargs):
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

    @staticmethod
    def relative_maximum(*args, **kwargs):
        """
        Not yet implemented
        
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        pass

    @staticmethod
    def gaussian_processes(*args, **kwargs):
        """
        Not yet implemented
        
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        pass

    @staticmethod
    def complex_gaussian(*args, **kwargs):
        """
        Sets up the needed parameters for the analytic approximation for the
        Gaussian fit of complex data.

        Parameters
        ----------
        args: numpy arrays.

        kwargs: Passed to SHOEstimateFit().

        Returns
        -------
        sho_guess: callable function.

        """
        try:
            w_vec = kwargs.pop('frequencies')
            # lower_bounds = kwargs.pop('lower_bounds', [0, np.min(w_vec), -1e5, -np.pi])
            # upper_bounds = kwargs.pop('upper_bounds', [1e5, np.max(w_vec), 1e5, np.pi])
            num_points = kwargs.pop('num_points', 5)

            def sho_guess(resp_vec):

                guess = SHOestimateGuess(w_vec, resp_vec, num_points)

                guess = np.hstack([guess, np.array(r_square(resp_vec, SHOfunc, guess, w_vec))])

                return guess

            return sho_guess
        except KeyError:
            warn('Error: Please specify "peak_widths" kwarg to use this method')


def r_square(data_vec, func, *args, **kwargs):
    """
    R-square for estimation of the fitting quality
    Typical result is in the range (0,1), where 1 is the best fitting

    Parameters
    ----------
    data_vec : array_like
        Measured data points
    func : callable function
        Should return a numpy.ndarray of the same shape as data_vec
    args :
        Parameters to be pased to func
    kwargs :
        Keyword parameters to be pased to func

    Returns
    -------
    r_squared : float
        The R^2 value for the current data_vec and parameters
    """
    data_mean = np.mean(data_vec)
    ss_tot = sum(abs(data_vec - data_mean) ** 2)
    ss_res = sum(abs(data_vec - func(*args, **kwargs)) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return r_squared
