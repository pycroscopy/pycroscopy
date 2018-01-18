"""
Created on 12/15/16 3:44 PM
@author: Numan Laanait -- nlaanait@gmail.com
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from .utils.be_loop import loop_fit_function


class Fit_Methods(object):
    """
    This is a container class for the different objective functions used in BE fitting
    To implement a new guess generation strategy, add it following exactly how it's done below.

    In essence, the guess methods here need to return a callable function that will take a feature vector as the sole
    input and return the guess parameters. The guess methods here use the keyword arguments to configure the returned
    function.
    """
    def __init__(self):
        self.methods = ['SHO']

    @staticmethod
    def SHO(guess, data_vec, freq_vector, *args):
        """
        Generates the single Harmonic Oscillator response over the given vector

        Parameters
        ----------
        guess : array-like
            The set of guess parameters to be tested
        data_vec : numpy.ndarray
            The data vector to compare the current guess against
        freq_vector : numpy.ndarray
            The frequencies that correspond to each data point in `data_vec`
        args : list or tuple
            SHO parameters=(Amp,w0,Q,phi,vector). vector: 1D np.array of frequency values.
            Amp: amplitude.
            w0: resonant frequency.
            Q: Quality Factor.
            phi: Phase.
            vector:

        Returns
        -------
        SHO_func: callable function.
        """

        if guess.size < 4:
            raise ValueError('Error: The Single Harmonic Oscillator requires 4 parameter guesses!')
        data_mean = np.mean(data_vec)

        Amp, w_0, Q, phi = guess[:4]
        func = Amp * np.exp(1.j * phi) * w_0 ** 2 / (freq_vector ** 2 - 1j * freq_vector * w_0 / Q - w_0 ** 2)

        ss_tot = sum(abs(data_vec - data_mean) ** 2)
        ss_res = sum(abs(data_vec - func) ** 2)

        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return 1-r_squared


class BE_Fit_Methods(object):
    """
    Contains fit methods that are specific to BE data.
    """
    def __init__(self):
        self.methods = ['BE_LOOP']

    @staticmethod
    def BE_LOOP(coef_vec, data_vec, dc_vec, *args):
        """

        Parameters
        ----------
        coef_vec : numpy.ndarray
        data_vec : numpy.ndarray
        dc_vec : numpy.ndarray
            The DC offset vector
        args : list

        Returns
        -------
        fitness : float
            The 1-r^2 value for the current set of loop coefficients

        """

        if coef_vec.size < 9:
            raise ValueError('Error: The Loop Fit requires 9 parameter guesses!')

        data_mean = np.mean(data_vec)

        func = loop_fit_function(dc_vec, coef_vec)

        ss_tot = sum(abs(data_vec - data_mean) ** 2)
        ss_res = sum(abs(data_vec - func) ** 2)

        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return 1 - r_squared


class forc_iv_fit_methods(Fit_Methods):
    """
    Any fitting methods specific to FORC_IV should go here.
    """
    pass
