"""
functions that facilitate functional fitting of spectroscopic and imaging data

Created on 12/15/16 3:44 PM
@author: Numan Laanait -- nlaanait@gmail.com
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
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


def exp(x, a, k, c):
    return (a * np.exp(-(x/k))) + c

def fit_exp_curve(x, y):
    popt, _ = curve_fit(exp, x, y, maxfev=25000)
    return popt

def double_exp(x, a, k, a2, k2, c):
    return (a * np.exp(-k*x)) + (a2 * np.exp(-k2*x) + c )

def fit_double_exp(x, y):
    """
    fit spectrum to double exp using differential evolution
    """
    time_ax = x
    spectrum = y
    def cost_func_double_exp(params):
        a = params[0]; k = params[1]; a2 = params[2]; k2 = params[3]; c = params[4]
        double_exp_model = double_exp(time_ax, a, k, a2, k2, c)
        return np.sum((spectrum - double_exp_model)**2)
    popt = differential_evolution(cost_func_double_exp,
                                  bounds=([-100,100],[-100, 100],[-200,200],[-100,100],[-200,200])).x
    return popt

def str_exp(x, a, k, c):
    return a * np.exp(x ** k) + c

def fit_str_exp(x,y):
    popt, _ = curve_fit(str_exp, x, y, maxfev=25000, bounds=([-np.inf,0,-np.inf], [np.inf,1,np.inf]))
    return popt

def sigmoid(x, A, K, B, v, Q, C):
    return A + (K-A)/(C+Q*np.exp(-B*x)**(1/v))

def fit_sigmoid(x, y):
    popt, pcov = curve_fit(sigmoid, x, y, maxfev=2500)
    return popt
