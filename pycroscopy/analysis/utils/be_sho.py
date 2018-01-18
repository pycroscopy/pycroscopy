# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:35:57 2015

@author: Stephen Jesse, Anton Ievlev, Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from numpy import exp, abs, sqrt, sum, real, imag, arctan2, append


def SHOfunc(parms, w_vec):
    """
    Generates the SHO response over the given frequency band

    Parameters
    -----------
    parms : list or tuple
        SHO parae=(A,w0,Q,phi)
    w_vec : 1D numpy array
        Vector of frequency values
    """
    return parms[0] * exp(1j * parms[3]) * parms[1] ** 2 / \
        (w_vec ** 2 - 1j * w_vec * parms[1] / parms[2] - parms[1] ** 2)


def SHOestimateGuess(resp_vec, w_vec, num_points=5):
    """
    Generates good initial guesses for fitting

    Parameters
    ------------
    w_vec : 1D numpy array or list
        Vector of BE frequencies
    resp_vec : 1D complex numpy array or list
        BE response vector as a function of frequency
    num_points : (Optional) unsigned int
        Quality factor of the SHO peak

    Returns
    ---------
    retval : tuple
        SHO fit parameters arranged as amplitude, frequency, quality factor, phase
    """

    ii = np.argsort(abs(resp_vec))[::-1]

    a_mat = np.array([])
    e_vec = np.array([])

    for c1 in range(num_points):
        for c2 in range(c1 + 1, num_points):
            w1 = w_vec[ii[c1]]
            w2 = w_vec[ii[c2]]
            X1 = real(resp_vec[ii[c1]])
            X2 = real(resp_vec[ii[c2]])
            Y1 = imag(resp_vec[ii[c1]])
            Y2 = imag(resp_vec[ii[c2]])

            denom = (w1 * (X1 ** 2 - X1 * X2 + Y1 * (Y1 - Y2)) + w2 * (-X1 * X2 + X2 ** 2 - Y1 * Y2 + Y2 ** 2))
            if denom > 0:
                a = ((w1 ** 2 - w2 ** 2) * (w1 * X2 * (X1 ** 2 + Y1 ** 2) - w2 * X1 * (X2 ** 2 + Y2 ** 2))) / denom
                b = ((w1 ** 2 - w2 ** 2) * (w1 * Y2 * (X1 ** 2 + Y1 ** 2) - w2 * Y1 * (X2 ** 2 + Y2 ** 2))) / denom
                c = ((w1 ** 2 - w2 ** 2) * (X2 * Y1 - X1 * Y2)) / denom
                d = (w1 ** 3 * (X1 ** 2 + Y1 ** 2) -
                     w1 ** 2 * w2 * (X1 * X2 + Y1 * Y2) -
                     w1 * w2 ** 2 * (X1 * X2 + Y1 * Y2) +
                     w2 ** 3 * (X2 ** 2 + Y2 ** 2)) / denom

                if d > 0:
                    a_mat = append(a_mat, [a, b, c, d])

                    A_fit = abs(a + 1j * b) / d
                    w0_fit = sqrt(d)
                    Q_fit = -sqrt(d) / c
                    phi_fit = arctan2(-b, -a)

                    H_fit = A_fit * w0_fit ** 2 * exp(1j * phi_fit) / (
                        w_vec ** 2 - 1j * w_vec * w0_fit / Q_fit - w0_fit ** 2)

                    e_vec = append(e_vec,
                                   sum((real(H_fit) - real(resp_vec)) ** 2) +
                                   sum((imag(H_fit) - imag(resp_vec)) ** 2))
    if a_mat.size > 0:
        a_mat = a_mat.reshape(-1, 4)

        weight_vec = (1 / e_vec) ** 4
        w_sum = sum(weight_vec)

        a_w = sum(weight_vec * a_mat[:, 0]) / w_sum
        b_w = sum(weight_vec * a_mat[:, 1]) / w_sum
        c_w = sum(weight_vec * a_mat[:, 2]) / w_sum
        d_w = sum(weight_vec * a_mat[:, 3]) / w_sum

        A_fit = abs(a_w + 1j * b_w) / d_w
        w0_fit = sqrt(d_w)
        Q_fit = -sqrt(d_w) / c_w
        phi_fit = np.arctan2(-b_w, -a_w)

        H_fit = A_fit * w0_fit ** 2 * exp(1j * phi_fit) / (w_vec ** 2 - 1j * w_vec * w0_fit / Q_fit - w0_fit ** 2)

        if np.std(abs(resp_vec)) / np.std(abs(resp_vec - H_fit)) < 1.2 or w0_fit < np.min(w_vec) or w0_fit > np.max(
                w_vec):
            p0 = SHOfastGuess(w_vec, resp_vec)
        else:
            p0 = np.array([A_fit, w0_fit, Q_fit, phi_fit])
    else:
        p0 = SHOfastGuess(w_vec, resp_vec)

    return p0


def SHOfastGuess(w_vec, resp_vec, qual_factor=200):
    """
    Default SHO guess from the maximum value of the response

    Parameters
    ------------
    w_vec : 1D numpy array or list
        Vector of BE frequencies
    resp_vec : 1D complex numpy array or list
        BE response vector as a function of frequency
    qual_factor : float
        Quality factor of the SHO peak

    Returns
    -------
    retval : 1D numpy array
        SHO fit parameters arranged as [amplitude, frequency, quality factor, phase]
    """
    amp_vec = abs(resp_vec)
    i_max = int(len(resp_vec) / 2)
    return np.array([np.mean(amp_vec) / qual_factor, w_vec[i_max], qual_factor, np.angle(resp_vec[i_max])])


def SHOlowerBound(w_vec):
    """
    Provides the lower bound for the SHO fitting function

    Parameters
    ----------
    w_vec : 1D numpy array or list
        Vector of BE frequencies

    Returns
    -------
    retval : tuple
        SHO fit parameters arranged as amplitude, frequency, quality factor, phase
    """
    return 0, np.min(w_vec), -1e5, -np.pi


def SHOupperBound(w_vec):
    """
    Provides the upper bound for the SHO fitting function

    Parameters
    ----------
    w_vec: 1D numpy array or list
        Vector of BE frequencies

    Returns
    -------
    retval : tuple
        SHO fit parameters arranged as amplitude, frequency, quality factor, phase
    """
    return 1e5, np.max(w_vec), 1e5, np.pi
