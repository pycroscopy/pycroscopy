# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:31:55 2017

@author: Kody Law, Rama K. Vasudevan
"""
from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import recommendCores
from ..io.microdata import MicroDataGroup, MicroDataset
from ..io.hdf_utils import getH5DsetRefs, getAuxData
from multiprocessing import Pool
from _warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def do_bayesian_inference(V, IV_point, freq, dx=0.01, gam=0.03, e=10.0, sigma=10., sigmaC=1.,
                          num_samples=1E4, show_plots=False):
    """
    this function accepts a Voltage vector and current vector
    and returns a Bayesian inferred result for R(V) and capacitance
    Used for solving the situation I = V/R(V) + CdV/dt
    to recover R(V) and C, where C is constant.

    Parameters
    ----------
    V : 1D array or list
        voltage values
    IV_point : 1D array or list
        current values, should be in nA
    freq : float
        frequency of applied waveform
    dx : float (Optional, Default = 0.01)
        step in x vector (interpolating V)
    gam : float (Optional, Default = 0.03)
        gamma value for reconstruction
    e : float (Optional, Default = 10.0)
        Ask Kody
    sigma : float (Optional, Default = 10.0)
        Ask Kody
    sigmaC : float (Optional, Default = 1.0)
        Ask Kody
    num_samples : unsigned int (Optional, Default = 1E4)
        Number of samples. 1E+4 is more than sufficient
    show_plots : Boolean (Optional, Default = False)
        Whether or not to show plots

    Returns
    -------
    results_dict : Dictionary
        'x' : 1D array or float
            Voltage vector interpolated with num_samples number of points
        'm' : Ask Kody
        'mR' : Ask Kody
        'vR' : Ask Kody
        'Irec' : 1D array or float
            Reconstructed current without capacitance
        'Sigma' : Ask Kody
        'cValue' : float
            Capacitance value
        'm2R' : Ask Kody
        'SI' : Ask Kody
    """
    num_samples = int(num_samples)

    # Organize, set up the problem
    t_max = 1. / freq
    t = np.linspace(0, t_max, len(V))
    dt = t[2] - t[1]
    dv = np.diff(V) / dt
    dv = np.append(dv, dv[-1])
    max_volts = max(V)
    num_x_steps = int(round(2 * round(max_volts / dx, 1) + 1, 0))
    x = np.linspace(-max_volts, max_volts, num_x_steps)
    # M = len(x)
    num_volt_points = len(V)

    # Build A
    A = np.zeros(shape=(num_volt_points, num_x_steps + 1))
    for j in range(num_volt_points):
        ix = int(round(np.floor((V[j] + max_volts) / dx) + 1))
        ix = min(ix, len(x)-1)
        ix = max(ix, 1)
        A[j, ix] = V[j] * (V[j] - x[ix - 1]) / (x[ix] - x[ix - 1])
        A[j, ix - 1] = V[j] * (1. - (V[j] - x[ix - 1]) / (x[ix] - x[ix - 1]))

    A[:, num_x_steps] = dv

    # generate simulated observations
    Lapt = (-1. * np.diag((t[:-1]) ** 0, -1) - np.diag(t[:-1] ** 0, 1) + 2. * np.diag(t ** 0, 0)) / dt / dt
    Lapt[0, 0] = 1. / dt / dt
    Lapt[-1, -1] = 1. / dt / dt
    O = (1. / gam ** 2) * (np.eye(num_volt_points))
    # noise_term = np.linalg.lstsq(sqrtm(O),np.random.randn(N,1))[0]
    # y = IV_point
    #  Itrue + noise_term.ravel()

    Lap = (-1. * np.diag((x[:-1]) ** 0, -1) - np.diag(x[:-1] ** 0, 1) + 2. * np.diag(x ** 0, 0)) / dx / dx
    Lap[0, 0] = 1. / dx / dx
    Lap[-1, -1] = 1. / dx / dx

    m0 = 3. * np.ones((num_x_steps, 1))
    m0 = np.append(m0, 0)

    P0 = np.zeros(shape=(num_x_steps + 1, num_x_steps + 1))
    P0[:num_x_steps, :num_x_steps] = 1. / sigma ** 2 * (1. * np.eye(num_x_steps) + np.linalg.matrix_power(Lap, 3))
    P0[num_x_steps, num_x_steps] = 1. / sigmaC ** 2

    Sigma = np.linalg.inv(np.dot(A.T, np.dot(O, A)) + P0)
    m = np.dot(Sigma, (np.dot(A.T, np.dot(O, IV_point)) + np.dot(P0, m0)))

    # Reconstructed current
    Irec = np.dot(A, m)  # This includes the capacitance

    # Draw samples from S
    # SI = (np.matlib.repmat(m[:M], num_samples, 1).T) + np.dot(sqrtm(Sigma[:M, :M]), np.random.randn(M, num_samples))
    SI = np.tile(m[:num_x_steps], (num_samples, 1)).T + np.dot(sqrtm(Sigma[:num_x_steps, :num_x_steps]),
                                                               np.random.randn(num_x_steps, num_samples))
    # approximate mean and covariance of R
    mR = 1. / num_samples * np.sum(1. / SI, 1)
    m2R = 1. / num_samples * np.dot(1. / SI, (1. / SI).T)
    # m2R=1./num_samples*(1./SI)*(1./SI).T
    # vR=m2R-np.dot(mR,mR.T)
    vR = m2R - mR * mR.T
    cValue = m[-1]
    results_dict = {'x': x, 'm': m, 'mR': mR, 'vR': vR, 'Irec': Irec, 'Sigma': Sigma, 'cValue': cValue, 'm2R': m2R,
                    'SI': SI}

    if show_plots:
        # Do some plotting
        plt.figure(101)
        plt.plot(x, mR, 'b', linewidth=3)
        plt.plot(x, mR + np.sqrt(np.diag(vR)), 'r-', linewidth=3)
        plt.plot(x, mR - np.sqrt(np.diag(vR)), 'r-', linewidth=3)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Resistance (GOhm)')
        plt.title('R(V)')
        plt.legend(('R(V)', 'R(V)+sigma', 'R(V)-sigma'), loc='best')
        # plt.ylim((0,3))
        plt.xlim((-max_volts, max_volts))

        plt.figure(102)
        plt.plot(V, IV_point)
        plt.plot(x, x / mR)
        plt.xlabel('Voltage')
        plt.ylabel('Current')
        plt.legend(('measured current', 'reconstructed I (no C)'), loc='best')

        plt.figure(103)
        plt.plot(V, Irec)
        plt.plot(V, IV_point)
        plt.legend(('I$_{rec}$', 'I$_{true}$'), loc='best')

        plt.figure(104)
        cx = np.arange(0, 2, 0.01)
        dens_cx = 1. / np.sqrt(Sigma[num_x_steps, num_x_steps] * 2 * np.pi) * np.exp(
            -(cx - m[num_x_steps]) ** 2 / 2 / Sigma[num_x_steps, num_x_steps])
        plt.plot(cx, dens_cx)
        plt.ylabel('p(C)')
        plt.xlabel('C')

        print("The value of the capacitance is ", str(round(m[-1] * 1E3, 2)) + "pF")

    return results_dict