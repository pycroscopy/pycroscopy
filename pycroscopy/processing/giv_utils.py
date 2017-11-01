# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:31:55 2017

@author: Kody Law, Suhas Somnath, Rama K. Vasudevan
"""

from __future__ import division, print_function, absolute_import
from multiprocessing import Pool
from _warnings import warn
import itertools
import time as tm
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import recommendCores
from ..io.microdata import MicroDataGroup, MicroDataset
from ..io.hdf_utils import getH5DsetRefs, getAuxData, link_as_main, copyAttributes, linkRefAsAlias
from ..viz.plot_utils import set_tick_font_size


def do_bayesian_inference(i_meas, bias, freq, num_x_steps=251, gam=0.03, e=10.0, sigma=10., sigmaC=1.,
                          num_samples=2E3, show_plots=False, econ=False):
    """
    this function accepts a Voltage vector and current vector
    and returns a Bayesian inferred result for R(V) and capacitance
    Used for solving the situation I = V/R(V) + CdV/dt
    to recover R(V) and C, where C is constant.

    Parameters
    ----------
    i_meas : 1D array or list
        current values, should be in nA
    bias : 1D array or list
        voltage values
    freq : float
        frequency of applied waveform
    num_x_steps : unsigned int (Optional, Default = 251)
        Number of steps in x vector (interpolating V)
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
    econ : Boolean (Optional, Default = False)
        Whether or not extra datasets are returned. Turn this on when running on multiple datasets

    Returns
    -------
    results_dict : Dictionary
        Dictionary iterms are
        'x' : 1D float array.  Voltage vector interpolated with num_samples number of points
        'm' : Ask Kody
        'mR' : 1D float array.  Bayesian inference of the resistance. This is the one you want
        'vR' : 2D float array.  varaiance ? of inferred resistance
        'Irec' : 1D array or float.  Reconstructed current without capacitance
        'Sigma' : Ask Kody
        'cValue' : float.  Capacitance value
        'm2R' : Ask Kody
        'SI' : Ask Kody

    """
    num_samples = int(num_samples)
    num_x_steps = int(num_x_steps)
    if num_x_steps % 2 == 0:
        num_x_steps += 1  # Always keep it odd

    # Organize, set up the problem
    t_max = 1. / freq
    t = np.linspace(0, t_max, len(bias))
    dt = t[2] - t[1]
    dv = np.diff(bias) / dt
    dv = np.append(dv, dv[-1])
    max_volts = max(bias)
    # num_x_steps = int(round(2 * round(max_volts / dx, 1) + 1, 0))
    x = np.linspace(-max_volts, max_volts, num_x_steps)
    dx = x[1] - x[0]
    # M = len(x)
    num_volt_points = len(bias)

    # Build A
    A = np.zeros(shape=(num_volt_points, num_x_steps + 1))
    for j in range(num_volt_points):
        ix = int(round(np.floor((bias[j] + max_volts) / dx) + 1))
        ix = min(ix, len(x)-1)
        ix = max(ix, 1)
        A[j, ix] = bias[j] * (bias[j] - x[ix - 1]) / (x[ix] - x[ix - 1])
        A[j, ix - 1] = bias[j] * (1. - (bias[j] - x[ix - 1]) / (x[ix] - x[ix - 1]))

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
    m = np.dot(Sigma, (np.dot(A.T, np.dot(O, i_meas)) + np.dot(P0, m0)))

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

    if econ:
        results_dict = {'x': x, 'mR': mR, 'vR': np.diag(vR), 'Irec': Irec, 'cValue': cValue}
    else:
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
        plt.plot(bias, i_meas)
        plt.plot(x, x / mR)
        plt.xlabel('Voltage')
        plt.ylabel('Current')
        plt.legend(('measured current', 'reconstructed I (no C)'), loc='best')

        plt.figure(103)
        plt.plot(bias, Irec)
        plt.plot(bias, i_meas)
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


def bayesian_inference_on_period(i_meas, excit_wfm, ex_freq, r_extra=220, num_x_steps=500, show_plots=False,
                                 r_max=None, **kwargs):
    """
    Performs Bayesian Inference on a single I-V curve.
    The excitation waveform must be a single period of a sine wave.
    This algorithm splits the curve into the forward and reverse sections, performs inference on each of the sections,
    stitches the results back again, and corrects the resistance which is not handled in the main bayesian function.
    Parameters
    ----------
    i_meas : array-like
        Current corresponding to a single period of sinusoidal excitation bias
    excit_wfm : array-like
        Single period of the sinusoidal excitation waveform
    ex_freq : float
        Frequency of the excitation waveform
    r_extra : float (Optional, default = 220 [Ohms])
        Extra resistance in the RC circuit that will provide correct current and resistance values
    num_x_steps : uint (Optional, default = 500)
        Number of steps for the inferred results. Note: this may be different from what is specified.
    show_plots : Boolean (Optional, Default = False)
        Whether or not to show plots
    r_max : float (Optional, Default = None)
        Maximum limit of the resistance plots.
    kwargs : dict
        Other parameters that will be passed on to the do_bayesian_inference function
    Returns
    -------
    capacitance : array-like - 2 elements
        Capacitance on the forward and reverse sections
    triangular_bias : array-like
        Interpolated bias from bayesian inference of length num_x_steps
    resistance : array-like
        Resistance of sample infered by Bayesian Inference of length num_x_steps
    variance : array-like
        Variance of the inferred resistance of length num_x_steps
    i_corr_sine : array-like
        Measured current with the capacitance correctly subtracted.
    """
    roll_val = -0.25
    num_v_steps = excit_wfm.size
    cos_omega_t = np.roll(excit_wfm, int(num_v_steps * roll_val))
    y_val = np.roll(i_meas, int(num_v_steps * roll_val))
    half_x_steps = num_x_steps // 2
    forw_results = do_bayesian_inference(y_val[:int(0.5 * num_v_steps)], cos_omega_t[:int(0.5 * num_v_steps)],
                                         ex_freq, num_x_steps=half_x_steps,
                                         econ=True, show_plots=False, **kwargs)
    rev_results = do_bayesian_inference(y_val[int(0.5 * num_v_steps):], cos_omega_t[int(0.5 * num_v_steps):],
                                        ex_freq, num_x_steps=half_x_steps,
                                        econ=True, show_plots=False, **kwargs)
    # putting the split inference together:
    full_results = dict()
    for item in ['Irec', 'cValue']:
        full_results[item] = np.hstack((forw_results[item], rev_results[item]))
        # print(item, full_results[item].shape)

    # Rolling back Irec - (this is the only one in cosine):
    # A wasteful quantity since it is a duplicate of i_meas
    # full_results['Irec'] = np.roll(full_results['Irec'], int(num_v_steps * roll_val * -1))

    # Capacitance is always doubled - halve it now:
    full_results['cValue'] *= 0.5
    cap_val = np.mean(full_results['cValue'])

    # Compensating the resistance..
    omega = 2 * np.pi * ex_freq
    i_meas = i_meas  # from nA to A
    i_cap = cap_val * omega * cos_omega_t  # from nF to F
    i_extra = r_extra * cap_val * excit_wfm  # from nF to F, ohms to ohms (1)
    i_corr_sine = i_meas - i_cap - i_extra

    # It is a lot simpler to correct the bayesian current,
    # since the capacitance contribution has already been removed, and
    # i_extra when plotted against bias is just a straight line!
    """i_extra = r_extra * cap_val * forw_results['x']
    i_forw = (forw_results['x']/forw_results['mR']) - i_extra
    i_rev = (rev_results['x']/rev_results['mR']) - i_extra"""

    # Now also correct the inferred resistance
    # old_resistance = np.hstack((forw_results['mR'], np.flipud(rev_results['mR'])))
    old_r_forw = forw_results['mR']
    old_r_rev = rev_results['mR']
    forw_results['mR'] = forw_results['mR'] / (1 - (forw_results['mR'] * r_extra * cap_val))
    rev_results['mR'] = rev_results['mR'] / (1 - (rev_results['mR'] * r_extra * cap_val))

    # by default Bayesian inference will sort bias in ascending order
    for item in ['x', 'mR', 'vR']:
        full_results[item] = np.hstack((forw_results[item], np.flipud(rev_results[item])))
        # print(item, full_results[item].shape)

    # Plot to make sure things are indeed correct:
    if show_plots:
        fig, axis = plt.subplots(figsize=(8, 8))
        axis.plot(excit_wfm, i_meas, color='green', label='Meas')
        axis.plot(excit_wfm, i_corr_sine, color='k', label='Sine corr')  # should not be able to see this.
        axis.plot(excit_wfm, i_extra, '--', color='grey', label='I extra')
        # axis.plot(full_results['x'], full_results['x'] / old_resistance, label='Bayes orig')
        # axis.plot(full_results['x'], full_results['x'] / full_results['mR'], label='Bayes corr')
        axis.plot(forw_results['x'], forw_results['x'] / old_r_forw, '--', color='cyan', label='Bayes orig F')
        axis.plot(rev_results['x'], rev_results['x'] / old_r_rev, '--', color='orange', label='Bayes orig R')
        axis.plot(forw_results['x'], forw_results['x'] / forw_results['mR'], color='blue', label='Bayes corr F')
        axis.plot(rev_results['x'], rev_results['x'] / rev_results['mR'], color='red', label='Bayes corr R')
        axis.set_xlabel('Bias (V)')
        axis.set_ylabel('Current')
        axis.legend()
        axis.axhline(y=0, xmin=np.min(excit_wfm), xmax=np.max(excit_wfm), ls=':')
        fig.tight_layout()

        def _plot_resistance(axis, bias_triang, res_vec, variance_vec, forward=True):
            st_dev = np.sqrt(variance_vec)
            good_pts = np.where(st_dev < 10)[0]
            good_pts = good_pts[np.where(good_pts < old_r_forw.size)[0]]
            pos_limits = res_vec + st_dev
            neg_limits = res_vec - st_dev
            if forward:
                cols_set = ['blue', 'cyan']
            else:
                cols_set = ['red', 'orange']

            axis.plot(bias_triang[good_pts], res_vec[good_pts], color=cols_set[0], label='R(V)')
            axis.fill_between(bias_triang[good_pts], pos_limits[good_pts], neg_limits[good_pts],
                              alpha=0.25, color=cols_set[1], label='R(V)+-$\sigma$')

        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        for axis, res_vec, variance_vec, name, direction in zip(axes.flat,
                                                                [old_r_forw, old_r_rev, forw_results['mR'],
                                                                 rev_results['mR']],
                                                                [forw_results['vR'], rev_results['vR'],
                                                                 forw_results['vR'], rev_results['vR']],
                                                                ['Forw Orig', 'Rev Orig', 'Forw Corr', 'Rev Corr'],
                                                                [True, False, True, False]):
            _plot_resistance(axis, forw_results['x'], res_vec, variance_vec, forward=direction)
            y_lims = axis.get_ylim()
            if r_max is not None:
                axis.set_ylim([0, min(r_max, y_lims[1])])
            else:
                axis.set_ylim([0, y_lims[1]])
            axis.set_title(name)
            axis.set_xlabel('Bias (V)')
            axis.set_ylabel('Resistance')
            axis.legend()
        fig.tight_layout()

    return full_results['cValue'], full_results['x'], full_results['mR'], full_results['vR'], i_corr_sine


def plot_bayesian_spot_from_h5(h5_bayesian_grp, h5_resh, pix_ind, r_extra_override=None, **kwargs):
    """
    Plots the basic Bayesian Inference results for a specific pixel

    Parameters
    ----------
    h5_bayesian_grp : h5py.Datagroup reference
        Group containing the Bayesian Inference results
    h5_resh : h5py.Dataset reference
        Dataset containing the raw / filtered measured current split by pixel
    pix_ind : unsigned int
        Integer index of the desired pixel
    r_extra_override : float, Optional
        Default - will not override

    Returns
    -------
    fig : matplotlib.pyplot figure handle
        Handle to figure
    """
    bias_interp = np.squeeze(h5_bayesian_grp['Spectroscopic_Values'][()])
    h5_mr = h5_bayesian_grp['mr']
    h5_vr = h5_bayesian_grp['vr']
    h5_irec = h5_bayesian_grp['irec']
    h5_cap = h5_bayesian_grp['capacitance']
    freq = h5_bayesian_grp.attrs['freq']
    try:
        r_extra = h5_bayesian_grp['Rextra']  # should be 220 Ohms according to calibration
    except KeyError:
        # Old / incorrect inference model
        r_extra = 0
    if r_extra_override is not None:
        r_extra = 220

    possibly_rolled_bias = getAuxData(h5_irec, auxDataName=['Spectroscopic_Values'])[0]
    split_directions = h5_bayesian_grp.attrs['split_directions']

    i_meas = np.squeeze(h5_resh[pix_ind])
    orig_bias = np.squeeze(getAuxData(h5_resh, auxDataName=['Spectroscopic_Values'])[0])
    h5_pos = getAuxData(h5_resh, auxDataName=['Position_Indices'])[0]

    mr_vec = h5_mr[pix_ind]
    i_recon = h5_irec[pix_ind]
    vr_vec = h5_vr[pix_ind]
    cap_val = h5_cap[pix_ind]

    return plot_bayesian_results(orig_bias, possibly_rolled_bias, i_meas, bias_interp, mr_vec, i_recon, vr_vec,
                                 split_directions, cap_val, freq, r_extra, pix_pos=h5_pos[pix_ind], **kwargs)


def plot_bayesian_results(orig_bias, possibly_rolled_bias, i_meas, bias_interp, mr_vec, i_recon, vr_vec,
                          split_directions, cap_val, freq, r_extra, pix_pos=[0, 0], broken_resistance=True, **kwargs):
    """
    Plots the basic Bayesian Inference results for a specific pixel

    Parameters
    ----------
    orig_bias : 1D float numpy array
        Original bias vector used for experiment
    possibly_rolled_bias : 1D float numpy array
        Bias vector used for Bayesian inference
    i_meas : 1D float numpy array
        Current measured from experiment
    bias_interp : 1D float numpy array
        Interpolated bias
    mr_vec : 1D float numpy array
        Inferred resistance
    i_recon : 1D float numpy array
        Reconstructed current
    vr_vec : 1D float numpy array
        Variance of the resistance
    split_directions : Boolean
        Whether or not to compute the forward and reverse portions of the loop separately
    cap_val : float
        Inferred capacitance in nF
    freq : float
        Excitation frequency
    r_extra : float
        Resistance of extra resistor [Ohms] necessary to get correct resistance values
    pix_pos : list of two numbers
        Pixel row and column positions or values
    broken_resistance : bool, Optional
        Whether or not to break the resistance plots into sections so as to avoid plotting areas with high variance

    Returns
    -------
    fig : matplotlib.pyplot figure handle
        Handle to figure
    """

    font_size_1 = 14
    font_size_2 = 16

    half_x_ind = int(0.5 * bias_interp.size)

    ex_amp = np.max(bias_interp)

    colors = [['red', 'orange'], ['blue', 'cyan']]
    syms = [['-', '--', '--'], ['-', ':', ':']]
    names = ['Forward', 'Reverse']
    if not split_directions:
        colors = colors[0]
        syms = syms[0]
        names = ['']

    # Need to calculate the correct resistance in the original domain:
    omega = 2 * np.pi * freq
    cos_omega_t = np.roll(orig_bias, int(-0.25 * orig_bias.size))
    mean_cap_val = 0.5 * np.mean(cap_val)  # Not sure why we need the 0.5
    i_cap = mean_cap_val * omega * cos_omega_t  # * nF -> nA
    i_extra = r_extra * mean_cap_val * orig_bias  # ohms * nF * V -> nA
    i_correct = i_meas - i_cap - i_extra
    i_correct_rolled = np.roll(i_correct, int(-0.25 * orig_bias.size))
    orig_half_pt = int(0.5 * orig_bias.size)

    st_dev = np.sqrt(vr_vec)
    good_pts = np.where(st_dev < 10)[0]
    good_forw = good_pts[np.where(good_pts < half_x_ind)[0]]
    good_rev = good_pts[np.where(good_pts >= half_x_ind)[0]]
    pos_limits = mr_vec + st_dev
    neg_limits = mr_vec - st_dev

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    # fig.subplots_adjust(wspace=3.5)

    axes[0].set_ylabel('Resistance (G$\Omega$)', fontsize=font_size_2)

    if split_directions:
        pts_to_plot = [good_forw, good_rev]
    else:
        pts_to_plot = [good_pts]

    for type_ind, axis, pts_list, cols_set, sym_set, set_name in zip(range(len(names)),
                                                                     axes[:2], pts_to_plot,
                                                                     colors, syms, names):
        axis.set_title('$R(V)$ ' + set_name + ' at Row = ' + str(pix_pos[0]) +
                       ' Col =' + str(pix_pos[1]), fontsize=font_size_2)

        single_plot = not broken_resistance
        if broken_resistance:
            diff = np.diff(pts_list)
            jump_inds = np.argwhere(diff > 4) + 1
            if jump_inds.size < 1:
                single_plot = True

        if not single_plot:
            jump_inds = np.append(np.append(0, jump_inds), pts_list[-1])
            for ind in range(1, jump_inds.size):
                cur_range = pts_list[jump_inds[ind - 1]:jump_inds[ind]]
                axis.plot(bias_interp[cur_range], mr_vec[cur_range], cols_set[0],
                          linestyle=sym_set[0], linewidth=3)
                axis.fill_between(bias_interp[cur_range], pos_limits[cur_range], neg_limits[cur_range],
                                  alpha=0.25, color=cols_set[1])
                if ind == 1:
                    axis.legend(['R(V)', 'R(V)+-$\sigma$'], loc='upper left', fontsize=font_size_1)
        else:
            axis.plot(bias_interp[pts_list], mr_vec[pts_list], cols_set[0],
                      linestyle=sym_set[0], linewidth=3, label='R(V)')
            axis.fill_between(bias_interp[pts_list], pos_limits[pts_list], neg_limits[pts_list],
                              alpha=0.25, color=cols_set[1], label='R(V)+-$\sigma$')
            axis.legend(loc='upper left', fontsize=font_size_1)
        axis.set_xlabel('Voltage (V)', fontsize=font_size_2)

        axis.set_xlim((-ex_amp, ex_amp))

    # ################### CURRENT PLOT ##########################

    axes[2].plot(orig_bias, i_meas, 'b', linewidth=2, label='I$_{meas}$')
    axes[2].plot(possibly_rolled_bias, i_recon, 'cyan', linewidth=2, linestyle='--', label='I$_{rec}$')

    if split_directions:
        axes[2].plot(cos_omega_t[:orig_half_pt], i_correct_rolled[:orig_half_pt],
                     'r', linewidth=3, label='I$_{Bayes} Forw$')
        axes[2].plot(cos_omega_t[orig_half_pt:], i_correct_rolled[orig_half_pt:],
                     'orange', linewidth=3, label='I$_{Bayes} Rev$')
    else:
        axes[1].plot(orig_bias, i_correct, 'g', label='I$_{Bayes}$')

        # axes[2].legend(loc='upper right', bbox_to_anchor=(-.1, 0.30), fontsize=font_size_1)
    axes[2].legend(loc='best', fontsize=font_size_1)
    axes[2].set_xlabel('Voltage(V)', fontsize=font_size_2)
    axes[2].set_title('$I(V)$ at row ' + str(pix_pos[0]) + ', col ' + str(pix_pos[1]),
                      fontsize=font_size_2)

    axes[2].set_ylabel('Current (nA)', fontsize=font_size_2)

    set_tick_font_size(axes, font_size_1)

    fig.tight_layout()

    return fig