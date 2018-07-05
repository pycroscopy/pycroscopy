# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:31:55 2017

@author: Suhas Somnath, Kody Law, Rama K. Vasudevan
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

from pyUSID.io.hdf_utils import get_auxiliary_datasets
from pyUSID.viz.plot_utils import set_tick_font_size


def do_bayesian_inference(i_meas, bias, freq, num_x_steps=251, r_extra=110, gam=0.03, e=10.0, sigma=10., sigmaC=1.,
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
    r_extra : float (Optional, default = 220 [Ohms])
        Extra resistance in the RC circuit that will provide correct current and resistance values
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
    Written by Kody J. Law (Matlab) and translated to Python by Rama K. Vasudevan
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
        ix = min(ix, len(x) - 1)
        ix = max(ix, 1)
        A[j, ix] = bias[j] * (bias[j] - x[ix - 1]) / (x[ix] - x[ix - 1])
        A[j, ix - 1] = bias[j] * (1. - (bias[j] - x[ix - 1]) / (x[ix] - x[ix - 1]))

    A[:, num_x_steps] = dv + r_extra * bias

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


def bayesian_inference_on_period(i_meas, excit_wfm, ex_freq, r_extra=110, num_x_steps=500, show_plots=False,
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
    results : dictionary
        Dictionary iterms are
        'cValue' : array-like - 2 elements
            Capacitance on the forward and reverse sections
        'x' : array-like
            Interpolated bias from bayesian inference of length num_x_steps
        'mR' : array-like
            Resistance of sample infered by Bayesian Inference of length num_x_steps
        'vR' : array-like
            Variance of the inferred resistance of length num_x_steps
        'IcorrSine' : array-like
            Measured current with the capacitance correctly subtracted.
        'Irec' : array-like
            Current reconstructed via Bayesian Inference
    """
    roll_val = -0.25
    num_v_steps = excit_wfm.size
    cos_omega_t = np.roll(excit_wfm, int(num_v_steps * roll_val))
    y_val = np.roll(i_meas, int(num_v_steps * roll_val))
    half_x_steps = num_x_steps // 2
    rev_results = do_bayesian_inference(y_val[:int(0.5 * num_v_steps)] * -1,
                                        cos_omega_t[:int(0.5 * num_v_steps)] * -1,
                                        ex_freq, num_x_steps=half_x_steps,
                                        econ=True, show_plots=False, r_extra=r_extra, **kwargs)
    forw_results = do_bayesian_inference(y_val[int(0.5 * num_v_steps):], cos_omega_t[int(0.5 * num_v_steps):],
                                         ex_freq, num_x_steps=half_x_steps,
                                         econ=True, show_plots=False, r_extra=r_extra, **kwargs)

    # putting the split inference together:
    full_results = dict()
    for item in ['cValue']:
        full_results[item] = np.hstack((forw_results[item], rev_results[item]))

    # Capacitance is always doubled - halve it now:
    full_results['cValue'] *= 0.5
    cap_val = np.mean(full_results['cValue'])

    # Compensating the resistance..
    # omega = 2 * np.pi * ex_freq
    """t_max = 1. / ex_freq
    t = np.linspace(0, t_max, len(excit_wfm))
    dt = t[2] - t[1]"""
    # dt = period time / points per period
    dt = 1 / (ex_freq * excit_wfm.size)
    dv = np.diff(excit_wfm) / dt
    dv = np.append(dv, dv[-1])
    i_cap = cap_val * dv
    i_extra = r_extra * 2 * cap_val * excit_wfm
    i_corr_sine = i_meas - i_cap - i_extra
    full_results['IcorrSine'] = i_corr_sine

    # by default Bayesian inference will sort bias in ascending order
    rev_results['x'] *= -1
    rev_results['Irec'] *= -1

    for item in ['x', 'mR', 'vR', 'Irec']:
        full_results[item] = np.hstack((forw_results[item], rev_results[item]))
        # print(item, full_results[item].shape)

    full_results['Irec'] = np.roll(full_results['Irec'], int(num_v_steps * roll_val))

    # Plot to make sure things are indeed correct:
    if show_plots:
        fig, axis = plt.subplots(figsize=(8, 8))
        axis.plot(excit_wfm, i_meas, color='green', label='Meas')
        axis.plot(excit_wfm, i_corr_sine, color='k', label='Sine corr')  # should not be able to see this.
        axis.plot(excit_wfm, i_extra, '--', color='grey', label='I extra')
        axis.plot(excit_wfm, full_results['Irec'], '--', color='orange', label='I rec')
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
            good_pts = good_pts[np.where(good_pts < forw_results['x'].size)[0]]
            pos_limits = res_vec + st_dev
            neg_limits = res_vec - st_dev
            if forward:
                cols_set = ['blue', 'cyan']
            else:
                cols_set = ['red', 'orange']

            axis.plot(bias_triang[good_pts], res_vec[good_pts], color=cols_set[0], label='R(V)')
            axis.fill_between(bias_triang[good_pts], pos_limits[good_pts], neg_limits[good_pts],
                              alpha=0.25, color=cols_set[1], label='R(V)+-$\sigma$')

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        for axis, res_vec, variance_vec, name, direction in zip(axes.flat,
                                                                [forw_results['mR'], rev_results['mR']],
                                                                [forw_results['vR'], rev_results['vR']],
                                                                ['Forw', 'Rev'], [True, False, ]):
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

    return full_results


def plot_bayesian_spot_from_h5(h5_bayesian_grp, h5_resh, pix_ind, **kwargs):
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
    Returns
    -------
    fig : matplotlib.pyplot figure handle
        Handle to figure
    """
    bias_triang = np.squeeze(h5_bayesian_grp['Spectroscopic_Values'][()])
    h5_resistance = h5_bayesian_grp['Resistance']
    h5_r_variance = h5_bayesian_grp['R_variance']
    h5_i_corrected = h5_bayesian_grp['Corrected_Current']

    i_meas = np.squeeze(h5_resh[pix_ind])
    orig_bias = np.squeeze(get_auxiliary_datasets(h5_resh, aux_dset_name=['Spectroscopic_Values'])[0])
    h5_pos = get_auxiliary_datasets(h5_resh, aux_dset_name=['Position_Indices'])[0]

    resistance = h5_resistance[pix_ind]
    i_correct = h5_i_corrected[pix_ind]
    r_variance = h5_r_variance[pix_ind]

    return plot_bayesian_results(orig_bias, i_meas, i_correct, bias_triang, resistance, r_variance,
                                 pix_pos=h5_pos[pix_ind], **kwargs)


def plot_bayesian_results(bias_sine, i_meas, i_corrected, bias_triang, resistance, r_variance, i_recon=None,
                          pix_pos=[0, 0], broken_resistance=True, r_max=None, res_scatter=False, **kwargs):
    """
    Plots the basic Bayesian Inference results for a specific pixel
    Parameters
    ----------
    bias_sine : 1D float numpy array
        Original bias vector used for experiment
    i_meas : 1D float numpy array
        Current measured from experiment
    i_corrected : 1D float numpy array
        current with capacitance and R extra compensated
    i_recon : 1D float numpy array
        Reconstructed current
    bias_triang : 1D float numpy array
        Interpolated bias
    resistance : 1D float numpy array
        Inferred resistance
    r_variance : 1D float numpy array
        Variance of the resistance
    pix_pos : list of two numbers
        Pixel row and column positions or values
    broken_resistance : bool, Optional
        Whether or not to break the resistance plots into sections so as to avoid plotting areas with high variance
    r_max : float, Optional
        Maximum value of resistance to plot
    res_scatter : bool, Optional
        Use scatter instead of line plots for resistance
    Returns
    -------
    fig : matplotlib.pyplot figure handle
        Handle to figure
    """

    font_size_1 = 14
    font_size_2 = 16

    half_x_ind = int(0.5 * bias_triang.size)

    ex_amp = np.max(bias_triang)

    colors = [['red', 'orange'], ['blue', 'cyan']]
    syms = [['-', '--', '--'], ['-', ':', ':']]
    names = ['Forward', 'Reverse']

    cos_omega_t = np.roll(bias_sine, int(-0.25 * bias_sine.size))
    orig_half_pt = int(0.5 * bias_sine.size)
    i_correct_rolled = np.roll(i_corrected, int(-0.25 * bias_sine.size))

    st_dev = np.sqrt(r_variance)
    tests = [st_dev < 10, resistance > 0]
    if r_max is not None:
        tests.append(resistance < r_max)
    good_pts = np.ones(resistance.shape, dtype=bool)
    for item in tests:
        good_pts = np.logical_and(good_pts, item)
    good_pts = np.where(good_pts)[0]
    good_forw = good_pts[np.where(good_pts < half_x_ind)[0]]
    good_rev = good_pts[np.where(good_pts >= half_x_ind)[0]]
    pos_limits = resistance + st_dev
    neg_limits = resistance - st_dev

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    # fig.subplots_adjust(wspace=3.5)

    axes[0].set_ylabel('Resistance (G$\Omega$)', fontsize=font_size_2)

    pts_to_plot = [good_forw, good_rev]

    for type_ind, axis, pts_list, cols_set, sym_set, set_name in zip(range(len(names)),
                                                                     axes[:2], pts_to_plot,
                                                                     colors, syms, names):
        axis.set_title('$R(V)$ ' + set_name + ' at Row = ' + str(pix_pos[1]) +
                       ' Col =' + str(pix_pos[0]), fontsize=font_size_2)

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
                if res_scatter:
                    axis.scatter(bias_triang[cur_range], resistance[cur_range],
                                 color=cols_set[0], s=30)
                else:
                    axis.plot(bias_triang[cur_range], resistance[cur_range], cols_set[0],
                              linestyle=sym_set[0], linewidth=3)
                axis.fill_between(bias_triang[cur_range], pos_limits[cur_range], neg_limits[cur_range],
                                  alpha=0.25, color=cols_set[1])
                if ind == 1:
                    axis.legend(['R(V)', 'R(V)+-$\sigma$'], loc='upper center', fontsize=font_size_1)
        else:
            if res_scatter:
                axis.scatter(bias_triang[pts_list], resistance[pts_list],
                             color=cols_set[0], s=30)
            else:
                axis.plot(bias_triang[pts_list], resistance[pts_list], cols_set[0],
                          linestyle=sym_set[0], linewidth=3, label='R(V)')
            axis.fill_between(bias_triang[pts_list], pos_limits[pts_list], neg_limits[pts_list],
                              alpha=0.25, color=cols_set[1], label='R(V)+-$\sigma$')
            axis.legend(loc='upper center', fontsize=font_size_1)
        axis.set_xlabel('Voltage (V)', fontsize=font_size_2)

        axis.set_xlim((-ex_amp, ex_amp))

    # ################### CURRENT PLOT ##########################

    axes[2].plot(bias_sine, i_meas, 'green', linewidth=3, label='I$_{meas}$')
    if i_recon is not None:
        axes[2].plot(bias_sine, i_recon, 'c--', linewidth=3, label='I$_{recon}$')
    axes[2].plot(cos_omega_t[orig_half_pt:], i_correct_rolled[orig_half_pt:],
                 'blue', linewidth=3, label='I$_{Bayes} Forw$')
    axes[2].plot(cos_omega_t[:orig_half_pt], i_correct_rolled[:orig_half_pt],
                 'red', linewidth=3, label='I$_{Bayes} Rev$')

    # axes[2].legend(loc='upper right', bbox_to_anchor=(-.1, 0.30), fontsize=font_size_1)
    axes[2].legend(loc='best', fontsize=font_size_1)
    axes[2].set_xlabel('Voltage(V)', fontsize=font_size_2)
    axes[2].set_title('$I(V)$ at row ' + str(pix_pos[0]) + ', col ' + str(pix_pos[1]),
                      fontsize=font_size_2)

    axes[2].set_ylabel('Current (nA)', fontsize=font_size_2)

    set_tick_font_size(axes, font_size_1)

    fig.tight_layout()

    return fig
