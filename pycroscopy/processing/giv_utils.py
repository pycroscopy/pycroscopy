# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:31:55 2017

@author: Kody Law, Suhas Somnath, Rama K. Vasudevan
"""
from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import recommendCores
from ..io.microdata import MicroDataGroup, MicroDataset
from ..io.hdf_utils import getH5DsetRefs, getAuxData, link_as_main, copyAttributes, linkRefAsAlias
from multiprocessing import Pool
from _warnings import warn
import itertools
import time as tm

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def do_bayesian_inference(V, IV_point, freq, num_x_steps=251, gam=0.03, e=10.0, sigma=10., sigmaC=1.,
                          num_samples=2E3, show_plots=False, econ=False):
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
        'x' : 1D float array
            Voltage vector interpolated with num_samples number of points
        'm' : Ask Kody
        'mR' : 1D float array
            Bayesian inference of the resistance. This is the one you want
        'vR' : 2D float array
            varaiance ? of inferred resistance
        'Irec' : 1D array or float
            Reconstructed current without capacitance
        'Sigma' : Ask Kody
        'cValue' : float
            Capacitance value
        'm2R' : Ask Kody
        'SI' : Ask Kody
    """
    num_samples = int(num_samples)
    num_x_steps = int(num_x_steps)
    if num_x_steps % 2 == 0:
        num_x_steps += 1  # Always keep it odd

    # Organize, set up the problem
    t_max = 1. / freq
    t = np.linspace(0, t_max, len(V))
    dt = t[2] - t[1]
    dv = np.diff(V) / dt
    dv = np.append(dv, dv[-1])
    max_volts = max(V)
    # num_x_steps = int(round(2 * round(max_volts / dx, 1) + 1, 0))
    x = np.linspace(-max_volts, max_volts, num_x_steps)
    dx = x[1] - x[0]
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


def bayesian_inference_unit(single_parm):
    """
    Wrapper around the original Bayesian inference function for parallel computing purposes

    Parameters
    ----------
    single_parm : tuple
        The first index of the tuple should contain the IV data to be processed
        The second index of the tuple should contain the parameter dictionary necessary for the bayesian function.

    Returns
    -------
    See the econ results of the original Bayesian Inference function
    """
    iv_point = single_parm[0]
    parm_dict = dict(single_parm[1])
    return do_bayesian_inference(parm_dict['volt_vec'], iv_point, parm_dict['freq'],
                                 num_x_steps=parm_dict['num_x_steps'], gam=parm_dict['gam'], e=parm_dict['e'],
                                 sigma=parm_dict['sigma'], sigmaC=parm_dict['sigmaC'],
                                 num_samples=parm_dict['num_samples'], show_plots=False, econ=True)


def bayesian_inference_dataset(h5_main, ex_freq, num_cores=None, num_x_steps=251, gam=0.03, e=10.0, sigma=10.,
                               sigmaC=1., num_samples=2E3, verbose=False):
    """
    Parameters
    ----------
    h5_main : h5py.Dataset
        Reference to the dataset containing the IV spectroscopy data
    ex_freq : float
        frequency of applied waveform
    num_cores : unsigned int (Optional. Default = None)
        Number of cores to use for computation. Leave as None for adaptive decision.
    num_x_steps : unsigned int (Optional, Default = 1E+3)
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
    verbose : Boolean (Optional, Default = False)
        Whether or not to print the status messages for debugging purposes

    Returns
    -------
    h5_bayes_grp : h5py.DataGroup object
        Reference to the group containing all the results of the Bayesian Inference
    """

    num_samples = int(num_samples)
    num_x_steps = int(num_x_steps)
    if num_x_steps % 2 == 0:
        num_x_steps += 1

    if h5_main.file.mode != 'r+':
        warn('Need to ensure that the file is in r+ mode to write results back to the file')
        raise TypeError

    h5_spec_vals = getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0]
    single_ao = np.squeeze(h5_spec_vals[()])

    num_pos = h5_main.shape[0]

    # create all h5 datasets here:
    bayes_grp = MicroDataGroup(h5_main.name.split('/')[-1] + '-Bayesian_Inference_', parent=h5_main.parent.name)
    # num_x_steps = int(round(2 * round(np.max(single_ao) / dx, 1) + 1, 0))
    if verbose:
        print('Now creating the datasets')
    ds_spec_vals = MicroDataset('Spectroscopic_Values', data=np.atleast_2d(np.arange(num_x_steps, dtype=np.float32)))
    ds_spec_inds = MicroDataset('Spectroscopic_Indices', data=np.atleast_2d(np.arange(num_x_steps, dtype=np.uint32)))
    ds_cap = MicroDataset('capacitance', data=[], maxshape=num_pos, dtype=np.float32, chunking=num_pos,
                          compression='gzip')
    ds_vr = MicroDataset('vr', data=[], maxshape=(num_pos, num_x_steps), dtype=np.float32,
                         chunking=(1, num_x_steps), compression='gzip')
    ds_mr = MicroDataset('mr', data=[], maxshape=(num_pos, num_x_steps), dtype=np.float32,
                         chunking=(1, num_x_steps), compression='gzip')
    ds_irec = MicroDataset('irec', data=[], maxshape=(num_pos, single_ao.size), dtype=np.float32,
                           chunking=(1, single_ao.size), compression='gzip')
    """
    # The following datasets will NOT be written because the data size becomes simply too big
    ds_vr = MicroDataset('vr', data=[], maxshape=(num_pos, num_x_points, num_x_points), dtype=np.float32,
                         chunking=(1, 1, num_x_points), compression='gzip')
    ds_m2r = MicroDataset('m2r', data=[], maxshape=ds_vr.maxshape, dtype=np.float32, compression='gzip',
                          chunking=ds_vr.chunking)
    ds_sigma = MicroDataset('sigma', data=[], maxshape=(num_pos, num_x_points + 1, num_x_points + 1),
                            dtype=np.float32, chunking=(1, 1, num_x_points + 1),
                            compression='gzip')
    ds_si = MicroDataset('si', data=[], maxshape=(num_pos, num_x_points, num_samples), dtype=np.float32,
                         chunking=(1, 1, num_samples), compression='gzip')


    ds_m = MicroDataset('m', data=[], maxshape=(num_pos, num_x_points + 1), dtype=np.float32,
                        chunking=(1, num_x_points + 1), compression='gzip')
    bayes_grp.addChildren([ds_x, ds_cap, ds_vr, ds_m2r, ds_sigma, ds_si, ds_mr, ds_m, ds_irec])
    """

    bayes_grp.addChildren([ds_spec_inds, ds_spec_vals, ds_cap, ds_vr, ds_mr, ds_irec])

    if verbose:
        bayes_grp.showTree()

    hdf = ioHDF5(h5_main.file)
    h5_refs = hdf.writeData(bayes_grp, print_log=verbose)

    h5_new_spec_vals = getH5DsetRefs(['Spectroscopic_Values'], h5_refs)[0]
    h5_new_spec_inds = getH5DsetRefs(['Spectroscopic_Indices'], h5_refs)[0]
    h5_cap = getH5DsetRefs(['capacitance'], h5_refs)[0]
    h5_vr = getH5DsetRefs(['vr'], h5_refs)[0]
    h5_mr = getH5DsetRefs(['mr'], h5_refs)[0]
    h5_irec = getH5DsetRefs(['irec'], h5_refs)[0]
    """
    h5_m2r = getH5DsetRefs(['m2r'], h5_refs)[0]
    h5_sigma = getH5DsetRefs(['sigma'], h5_refs)[0]
    h5_si = getH5DsetRefs(['si'], h5_refs)[0]
    h5_m = getH5DsetRefs(['m'], h5_refs)[0]
    """

    if verbose:
        print('Finished making room for the datasets. Now linking them')

    # Now link the datasets appropriately so that they become hubs:
    h5_pos_vals = getAuxData(h5_main, auxDataName=['Position_Values'])[0]
    h5_pos_inds = getAuxData(h5_main, auxDataName=['Position_Indices'])[0]

    # We don't have spectroscopic values for this dataset
    linkRefAsAlias(h5_cap, h5_pos_inds, 'Position_Indices')
    linkRefAsAlias(h5_cap, h5_pos_vals, 'Position_Values')

    # this dataset is the same as the main dataset in every way
    h5_irec = copyAttributes(h5_main, h5_irec, skip_refs=False)

    # These datasets get new spec datasets but reuse the old pos datasets:
    for new_dset in [h5_mr, h5_vr]:
        link_as_main(new_dset, h5_pos_inds, h5_pos_vals, h5_new_spec_inds, h5_new_spec_vals)

    if verbose:
        print('Finished linking all datasets!')

    # setting up parameters for parallel function:
    parm_dict = {'volt_vec': single_ao, 'freq': ex_freq, 'num_x_steps': num_x_steps, 'gam': gam, 'e': e, 'sigma': sigma, 'sigmaC': sigmaC,
                 'num_samples': num_samples}

    max_pos_per_chunk = 500  # Need a better way of figuring out a more appropriate estimate

    start_pix = 0

    while start_pix < num_pos:

        last_pix = min(start_pix + max_pos_per_chunk, num_pos)
        print('Working on pixels {} to {} of {}'.format(start_pix, last_pix, num_pos))

        t_start = tm.time()
        sing_parm = itertools.izip(h5_main[start_pix:last_pix], itertools.repeat(parm_dict))
        current_num_cores = recommendCores(last_pix-start_pix, requested_cores=num_cores, lengthy_computation=True)
        # Start parallel processing:
        if verbose:
            print('Starting a pool of {} cores'.format(current_num_cores))
        pool = Pool(processes=current_num_cores)
        jobs = pool.imap(bayesian_inference_unit, sing_parm)  # , chunksize=num_chunks)
        bayes_results = [j for j in jobs]
        pool.close()
        tot_time = np.round(tm.time() - t_start)

        if verbose:
            print('Done parallel computing in {} sec or {} sec per pixel'.format(tot_time, tot_time/max_pos_per_chunk))

        if verbose:
            print('Started accumulating all results')

        chunk_pos = len(bayes_results)
        cap_vec = np.zeros(chunk_pos, dtype=np.float32)
        vr_mat = np.zeros(shape=(chunk_pos, num_x_steps), dtype=np.float32)
        mr_mat = np.zeros(shape=(chunk_pos, num_x_steps), dtype=np.float32)
        irec_mat = np.zeros(shape=(chunk_pos, single_ao.size), dtype=np.float32)

        """
        m2r_mat = np.zeros(shape=vr_mat.shape, dtype=np.float32)
        sigma_mat = np.zeros(shape=(max_pos_per_chunk, num_x_points + 1, num_x_points + 1), dtype=np.float32)
        si_mat = np.zeros(shape=(max_pos_per_chunk, num_x_points, num_samples), dtype=np.float32)
        m_mat = np.zeros(shape=(max_pos_per_chunk, num_x_points + 1), dtype=np.float32)
        """

        # filling in all the results:
        x_vec = bayes_results[0]['x']
        for pix_ind, pix_results in enumerate(bayes_results):
            cap_vec[pix_ind] = pix_results['cValue']
            vr_mat[pix_ind] = pix_results['vR']
            mr_mat[pix_ind] = pix_results['mR']
            irec_mat[pix_ind] = pix_results['Irec']
            """m2r_mat[pix_ind] = pix_results['m2R']
            m_mat[pix_ind] = pix_results['m']
            sigma_mat[pix_ind] = pix_results['Sigma']
            si_mat[pix_ind] = pix_results['SI']"""

        t_accum_end = tm.time()

        if verbose:
            print('Finished accumulating results')
            print('Writing to h5')

        h5_cap[start_pix: last_pix] = cap_vec
        h5_vr[start_pix: last_pix] = vr_mat
        h5_mr[start_pix: last_pix] = mr_mat
        h5_irec[start_pix: last_pix] = irec_mat
        """
        h5_m2r[start_pix: last_pix] = m2r_mat
        h5_m[start_pix: last_pix] = m_mat
        h5_sigma[start_pix: last_pix] = sigma_mat
        h5_si[start_pix: last_pix] = si_mat
        """

        if verbose:
            print('Finished writing to file in {} sec'.format(np.round(tm.time() - t_accum_end)))

        hdf.flush()

        start_pix = last_pix

    h5_new_spec_vals[0, :] = x_vec  # Technically this needs to only be done once

    if verbose:
        print('Finished processing the dataset completely')

    return h5_cap.parent
