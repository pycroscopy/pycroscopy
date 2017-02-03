# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:13:22 2017

@author: Suhas Somnath, Stephen Jesse
"""

import numpy as np
from scipy.optimize import least_squares
import itertools as itt
import multiprocessing as mp
import time as tm
from _warnings import warn

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ...io.io_utils import recommendCores, realToCompound
from ...io.microdata import MicroDataset, MicroDataGroup
from ...io.io_hdf5 import ioHDF5

atom_dtype = np.dtype([('x', np.float32),
                       ('y', np.float32),
                       ('type', np.uint32)])

atom_coeff_dtype = np.dtype([('Amplitude', np.float32),
                             ('x', np.float32),
                             ('y', np.float32),
                             ('Sigma', np.float32)])


def multi_gauss_surface_fit(coef_mat, s_mat):
    """
    Evaluates the provided coefficients for N gaussian peaks to generate a 2D matrix

    Parameters
    ----------
    coef_mat : 2D numpy array
        Coefficients arranged as [atom, parameter] where the parameters are:
            height, row, column, sigma (width of the gaussian)
    s_mat : 3D numpy array
        Stack of the mesh grid

    Returns
    -------
    multi_gauss : 2D numpy array
        2D matrix with the N gaussian peaks whose properties are specified in the coefficients matrix
    """
    x = s_mat[:, :, 0]
    y = s_mat[:, :, 1]
    num_peaks = coef_mat.shape[0]
    multi_gauss = np.zeros(shape=x.shape, dtype=np.float32)

    for peak_ind in range(num_peaks):
        amp = coef_mat[peak_ind, 0]
        x_val = coef_mat[peak_ind, 1]
        y_val = coef_mat[peak_ind, 2]
        sigma = coef_mat[peak_ind, 3]
        gauss = amp * np.exp(-((x - x_val) ** 2 + (y - y_val) ** 2) / sigma ** 2)
        multi_gauss += gauss

    return multi_gauss


def fit_atom_pos(single_parm):
    """
    Fits the position of a single atom.

    Parameters
    ----------
    single_parm : tuple
        atom_ind : unsigned integer
            Index of the atom being fitted
        parm_dict : dictionary
            Dictionary containing all the guess values, table of nearest neighbors for each atom, and the original image
        fitting_parms : dictionary
            Dictionary of the many fitting parameters

    Returns
    -------
    coef_guess_mat : 2D numpy array
        guess coefficients for the set of N atoms
    coef_fit_mat : 2D numpy array
        Fit coefficients for the set of N atoms

    This function also returns all intermediate results for debugging purposes if parm_dict['verbose']=True
    """
    atom_ind = single_parm[0]
    parm_dict = single_parm[1]
    fitting_parms = single_parm[2]

    all_atom_guesses = parm_dict['atom_pos_guess']
    closest_neighbors_mat = parm_dict['nearest_neighbors']
    cropped_clean_image = parm_dict['cropped_cleaned_image']

    fit_region_size = fitting_parms['fit_region_size']
    gauss_width_guess = fitting_parms['gauss_width_guess']
    num_nearest_neighbors = fitting_parms['num_nearest_neighbors']
    min_amplitude = fitting_parms['min_amplitude']
    max_amplitude = fitting_parms['max_amplitude']
    position_range = fitting_parms['position_range']
    max_function_evals = fitting_parms['max_function_evals']
    min_gauss_width_ratio = fitting_parms['min_gauss_width_ratio']
    max_gauss_width_ratio = fitting_parms['max_gauss_width_ratio']
    verbose = False
    if 'verbose' in parm_dict:
        verbose = parm_dict['verbose']

    x_center_atom = all_atom_guesses[atom_ind, 0]
    y_center_atom = all_atom_guesses[atom_ind, 1]
    x_neighbor_atoms = all_atom_guesses[closest_neighbors_mat[atom_ind], 0]
    y_neighbor_atoms = all_atom_guesses[closest_neighbors_mat[atom_ind], 1]
    x_range = slice(max(int(np.round(x_center_atom - fit_region_size)), 0),
                    min(int(np.round(x_center_atom + fit_region_size)),
                        cropped_clean_image.shape[0]))
    y_range = slice(max(int(np.round(y_center_atom - fit_region_size)), 0),
                    min(int(np.round(y_center_atom + fit_region_size)),
                        cropped_clean_image.shape[1]))

    will_fail = False
    # Stephen says that it does not matter if guesses are outside but the fit does not work
    # well when guesses are outside the window
    x_outside = np.hstack((np.where(x_neighbor_atoms < x_range.start)[0],
                           np.where(x_neighbor_atoms > x_range.stop)[0]))
    y_outside = np.hstack((np.where(y_neighbor_atoms < y_range.start)[0],
                           np.where(y_neighbor_atoms > y_range.stop)[0]))
    guesses_outside = np.unique(np.hstack((x_outside, y_outside)))
    if guesses_outside.size >= 0.5 * num_nearest_neighbors:
        if verbose:
            warn('Atom {}: Too few ({} of {}) neighbors within window to fit'.format(atom_ind, num_nearest_neighbors -
                                                                                     guesses_outside.size,
                                                                                     num_nearest_neighbors))
        will_fail = True
    else:
        guesses_inside = np.invert(np.in1d(np.arange(num_nearest_neighbors), guesses_outside))
        x_neighbor_atoms = x_neighbor_atoms[guesses_inside]
        y_neighbor_atoms = y_neighbor_atoms[guesses_inside]
        num_nearest_neighbors = x_neighbor_atoms.size

    fit_region = cropped_clean_image[x_range, y_range]

    # define x and y fitting range
    s1, s2 = np.meshgrid(range(x_range.start, x_range.stop),
                         range(y_range.start, y_range.stop))
    s_mat = np.dstack((s1.T, s2.T))

    # initial guess values
    x_guess = np.hstack((x_center_atom, x_neighbor_atoms))
    y_guess = np.hstack((y_center_atom, y_neighbor_atoms))
    a_guess = cropped_clean_image[np.uint32(x_guess), np.uint32(y_guess)]
    sigma_guess = gauss_width_guess * np.ones(num_nearest_neighbors + 1)

    coef_guess_mat = np.transpose(np.vstack((a_guess, x_guess,
                                             y_guess, sigma_guess)))
    # Set up upper and lower bounds:
    lb_mat = [min_amplitude * np.ones(num_nearest_neighbors + 1),
              coef_guess_mat[:, 1] - position_range,
              coef_guess_mat[:, 2] - position_range,
              min_gauss_width_ratio * gauss_width_guess * np.ones(num_nearest_neighbors + 1)]

    ub_mat = [max_amplitude * np.ones(num_nearest_neighbors + 1),
              coef_guess_mat[:, 1] + position_range,
              coef_guess_mat[:, 2] + position_range,
              max_gauss_width_ratio * gauss_width_guess * np.ones(num_nearest_neighbors + 1)]
    lb_mat = np.transpose(lb_mat)
    ub_mat = np.transpose(ub_mat)

    if will_fail:
        coef_fit_mat = coef_guess_mat
        plsq = None
    else:
        # Now refine the positions!

        def gauss_2d_residuals(parms_vec, orig_data_mat, x_data_mat):
            """
            Calculates the residual
            Parameters
            ----------
            parms_vec : 1D numpy array
                Raveled version of the parameters matrix
            orig_data_mat : 2D numpy array
                Section of the image being fitted
            x_data_mat : 3D numpy array

            Returns
            -------
            err_vec : 1D numpy array
                Difference between the original data and the matrix obtained by evaluating parms_vec with x_data_mat
            """
            # Only need to reshape the parms from 1D to 2D
            parms_mat = np.reshape(parms_vec, (-1, 4))

            err = orig_data_mat - multi_gauss_surface_fit(parms_mat, x_data_mat)
            return err.ravel()

        plsq = least_squares(gauss_2d_residuals,
                             coef_guess_mat.ravel(),
                             args=(fit_region, s_mat),
                             bounds=(lb_mat.ravel(), ub_mat.ravel()),
                             jac='2-point', max_nfev=max_function_evals)
        coef_fit_mat = np.reshape(plsq.x, (-1, 4))

    if verbose:
        return coef_guess_mat, lb_mat, ub_mat, coef_fit_mat, fit_region, s_mat, plsq
    else:
        return coef_guess_mat, coef_fit_mat


def fit_atom_positions_parallel(parm_dict, fitting_parms, num_cores=None):
    """
    Fits the positions of N atoms in parallel

    Parameters
    ----------
    parm_dict : dictionary
        Dictionary containing the guess positions, nearest neighbors and original image
    fitting_parms : dictionary
        Parameters used for atom position fitting
    num_cores : unsigned int (Optional. Default = available logical cores - 2)
        Number of cores to compute with

    Returns
    -------
    results : list of tuples
        Guess and fit coefficients
    """
    parm_dict['verbose'] = False
    all_atom_guesses = parm_dict['atom_pos_guess']
    t_start = tm.time()
    num_cores = recommendCores(all_atom_guesses.shape[0], requested_cores=num_cores, lengthy_computation=False)
    if num_cores>1:
        pool = mp.Pool(processes=num_cores)
        parm_list = itt.izip(range(all_atom_guesses.shape[0]), itt.repeat(parm_dict), itt.repeat(fitting_parms))
        chunk = int(all_atom_guesses.shape[0] / num_cores)
        jobs = pool.imap(fit_atom_pos, parm_list, chunksize=chunk)
        results = [j for j in jobs]
        pool.close()
    else:
        parm_list = itt.izip(range(all_atom_guesses.shape[0]), itt.repeat(parm_dict), itt.repeat(fitting_parms))
        results = [fit_atom_pos(parm) for parm in parm_list]

    tot_time = np.round(tm.time() - t_start)
    print('Took {} sec to find {} atoms with {} cores'.format(tot_time, len(results), num_cores))

    return results


def fit_atom_positions_dset(h5_grp, fitting_parms=None, num_cores=None):
    """
    A temporary substitute for a full-fledged process class.
    Computes the guess and fit coefficients for the provided atom guess positions and writes these results to the
    given h5 group

    Parameters
    ----------

    h5_grp : h5py.Group reference
        Group containing the atom guess positions, cropped clean image and some necessary parameters
    fitting_parms : dictionary
        Parameters used for atom position fitting
    num_cores : unsigned int (Optional. Default = available logical cores - 2)
        Number of cores to compute with

    Returns
    -------
    h5_grp : h5py.Group reference
        Same group as the parameter but now with the 'Guess' and 'Fit' datasets
    """

    cropped_clean_image = h5_grp['Cropped_Clean_Image'][()]
    h5_guess = h5_grp['Guess_Positions']
    all_atom_guesses = np.transpose(np.vstack((h5_guess['x'], h5_guess['y'])))  # leave out the atom type for now
    win_size = h5_grp.attrs['motif_win_size']
    psf_width = h5_grp.attrs['psf_width']

    num_atoms = all_atom_guesses.shape[0]  # number of atoms

    # build distance matrix
    pos_vec = all_atom_guesses[:, 0] + 1j * all_atom_guesses[:, 1]

    pos_mat1 = np.tile(np.transpose(np.atleast_2d(pos_vec)), [1, num_atoms])
    pos_mat2 = np.transpose(pos_mat1)
    d_mat = np.abs(pos_mat2 - pos_mat1)  # matrix of distances between all atoms
    # sort the distance matrix and keep only the atoms within the nearest neighbor limit
    neighbor_dist_order = np.argsort(d_mat)

    if fitting_parms is None:
        num_nearest_neighbors = 6  # to consider when fitting
        fitting_parms = {'fit_region_size': win_size * 0.80,  # region to consider when fitting
                         'gauss_width_guess': psf_width * 2,
                         'num_nearest_neighbors': num_nearest_neighbors,
                         'min_amplitude': 0,  # min amplitude limit for gauss fit
                         'max_amplitude': 2,  # max amplitude limit for gauss fit
                         'position_range': win_size / 2,
                         # range that the fitted position can go from initial guess position[pixels]
                         'max_function_evals': 100,
                         'min_gauss_width_ratio': 0.5,  # min width of gauss fit ratio,
                         'max_gauss_width_ratio': 2,  # max width of gauss fit ratio
                         'fitting_tolerance': 1E-4}

    num_nearest_neighbors = fitting_parms['num_nearest_neighbors']

    # neighbor dist order has the (indices of the) neighbors for each atom sorted by distance
    closest_neighbors_mat = neighbor_dist_order[:, 1:num_nearest_neighbors + 1]

    parm_dict = {'atom_pos_guess': all_atom_guesses,
                 'nearest_neighbors': closest_neighbors_mat,
                 'cropped_cleaned_image': cropped_clean_image}

    # do the parallel fitting
    fitting_results = fit_atom_positions_parallel(parm_dict, fitting_parms, num_cores=num_cores)

    # Make datasets to write back to file:
    guess_parms = np.zeros(shape=(num_atoms, num_nearest_neighbors+1), dtype=atom_coeff_dtype)
    fit_parms = np.zeros(shape=guess_parms.shape, dtype=guess_parms.dtype)

    for atom_ind, single_atom_results in enumerate(fitting_results):
        guess_coeff, fit_coeff = single_atom_results
        num_neighbors_used = guess_coeff.shape[0]
        guess_parms[atom_ind, :num_neighbors_used] = np.squeeze(realToCompound(guess_coeff, guess_parms.dtype))
        fit_parms[atom_ind, :num_neighbors_used] = np.squeeze(realToCompound(fit_coeff, guess_parms.dtype))

    ds_atom_guesses = MicroDataset('Guess', data=guess_parms)
    ds_atom_fits = MicroDataset('Fit', data=fit_parms)
    dgrp_atom_finding = MicroDataGroup(h5_grp.name.split('/')[-1], parent=h5_grp.parent.name)
    dgrp_atom_finding.attrs = fitting_parms
    dgrp_atom_finding.addChildren([ds_atom_guesses, ds_atom_fits])

    hdf = ioHDF5(h5_grp.file)
    h5_atom_refs = hdf.writeData(dgrp_atom_finding)
    return h5_grp


def visualize_atom_fit(atom_rough_pos, all_atom_guesses, parm_dict, fitting_parms, cropped_clean_image):
    """
    Computes the fit for a given atom and plots the results
    
    Parameters
    ----------
    atom_rough_pos : tuple
        row, column position of the atom from the guess
    all_atom_guesses : 2D numpy array
        Guesses of atom positions arranged as [atom index, row(0) and column(1)]
    parm_dict : dictionary
        Dictionary containing the guess positions, nearest neighbors and original image
    fitting_parms : dictionary
        Parameters used for atom position fitting
    cropped_clean_image : 2D numpy array
        original image to fit to

    Returns
    -------
    coef_guess_mat : 2D numpy array
        Coefficients arranged as [atom, parameter] where the parameters are:
            height, row, column, sigma (width of the gaussian)
    lb_mat  : 2D numpy array
        Lower bounds for the fits
    ub_mat : 2D numpy array
        Upper bounds for the fits
    coef_fit_mat : 2D numpy array
        Coefficients arranged as [atom, parameter] where the parameters are:
            height, row, column, sigma (width of the gaussian)
    fit_region : 2D numpy array
        Section of the image being fitted
    s_mat : 3D numpy array
        Stack of the mesh grid
    plsq : Least squares fit object
        Use this to find if the fitting went well
    fig_01 : matplotlib.pyplot.figure handle
        Handle to figure 1
    fig_02 : matplotlib.pyplot.figure handle
        Handle to figure 2
    """
    temp_dist = np.abs(
        all_atom_guesses[:, 0] + 1j * all_atom_guesses[:, 1] - (atom_rough_pos[0] + 1j * atom_rough_pos[1]))
    atom_ind = np.argsort(temp_dist)[0]

    parm_dict['verbose'] = True
    coef_guess_mat, lb_mat, ub_mat, coef_fit_mat, fit_region, s_mat, plsq = fit_atom_pos((atom_ind, parm_dict, fitting_parms))

    print('\tAmplitude\tx position\ty position\tsigma')
    print('-------------------GUESS---------------------')
    print(coef_guess_mat)
    print('-------------------LOWER---------------------')
    print(lb_mat)
    print('-------------------UPPER---------------------')
    print(ub_mat)
    print('--------------------FIT----------------------')
    print(coef_fit_mat)
    print('-------------- LEAST SQUARES ----------------')
    print(plsq.message)
    print('Function evaluations: {}\nJacobian evaluations: {}'.format(plsq.nfev, plsq.njev))

    gauss_2d_guess = multi_gauss_surface_fit(coef_guess_mat, s_mat)
    gauss_2d_fit = multi_gauss_surface_fit(coef_fit_mat, s_mat)

    fit_region_size = fitting_parms['fit_region_size']

    fig_01, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(cropped_clean_image, interpolation='none', cmap="gray")
    axis.add_patch(patches.Rectangle((all_atom_guesses[atom_ind, 1] - fit_region_size,
                                      all_atom_guesses[atom_ind, 0] - fit_region_size),
                                     2 * fit_region_size, 2 * fit_region_size, fill=False,
                                     color='orange', linewidth=2))
    axis.scatter(all_atom_guesses[:, 1], all_atom_guesses[:, 0], color='yellow')
    axis.scatter(all_atom_guesses[atom_ind, 1], all_atom_guesses[atom_ind, 0], color='red')
    axis.scatter(coef_guess_mat[1:, 2], coef_guess_mat[1:, 1], color='green')
    fig_01.tight_layout()

    fig_02, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    for axis, img_mat, coeff_mat, pos_mat, img_title in zip(axes.flat,
                                                            [fit_region, fit_region, gauss_2d_guess, gauss_2d_fit],
                                                            [coef_guess_mat, coef_fit_mat, coef_guess_mat,
                                                             coef_fit_mat],
                                                            [all_atom_guesses, all_atom_guesses, all_atom_guesses,
                                                             all_atom_guesses],
                                                            ['Original + guess pos', 'Original + fit pos', 'Guess',
                                                             'Fit']):
        centered_pos_mat = np.copy(coeff_mat[:, 1:3])
        # TODO: This is not necessarily correct, especially when the window extends beyond the image
        centered_pos_mat[:, 0] -= pos_mat[atom_ind, 0] - (0.5 * fit_region.shape[0])
        centered_pos_mat[:, 1] -= pos_mat[atom_ind, 1] - (0.5 * fit_region.shape[1])
        axis.hold(True)  # Without this, plots do not show up on the notebooks
        axis.imshow(img_mat, cmap="gray")
        axis.set_title(img_title)
        axis.scatter(centered_pos_mat[1:, 1], centered_pos_mat[1:, 0], color='orange')
        axis.scatter(centered_pos_mat[0, 1], centered_pos_mat[0, 0], color='red')
    fig_02.tight_layout()

    return coef_guess_mat, lb_mat, ub_mat, coef_fit_mat, fit_region, s_mat, plsq, fig_01, fig_02
