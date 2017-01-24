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


def multi_gauss_surface_fit(coef_mat, s_mat):
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


def gauss_2d_residuals_new(parms_vec, orig_data_mat, x_data_mat):
    # Only need to reshape the parms from 1D to 2D
    parms_mat = np.reshape(parms_vec, (-1, 4))

    err = orig_data_mat - multi_gauss_surface_fit(parms_mat, x_data_mat)
    return err.ravel()


def fit_atom_pos(single_parm):
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
    else:
        # Now refine the positions!
        plsq = least_squares(gauss_2d_residuals_new,
                             coef_guess_mat.ravel(),
                             args=(fit_region, s_mat),
                             bounds=(lb_mat.ravel(), ub_mat.ravel()),
                             jac='3-point', max_nfev=max_function_evals)
        coef_fit_mat = np.reshape(plsq.x, (-1, 4))

    if verbose:
        return coef_guess_mat, lb_mat, ub_mat, coef_fit_mat, fit_region, s_mat
    else:
        # only return position of central atom
        return coef_fit_mat[0, 1: 3]


def fit_atom_positions_parallel(num_cores, parm_dict, fitting_parms):
    parm_dict['verbose'] = False
    all_atom_guesses = parm_dict['atom_pos_guess']
    parm_list = itt.izip(range(all_atom_guesses.shape[0]), itt.repeat(parm_dict), itt.repeat(fitting_parms))
    t_start = tm.time()
    pool = mp.Pool(processes=num_cores)
    jobs = pool.imap(fit_atom_pos, parm_list)
    results = [j for j in jobs]
    pool.close()
    tot_time = np.round(tm.time() - t_start)
    print('Took {} sec to find {} atoms with {} cores'.format(tot_time, len(results), num_cores))
    return np.array(results)


def visualize_atom_fit(atom_rough_pos, all_atom_guesses, parm_dict, fitting_parms, cropped_clean_image):
    temp_dist = np.abs(
        all_atom_guesses[:, 0] + 1j * all_atom_guesses[:, 1] - (atom_rough_pos[0] + 1j * atom_rough_pos[1]))
    atom_ind = np.argsort(temp_dist)[0]

    parm_dict['verbose'] = True
    coef_guess_mat, lb_mat, ub_mat, coef_fit_mat, fit_region, s_mat = fit_atom_pos((atom_ind, parm_dict, fitting_parms))
    print('\tAmplitude\t\tx position\ty position\tsigma')
    print('-------------------GUESS---------------------')
    print(coef_guess_mat)
    print('-------------------LOWER---------------------')
    print(lb_mat)
    print('-------------------UPPER---------------------')
    print(ub_mat)
    print('--------------------FIT----------------------')
    print(coef_fit_mat)
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

        axis.imshow(img_mat, cmap="gray")
        axis.set_title(img_title)
        axis.scatter(centered_pos_mat[1:, 1], centered_pos_mat[1:, 0], color='orange')
        axis.scatter(centered_pos_mat[0, 1], centered_pos_mat[0, 0], color='red')
    fig_02.tight_layout()

    return coef_guess_mat, lb_mat, ub_mat, coef_fit_mat, fit_region, s_mat, fig_01, fig_02
