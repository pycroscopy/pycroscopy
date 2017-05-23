# -*- coding: utf-8 -*-
"""
@author: Ondrej Dyck
"""

from __future__ import division, print_function, absolute_import
import os
import numpy as np
from scipy.optimize import least_squares
import itertools as itt
import multiprocessing as mp
import time as tm
from _warnings import warn
from sklearn.neighbors import KNeighborsClassifier
import h5py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

from ...io.io_utils import recommendCores, realToCompound
from ...io.microdata import MicroDataset, MicroDataGroup
from ...io.io_hdf5 import ioHDF5
from ...viz import plot_utils
from ..model import Model

def do_fit(single_parm):
    parms = single_parm[0]
    coef_guess_mat = parms[1]
    fit_region = parms[2]
    s1 = parms[3]
    s2 = parms[4]
    lb_mat = parms[5]
    ub_mat = parms[6]
    kwargs = single_parm[1]
    max_function_evals = kwargs['max_function_evals']
    plsq = least_squares(gauss_2d_residuals,
                         coef_guess_mat.ravel(),
                         args=(fit_region.ravel(), s1.T, s2.T),
                         kwargs=kwargs,
                         bounds=(lb_mat.ravel(), ub_mat.ravel()),
                         jac='2-point', max_nfev=max_function_evals)

    coef_fit_mat = np.reshape(plsq.x, (-1, 7))

    #if verbose:
    #    return coef_guess_mat, lb_mat, ub_mat, coef_fit_mat, fit_region, s_mat, plsq
    #else:
    return coef_fit_mat


def gauss_2d_residuals(parms_vec, orig_data_mat, x_data, y_data, **kwargs):
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
    parms_mat = np.reshape(parms_vec, (-1, 7))
    # print(parms_mat)

    err = orig_data_mat - gauss2d(x_data, y_data, *parms_mat, **kwargs).ravel()
    return err


def gauss2d(X, Y, *parms, **kwargs):
    """
    Calculates a general 2d elliptic gaussian

    Parameters
    ----------

    X, Y : the x and y matrix values from the call "X, Y = np.meshgrid(x,y)" where x and y are
        defined by x = np.arange(-width/2,width/2) and y = np.arange(-height/2,height/2). 

    params: List of 7 parameters defining the gaussian.
        The parameters are [A, x0, y0, sigma_x, sigma_y, theta, background]
            A : amplitude
            x0: x position
            y0: y position
            sigma_x: standard deviation in x
            sigma_y: standard deviation in y
            theta: rotation angle
            background: a background constant

    Returns
    -------

    Returns a width x height matrix of values representing the call to the gaussian function at each position. 
    """
    symmetric = kwargs['symmetric']
    background = kwargs['background']
    Z = np.zeros(np.shape(X))
    background_value = parms[0][-1]  # we can only have one background value for the fit region
    for guess in parms:
        # each gaussian has a background associated with it but we only use the center atom background
        A, x0, y0, sigma_x, sigma_y, theta, background_unused = guess

        # determine which type of gaussian we want
        if symmetric:
            sigma_y = sigma_x
            if not background:
                background = 0
        else:
            if not background:
                background = 0

        # define some variables
        a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
        b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
        c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)

        # calculate the final value
        Z += A * np.exp(- (a * (X - x0) ** 2 - 2 * b * (X - x0) * (Y - y0) + c * (Y - y0) ** 2)) + background_value
    return Z

class Gauss_Fit(object):
        """
        Initializes the gaussian fitting routines:
            fit_motif()
            fit_atom_positions_parallel()
            write_to_disk()
            

        Parameters
        ----------

        atom_grp : h5py.Group reference
            Parent group containing the atom guess positions, cropped clean image and motif positions
            
        fitting_parms : dictionary
            Parameters used for atom position fitting
            Example:
                fitting_parms = {'fit_region_size': win_size * 0.5,                 # region to consider when fitting. Should be large enough to see the nearest neighbors.
                                 'num_nearest_neighbors': num_nearest_neighbors,    # the number of nearest neighbors to fit
                                 'sigma_guess': 3,                                  # starting guess for gaussian standard deviation. Should be about the size of an atom width in pixels.
                                 'position_range': win_size / 4,                    # range that the fitted position can move from initial guess position in pixels
                                 'max_function_evals': 100,                         # maximum allowed function calls; passed to the least squares fitter
                                 'fitting_tolerance': 1E-4,                         # target difference between the fit and the data
                                 'symmetric': True,                                 # flag to signal if a symmetric gaussian is desired (i.e. sigma_x == sigma_y)
                                 'background': True,                                # flag to signal if a background constant is desired
                                 'movement_allowance': 5.0}                         # percent of movement allowed (on all parameters except x and y positions
        
        """

        def __init__(self, atom_grp, fitting_parms):
            # we should do some initial checks here to ensure the data is at the correct stage for atom fitting
            print('Initializing Gauss Fit')
            # check that the data is appropriate (does nothing yet)
            self.check_data(atom_grp)

            # set dtypes
            self.atom_coeff_dtype = np.dtype([('type', np.float32),
                                              ('amplitude', np.float32),
                                              ('x', np.float32),
                                              ('y', np.float32),
                                              ('sigma_x', np.float32),
                                              ('sigma_y', np.float32),
                                              ('theta', np.float32),
                                              ('background', np.float32)])

            self.motif_coeff_dtype = np.dtype([('amplitude', np.float32),
                                              ('x', np.float32),
                                              ('y', np.float32),
                                              ('sigma_x', np.float32),
                                              ('sigma_y', np.float32),
                                              ('theta', np.float32),
                                              ('background', np.float32)])

            # initialize some variables
            self.atom_grp = atom_grp
            self.cropped_clean_image = self.atom_grp['Cropped_Clean_Image'][()]
            self.h5_guess = self.atom_grp['Guess_Positions']
            self.fitting_parms = fitting_parms
            self.win_size = self.atom_grp.attrs['motif_win_size']
            half_wind = int(self.win_size * 0.5)
            self.motif_centers = self.atom_grp['Motif_Centers'][:] - half_wind  # << correction for double cropped image
            self.psf_width = self.atom_grp.attrs['psf_width']

            # grab the initial guesses
            self.all_atom_guesses = np.transpose(np.vstack((self.h5_guess['x'],
                                                            self.h5_guess['y'],
                                                            self.h5_guess['type'])))

            self.num_atoms = self.all_atom_guesses.shape[0]  # number of atoms

            # build distance matrix
            pos_vec = self.all_atom_guesses[:, 0] + 1j * self.all_atom_guesses[:, 1]

            pos_mat1 = np.tile(np.transpose(np.atleast_2d(pos_vec)), [1, self.num_atoms])
            pos_mat2 = np.transpose(pos_mat1)
            d_mat = np.abs(pos_mat2 - pos_mat1)  # matrix of distances between all atoms
            # sort the distance matrix and keep only the atoms within the nearest neighbor limit
            neighbor_dist_order = np.argsort(d_mat)

            self.num_nearest_neighbors = self.fitting_parms['num_nearest_neighbors']

            # neighbor dist order has the (indices of the) neighbors for each atom sorted by distance
            self.closest_neighbors_mat = neighbor_dist_order[:, 1:self.num_nearest_neighbors + 1]

            # find which atoms are at the centers of the motifs
            self.center_atom_vec = (self.motif_centers[:, 0]) + (1j * self.motif_centers[:, 1])
            self.center_atom_dists = [np.abs(pos_vec - item) for item in self.center_atom_vec]
            self.center_atom_sorted = np.argsort(self.center_atom_dists)
            self.center_atom_indices = self.center_atom_sorted[:, 0]

        def fit_atom_positions_parallel(self, plot_results=True, num_cores=None):
            """
            Fits the positions of N atoms in parallel

            Parameters
            ----------
            plot_results : optional boolean (default is True)
                Specifies whether to output a visualization of the fitting results
                
            num_cores : unsigned int (Optional. Default = available logical cores - 2)
                Number of cores to compute with

            Creates guess_dataset and fit_dataset with the results.
            
            Returns
            -------
            
            fit_dataset: NxM numpy array of tuples where N is the number of atoms fit and M is the number of nearest
                neighbors considered. Each tuple contains the converged values for each gaussian.
                The value names are stored in the dtypes.
            """

            t_start = tm.time()
            if num_cores is None:
                num_cores = recommendCores(self.num_atoms, requested_cores=num_cores, lengthy_computation=False)

            print('Setting up guesses')
            self.guess_parms = []
            for i in range(self.num_atoms):
                self.guess_parms.append(self.do_guess(i))

            print('Fitting...')
            if num_cores > 1:
                pool = mp.Pool(processes=num_cores)
                parm_list = itt.izip(self.guess_parms, itt.repeat(self.fitting_parms))
                chunk = int(self.num_atoms / num_cores)
                jobs = pool.imap(do_fit, parm_list, chunksize=chunk)
                self.fitting_results = [j for j in jobs]
                pool.close()
            else:
                parm_list = itt.izip(self.guess_parms, itt.repeat(self.fitting_parms))
                self.fitting_results = [do_fit(parm) for parm in parm_list]

            print ('Finalizing datasets...')
            self.guess_dataset = np.zeros(shape=(self.num_atoms, self.num_nearest_neighbors + 1), dtype=self.atom_coeff_dtype)
            self.fit_dataset = np.zeros(shape=self.guess_dataset.shape, dtype=self.guess_dataset.dtype)

            for atom_ind, single_atom_results in enumerate(self.fitting_results):
                types = np.hstack((self.h5_guess['type'][atom_ind], [self.h5_guess['type'][neighbor] for neighbor in self.closest_neighbors_mat[atom_ind]]))
                atom_data = np.hstack((np.vstack(types), single_atom_results))
                atom_data = [tuple(element) for element in atom_data]
                self.fit_dataset[atom_ind] = atom_data

                single_atom_guess = self.guess_parms[atom_ind]
                atom_guess_data = np.hstack((np.vstack(types), single_atom_guess[1]))
                atom_guess_data = [tuple(element) for element in atom_guess_data]
                self.guess_dataset[atom_ind] = atom_guess_data

            tot_time = np.round(tm.time() - t_start)
            print('Took {} sec to find {} atoms with {} cores'.format(tot_time, len(self.fitting_results), num_cores))

            # if plotting is desired
            if plot_results:
                fig, axis = plt.subplots(figsize=(14, 14))
                axis.hold(True)
                axis.imshow(self.cropped_clean_image, interpolation='none', cmap="gray")
                axis.scatter(self.guess_dataset[:, 0]['y'], self.guess_dataset[:, 0]['x'], color='yellow', label='Guess')
                axis.scatter(self.fit_dataset[:, 0]['y'], self.fit_dataset[:, 0]['x'], color='red', label='Fit')
                axis.legend()
                fig.tight_layout()
                fig.show()

            return self.fit_dataset

        def do_guess(self, atom_ind, initial_motifs=False):
            """
            Fits the position of a single atom.

            Parameters
            ----------
            atom_ind : int
                The index of the atom to generate guess parameters for
                
            initial_motifs : optional boolean (default is False)
                Specifies whether we are generating guesses for the initial motifs. Subsequent guesses
                have the advantage of the fits from the motifs and will be much better starting values.

            Returns
            -------
            atom_ind : int
                The index of the atom to generate guess parameters for
                
            coef_guess_mat : 2D numpy array
                Initial guess parameters for all the gaussians.
                
            fit_region : 2D numpy array
                The fit region cropped from the image
                
            s1 and s2 : 2D numpy arrays
                The required input for the X and Y parameters of gauss2d
                 
            lb_mat and ub_mat : 2D numpy arrays
                The lower and upper bounds for the fitting.
            """

            fit_region_size = self.fitting_parms['fit_region_size']
            movement_allowance = self.fitting_parms['movement_allowance']
            position_range = self.fitting_parms['position_range']

            # start writing down initial guesses
            x_center_atom = self.h5_guess['x'][atom_ind]
            y_center_atom = self.h5_guess['y'][atom_ind]
            x_neighbor_atoms = [self.h5_guess['x'][self.closest_neighbors_mat[atom_ind][i]] for i in
                                range(self.num_nearest_neighbors)]
            y_neighbor_atoms = [self.h5_guess['y'][self.closest_neighbors_mat[atom_ind][i]] for i in
                                range(self.num_nearest_neighbors)]

            # select the window we're going to be fitting
            x_range = slice(max(int(np.round(x_center_atom - fit_region_size)), 0),
                            min(int(np.round(x_center_atom + fit_region_size)),
                                self.cropped_clean_image.shape[0]))
            y_range = slice(max(int(np.round(y_center_atom - fit_region_size)), 0),
                            min(int(np.round(y_center_atom + fit_region_size)),
                                self.cropped_clean_image.shape[1]))
            fit_region = self.cropped_clean_image[x_range, y_range]

            # define x and y fitting range
            s1, s2 = np.meshgrid(range(x_range.start, x_range.stop),
                                 range(y_range.start, y_range.stop))

            # guesses are different if we're fitting the initial windows
            if initial_motifs:
                # If true, we need to generate more crude guesses
                # for the initial motif window fitting.
                # Once these have been fit properly they will act
                # as the starting point for future guesses.

                # put the initial guesses into the proper form
                x_guess = np.hstack((x_center_atom, x_neighbor_atoms))
                y_guess = np.hstack((y_center_atom, y_neighbor_atoms))
                sigma_x_center_atom = self.fitting_parms['sigma_guess']
                sigma_y_center_atom = self.fitting_parms['sigma_guess']
                sigma_x_neighbor_atoms = [self.fitting_parms['sigma_guess'] for i in
                                          range(self.num_nearest_neighbors)]
                sigma_y_neighbor_atoms = [self.fitting_parms['sigma_guess'] for i in
                                          range(self.num_nearest_neighbors)]
                theta_center_atom = 0
                theta_neighbor_atoms = np.zeros(self.num_nearest_neighbors)
                background_center_atom = np.min(fit_region)

                # The existence of a background messes up a straight forward gaussian amplitude guess,
                # so we add/subtract the background value from the straight forward guess depending
                # on if the background is positive or negative.
                if np.min(fit_region) < 0:
                    a_guess = self.cropped_clean_image[
                                  np.rint(x_guess).astype(int), np.rint(y_guess).astype(int)] - background_center_atom
                else:
                    a_guess = self.cropped_clean_image[
                                  np.rint(x_guess).astype(int), np.rint(y_guess).astype(int)] + background_center_atom

                sigma_x_guess = np.hstack((sigma_x_center_atom, sigma_x_neighbor_atoms))
                sigma_y_guess = np.hstack((sigma_y_center_atom, sigma_y_neighbor_atoms))
                theta_guess = np.hstack((theta_center_atom, theta_neighbor_atoms))
                background_guess = np.hstack([background_center_atom for num in range(
                    self.num_nearest_neighbors + 1)])  # we will only need one background
                coef_guess_mat = np.transpose(np.vstack((a_guess, x_guess, y_guess, sigma_x_guess, sigma_y_guess,
                                                         theta_guess, background_guess)))
            else:
                # otherwise better guesses are assumed to exist
                motif_type = self.h5_guess['type'][atom_ind]
                coef_guess_mat = np.copy(self.motif_converged_parms[motif_type])
                coef_guess_mat[:, 1] = self.h5_guess['x'][atom_ind] + coef_guess_mat[:, 1]
                coef_guess_mat[:, 2] = self.h5_guess['y'][atom_ind] + coef_guess_mat[:, 2]

            # Choose upper and lower bounds for the fitting
            #
            # Address negatives first
            lb_a = []
            ub_a = []
            for item in coef_guess_mat[:, 0]:  # amplitudes

                if item < 0:
                    lb_a.append(item + item * movement_allowance)
                    ub_a.append(item - item * movement_allowance)
                else:
                    lb_a.append(item - item * movement_allowance)
                    ub_a.append(item + item * movement_allowance)

            lb_background = []
            ub_background = []
            for item in coef_guess_mat[:, 6]:  # background
                if item < 0:
                    lb_background.append(item + item * movement_allowance)
                    ub_background.append(item - item * movement_allowance)
                else:
                    lb_background.append(item - item * movement_allowance)
                    ub_background.append(item + item * movement_allowance)

            # Set up upper and lower bounds:
            lb_mat = [lb_a,                                                              # amplitude
                      coef_guess_mat[:, 1] - position_range,                             # x position
                      coef_guess_mat[:, 2] - position_range,                             # y position
                      [np.max([0, value - value * movement_allowance]) for value in coef_guess_mat[:, 3]],  # sigma x
                      [np.max([0, value - value * movement_allowance]) for value in coef_guess_mat[:, 4]],  # sigma y
                      coef_guess_mat[:, 5] - 2 * 3.14159,                                # theta
                      lb_background]                                                     # background

            ub_mat = [ub_a,                                                              # amplitude
                      coef_guess_mat[:, 1] + position_range,                             # x position
                      coef_guess_mat[:, 2] + position_range,                             # y position
                      coef_guess_mat[:, 3] + coef_guess_mat[:, 3] * movement_allowance,  # sigma x
                      coef_guess_mat[:, 4] + coef_guess_mat[:, 4] * movement_allowance,  # sigma y
                      coef_guess_mat[:, 5] + 2 * 3.14159,                                # theta
                      ub_background]                                                     # background

            lb_mat = np.transpose(lb_mat)
            ub_mat = np.transpose(ub_mat)

            check_bounds = False
            if check_bounds:
                for i, item in enumerate(coef_guess_mat):
                    for j, value in enumerate(item):
                        if lb_mat[i][j] > value or ub_mat[i][j] < value:
                            print('Atom number: {}'.format(atom_ind))
                            print('Guess: {}'.format(item))
                            print('Lower bound: {}'.format(lb_mat[i]))
                            print('Upper bound: {}'.format(ub_mat[i]))
                            print('dtypes: {}'.format(self.atom_coeff_dtype.names))
                            raise ValueError('{} guess is out of bounds'.format(self.atom_coeff_dtype.names[j]))

            return atom_ind, coef_guess_mat, fit_region, s1, s2, lb_mat, ub_mat

        def check_data(self, atom_grp):
            # some data checks here
            try:
                img = atom_grp['Cropped_Clean_Image']
            except KeyError:
                raise KeyError('The data \'Cropped_Clean_Image\' must exist before fitting')

            try:
                guesses = atom_grp['Guess_Positions']
            except KeyError:
                raise KeyError('The data \'Guess_Positions\' must exist before fitting')

            try:
                motifs = atom_grp['Motif_Centers']
            except KeyError:
                raise KeyError('The data \'Motif_Centers\' must exist before fitting')

            if np.shape(img)[0] < 1 or np.shape(img)[1] < 1:
                raise Exception('\'Cropped_Clean_Image\' data must have two dimensions with lengths greater than one')

            if len(guesses) < 1:
                raise Exception('\'Guess_Positions\' data length must be greater than one')

            if len(guesses[0]) < 3:
                raise Exception('\'Guess_Positions\' data must have at least three values for each entry: '
                                'type, x position, and y position')

            if motifs.shape[0] < 1:
                raise Exception('\'Motif_Centers\' data must contain at least one motif')

            if motifs.shape[1] != 2:
                raise Exception('\'Motif_Centers\' data is expected to have a shape of (n, 2). '
                                'The second dimension is not 2.')

        def write_to_disk(self):
            """
            Writes the gaussian fitting results to disk

            Parameters
            ----------

                None

            Returns
            -------

                Returns the atom parent group containing the original data and the newly written data:
                    Gaussian_Guesses
                    Gaussian_Fits
                    Motif_Guesses
                    Motif_Fits
                    Nearest_Neighbor_Indices
            """

            ds_atom_guesses = MicroDataset('Gaussian_Guesses', data=self.guess_dataset)
            ds_atom_fits = MicroDataset('Gaussian_Fits', data=self.fit_dataset)
            ds_motif_guesses = MicroDataset('Motif_Guesses', data=self.motif_guess_dataset)
            ds_motif_fits = MicroDataset('Motif_Fits', data=self.motif_converged_dataset)
            ds_nearest_neighbors = MicroDataset('Nearest_Neighbor_Indices', data=self.closest_neighbors_mat, dtype=np.uint32)
            dgrp_atom_finding = MicroDataGroup(self.atom_grp.name.split('/')[-1], parent=self.atom_grp.parent.name)
            dgrp_atom_finding.attrs = self.fitting_parms
            dgrp_atom_finding.addChildren([ds_atom_guesses, ds_atom_fits, ds_motif_guesses, ds_motif_fits, ds_nearest_neighbors])

            hdf = ioHDF5(self.atom_grp.file)
            h5_atom_refs = hdf.writeData(dgrp_atom_finding)
            hdf.flush()
            return self.atom_grp

        def fit_motif(self, plot_results=True):
            '''
            Parameters
            ----------
                plot_results: boolean (default = True)
                    Flag to specify whether a result summary should be plotted
            
            Returns
            -------
                motif_converged_dataset: NxM numpy array of tuples where N is the number of motifs and M is the number
                    of nearest neighbors considered. Each tuple contains the converged parameters for a gaussian fit to
                    an atom in a motif window. 
            '''


            self.motif_guesses = []
            self.motif_parms = []
            self.motif_converged_parms = []
            self.fit_motifs = []
            fit_region = []

            # generate final dataset forms
            self.motif_guess_dataset = np.zeros(shape=(self.motif_centers.shape[0], self.num_nearest_neighbors + 1),
                                                dtype=self.motif_coeff_dtype)
            self.motif_converged_dataset = np.zeros(shape=(self.motif_centers.shape[0], self.num_nearest_neighbors + 1),
                                                    dtype=self.motif_coeff_dtype)

            for motif in range(len(self.motif_centers)):
                # get guesses
                self.motif_parms.append(self.do_guess(self.center_atom_indices[motif], initial_motifs=True))

                # pull out parameters for generating the gaussians
                coef_guess_mat = self.motif_parms[motif][1]
                s1 = self.motif_parms[motif][3].T
                s2 = self.motif_parms[motif][4].T
                fit_region.append(self.motif_parms[motif][2])

                # put guesses into final dataset form
                self.motif_guess_dataset[motif] = [tuple(element) for element in coef_guess_mat]

                # store the guess results for plotting
                self.motif_guesses.append(gauss2d(s1, s2, *coef_guess_mat, **self.fitting_parms))

                # fit the motif with num_nearest_neighbors + 1 gaussians
                parm_list = [self.motif_parms[motif], self.fitting_parms]
                fitting_results = do_fit(parm_list)

                # store the converged results
                self.motif_converged_parms.append(fitting_results)
                self.motif_converged_dataset[motif] = [tuple(element) for element in fitting_results]

                # store the images of the converged gaussians
                self.fit_motifs.append(gauss2d(s1, s2, *fitting_results, **self.fitting_parms))

                # calculate the relative atom positions (instead of absolute)
                fitting_results[:, 1] = fitting_results[:, 1] - self.motif_centers[motif][0]
                fitting_results[:, 2] = fitting_results[:, 2] - self.motif_centers[motif][1]

            # plot results if desired
            if plot_results:

                # initialize the figure
                fig, axes = plt.subplots(ncols=3, nrows=len(self.motif_centers), figsize=(14, 6 * len(self.motif_centers)))

                for i, ax_row in enumerate(np.atleast_2d(axes)):
                    # plot the original windows
                    ax_row[0].imshow(fit_region[i], interpolation='none',
                                     cmap=plot_utils.cmap_jet_white_center())
                    ax_row[0].set_title('Original Window')

                    # plot the initial guess windows
                    ax_row[1].imshow(self.motif_guesses[i], interpolation='none',
                                     cmap=plot_utils.cmap_jet_white_center())
                    ax_row[1].set_title('Initial Gaussian Guesses')

                    # plot the converged gaussians
                    ax_row[2].imshow(self.fit_motifs[i], interpolation='none',
                                     cmap=plot_utils.cmap_jet_white_center())
                    ax_row[2].set_title('Converged Gaussians')

                fig.show()

            return self.motif_converged_dataset


# if __name__=='__main__':
#     file_name = r"C:\Users\o2d\Documents\pycroscopy\\test_scripts\\testing_gauss_fit\image 04.h5"
#     folder_path, file_path = os.path.split(file_name)
#
#     file_base_name, file_extension = file_name.rsplit('.')
#     h5_file = h5py.File(file_name, mode='r+')
#     # look at the data tree in the h5
#     '''
#     # define a small function called 'print_tree' to look at the folder tree structure
#     def print_tree(parent):
#         print(parent.name)
#         if isinstance(parent, h5py.Group):
#             for child in parent:
#                 print_tree(parent[child])
#     '''
#
#     #print('Datasets and datagroups within the file:')
#     file_handle = h5_file
#     #print_tree(file_handle)
#
#     cropped_clean_image = h5_file['/Measurement_000/Channel_000/Raw_Data-Windowing_000/Image_Windows-SVD_000/U-Cluster_000/Labels-Atom_Finding_000/Cropped_Clean_Image']
#     atom_grp = cropped_clean_image.parent
#     guess_params = atom_grp['Guess_Positions']
#
#     num_nearest_neighbors = 4
#     psf_width = atom_grp.attrs['psf_width']
#     win_size = atom_grp.attrs['motif_win_size']
#
#     fitting_parms = {'fit_region_size': win_size * 0.5,  # region to consider when fitting
#                      'num_nearest_neighbors': num_nearest_neighbors,
#                      'sigma_guess': 3, # starting guess for gaussian standard deviation
#                      'position_range': win_size / 4,# range that the fitted position can go from initial guess position[pixels]
#                      'max_function_evals': 100,
#                      'fitting_tolerance': 1E-4,
#                      'symmetric': True,
#                      'background': True,
#                      'movement_allowance': 5.0} # percent of movement allowed (on some parameters)
#
#     foo = Gauss_Fit(atom_grp, fitting_parms)
#
#
