# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Chris Smith
"""
import numpy as np
import os

beps_image_folder = os.path.abspath(os.path.join(os.path.realpath(__file__), '../beps_data_gen_images'))


def combine_in_out_field_loops(in_vec, out_vec):
    """
    Stack the in-field and out-of-field loops

    Parameters
    ----------
    in_vec : numpy.ndarray
        1d array of in-field values
    out_vec : numpy.ndarray
        1d array of out-of-field values

    Returns
    -------
    field_mat : numpy.ndarray
        2d array of combined in-field and out-of-field vectors

    """
    return np.vstack((in_vec, out_vec))


def build_loop_from_mat(loop_mat, num_steps):
    """


    Parameters
    ----------
    loop_mat
    num_steps

    Returns
    -------

    """
    return np.vstack((loop_mat[0, :int(num_steps / 4) + 1],
                      loop_mat[1],
                      loop_mat[0, int(num_steps / 4) + 1: int(num_steps / 2)]))


def get_noise_vec(num_pts, noise_coeff):
    """
    Calculate a multiplicative noise vector from the `noise_coeff`

    Parameters
    ----------
    num_pts : uint
        number of points in the vector
    noise_coeff : float
        Noise coefficient that determines the variation in the noise vector

    Returns
    -------
    noise_vec : numpy.ndarray
        1d noise vector array

    """
    return np.ones(num_pts) * (1 + 0.5 * noise_coeff) - np.random.random(num_pts) * noise_coeff
