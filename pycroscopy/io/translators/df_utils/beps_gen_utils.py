

import numpy as np

def combine_in_out_field_loops(in_vec, out_vec):
    """

    Parameters
    ----------
    in_vec
    out_vec

    Returns
    -------

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

    Parameters
    ----------
    num_pts
    noise_coeff

    Returns
    -------

    """
    return np.ones(num_pts) * (1 + 0.5 * noise_coeff) - np.random.random(num_pts) * noise_coeff
