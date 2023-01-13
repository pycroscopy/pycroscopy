import sys

sys.path.append(r'C:\Users\4sv\PycharmProjects\sidpy')
sys.path.append(r'C:\Users\4sv\PycharmProjects\pycroscopy')

import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# import scipy.optimize as opt
# from copy import deepcopy
# from sklearn.mixture import GaussianMixture as GM
# import sidpy as sid
# import pycroscopy as px
# import ase
# from ase import Atom, Atoms
from scipy import spatial


class LocalCrystallography_n:
    def __init__(self, dset, window_size=25, border=0.03,
                 image_name='', comp=0, **kwargs):
        """
        Parameters
        ----------
        dset
            Sidpy.Dataset with structure attribute
        window_size
            Roughly, half the size of the unit cell in pixels, used for atomic refinement
        border
            (float between 0 and 1) border pixels size as a function of size of the image
        image_name
            Name of the image, default is blank
        comp
            Composition of the sample (optional)
            """

        self.dset = dset  # source dataset
        self.image_name = image_name  # text list of name of file
        self.neighbor_indices = None
        self.image_size_x, self.image_size_y = self.dset.shape
        self.window_size = window_size  # size of window, roughly equal to lattice parameter in pixels
        self.border = border  # border is percentage of width to chop off
        self.determine_border_indices(border=border)
        self.neighborhood_results = None
        self.comp = comp  # composition, usually this is the amount of atom B in the binary mixture.

    def determine_border_indices(self, border=0.03):
        """This will find the indices of all the non-border and border atoms so it will not screw up
        the local crystallography analysis. Basically we find atoms nearest the edge and then move the
        border*image_size px in, and if the atom is caught in that space, it gets marked as a border px"""

        border_pixel_ind = []
        nonborder_pixel_ind = []

        xlims = [border * self.dset.shape[1], self.dset.shape[1] - border * self.dset.shape[1]]
        ylims = [border * self.dset.shape[0], self.dset.shape[1] - border * self.dset.shape[0]]

        # print(xlims, ylims)

        for ind in range(len(self.dset.structures['initial'].positions)):

            x, y = self.dset.structures['initial'].positions[ind, 1], self.dset.structures['initial'].positions[ind, 0]

            if xlims[0] < x < xlims[1] and ylims[0] < y < ylims[1]:
                nonborder_pixel_ind.append(ind)
            else:
                border_pixel_ind.append(ind)

        self.dset.border_pixel_inds = border_pixel_ind
        self.dset.nonborder_pixel_inds = nonborder_pixel_ind

    @staticmethod
    def gauss_oval_2D(fitting_space, amplitude, xo, yo, sigmax, sigmay, offset):
        x = fitting_space[0]
        y = fitting_space[1]
        xo = float(xo)
        yo = float(yo)
        g = amplitude * np.exp(-((x - xo) ** 2 / (2 * sigmax ** 2) + ((y - yo) ** 2 / (2 * sigmay ** 2))));

        return g.ravel() + offset

    def compute_neighborhood_indices(self, num_neighbors=8):
        """
        Computes the local neighbors, returning the indices for each atom

        Input: - num_neighbors (int) (Default = 8): Number of neighbors

        Output: (None), results are stored as matrix of size (num_atoms, num_neighbors) in neighbor_indices.
        """
        atom_positions = self.dset.structures['initial'].get_positions()[:, 0:2]
        atom_types = self.dset.structures['initial'].get_atomic_numbers()
        # finds indices of the neighbors
        tree = spatial.cKDTree(atom_positions)
        dists, neighbors = tree.query(atom_positions, k=num_neighbors + 1)
        self.dset.neighbor_indices = neighbors[:, 1::]  # Indices of the neighbors
        # atomic numbers of the neighboring atoms
        self.dset.neighborhood_atom_types = atom_types[self.dset.neighbor_indices]
