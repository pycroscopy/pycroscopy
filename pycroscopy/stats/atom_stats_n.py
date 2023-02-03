# sys.path.append(r'C:\Users\4sv\PycharmProjects\sidpy')
# sys.path.append(r'C:\Users\4sv\PycharmProjects\pycroscopy')

import numpy as np
import sidpy as sid
from scipy import spatial


# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# import scipy.optimize as opt
# from copy import deepcopy
# from sklearn.mixture import GaussianMixture as GM
# import pycroscopy as px
# import ase
# from ase import Atom, Atoms


class LocalCrystallography_n:
    def __init__(self, dataset, struct=None, window_size=25, border=0.03,
                 image_name='', comp=0, **kwargs):
        """
        Parameters
        ----------
        dset
            Sidpy.Dataset with structure attribute
        struct
            str, Name of the structure to get the atomic positions and types,
            Default is None, the first structure in the
        window_size
            Roughly, half the size of the unit cell in pixels, used for atomic refinement
        border
            (float between 0 and 1) border pixels size as a function of size of the image
        image_name
            Name of the image, default is blank
        comp
            Composition of the sample (optional)
            """
        if isinstance(dataset, sid.Dataset):
            if hasattr(dataset, 'structures'):
                self.dset = dataset  # source dataset
            else:
                # What happens if we don't have a structures dictionary
                pass
        else:
            raise TypeError('The dataset provided is expected to be of the type sidpy.Dataset,'
                            'but {} is provided'.format(type(dataset)))

        if struct is not None:
            if struct in self.dset.structures.keys():
                self.struct = struct
            else:
                raise KeyError('The provided struct is not part of the structures dictionary of the dataset')
        else:
            self.struct = self.dset.structures.keys()[0]
            raise Warning('No structure has been provided. The first structure: {} will be used to compute the '
                          'statistics'.format(self.struct))

        self.image_name = image_name  # text list of name of file
        self.neighbor_indices = None
        self.image_size_x, self.image_size_y = self.dset.shape

        if isinstance(window_size, int):
            self.window_size = window_size  # size of window, roughly equal to lattice parameter in pixels
        else:
            try:
                self.window_size = int(window_size)
                raise Warning('The window size parameter is expected to of the type int, but received'
                              '{}, we will use int(window_size) = {} as the updated '
                              'window size'.format(type(window_size), int(window_size)))
            except:
                raise TypeError('The window size parameter is expected to of the type int, but received '
                                'type :{}'.format(type(window_size)))

        self.border = border  # border is percentage of width to chop off
        self.determine_border_indices(border=border)
        self.neighborhood_results = None
        self.comp = comp  # composition, usually this is the amount of atom B in the binary mixture.

    def determine_border_indices(self, border_prop=None, border_ind=None):
        """This method will find the indices of all the non-border and border atoms so it will not screw up
        the local crystallography analysis. Basically we find atoms nearest the edge and then move the
        border*image_size px in, and if the atom is caught in that space, it gets marked as a border px

        Parameters
        ----------
        Requires either border_prop or border_ind, ignores border_ind when both are provided

        border_prop: float or List[float, float] 
            proportion of the image size to be used as the border
            when given float, uses the same value for x and y axes
            When given list, the first item is used for x-axis or the y-axis
        border_ind: int or List[int, int]
            Number of indices to be used as the border
            when given int, uses the same value for x and y axes
            When given list, the first item is used for x-axis or the y-axis
        
        Returns: None
        ----------
            Stores border_pixel_inds and nonborder_pixel_ind as the attributes to the dataset

        """

        if border_prop is not None:
            if isinstance(border_prop, float):
                b_p = [border_prop, border_prop]
            elif isinstance(border_prop, list):
                b_p = border_prop
            b_inds = [b_p[0] * self.dset.shape[1], border_ind * self.dset.shape[0]]

        if border_prop is None and border_ind is not None:
            if isinstance(border_ind, int):
                b_inds = [border_ind, border_ind]
            elif isinstance(border_ind, list):
                b_inds = border_ind

        xlims = [b_inds[0] * self.dset.shape[1], self.dset.shape[1] - b_inds[0] * self.dset.shape[1]]
        ylims = [b_inds[1] * self.dset.shape[0], self.dset.shape[1] - b_inds[1] * self.dset.shape[0]]

        border_pixel_ind = []
        nonborder_pixel_ind = []

        for ind in range(len(self.dset.structures[self.struct].positions)):

            x, y = self.dset.structures[self.struct].positions[ind, 1], \
                   self.dset.structures[self.struct].positions[ind, 0]

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
        Computes the local neighbors, returning the indices and atom types for the neighborhood atoms

        Parameters
        ----------
            num_neighbors (int) (Default = 8): Number of neighbors

            Returns: (None), results are stored as neighbor_indices and neighborhood_atom_types attributes.
        """
        atom_positions = self.dset.structures[self.struct].get_positions()[:, 0:2]
        atom_types = self.dset.structures[self.struct].get_atomic_numbers()
        # finds indices of the neighbors
        tree = spatial.cKDTree(atom_positions)
        dists, neighbors = tree.query(atom_positions, k=num_neighbors + 1)
        self.dset.neighbor_indices = neighbors[:, 1::]  # Indices of the neighbors
        # atomic numbers of the neighboring atoms
        self.dset.neighborhood_atom_types = atom_types[self.dset.neighbor_indices]
