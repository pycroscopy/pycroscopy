import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import scipy.optimize as opt
from copy import deepcopy
from sklearn.mixture import GaussianMixture as GM


# simple helper function for intersecting lists
def intersection(list1, list2):
    list_intersected = [value for value in list1 if value in list2]
    return list_intersected


class LocalCrystallography:
    def __init__(self, image_source, atom_positions, atom_descriptors=dict(),
                 window_size=25, atom_types=None, border=0.03,
                 image_name='', comp=0, **kwargs):
        """
        Parameters
        ----------
        image_source
            Image from which N atoms or objects are derived
        atom_positions
            Positions of the objects/atoms as an (N,2) array
        atom_descriptors
            Dictionary with keys being object types and values being index
        window_size
            Roughly, half the size of the unit cell in pixdls, used for atomic refinement
        atom_types
            Default is None. List of size N with indices indicting atom or object type
        border
            (float between 0 and 1) border pixels size as a function of size of the image ()
        image_name
            Name of the image, default is blank
        comp
            Composition of the sample (optional)

            """

        self.image_source = image_source  # source dataset
        self.image_name = image_name  # text list of name of file
        self.atom_positions = atom_positions  # atomic positions as an Nx2 array
        self.num_atoms = len(self.atom_positions)
        # Pass a vector of length num_atoms with index of atom type, if system has
        # multiple types of atoms
        self.atom_types = atom_types
        if atom_types is None: self.atom_types = np.zeros(self.num_atoms)
        self.atom_descriptors = atom_descriptors
        self.neighbor_indices = None
        self.image_size_x, self.image_size_y = self.image_source.shape
        self.window_size = window_size  # size of window, roughly equal to lattice parameter in pixels
        self.border = border  # border is percentage of width to chop off
        self.determine_border_indices(border=border)
        self.neighborhood_results = None
        self.comp = comp  # composition, usually this is the amount of atom B in the binary mixture.

    def determine_border_indices(self, border=0.03):
        '''This will find the index of all the non-border and border atoms so it will not screw up
        the local crystallography analysis. Basically we find atoms nearest the edge and then move the 
        border*image_size px in, and if the atom is caught in that space, it gets marked as a border px'''

        border_pixel_ind = []
        nonborder_pixel_ind = []

        xlims = [border * self.image_source.shape[1], self.image_source.shape[1] - border * self.image_source.shape[1]]
        ylims = [border * self.image_source.shape[0], self.image_source.shape[1] - border * self.image_source.shape[0]]

        # print(xlims, ylims)

        for ind in range(len(self.atom_positions)):

            x, y = self.atom_positions[ind, 1], self.atom_positions[ind, 0]

            if x > xlims[0] and x < xlims[1] and y > ylims[0] and y < ylims[1]:
                nonborder_pixel_ind.append(ind)
            else:
                border_pixel_ind.append(ind)

        self.border_pixel_inds = border_pixel_ind
        self.nonborder_pixel_inds = nonborder_pixel_ind

        return

    def refine_atomic_positions(self):
        # This will refine the given atomic positions
        # Create fitting space

        x_vec = np.linspace(-self.window_size / 2, self.window_size / 2, self.window_size)
        y_vec = np.linspace(-self.window_size / 2, self.window_size / 2, self.window_size)

        x_mat, y_mat = np.meshgrid(x_vec, y_vec)

        # print(x_mat.shape)

        atomic_positions_corrections = np.zeros(self.atom_positions.shape, dtype=np.float32)
        num_atoms = self.atom_positions.shape[0]

        # do a fit of each atom using a gaussian
        opt_fits = []
        for k1 in range(0, num_atoms):

            ax = self.atom_positions[k1, 0]
            ay = self.atom_positions[k1, 1]
            t1 = int((ax > self.window_size))
            t2 = int(ax < (self.image_size_x - self.window_size))
            t3 = int((ay > self.window_size))
            t4 = int(ay < (self.image_size_y - self.window_size))

            if (t1 + t2 + t3 + t4) == 4:

                ROI = self.image_source[int(ay - self.window_size / 2):int(ay + self.window_size / 2),
                      int(ax - self.window_size / 2):int(ax + self.window_size / 2)]
                amp_guess = ROI[int(self.window_size / 2), int(self.window_size / 2)]

                initial_guess = (amp_guess, 0.1, 0.1, 2., 2., 0.001)
                guess = self.gauss_oval_2D((x_mat, y_mat), *initial_guess)

                guess_surface = guess.reshape(self.window_size, self.window_size)

                try:
                    popt, pcov = opt.curve_fit(self.gauss_oval_2D, (x_mat, y_mat), ROI.ravel(), p0=initial_guess)

                except RuntimeError:

                    popt = guess
                    pcov = [np.nan, np.nan, np.nan, np.nan, np.nan]
                    print('Failed Fit')

                # need to make sure x and y are correct
                if ~np.isnan(popt[1]) and ~np.isnan(popt[2]):

                    atomic_positions_corrections[k1, 0] = popt[1]
                    atomic_positions_corrections[k1, 1] = popt[2]
                else:
                    atomic_positions_corrections[k1, 0] = 0.0
                    atomic_positions_corrections[k1, 1] = 0.0
                opt_fits.append((popt, pcov))

        atomic_positions_corrected = self.atom_positions + atomic_positions_corrections
        self.atom_positions = atomic_positions_corrected
        self.corrections = atomic_positions_corrections

        return atomic_positions_corrected, opt_fits

    @staticmethod
    def gauss_oval_2D(fitting_space, amplitude, xo, yo, sigmax, sigmay, offset):
        x = fitting_space[0]
        y = fitting_space[1]
        xo = float(xo)
        yo = float(yo)
        g = amplitude * np.exp(-((x - xo) ** 2 / (2 * sigmax ** 2) + ((y - yo) ** 2 / (2 * sigmay ** 2))));

        return g.ravel() + offset

    def compute_neighborhood_indices(self, num_neighbors=8):
        '''
        Computes the local neighbors, returning the indices for each atom
        
        Input: - num_neighbors (int) (Default = 8): Number of neighbors
        
        Output: (None), results are stored as matrix of size (num_atoms, num_neighbors) in neighbor_indices.
        '''

        # finds indices of the neighbors
        nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='brute').fit(self.atom_positions)
        distance_vec, full_index = nbrs.kneighbors(self.atom_positions)
        self.neighbor_indices = full_index  # matrix of size (num_atoms, num_neighbors) with indices of neighbors
        # Once we have the neighbor indices, we should also just write an array of the atom types by neighbor as this will be needed for local analysis down the line!

        neighbor_types_mat = []
        for ind in range(self.neighbor_indices.shape[0]):
            n_indices = self.neighbor_indices[ind, :]
            neighbor_types_mat.append(self.atom_types[n_indices])

        self.neighborhood_atom_types = np.array(neighbor_types_mat)

        return

    def compute_neighborhood(self, num_neighbors=8, atom_type=None):
        '''
        Computes the distance and angle to local neighbors.
        
        Inputs: -num_neighbors (int) (default = 8): Number of neighbors
                - atom_type: (int or str of key in atom_descriptors) (Default = None
                ). If provided, neighborhood will be computed for the atom type given
                If none, then atom types will be ignored.
                
        Output: None, but a dictionary of results is stored as neighborhood_results
        
        Note that the border pixels will be ignored in the neighborhood calculation.
        It is also reccomended to run compute_neighborhood_indices before this, but it will 
        run anyway if it has not already.
        
        '''

        if self.neighbor_indices is None:
            self.compute_neighborhood_indices(num_neighbors)
        else:
            if self.neighbor_indices.shape[-1] != num_neighbors + 1:
                print('Warning - number of neighbors given to this function differs ' + \
                      'from previously computed, will recompute with {} neighbors'.format(num_neighbors))

                self.compute_neighborhood_indices(num_neighbors)

        if atom_type is not None:
            if type(atom_type == 'str'): atom_type = self.atom_descriptors[atom_type]
            # num_atoms = len(self.atom_types==atom_type)
            num_atoms = len(self.atom_types)
        else:
            atom_type = 0

        atom_types = self.atom_types
        num_atoms = self.num_atoms

        d_vec = np.zeros((1, num_neighbors), dtype=float)
        a_vec = d_vec.copy()
        xd_vec = d_vec.copy()
        yd_vec = d_vec.copy()
        d_mat = np.zeros((num_atoms, num_neighbors), dtype=float)
        a_mat = d_mat.copy()
        xd_mat = d_mat.copy()
        yd_mat = d_mat.copy()
        atom_neighbor_pos = np.zeros((num_atoms, num_neighbors, 2))

        # build a matrix of measurements from each atom to its nearest neighbors
        for k1 in range(0, num_atoms):

            # compute it only if the atom is a nonborder atom
            if k1 in self.nonborder_pixel_inds:

                # Finally, compute only if the atom type matches
                if atom_types[k1] == atom_type:

                    x0 = self.atom_positions[k1, 0]
                    y0 = self.atom_positions[k1, 1]

                    if ((int(x0 == 0) + int(y0 == 0)) == 0):

                        for k2 in range(0, num_neighbors):
                            x1 = self.atom_positions[self.neighbor_indices[k1, k2 + 1], 0]
                            y1 = self.atom_positions[self.neighbor_indices[k1, k2 + 1], 1]
                            d_vec[0, k2] = np.abs((x0 - x1) + 1j * (
                                        y0 - y1))  # array of distances from each atom to its nearest neighbors
                            a_vec[0, k2] = np.angle(
                                (x0 - x1) + 1j * (y0 - y1))  # array of angles from each atom to its nearest neighbors
                            xd_vec[0, k2] = (x0 - x1)
                            yd_vec[0, k2] = (y0 - y1)
                            atom_neighbor_pos[k1, k2, :] = [x1, y1]

                            # sort neighbors based on angle
                        sort_ind = np.argsort(a_vec[0, :], axis=None)
                        d_mat[k1, :] = d_vec[0, sort_ind]
                        a_mat[k1, :] = a_vec[0, sort_ind]
                        xd_mat[k1, :] = xd_vec[0, sort_ind]
                        yd_mat[k1, :] = yd_vec[0, sort_ind]

        neighborhood_results = {'distance_mat': d_mat, 'angles_mat': a_mat,
                                'xdistance_mat': xd_mat, 'ydistance_mat': yd_mat,
                                'atom_neighbor_positions': atom_neighbor_pos,
                                'atom_type': atom_type}

        self.neighborhood_results = deepcopy(neighborhood_results)

        self.atom_neighbor_positions = atom_neighbor_pos

        return neighborhood_results

    def compute_pca_of_neighbors(self):
        '''
        PCA of neighorhood based on results computed from compute_neighborhood
        This means that the results will depend on the settings chosen there.
        So for instance if you change number of neighbors or atom type to look for, 
        then this will be reflected once you execute this method.
        
        Inputs: - (None)
        Output: - (list) (Handles to three figures with the results).
        '''

        if self.neighborhood_results is None:
            raise ValueError("Warning, neighborhood hasn't been computed yet. Run compute_neighborhood() first")

        # Access the data

        # we need to pick the overlap of sets between the atom_idx of the chosen type,
        # and the nonborder pixel indices
        atom_type = self.neighborhood_results['atom_type']
        atom_idx = np.where(self.atom_types == atom_type)[0]
        atoms_to_select = intersection(atom_idx, self.nonborder_pixel_inds)

        xd_mat = self.neighborhood_results['xdistance_mat'][atoms_to_select]
        yd_mat = self.neighborhood_results['ydistance_mat'][atoms_to_select]
        d_mat = self.neighborhood_results['distance_mat'][atoms_to_select]
        a_mat = self.neighborhood_results['angles_mat'][atoms_to_select]

        fig0, ax0 = plt.subplots(figsize=(8, 8))
        ax0.scatter(xd_mat, yd_mat, c='k', s=2)
        plt.axis('equal')

        [ud, sd, vd] = np.linalg.svd(d_mat, full_matrices=0)  # SVD on radial distance of nearest neighbors
        [ua, sa, va] = np.linalg.svd(a_mat, full_matrices=0)  # SVD on anglular displacement of nearest neighbors

        [us, ss, vs] = np.linalg.svd(xd_mat + 1j * yd_mat,
                                     full_matrices=0)  # SVD on relative position of nearest neighbors (complex valued)

        # convert complex results of
        usm = np.abs(us)
        usa = np.angle(us)
        usx = np.real(us)
        usy = np.imag(us)

        vsm = np.abs(vs)
        vsa = np.angle(vs)
        vsx = np.real(vs)
        vsy = np.imag(vs)

        # plot eigenvalue radial displacements
        fig1, axs1 = plt.subplots(2, 3, figsize=(12, 8), facecolor='black', edgecolor='w')
        fig1.subplots_adjust(hspace=.1, wspace=.1)
        axs1 = axs1.ravel()
        for k1 in range(0, min(6, ud.shape[-1])):
            up = ud[:, k1]
            up = up - up.min()
            up = up / up.max()
            colors = plt.cm.bwr(up)

            axs1[k1].imshow(1 - self.image_source, cmap='Greys',
                            interpolation='none', alpha=1)

            axs1[k1].scatter(self.atom_positions[atoms_to_select][:, 0],
                             self.atom_positions[atoms_to_select][:, 1], c=colors, s=8)

        # plot eigenvector maps of radial displacement eigenvectors
        fig2, axs2 = plt.subplots(2, 3, figsize=(12, 8))
        fig2.subplots_adjust(hspace=.2, wspace=.2)
        axs2 = axs2.ravel()
        for k1 in range(0, min(6, vd.shape[0])):
            axs2[k1].plot(0, 0, 'bo')
            axs2[k1].plot(vsx[0, :], vsy[0, :], 'ro')
            axs2[k1].axis('equal')
            c = 1.0
            axs2[k1].axis([-c, c, -c, c])
            axs2[k1].set_title(str(k1))
            axs2[k1].quiver(vsx[0, :], vsy[0, :], vsx[0, :] * vd[k1, :], vsy[0, :] * vd[k1, :])

        return [fig0, fig1, fig2], ud

    def compute_kmeans_neighbors(self, num_clusters=4):
        '''
        K-Means of neighorhood based on results computed from compute_neighborhood
        This means that the results will depend on the settings chosen there.
        So for instance if you change number of neighbors or atom type to look for, 
        then this will be reflected once you execute this method.
        
        Inputs: - num_clusters (int) (Default = 4): number of clusters in the k-means decomposition
        Output: - (list) Results arranged as ([cluster_centers, labels, figure_handle]).
        
        '''

        atom_type = self.neighborhood_results['atom_type']
        atom_idx = np.where(self.atom_types == atom_type)[0]
        atoms_to_select = intersection(atom_idx, self.nonborder_pixel_inds)

        xd_mat2 = self.neighborhood_results['xdistance_mat'][atoms_to_select]
        yd_mat2 = self.neighborhood_results['ydistance_mat'][atoms_to_select]

        zd_mat2 = np.zeros(shape=(xd_mat2.shape[0] - 1, xd_mat2.shape[1], 2))
        zd_mat2[:, :, 0] = xd_mat2[:-1, :]
        zd_mat2[:, :, 1] = yd_mat2[:-1, :]
        zd_mat2 = zd_mat2.reshape(xd_mat2.shape[0] - 1, -1)

        km = KMeans(n_clusters=num_clusters)
        km.fit(zd_mat2)
        cluster_centroids = km.cluster_centers_
        labels = km.labels_

        my_colors = ['r', 'b', 'c', 'k', 'g', 'y', 'gray', 'm', '#eeefff',
                     '#eeefff', '#eeefff', '#eeefff', '#eeefff', '#eeefff']

        fig, axes = plt.subplots(figsize=(10, 10))
        axes.imshow(self.image_source)
        axes.set_title('KMeans Labels', fontsize=18)

        for i in range(min(len(self.atom_positions[atoms_to_select]), len(labels))):
            x, y = self.atom_positions[atoms_to_select][i]
            axes.scatter(x, y, c=my_colors[labels[i]])

        if num_clusters <= 4:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
        elif num_clusters > 4 and num_clusters <= 6:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 4))
        elif num_clusters > 6:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

        for ind, ax in enumerate(axes.flat):
            if ind < num_clusters:
                pt = cluster_centroids[ind, :]
                xpts = pt[::2]
                ypts = pt[1::2]
                my_color = my_colors[ind]
                ax.scatter(xpts, ypts, s=50, c=my_color)
                ax.scatter(0, 0, c='b', s=40)
                ax.set_xlim(-100, 100)
                ax.set_ylim(-100, 100)
                ax.set_title('Cluster {}'.format(ind), fontsize=14)
        fig.tight_layout()

        # Plot as point clouds

        if num_clusters <= 4:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        elif num_clusters > 4 and num_clusters <= 6:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        elif num_clusters > 6:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

        for ind, ax in enumerate(axes.flat):
            if ind < num_clusters:
                cluster_pts = zd_mat2[labels == ind, :]
                my_color = my_colors[ind]
                for atom_index in range(cluster_pts.shape[0]):
                    xpts = cluster_pts[atom_index, ::2]
                    ypts = cluster_pts[atom_index, 1::2]
                    ax.scatter(xpts, ypts, color=my_color, s=2)

                    ax.set_xlim(-100, 100)
                    ax.set_ylim(-100, 100)

            ax.set_title('Cluster {} Point Cloud'.format(ind), fontsize=14)
        fig.tight_layout()

        return [cluster_centroids, labels, fig]
