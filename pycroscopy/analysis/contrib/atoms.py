import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import scipy.optimize as opt
from copy import deepcopy
from sklearn.mixture import GaussianMixture as GM
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch
import warnings
pyro.set_rng_seed(1)

#simple helper function for intersecting lists
def intersection(list1, list2): 
    list_intersected = [value for value in list1 if value in list2] 
    return list_intersected

#Main class
class Atoms:
    def __init__(self, image_source, atom_positions, atom_descriptors = dict(), 
                 window_size = 25, atom_types = None, border = 0.03, 
                 image_name = '', comp = 0, **kwargs):
        
        self.image_source = image_source #source dataset
        self.image_name = image_name#text list of name of file
        self.atom_positions = atom_positions #atomic positions as an Nx2 array
        self.num_atoms = len(self.atom_positions)
        self.atom_types = atom_types #Pass a vector of length num_atoms with index of atom type, if system has multiple types of atoms
        if atom_types is None: self.atom_types = np.zeros(self.num_atoms)
        self.atom_descriptors = atom_descriptors
        self.neighbor_indices = None
        self.image_size_x, self.image_size_y = self.image_source.shape
        self.window_size = window_size #size of window, roughly equal to lattice parameter in pixels
        self.border = border #border is percentage of width to chop off
        self.determine_border_indices(border = border)
        self.neighborhood_results = None
        self.comp = comp #composition, usually this is the amount of atom B in the binary mixture.
        
    def determine_border_indices(self, border=0.03):
        '''This will find the index of all the non-border and border atoms so it will not screw up
        the local crystallography analysis. Basically we find atoms nearest the edge and then move the 
        border*image_size px in, and if the atom is caught in that space, it gets marked as a border px'''
        
        border_pixel_ind = []
        nonborder_pixel_ind = []
        
        xlims = [border*self.image_source.shape[1], self.image_source.shape[1]-border*self.image_source.shape[1]]
        ylims = [border*self.image_source.shape[0], self.image_source.shape[1]-border*self.image_source.shape[0]]
        
        #print(xlims, ylims)
        
        for ind in range(len(self.atom_positions)):
            
            x,y = self.atom_positions[ind,1], self.atom_positions[ind,0]
            
            if x>xlims[0] and x<xlims[1] and y>ylims[0] and y<ylims[1]:
                nonborder_pixel_ind.append(ind)
            else:
                border_pixel_ind.append(ind)
                
        self.border_pixel_inds = border_pixel_ind
        self.nonborder_pixel_inds = nonborder_pixel_ind
        
        return
        
        
    def refine_atomic_positions(self):
        #This will refine the given atomic positions
        # Create fitting space
       
        
        x_vec = np.linspace(-self.window_size/2, self.window_size/2, self.window_size)
        y_vec = np.linspace(-self.window_size/2, self.window_size/2, self.window_size)
        
        x_mat, y_mat = np.meshgrid(x_vec, y_vec)
       
        #print(x_mat.shape)

        atomic_positions_corrections = np.zeros(self.atom_positions.shape, dtype = np.float32)
        num_atoms = self.atom_positions.shape[0]
        
        # do a fit of each atom using a gaussian
        opt_fits = []
        for k1 in range(0,num_atoms):
            
            ax = self.atom_positions[k1,0]
            ay = self.atom_positions[k1,1]
            t1 = int((ax>self.window_size)) 
            t2 = int(ax<(self.image_size_x-self.window_size))
            t3 = int((ay>self.window_size)) 
            t4 = int(ay<(self.image_size_y-self.window_size))

            if (t1+t2+t3+t4)==4:
                
                ROI =  self.image_source[int(ay-self.window_size/2):int(ay+self.window_size/2),
                                    int(ax-self.window_size/2):int(ax+self.window_size/2)]
                amp_guess = ROI[int(self.window_size/2),int(self.window_size/2)]
                

                initial_guess = (amp_guess, 0.1, 0.1, 2., 2., 0.001)
                guess = self.gauss_oval_2D((x_mat, y_mat), *initial_guess)
                    
                guess_surface = guess.reshape(self.window_size,self.window_size)
                
                try:
                    popt, pcov = opt.curve_fit(self.gauss_oval_2D, (x_mat, y_mat), ROI.ravel(), p0=initial_guess)
                        
                except RuntimeError:

                    popt=guess
                    pcov=[np.nan, np.nan, np.nan, np.nan, np.nan]
                    print('Failed Fit')
                    
                # need to make sure x and y are correct
                if ~np.isnan(popt[1]) and ~np.isnan(popt[2]):

                    atomic_positions_corrections[k1,0] = popt[1]
                    atomic_positions_corrections[k1,1] = popt[2]
                else:
                    atomic_positions_corrections[k1,0] = 0.0
                    atomic_positions_corrections[k1,1] = 0.0
                opt_fits.append((popt,pcov))
        
        
        atomic_positions_corrected = self.atom_positions + atomic_positions_corrections
        self.atom_positions = atomic_positions_corrected
        self.corrections = atomic_positions_corrections

        return atomic_positions_corrected , opt_fits
    
    @staticmethod 
    def gauss_oval_2D(fitting_space, amplitude, xo, yo, sigmax, sigmay, offset):
        x = fitting_space[0]
        y = fitting_space[1]
        xo = float(xo)
        yo = float(yo)    
        g = amplitude*np.exp(-((x-xo)**2/(2*sigmax**2) + ((y-yo)**2 /(2*sigmay**2))));

        return g.ravel() + offset
                            

    def compute_neighborhood_indices (self, num_neighbors = 8):
        '''
        Computes the local neighbors, returning the indices for each atom
        
        Input: - num_neighbors (int) (Default = 8): Number of neighbors
        
        Output: (None), results are stored as matrix of size (num_atoms, num_neighbors) in neighbor_indices.
        '''
        
        #finds indices of the neighbors
        nbrs = NearestNeighbors(n_neighbors=num_neighbors+1, algorithm='brute').fit(self.atom_positions)
        distance_vec, full_index = nbrs.kneighbors(self.atom_positions)
        self.neighbor_indices = full_index #matrix of size (num_atoms, num_neighbors) with indices of neighbors
        #Once we have the neighbor indices, we should also just write an array of the atom types by neighbor as this will be needed for local analysis down the line!
        
        neighbor_types_mat = []
        for ind in range(self.neighbor_indices.shape[0]):
            n_indices = self.neighbor_indices[ind,:]
            neighbor_types_mat.append(self.atom_types[n_indices])
       
        self.neighborhood_atom_types = np.array(neighbor_types_mat)
        
        return 
    
    def compute_neighborhood (self, num_neighbors=8, atom_type = None):
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
            if self.neighbor_indices.shape[-1] != num_neighbors+1:
                print('Warning - number of neighbors given to this function differs ' + \
                      'from previously computed, will recompute with {} neighbors'.format(num_neighbors))
                
                self.compute_neighborhood_indices(num_neighbors)
        
        
            
        if atom_type is not None: 
            if type(atom_type=='str'): atom_type = self.atom_descriptors[atom_type]
            #num_atoms = len(self.atom_types==atom_type)
            num_atoms = len(self.atom_types)
        else:
            atom_type = 0
            
        atom_types = self.atom_types
        num_atoms = self.num_atoms
        
        d_vec = np.zeros((1,num_neighbors),dtype=float)
        a_vec = d_vec.copy()
        xd_vec = d_vec.copy()
        yd_vec = d_vec.copy()
        d_mat = np.zeros((num_atoms,num_neighbors),dtype=float)
        a_mat = d_mat.copy()
        xd_mat = d_mat.copy()
        yd_mat = d_mat.copy()
        atom_neighbor_pos = np.zeros((num_atoms,num_neighbors,2))
        
        #build a matrix of measurements from each atom to its nearest neighbors
        for k1 in range(0,num_atoms):
            
            #compute it only if the atom is a nonborder atom
            if k1 in self.nonborder_pixel_inds:
                
                #Finally, compute only if the atom type matches
                if atom_types[k1]==atom_type:
                    
                    x0 = self.atom_positions[k1,0]
                    y0 = self.atom_positions[k1,1]

                    if ((int(x0==0)+int(y0==0))==0):

                        for k2 in range(0,num_neighbors):            
                            x1 = self.atom_positions[self.neighbor_indices[k1,k2+1],0]
                            y1 = self.atom_positions[self.neighbor_indices[k1,k2+1],1]
                            d_vec[0,k2] = np.abs((x0-x1)+1j*(y0-y1)) #array of distances from each atom to its nearest neighbors
                            a_vec[0,k2] = np.angle((x0-x1)+1j*(y0-y1)) #array of angles from each atom to its nearest neighbors
                            xd_vec[0,k2] = (x0-x1)
                            yd_vec[0,k2] = (y0-y1)
                            atom_neighbor_pos[k1,k2,:] =[x1,y1] 

                        # sort neighbors based on angle
                        sort_ind = np.argsort(a_vec[0,:],axis=None)
                        d_mat[k1,:] = d_vec[0,sort_ind]
                        a_mat[k1,:] = a_vec[0,sort_ind]
                        xd_mat[k1,:] = xd_vec[0,sort_ind]
                        yd_mat[k1,:] = yd_vec[0,sort_ind]

        neighborhood_results = {'distance_mat':d_mat, 'angles_mat':a_mat,
                                        'xdistance_mat':xd_mat, 'ydistance_mat':yd_mat, 
                                         'atom_neighbor_positions': atom_neighbor_pos,
                                    'atom_type':atom_type}
        
        self.neighborhood_results = deepcopy(neighborhood_results)
        
        self.atom_neighbor_positions  = atom_neighbor_pos
        
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

        #Access the data
        
        #we need to pick the overlap of sets between the atom_idx of the chosen type, 
        #and the nonborder pixel indices
        atom_type = self.neighborhood_results['atom_type']
        atom_idx = np.where(self.atom_types==atom_type)[0]        
        atoms_to_select = intersection(atom_idx, self.nonborder_pixel_inds)
        
        xd_mat = self.neighborhood_results['xdistance_mat'][atoms_to_select]
        yd_mat = self.neighborhood_results['ydistance_mat'][atoms_to_select]
        d_mat = self.neighborhood_results['distance_mat'][atoms_to_select]
        a_mat = self.neighborhood_results['angles_mat'][atoms_to_select]
        
        
        fig0, ax0 = plt.subplots(figsize=(8,8))
        ax0.scatter(xd_mat,yd_mat,c='k',s=2)
        plt.axis('equal')

        [ud,sd,vd] = np.linalg.svd(d_mat,full_matrices=0) #SVD on radial distance of nearest neighbors
        [ua,sa,va] = np.linalg.svd(a_mat,full_matrices=0) #SVD on anglular displacement of nearest neighbors

        [us,ss,vs] = np.linalg.svd(xd_mat+1j*yd_mat,full_matrices=0) #SVD on relative position of nearest neighbors (complex valued)

        #convert complex results of 
        usm = np.abs(us)
        usa = np.angle(us)
        usx = np.real(us)
        usy = np.imag(us)

        vsm = np.abs(vs)
        vsa = np.angle(vs)
        vsx = np.real(vs)
        vsy = np.imag(vs)
        
        # plot eigenvalue radial displacements
        fig1, axs1 = plt.subplots(2,3, figsize=(12, 8), facecolor='black', edgecolor='w')
        fig1.subplots_adjust(hspace = .1, wspace=.1)
        axs1 = axs1.ravel()
        for k1 in range(0,min(6,ud.shape[-1])):
            up = ud[:,k1]
            up = up-up.min()
            up = up/up.max()
            colors = plt.cm.bwr(up)
            
            axs1[k1].imshow(1-self.image_source, cmap='Greys', 
                            interpolation='none',alpha=1)
            
            axs1[k1].scatter(self.atom_positions[atoms_to_select][:,0],
                             self.atom_positions[atoms_to_select][:,1],c=colors,s=8)

        # plot eigenvector maps of radial displacement eigenvectors
        fig2, axs2 = plt.subplots(2,3, figsize=(12, 8))
        fig2.subplots_adjust(hspace = .2, wspace=.2)
        axs2 = axs2.ravel()
        for k1 in range(0,min(6,vd.shape[0])):
            axs2[k1].plot(0,0,'bo')
            axs2[k1].plot(vsx[0,:],vsy[0,:],'ro')
            axs2[k1].axis('equal')
            c = 1.0
            axs2[k1].axis([-c, c, -c, c])
            axs2[k1].set_title(str(k1))
            axs2[k1].quiver(vsx[0,:],vsy[0,:],vsx[0,:]*vd[k1,:],vsy[0,:]*vd[k1,:])

        return [fig0, fig1, fig2], ud
    
    def compute_kmeans_neighbors(self, num_clusters = 4):
        '''
        K-Means of neighorhood based on results computed from compute_neighborhood
        This means that the results will depend on the settings chosen there.
        So for instance if you change number of neighbors or atom type to look for, 
        then this will be reflected once you execute this method.
        
        Inputs: - num_clusters (int) (Default = 4): number of clusters in the k-means decomposition
        Output: - (list) Results arranged as ([cluster_centers, labels, figure_handle]).
        
        '''
    
        atom_type = self.neighborhood_results['atom_type']
        atom_idx = np.where(self.atom_types==atom_type)[0]        
        atoms_to_select = intersection(atom_idx, self.nonborder_pixel_inds)
        
        xd_mat2 = self.neighborhood_results['xdistance_mat'][atoms_to_select]
        yd_mat2 = self.neighborhood_results['ydistance_mat'][atoms_to_select]
        
        zd_mat2 = np.zeros(shape=(xd_mat2.shape[0]-1, xd_mat2.shape[1],2))
        zd_mat2[:,:,0] = xd_mat2[:-1,:]
        zd_mat2[:,:,1] = yd_mat2[:-1,:]
        zd_mat2 = zd_mat2.reshape(xd_mat2.shape[0]-1, -1)

        km = KMeans(n_clusters = num_clusters)
        km.fit(zd_mat2)
        cluster_centroids = km.cluster_centers_
        labels=km.labels_
  
        my_colors = ['r', 'b', 'c', 'k', 'g', 'y', 'gray', 'm', '#eeefff',
                    '#eeefff', '#eeefff', '#eeefff', '#eeefff','#eeefff']

        fig, axes = plt.subplots(figsize =(10,10))
        axes.imshow(self.image_source)
        axes.set_title('KMeans Labels', fontsize = 18)
        
        for i in range(min(len(self.atom_positions[atoms_to_select]),len(labels))):
            x,y = self.atom_positions[atoms_to_select][i]
            axes.scatter(x,y,c = my_colors[labels[i]])

        if num_clusters <=4:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (4,4))
        elif num_clusters>4 and num_clusters<=6:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (6,4))
        elif num_clusters>6:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (8,8))

        for ind, ax in enumerate(axes.flat):
            if ind< num_clusters:
                pt = cluster_centroids[ind,:]
                xpts = pt[::2]
                ypts = pt[1::2]
                my_color = my_colors[ind]
                ax.scatter(xpts, ypts, s=50, c = my_color)
                ax.scatter(0,0,c='b', s = 40)
                ax.set_xlim(-100,100)
                ax.set_ylim(-100,100)
                ax.set_title('Cluster {}'.format(ind), fontsize = 14)
        fig.tight_layout()

        #Plot as point clouds

        if num_clusters <=4:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (8,8))
        elif num_clusters>4 and num_clusters<=6:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (12,8))
        elif num_clusters>6:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (12,12))

        for ind, ax in enumerate(axes.flat):
            if ind< num_clusters:
                cluster_pts = zd_mat2[labels==ind,:]
                my_color = my_colors[ind]
                for atom_index in range(cluster_pts.shape[0]):
                    xpts = cluster_pts[atom_index,::2]
                    ypts = cluster_pts[atom_index, 1::2]
                    ax.scatter(xpts, ypts,color = my_color, s=2 )

                    ax.set_xlim(-100,100)
                    ax.set_ylim(-100,100)

            ax.set_title('Cluster {} Point Cloud'.format(ind), fontsize = 14)
        fig.tight_layout()
    
        return [cluster_centroids, labels, fig]
    
#Main class
class AtomSeries:
    ###This class will deal with multiple objects of the Atoms class, for combined statistics
    #Must pass them as a list, i.e. [atoms_01, atoms_02] etc.
    
    def __init__(self, AtomsObjectsList, component_select = 1):
        self.atomseries = AtomsObjectsList 
        self.process_data() #process the initial list
        self.num_comps = len(self.atomseries)
        self.comps = [self.atomseries[k].comp for k in range(len(self.atomseries))]
        
    def process_data(self):
        #Go through and collect the descriptors about the types of atoms
        atomseries_descriptors = dict()
        for obj in self.atomseries:
            atom_descriptors = obj.atom_descriptors
            for key in atom_descriptors.keys():
                if key not in atomseries_descriptors:
                    atomseries_descriptors[key] = atom_descriptors[key]

        #Ok so now we know the number of types of atoms we have...
        #Next we need to iterate through
        all_results = []
        
        for atom_type in atomseries_descriptors:
            
            series_results = []
            print(atom_type)
            
            for obj in self.atomseries:
                obj.compute_neighborhood(num_neighbors=6, atom_type = atom_type)
                atom_type_id = obj.neighborhood_results['atom_type']
                atom_idx = np.where(obj.atom_types==atom_type_id)[0]        
                atoms_to_select = intersection(atom_idx, obj.nonborder_pixel_inds)
                #print(atoms_to_select)
                xd_mat = obj.neighborhood_results['xdistance_mat'][atoms_to_select]
                yd_mat = obj.neighborhood_results['ydistance_mat'][atoms_to_select]
                d_mat = obj.neighborhood_results['distance_mat'][atoms_to_select]
                a_mat = obj.neighborhood_results['angles_mat'][atoms_to_select]
                nn_types = obj.neighborhood_atom_types[atoms_to_select]
                series_results.append([xd_mat, yd_mat, d_mat, a_mat, nn_types])

            all_results.append(series_results)

        self.all_results = all_results
        
        #Add compositions to a list
        self.comps = [obj.comp for obj in self.atomseries]
        
        return all_results
    
    def compute_pca(self):
        all_eigenvecs = []
        all_eigenvals = []
        xypositions = []
        #get the data in the right form
        for ind, result in enumerate(self.all_results):
            
            #stored as xd, yd, distance, angle
            xd_mats = [res[0] for res in result]
            yd_mats = [res[1] for res in result]
            d_mats = [res[2] for res in result]
            a_mats = [res[3] for res in result]
            
            xd_mats_arr = np.vstack(xd_mats)
            yd_mats_arr = np.vstack(yd_mats)
            d_mats_arr = np.vstack(d_mats)
            a_mats_arr = np.vstack(a_mats)
            
            [ud,sd,vd] = np.linalg.svd(d_mats_arr,full_matrices=0) #SVD on radial distance of nearest neighbors
            [ua,sa,va] = np.linalg.svd(a_mats_arr,full_matrices=0) #SVD on anglular displacement of nearest neighbors

            [us,ss,vs] = np.linalg.svd(xd_mats_arr+1j*yd_mats_arr,full_matrices=0) #SVD on relative position of nearest neighbors (complex valued)

            #convert complex results of 
            usm = np.abs(us)
            usa = np.angle(us)
            usx = np.real(us)
            usy = np.imag(us)

            vsm = np.abs(vs)
            vsa = np.angle(vs)
            vsx = np.real(vs)
            vsy = np.imag(vs)
            
            all_eigenvecs.append(vd)
            all_eigenvals.append(ud)
            if ind==0: xypositions = [vsx, vsy] #save for plotting
        
        self.all_eigenvals = all_eigenvals
        
        #Plot the results
        # plot eigenvector maps of radial displacement eigenvectors
        vsx = xypositions[0]
        vsy = xypositions[1]
        
        fig, axs = plt.subplots(2,3, figsize=(12, 8), sharex=True, sharey=True)
        fig.subplots_adjust(hspace = .2, wspace=.2)
        axs = axs.ravel()
        
        if len(self.all_results)==2:
            colors = ['g', 'r']
        else:
            cm = plt.cm.get_cmap('RdBu', len(self.all_results))
            colors = [cm(ind) for ind in range(len(self.all_results))]
            
        for ind, vd in enumerate(all_eigenvecs):
            for k1 in range(0,min(6,vd.shape[0])):
                axs[k1].plot(0,0,'o', color = 'k')
                axs[k1].plot(vsx[0,:],vsy[0,:],'o', color = 'b') 
                axs[k1].axis('equal')
                c = 1.0
                axs[k1].axis([-c, c, -c, c])
                axs[k1].set_title(str(k1))
                axs[k1].quiver(vsx[0,:],vsy[0,:],vsx[0,:]*vd[k1,:],vsy[0,:]*vd[k1,:], color = colors[ind])
        
        fig.suptitle('PCA Components for both atom neighborhoods', fontsize = 18, y = 1.05)
        fig.tight_layout()
        
        #Now comb through the eigenvalues, splitting it up by composition in the series
        reshaped_eigenvalues = []
        reshaped_nn_sums = []
        #Go through the different types of atoms
        for k in range(len(self.all_results)):
            
            #for each type of atom, go through its eigenvalues
            
            at_eigenvals = all_eigenvals[k]
            
            nn_info = self.all_results[k] #this is already split up by composition
             
            inds_at = [self.all_results[k][ind][0].shape[0] for ind in range(len(self.all_results[k]))]
            inds_at.insert(0,0)
            inds_at.append(len(inds_at))
            
            #Here we want to chop it up into the composition series...
            at_reshaped = []
           
            for ind, val in enumerate(inds_at):
                if ind<len(inds_at)-2:
                    start_ind = int(np.sum(inds_at[:ind+1]))
                    end_ind = int(np.sum(inds_at[:ind+1]) + inds_at[ind+1])
                    at_reshaped.append(at_eigenvals[start_ind:end_ind,:])
            
            reshaped_eigenvalues.append(at_reshaped)

        self.eigenvals_reshaped = reshaped_eigenvalues
       
        
        return fig
    
    def plot_eigenvalue_violin(self):
        #this will plot the eigenvalues for each of the four compositions
        for ind in range(len(self.eigenvals_reshaped)):
            
            key = list(self.atomseries[0].atom_descriptors.keys())[ind]
            atom_type_selection = self.atomseries[0].atom_descriptors[key]
            
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (12,8), sharex=True) #assuming 4 compositions. In future, would want this to be flexible 
            axflat = axes.flat
            for k in range(len(self.eigenvals_reshaped[ind])):
                axflat[k].violinplot(self.eigenvals_reshaped[ind][k])
                axflat[k].set_xlabel('PCA Component #', fontsize = 12)
                axflat[k].set_ylabel('PCA Score', fontsize = 12)
                axflat[k].set_title(self.atomseries[k].image_name, fontsize = 14)
                axflat[k].tick_params(axis='x', labelsize=12 )
                axflat[k].tick_params(axis='y', labelsize=12 )
            
            fig.suptitle('PCA Eigenvalues with ' + key + ' atom centered', fontsize = 18, y = 1.05)
            fig.tight_layout()
        return
    
    def plot_eigenvalue_composition(self, component_select = 1, plot_all=True):
        #Plot the eigenvalues colored by the composition
        #If plot_all is True, the individual compositions will be overlaid on teh same plot
        #Otherwise we will have one plot for each composition
        
        atom_neighborhood_types = []
        figs = []
        atomsums = []
        #Plot it for the Mo and the Re neighbors
        for ind in range(len(self.eigenvals_reshaped)):
            atom_comp_types=[]
            key = list(self.atomseries[0].atom_descriptors.keys())[ind]
            not_key = [itm for itm in list(self.atomseries[0].atom_descriptors.keys()) if key not in itm]
            
            #nearest neighbor atoms
            #Find out how many atoms of another type are around the atom
            all_atom_sums = []
            atom_type_selection = self.atomseries[0].atom_descriptors[key]
            for comp in range(len(self.eigenvals_reshaped[ind])):
                atom_sums = [len(np.where(self.all_results[ind][comp][-1][k,1:]!=atom_type_selection)[0]) 
                             for k in range(self.all_results[ind][comp][-1].shape[0])]
                all_atom_sums.append(atom_sums)
                atom_comp_types.append(all_atom_sums)
            
            atomsums.append(all_atom_sums)
            
            atom_neighborhood_types.append(atom_comp_types)    
            
            if plot_all:
                fig, axes = plt.subplots(figsize = (10,7))
                colors = ['r', 'k', 'b', 'g']
                for k in range(len(self.all_results[ind])):

                    axes.scatter(all_atom_sums[k], self.eigenvals_reshaped[ind][k][:,component_select], color = colors[k],
                                label = self.atomseries[k].image_name, alpha = 0.5)
                            
                axes.set_xlabel('# NN of ' + not_key[0], fontsize = 14)
                axes.set_ylabel('PCA Score')
                axes.tick_params(axis='x', labelsize=12 )
                axes.tick_params(axis='y', labelsize=12 )

                fig_title = 'PCA Component = ' + str(component_select) + ' with ' + key + ' as central atom' 
                axes.set_title(fig_title, fontsize = 16)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize = 14)
                fig.tight_layout()                         
                
                figs.append(fig)
            else:
                for k in range(len(self.all_results[ind])):
                    fig, axes = plt.subplots(figsize = (10,7))
                    axes.scatter(all_atom_sums[k], self.eigenvals_reshaped[ind][k][:,component_select], 
                                label = self.atomseries[k].image_name, alpha = 0.5)
                    axes.set_xlabel('# NN of ' + not_key[0], fontsize = 14)
                    axes.set_ylabel('PCA Score')
                    axes.tick_params(axis='x', labelsize=12 )
                    axes.tick_params(axis='y', labelsize=12 )
                    axes.set_title(self.atomseries[k].image_name, fontsize = 16)

                    fig.tight_layout()
                    figs.append(fig)
            
        self.atom_neighborhood_types = atom_neighborhood_types
        self.atomsums = atomsums
        return figs
    
    #Let's plot the probability of finding atom clusters per composition
    def plot_atom_cluster_prob(self, atom_choice = 'Mo'):
        
        key = atom_choice
        atom_type_selection = self.atomseries[0].atom_descriptors[atom_choice]
        sum_nn_index_Mo_all = self.atom_neighborhood_types[atom_type_selection][0]
        not_key = [itm for itm in list(self.atomseries[0].atom_descriptors.keys()) if key not in itm]
        
        fig = plt.subplots()
        if self.num_comps==4:
            colors = ['r', 'k', 'b', 'g']
        else:
            cm = plt.cm.get_cmap('jet', self.num_comps)
            colors = [cm(ind) for ind in range(self.num_comps)]
            
        for ind in range(self.num_comps):
            unique_nos = np.unique(sum_nn_index_Mo_all[ind])
            probs = [len(np.where(np.array(sum_nn_index_Mo_all[ind])==val)[0]) / len(np.array(sum_nn_index_Mo_all[ind])) for val in unique_nos]
            plt.plot(unique_nos, probs, 'o-', color = colors[ind], label = self.atomseries[ind].image_name)
            plt.legend(loc='best', fontsize = 12)
            plt.xlabel('NN of ' + not_key[0], fontsize = 14)
            plt.ylabel('Probability of finding # ' + not_key[0] + ' Neighbors', fontsize = 14)
            plt.title('Cluster probabilities, ' + key + ' Neighbors', fontsize = 16)
        return fig
    #Let's try violin plots

    def plot_eigenvalue_composition_violin(self, component_select = 1):
        #Plot the eigenvalues colored by the composition, as a violinplot
        self.component_select = component_select
        
        atom_neighborhood_types = []
        figs = []
        #Plot it for the Mo and the Re neighbors
        atomsums = []
        nn_eigenvalue_data = []
        
        temp_list = []
        for ind in range(len(self.eigenvals_reshaped)):

            key = list(self.atomseries[0].atom_descriptors.keys())[ind]
            not_key = [itm for itm in list(self.atomseries[0].atom_descriptors.keys()) if key not in itm]
            #nearest neighbor atoms
            #Find out how many atoms of another type are around the atom
            all_atom_sums = []
            atom_type_selection = self.atomseries[0].atom_descriptors[key]
            for comp in range(len(self.eigenvals_reshaped[ind])):
       
                atom_sums = [len(np.where(self.all_results[ind][comp][-1][k,1:]!=atom_type_selection)[0]) 
                             for k in range(self.all_results[ind][comp][-1].shape[0])]
                all_atom_sums.append(atom_sums)


            nneighbors = self.atomseries[0].atom_neighbor_positions.shape[-2]
            reshaped_comp_data = []
            temp_list.append(all_atom_sums)
            for k in range(len(self.all_results[ind])):

                reshaped_data = []
                ydata = self.eigenvals_reshaped[ind][k][:,1]

                for sind in range(nneighbors+1):
                    cur_dat = [val for idx,val in enumerate(ydata) if all_atom_sums[k][idx]==sind]
                    if len(cur_dat)>=2:
                        reshaped_data.append(cur_dat)
                    else:
                        reshaped_data.append([0])
                
                reshaped_comp_data.append(reshaped_data)
                fig, axes = plt.subplots(figsize = (10,7))

                axes.violinplot(reshaped_data, np.arange(nneighbors+1))
                axes.set_xlabel('# NN of ' + not_key[0], fontsize = 14)
                axes.set_ylabel('PCA Score Component #' + str(component_select))
                axes.tick_params(axis='x', labelsize=12 )
                axes.tick_params(axis='y', labelsize=12 )

                fig_title = self.atomseries[k].image_name + ' eigenvalues with ' + key + ' as central atom' 
                axes.set_title(fig_title, fontsize = 16)

                fig.tight_layout()                         

            atom_neighborhood_types.append(all_atom_sums)
            figs.append(fig)
            nn_eigenvalue_data.append(reshaped_comp_data)
            atomsums.append(all_atom_sums)
            self.atomsums = atomsums
        
        self.nn_eigenvalue_data = nn_eigenvalue_data
        self.temp_list = temp_list
        return figs, atomsums
    
    def perform_gmm(self, component_select = 1, gmm_comps = 2):
        
        #Perform gaussian mixture modeling, using the component selection and number of gmm components specified
        
        if component_select!=self.component_select: 
            print('You chose a different number of components to that used previously. Recalculating!')
            self.plot_eigenvalue_composition_violin(component_select = component_select)
           
        gm_list = []
        nneighbors = self.atomseries[0].atom_neighbor_positions.shape[-2]
       
        for ind in range(len(self.all_results)):
            gm_comp = []
            for comp in range(self.num_comps):
                gm_n = []
                for m in range(nneighbors+1):

                    y = np.array(self.nn_eigenvalue_data[ind][comp][m])

                    if y.shape[0]>2:
                        gm = GM(n_components = gmm_comps)
                        gm.fit(y.reshape(-1,1))
                        means = gm.means_
                        covariances = gm.covariances_
                    else:
                        means = [np.nan,np.nan]
                        covariances = np.zeros(shape=(2,1))
                    gm_n.append((means,covariances,m))
                gm_comp.append(gm_n)
            gm_list.append(gm_comp)
        
        self.gmm_list = gm_list #This list contains the gmm results
        
        
        if self.num_comps==4:
            colors = ['r', 'k', 'b', 'g']
        else:
            cm = plt.cm.get_cmap('jet', num_comps)
            colors = [cm(ind) for ind in range(num_comps)]
        
        for ind in range(len(self.all_results)):
            fig = plt.figure()   
            
            key = list(self.atomseries[0].atom_descriptors.keys())[ind]
            not_key = [itm for itm in list(self.atomseries[0].atom_descriptors.keys()) if key not in itm]
            
            for comp in range(self.num_comps):
                for neighbor in range(nneighbors+1):
                    for gmm_comp in range(gmm_comps):
                        x = np.array(neighbor).ravel()
                        y = np.array(gm_list[ind][comp][neighbor][0][gmm_comp]).ravel()
                        yerr = np.array(np.sqrt(gm_list[ind][comp][neighbor][1][gmm_comp])).ravel()
                        plt.errorbar(neighbor,y, yerr=yerr, color = colors[comp],capsize = 6);
                        plt.plot(neighbor, y, 'o', color = colors[comp], markersize = 5);
        
            plt.xlabel('NN of ' + not_key[0], fontsize = 14)
            plt.ylabel('PCA Comp. # ' + str(self.component_select) + ' Scores', fontsize = 14)
            plt.title('GMM with ' + key + ' Neighbors', fontsize = 16)
            
            #Need to create a custom legend
            
        return fig
    
    @staticmethod
    def GPReg(X, y, **kwargs):
        """
        Performs exact gaussian process regression 
        for variable with < 2000 data points.
        Performs sparse gaussian process regression
        for variable with > 2000 points.
        Args:
            X: numpy array
                ndarray of features, with dimensions (n_samples x n_features)
            y: numpy array
                ndarray with function falues, with dimensions (n_samples,)
        **Kwargs:
            epochs: int
                number of iterations
        Returns:
            calculated mean and variance
        """
        pyro.clear_param_store()
        epochs = kwargs.get('epochs', 500)
        X = X[:, np.newaxis] if np.ndim(X) == 1 else X
        y = y[:, 0] if np.ndim(y) > 1 else y

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            X, y = X.cuda(), y.cuda()
        if len(X) < 2000:
            gpr = gp.models.GPRegression(
                X, y, gp.kernels.RBF(input_dim=X.shape[-1]))
        else:
            indp_step = int(5e-4 * len(X))
            gpr = gp.models.SparseGPRegression(
                X, y, 
                gp.kernels.RBF(input_dim=X.shape[-1]), 
                Xu=torch.tensor(X[::indp_step]), 
                jitter=1.0e-5)
        if torch.cuda.is_available():
            gpr.cuda()
        optimizer = torch.optim.Adam(gpr.parameters(), lr=5e-2)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        for i in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            print('\rIteration {} / {}'.format(i+1, epochs), end="")
        with torch.no_grad():
                mean, cov = gpr(X, full_cov=False, noiseless=False)
        return mean.cpu().numpy(), cov.cpu().numpy()
      
    def perform_GP_fits(self, component_select = 1, scaleX = False, scaleY=False):
        '''This will perform a linear regression fit on the PCA component scores, with the x-axis being the local and global concentrations. Can choose to scale X and Y variables, False by default. Note that '''
        
        for ind in range(len(self.all_results)):
            key = list(self.atomseries[0].atom_descriptors.keys())[ind]
            not_key = [itm for itm in list(self.atomseries[0].atom_descriptors.keys()) if key not in itm]
            
            ud_Mo_reshaped = self.eigenvals_reshaped[ind]

            sum_nn_index_Mo_all = self.atom_neighborhood_types[ind][0]

            comps = self.comps
            
            X1 = np.array(sum_nn_index_Mo_all[0])
            X2 = np.array(sum_nn_index_Mo_all[1])
            X3 = np.array(sum_nn_index_Mo_all[2])
            X4 = np.array(sum_nn_index_Mo_all[3])

            samples_loc = {1 : X1, 2: X2, 
                           3: X3, 4: X4}

            samples_loc_glob = np.concatenate(
                [np.vstack((X, np.ones(X.shape)*comps[ind])).T
                for ind,X in enumerate([X1,X2,X3,X4])], axis=0)

            pred_all, var_all = [], []
            
            for k in samples_loc.keys():
                X = np.array(samples_loc[k], dtype = np.float64)
                y = np.abs(ud_Mo_reshaped[k-1][:,component_select])
                
                # Normalize data
                if scaleX:
                    X = (X - np.mean(X)) / np.std(X)
                if scaleY:
                    y = (y - np.mean(y)) / np.std(y)
                
                # Run GP regression    
                pred, var = self.GPReg(X, y)
                pred_all.append(pred)
                var_all.append(var)
                # Vizualize results
                print('\nAverage uncertainty:', np.mean(var))
                fig, axes = plt.subplots()
                axes.scatter(X, y, c='black', s=3)
                axes.scatter(X, pred, c=var, cmap='jet', marker='x', s=60)
                
                axes.set_xlabel('#NN of ' + not_key[0], fontsize = 14)
                axes.set_ylabel('PCA Component # ' + str(component_select) + 'Scores', fontsize = 14)
                axes.set_title('GP Fitting PCA(C,NN) for ' + self.atomseries[k-1].image_name + 'with ' + key + ' Centered', fontsize = 16)
            
            plt.figure()
            var_all_m = [np.mean(v) for v in var_all]
            
            plt.plot(comps, var_all_m,'-o')
            plt.xlabel('Composition')
            plt.ylabel('Uncertainty')
            plt.title('UQ as a function of compositons')
            plt.xticks(comps, [self.atomseries[ind].image_name for ind in range(len(self.atomseries))], fontsize = 12, rotation = 90)
    
    def perform_2d_GPR(self, component_select=1, scaleX=False, scaleY=False):
        figs = []
        for ind in range(len(self.all_results)):
            key = list(self.atomseries[0].atom_descriptors.keys())[ind]
            not_key = [itm for itm in list(self.atomseries[0].atom_descriptors.keys()) if key not in itm]

            ud_Mo_reshaped = self.eigenvals_reshaped[ind]

            sum_nn_index_Mo_all = self.atom_neighborhood_types[ind][0]

            comps = self.comps

            X1 = np.array(sum_nn_index_Mo_all[0])
            X2 = np.array(sum_nn_index_Mo_all[1])
            X3 = np.array(sum_nn_index_Mo_all[2])
            X4 = np.array(sum_nn_index_Mo_all[3])

            samples_loc = {1 : X1, 2: X2, 
                           3: X3, 4: X4}

            samples_loc_glob = np.concatenate(
                [np.vstack((X, np.ones(X.shape)*comps[ind])).T
                for ind,X in enumerate([X1,X2,X3,X4])], axis=0)
    
            # specify input vector
            X = samples_loc_glob
            # specify output vector
            y = np.abs(np.concatenate(ud_Mo_reshaped)[:, component_select])

            # Normalize data
            if scaleX:
                X = (X - np.mean(X)) / np.std(X)
            if scaleY:
                y = (y - np.mean(y)) / np.std(y)
            # Run GP regression    
            pred, var = self.GPReg(X, y)
            num_comps = self.num_comps
            nneighbors =  self.atomseries[0].atom_neighbor_positions.shape[-2]
            loc_glob_mat = np.zeros((num_comps, nneighbors+1))
            loc_glob_mat_var = np.zeros((num_comps, nneighbors+1))
            glob_list = self.comps
            loc_list = np.arange(0, nneighbors+1)
            for i1, i2 in enumerate(glob_list):
                for j1, j2 in enumerate(loc_list):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        loc_glob_mat[i1, j1] = np.mean(
                            pred[np.where((X[:,1]==i2) & (X[:,0]==j2))[0]])
                        loc_glob_mat_var[i1, j1] = np.mean(
                            var[np.where((X[:,1]==i2) & (X[:,0]==j2))[0]])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.matshow(loc_glob_mat, cmap='viridis', vmin = 0, vmax =0.10)
            ax1.set_xticks(np.arange(loc_glob_mat.shape[1]))
            ax1.set_yticks(np.arange(loc_glob_mat.shape[0]))
            ax1.set_xticklabels(loc_list, rotation='horizontal', fontsize=14)
            ax1.set_yticklabels(glob_list, rotation='horizontal', fontsize=14)
            ax1.set_title('Mean prediction ' + key + ' Centered', y=1.1, fontsize=14)
            for (j,i),output in np.ndenumerate(loc_glob_mat):
                ax1.text(i,j, np.around(output, 2), ha='center', va='center', c='w', fontsize = 14)

            ax2.matshow(loc_glob_mat_var, cmap='viridis', vmin = 0.000348, vmax = 0.000376)
            ax2.set_xticks(np.arange(loc_glob_mat_var.shape[1]))
            ax2.set_yticks(np.arange(loc_glob_mat_var.shape[0]))
            ax2.set_xticklabels(loc_list, rotation='horizontal', fontsize=14)
            ax2.set_yticklabels(glob_list, rotation='horizontal', fontsize=14)
            ax2.set_title('Mean uncertainty '+ key + ' Centered', y=1.1, fontsize=14)
            for (j,i),output in np.ndenumerate(loc_glob_mat_var):
                ax2.text(i,j, np.around(output**0.5, 3), ha='center', va='center', c='w', fontsize = 14)
            plt.subplots_adjust(wspace=.3)
            figs.append(fig)
            
        return figs
      


            
        
            
                
            
            
            
        
        
            
        
        
