# Matrix Factorization class
# Instantiate this object and perform matrix factorization on any sidpy dataset
# Note that matrix factorization can only operate on 2D matrices. So we will automatically reshape
# all datasets to be two dimensional (spatial v spectral)
import sidpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA, FastICA, KernelPCA
from pysptools.eea import nfindr
import pysptools.abundance_maps as amp


class MatrixFactor:
    def __init__(self, data, method='nmf', n_components=2, return_fit=False,
                 ) -> None:
        """
        Parameters:
            - data: (sidpy.Dataset)
            - method: Type of Matrix Decomposition to be performed. The User can be choose from:
                    PCA
                    SVD
                    NMF
                    ICA
                    nfindr
                    kernelpca
            - n_components: number of components, Default: None
            - return_fit: Whether to return the fitted dataset, Default: False

        Returns:


        """
        self.data = data
        self.ncomp = n_components
        self._allowed_methods = ['pca', 'svd', 'nmf', 'ica', 'nfindr', 'kernelpca']  # Might want to add others...
        assert method in self._allowed_methods, "Method must be one of {}".format(self._allowed_methods)
        self.method = method
        if self.method == 'svd':
            self.u, self.s, self.vh = None, None, None

        if self.data.ndim == 2:
            self.data_2d = data
        else:
            self.data_2d = self._return_2d_dataset(self.data)
        self.dim_order = self.data_2d.metadata['fold_attr']['dim_order']
        self.return_fit = return_fit
        self.results_computed = False

        # Datasets to be returned
        self.abundances = None
        self.components = None
        if self.return_fit:
            self.fit_dset = None

    def _return_2d_dataset(self, data):
        # Flattening the dataset as spatial-spectral
        folded_dset = data.fold(method='spaspec')

        return folded_dset

    def do_fit(self, **kwargs) -> sidpy.Dataset:
        """
        Parameters:
        (none)

        Returns: 
        Sidpy dataset after fit operation. Fit will calculate according to the method chosen.

        """
        abundances, components = None, None
        
        if self.method == 'svd':
            u, s, vh = np.linalg.svd(np.array(self.data_2d), full_matrices=False, compute_uv=True)
            abundances, components = None, None
            # abundances = u  # check...
            # components = (s * v.T).T  # check again
        elif self.method == 'nmf':
            # code goes here...
            nmf = NMF(n_components=self.ncomp, **kwargs)
            abundances = nmf.fit_transform(np.array(self.data_2d))
            components = nmf.components_
        elif self.method == 'pca':
            pca = PCA(n_components=self.ncomp, **kwargs)
            abundances = pca.fit_transform(np.array(self.data_2d))
            components = pca.components_
        elif self.method == 'ica':
            ica = FastICA(n_components=self.ncomp, **kwargs)
            abundances = ica.fit_transform(np.array(self.data_2d))
            components = ica.components_
        elif self.method == 'kernelpca':
            kpca = KernelPCA(n_components=self.ncomp, **kwargs)
            X_kpca = kpca.fit(np.array(self.data_2d).T)
            abundances = X_kpca.fit_transform(np.array(self.data_2d))
            components = X_kpca.eigenvectors_.T
        elif self.method == 'nfindr':
            nnls = amp.FCLS()
            a1 = nfindr.NFINDR(np.array(self.data_2d), self.ncomp)  # Find endmembers
            components = a1[0]
            data_spec_fold = self.data.fold(method='spec')  # This will fold all the spectral dimensions into 1
            data_amap = np.array(data_spec_fold)
            abundances = nnls.map(data_amap, components)  # Finding abundances
            abundances = abundances.reshape(-1, self.ncomp)

        if abundances is None and components is None:

            self.u = sidpy.Dataset.from_array(u, title='U')

            # We might want to write an unfolding command for s and vh
            self.s = sidpy.Dataset.from_array(s, title='singular values(s)')
            self.vh = sidpy.Dataset.from_array(vh, title='Vh')
            self.fit_dset = self.data_2d.like_data(np.dot(u * s, vh),
                                                   title_suffix='_factorized').unfold()

            if self.return_fit:
                return self.u, self.s, self.vh, self.fit_dset
            else:
                return self.u, self.s, self.vh
        else:
            # Getting the fit dataset i.e., abundances*components and unfolding it into the original shape
            self.fit_dset = self.data_2d.like_data(np.matmul(abundances, components),
                                                   title_suffix='_factorized').unfold()
            self.abundances = abundances
            self.components = components
            
            # Now we need to prepare an abundance dataset and a component dataset and make sure their spatial dimensions
            # are unfolded
            abundances_dset = self.data_2d.like_data(abundances, title='abundances',
                                                     check_dims=False)
            # Image stack
            # We will delete this for now and add this back manually
            del abundances_dset.metadata['fold_attr']
            abun_axes = {}
            for i, dim in enumerate(self.dim_order[0]):
                abun_axes[i] = self.data_2d.metadata['fold_attr']['_axes'][dim].copy()
            abun_axes[len(self.dim_order[0])] = sidpy.Dimension(np.arange(self.ncomp),
                                                                name='weights',
                                                                units='generic', quantity='generic',
                                                                dimension_type='spectral')

            abundances_dset.metadata['fold_attr'] = dict(
                dim_order_flattened=list(np.arange(len(self.dim_order[0]) + 1)),
                shape_transposed=self.data_2d.metadata['fold_attr'][
                                     'shape_transposed'][:len(self.dim_order[0])]
                                 + [self.ncomp], _axes=abun_axes)

            # Check the number of spectral dimensions
            # Check whether we will have the same datatype and other attributes
            components_dset = self.data_2d.like_data(components, title='components',
                                                     check_dims=False)
            del components_dset.metadata['fold_attr']
            comp_axes = {}
            comp_axes[0] = sidpy.Dimension(np.arange(self.ncomp),
                                           name='component_number',
                                           units='generic', quantity='generic',
                                           dimension_type='spectral')
            for i, dim in enumerate(self.dim_order[1]):
                comp_axes[i + 1] = self.data_2d.metadata['fold_attr']['_axes'][dim].copy()
            components_dset.metadata['fold_attr'] = {
                'dim_order_flattened': list(np.arange(len(self.dim_order[1]) + 1)),
                'shape_transposed': [self.ncomp] + self.data_2d.metadata['fold_attr']['shape_transposed'][
                                                   len(self.dim_order[0]):],
                '_axes': comp_axes

            }

            # Then we want to return components, abundances, and explained variances if available.
            # We should return a list of sidpy dataset objects
            self.abundances = abundances_dset.unfold()
            self.abundances.data_type = sidpy.DataType.IMAGE_STACK

            self.components = components_dset.unfold()
            self.components.data_type = sidpy.DataType.LINE_PLOT_FAMILY  # Check this again
            self.results_computed = True
            

            if self.return_fit:
                return self.abundances, self.components, self.fit_dset
            else:
                return self.abundances, self.components
