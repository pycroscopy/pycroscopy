# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import
import h5py
import numpy as np
import sklearn.decomposition as dec

from .process import Process
from ..io.hdf_utils import getH5DsetRefs, checkAndLinkAncillary
from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import check_dtype, transformToTargetType
from ..io.microdata import MicroDataGroup, MicroDataset


class Decomposition(Process):
    """
    Pycroscopy wrapper around the sklearn.decomposition classes
    """

    def __init__(self, h5_main, estimator):
        """
        Uses the provided (preconfigured) Decomposition object to 
        decompose the provided dataset
        
        Parameters
        ------------
        h5_main : HDF5 dataset object
            Main dataset with ancillary spectroscopic, position indices and values datasets
        estimator : sklearn.cluster estimator object
            configured decomposition object to apply to the data
        """
        
        allowed_methods = [dec.factor_analysis.FactorAnalysis,
                           dec.fastica_.FastICA,
                           dec.incremental_pca.IncrementalPCA,
                           dec.sparse_pca.MiniBatchSparsePCA,
                           dec.nmf.NMF,
                           dec.pca.PCA,
                           dec.sparse_pca.SparsePCA,
                           dec.truncated_svd.TruncatedSVD]
        
        # Store the decomposition object
        self.estimator = estimator
        
        # could not find a nicer way to extract the method name yet
        self.method_name = str(estimator)[:str(estimator).index('(')]

        if type(estimator) not in allowed_methods:
            raise NotImplementedError('Cannot work with {} yet'.format(self.method_name))
            
        # Done with decomposition-related checks, now call super init
        super(Decomposition, self).__init__(h5_main)
        
        # set up parameters
        self.parms_dict = {'decomposition_algorithm':self.method_name}
        self.parms_dict.update(self.estimator.get_params())
        
        # check for existing datagroups with same results 
        self.process_name = 'Decomposition'
        self.duplicate_h5_groups = self._check_for_duplicates()

        # figure out the operation that needs need to be performed to convert to real scalar
        (self.data_transform_func, self.data_is_complex, self.data_is_compound,
         self.data_n_features, self.data_n_samples, self.data_type_mult) = check_dtype(h5_main)

    def compute(self):
        """
        Decomposes the hdf5 dataset, and writes the ? back to the hdf5 file
        
        Returns
        -------
        h5_group : HDF5 Group reference
            Reference to the group that contains the decomposition results
        """
        self._fit()
        self._transform()
        return self._write_results_chunk(transformToTargetType(self.estimator.components_, self.h5_main.dtype),
                                 self.projection)

    def _fit(self):
        """
        Fits the provided dataset
        """
        # perform fit on the real dataset
        if self.method_name == 'NMF':
            self.estimator.fit(self.data_transform_func(np.abs(self.h5_main)))
        else:
            self.estimator.fit(self.data_transform_func(self.h5_main))

    def _transform(self, data=None):
        """
        Transforms the original OR provided dataset with previously computed fit
        
        Parameters
        --------
        data : (optional) HDF5 dataset
            Dataset to apply the transform to. 
            The number of elements in the first axis of this dataset should match that of the original
            dataset that was fitted
        """
        if data is None:
            if self.method_name == 'NMF':
                self.projection = self.estimator.transform(self.data_transform_func(np.abs(self.h5_main)))
            else:
                self.projection = self.estimator.transform(self.data_transform_func(self.h5_main))
        else:
            if isinstance(data, h5py.Dataset):
                if data.shape[0] == self.h5_main.shape[0]:
                    self.projection = self.estimator.transform(data)

    def _write_results_chunk(self, components, projection):
        """
        Writes the labels and mean response to the h5 file
        
        Parameters
        ------------
        labels : 1D unsigned int array
            Array of cluster labels as obtained from the fit
        mean_response : 2D numpy array
            Array of the mean response for each cluster arranged as [cluster number, response]
            
        Returns
        ---------
        h5_labels : HDF5 Group reference
            Reference to the group that contains the clustering results
        """
        ds_components = MicroDataset('Components', components)  # equivalent to V
        ds_projections = MicroDataset('Projection', np.float32(projection))  # equivalent of U compound

        decomp_ind_mat = np.transpose(np.atleast_2d(np.arange(components.shape[0])))

        ds_decomp_inds = MicroDataset('Decomposition_Indices', np.uint32(decomp_ind_mat))
        ds_decomp_vals = MicroDataset('Decomposition_Values', np.float32(decomp_ind_mat))

        # write the labels and the mean response to h5
        decomp_slices = {'Decomp': (slice(None), slice(0, 1))}
        ds_decomp_inds.attrs['labels'] = decomp_slices
        ds_decomp_inds.attrs['units'] = ['']
        ds_decomp_vals.attrs['labels'] = decomp_slices
        ds_decomp_vals.attrs['units'] = ['']

        decomp_grp = MicroDataGroup(self.h5_main.name.split('/')[-1] + '-Decomposition_', self.h5_main.parent.name[1:])
        decomp_grp.addChildren([ds_components, ds_projections, ds_decomp_inds, ds_decomp_vals])
        
        decomp_grp.attrs.update(self.parms_dict)
        decomp_grp.attrs.update({'n_components': components.shape[0],
                                 'n_samples': self.h5_main.shape[0]})
        
        hdf = ioHDF5(self.h5_main.file)
        h5_decomp_refs = hdf.writeData(decomp_grp)

        h5_components = getH5DsetRefs(['Components'], h5_decomp_refs)[0]
        h5_projections = getH5DsetRefs(['Projection'], h5_decomp_refs)[0]
        h5_decomp_inds = getH5DsetRefs(['Decomposition_Indices'], h5_decomp_refs)[0]
        h5_decomp_vals = getH5DsetRefs(['Decomposition_Values'], h5_decomp_refs)[0]

        checkAndLinkAncillary(h5_projections,
                              ['Position_Indices', 'Position_Values'],
                              h5_main=self.h5_main)

        checkAndLinkAncillary(h5_projections,
                              ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                              anc_refs=[h5_decomp_inds, h5_decomp_vals])

        checkAndLinkAncillary(h5_components,
                              ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                              h5_main=self.h5_main)

        checkAndLinkAncillary(h5_components,
                              ['Position_Indices', 'Position_Values'],
                              anc_refs=[h5_decomp_inds, h5_decomp_vals])

        # return the h5 group object
        return h5_components.parent
