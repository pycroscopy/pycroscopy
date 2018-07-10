# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import
import h5py
import time
import numpy as np
import sklearn.decomposition as dec

from pyUSID.processing.process import Process
from pyUSID.io.hdf_utils import reshape_to_n_dims, create_results_group, write_main_dataset, get_attr, \
    write_simple_attrs
from pyUSID.io.write_utils import Dimension
from pyUSID.io.dtype_utils import check_dtype, stack_real_to_target_dtype
from pyUSID.io.io_utils import format_time
from pyUSID import USIDataset


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
        # Partial groups don't make any sense for statistical learning algorithms....
        self.duplicate_h5_groups, self.h5_partial_groups = self._check_for_duplicates()

        # figure out the operation that needs need to be performed to convert to real scalar
        (self.data_transform_func, self.data_is_complex, self.data_is_compound,
         self.data_n_features, self.data_type_mult) = check_dtype(h5_main)

        # supercharge h5_main!
        self.h5_main = USIDataset(self.h5_main)
        
        self.__components = None
        self.__projection = None
        
    def test(self, override=False):
        """
        Decomposes the hdf5 dataset to calculate the components and projection. This function does NOT write results to
        the hdf5 file. Call compute() to  write to the file. Handles complex, compound datasets such that the
        components are of the same data-type as the input matrix.

        Parameters
        ----------
        override : bool, optional. default = False
            Set to true to recompute results if prior results are available. Else, returns existing results

        Returns
        -------
        components : numpy array
            Components
        projections : numpy array
            Projections
        """
        if not override:
            if isinstance(self.duplicate_h5_groups, list) and len(self.duplicate_h5_groups) > 0:
                self.h5_results_grp = self.duplicate_h5_groups[-1]
                print('Returning previously computed results from: {}'.format(self.h5_results_grp.name))
                print('set the "override" flag to True to recompute results')
                return USIDataset(self.h5_results_grp['Components']).get_n_dim_form(), \
                       USIDataset(self.h5_results_grp['Projection']).get_n_dim_form()

        self.h5_results_grp = None

        print('Performing Decomposition on {}.'.format(self.h5_main.name))

        t0 = time.time()
        self._fit()
        self._transform()
        print('Took {} to compute {}'.format(format_time(time.time() - t0), self.method_name))

        self.__components = stack_real_to_target_dtype(self.estimator.components_, self.h5_main.dtype)
        projection_mat, success = reshape_to_n_dims(self.__projection, h5_pos=self.h5_main.h5_pos_inds,
                                                    h5_spec=np.expand_dims(np.arange(self.__projection.shape[1]),
                                                                           axis=0))
        if not success:
            raise ValueError('Could not reshape projections to N-Dimensional dataset! Error:' + success)

        components_mat, success = reshape_to_n_dims(self.__components, h5_spec=self.h5_main.h5_spec_inds,
                                                    h5_pos=np.expand_dims(np.arange(self.__components.shape[0]),
                                                                          axis=1))

        if not success:
            raise ValueError('Could not reshape components to N-Dimensional dataset! Error:' + success)

        return components_mat, projection_mat

    def delete_results(self):
        """
        Deletes results from memory.
        """
        del self.__components, self.__projection
        self.__components = None
        self.__projection = None
        self.h5_results_grp = None

    def compute(self, override=False):
        """
        Decomposes the hdf5 dataset to calculate the components and projection (by calling test() if it hasn't already
        been called), and writes the results back to the hdf5 file

        Parameters
        ----------
        override : bool, optional. default = False
            Set to true to recompute results if prior results are available. Else, returns existing results

        Returns
        -------
        h5_group : HDF5 Group reference
            Reference to the group that contains the decomposition results
        """
        if self.__components is None and self.__projection is None:
            self.test(override=override)

        if self.h5_results_grp is None:
            h5_group = self._write_results_chunk()
            self.delete_results()
        else:
            h5_group = self.h5_results_grp

        return h5_group

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
                self.__projection = self.estimator.transform(self.data_transform_func(np.abs(self.h5_main)))
            else:
                self.__projection = self.estimator.transform(self.data_transform_func(self.h5_main))
        else:
            if isinstance(data, h5py.Dataset):
                if data.shape[0] == self.h5_main.shape[0]:
                    self.__projection = self.estimator.transform(data)

    def _write_results_chunk(self):
        """
        Writes the labels and mean response to the h5 file

        Returns
        ---------
        h5_group : HDF5 Group reference
            Reference to the group that contains the decomposition results
        """

        h5_decomp_group = create_results_group(self.h5_main, self.process_name)
        write_simple_attrs(h5_decomp_group, self.parms_dict)
        write_simple_attrs(h5_decomp_group, {'n_components': self.__components.shape[0],
                                             'n_samples': self.h5_main.shape[0], 'last_pixel': self.h5_main.shape[0]})

        decomp_desc = Dimension('Endmember', 'a. u.', self.__components.shape[0])

        # equivalent to V - compound / complex
        h5_components = write_main_dataset(h5_decomp_group, self.__components, 'Components',
                                           get_attr(self.h5_main, 'quantity')[0], 'a.u.', decomp_desc,
                                           None,
                                           h5_spec_inds=self.h5_main.h5_spec_inds,
                                           h5_spec_vals=self.h5_main.h5_spec_vals)

        # equivalent of U - real
        h5_projections = write_main_dataset(h5_decomp_group, np.float32(self.__projection), 'Projection', 'abundance',
                                            'a.u.', None, decomp_desc, dtype=np.float32,
                                            h5_pos_inds=self.h5_main.h5_pos_inds, h5_pos_vals=self.h5_main.h5_pos_vals)

        # return the h5 group object
        self.h5_results_grp = h5_decomp_group
        return self.h5_results_grp
