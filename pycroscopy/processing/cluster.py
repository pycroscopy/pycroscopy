# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith

"""
from __future__ import division, print_function, absolute_import
import time
import numpy as np
import sklearn.cluster as cls
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from .proc_utils import get_component_slice
from pyUSID.processing.process import Process, parallel_compute
from pyUSID.io.hdf_utils import reshape_to_n_dims, create_results_group, write_main_dataset, get_attr, \
    write_simple_attrs, link_h5_obj_as_alias, write_ind_val_dsets
from pyUSID import USIDataset
from pyUSID.io.io_utils import format_time
from pyUSID.io.write_utils import Dimension
from pyUSID.io.dtype_utils import check_dtype, stack_real_to_target_dtype


class Cluster(Process):
    """
    Pycroscopy wrapper around the sklearn.cluster classes.
    """

    def __init__(self, h5_main, estimator, num_comps=None, **kwargs):
        """
        Constructs the Cluster object
        Parameters
        ----------
        h5_main : HDF5 dataset object
            Main dataset with ancillary spectroscopic, position indices and values datasets
        estimator : sklearn.cluster estimator
            configured clustering algorithm to be applied to the data
        num_comps : (optional) unsigned int
            Number of features / spectroscopic indices to be used to cluster the data. Default = all
        args and kwargs : arguments to be passed to the estimator
        """

        allowed_methods = [cls.AgglomerativeClustering,
                           cls.Birch,
                           cls.KMeans,
                           cls.MiniBatchKMeans,
                           cls.SpectralClustering]

        # could not find a nicer way to extract the method name yet
        self.method_name = str(estimator)[:str(estimator).index('(')]

        if type(estimator) not in allowed_methods:
            raise TypeError('Cannot work with {} just yet'.format(self.method_name))

        # Done with decomposition-related checks, now call super init
        super(Cluster, self).__init__(h5_main, **kwargs)

        # Store the decomposition object
        self.estimator = estimator

        if num_comps is None:
            comp_attr = 'all'

        comp_slice, num_comps = get_component_slice(num_comps, total_components=self.h5_main.shape[1])

        self.num_comps = num_comps
        self.data_slice = (slice(None), comp_slice)

        if isinstance(comp_slice, slice):
            # cannot store slice as an attribute in hdf5
            # convert to list of integers!
            inds = comp_slice.indices(self.h5_main.shape[1])
            # much like range, inds are arranged as (start, stop, step)
            if inds[0] == 0 and inds[2] == 1:
                # starting from 0 with step of 1 = upto N components
                if inds[1] >= self.h5_main.shape[1] - 1:
                    comp_attr = 'all'
                else:
                    comp_attr = inds[1]
            else:
                comp_attr = range(*inds)
        elif comp_attr == 'all':
            pass
        else:
            # subset of spectral components specified as an array
            comp_attr = comp_slice

        # set up parameters
        self.parms_dict = {'cluster_algorithm': self.method_name,
                           'spectral_components': comp_attr}
        self.parms_dict.update(self.estimator.get_params())

        # update n_jobs according to the cores argument
        # print('cores reset to', self._cores)
        # different number of cores should not* be a reason for different results
        # so we update this flag only after checking for duplicates
        estimator.n_jobs = self._cores
        self.parms_dict.update({'n_jobs': self._cores})

        # check for existing datagroups with same results
        self.process_name = 'Cluster'
        # Partial groups don't make any sense for statistical learning algorithms....
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        # figure out the operation that needs need to be performed to convert to real scalar
        (self.data_transform_func, self.data_is_complex, self.data_is_compound,
         self.data_n_features, self.data_type_mult) = check_dtype(h5_main)

        # supercharge h5_main!
        self.h5_main = USIDataset(self.h5_main)

        self.__labels = None
        self.__mean_resp = None

    def test(self, rearrange_clusters=True, override=False):
        """
        Clusters the hdf5 dataset and calculates mean response for each cluster. This function does NOT write results to
        the hdf5 file. Call compute() to  write to the file. Handles complex, compound datasets such that the
        mean response vector for each cluster matrix is of the same data-type as the input matrix.

        Parameters
        ----------
        rearrange_clusters : bool, optional. Default = True
            Whether or not the clusters should be re-ordered by relative distances between the mean response
        override : bool, optional. default = False
            Set to true to recompute results if prior results are available. Else, returns existing results

        Returns
        -------
        labels : 1D unsigned int array
            Array of cluster labels as obtained from the fit
        mean_response : 2D numpy array
            Array of the mean response for each cluster arranged as [cluster number, response]
        """
        if not override:
            if isinstance(self.duplicate_h5_groups, list) and len(self.duplicate_h5_groups) > 0:
                self.h5_results_grp = self.duplicate_h5_groups[-1]
                print('Returning previously computed results from: {}'.format(self.h5_results_grp.name))
                print('set the "override" flag to True to recompute results')
                return np.squeeze(reshape_to_n_dims(self.h5_results_grp['Labels'])[0]), \
                       reshape_to_n_dims(self.h5_results_grp['Mean_Response'])[0]

        self.h5_results_grp = None

        t1 = time.time()

        print('Performing clustering on {}.'.format(self.h5_main.name))
        # perform fit on the real dataset
        results = self.estimator.fit(self.data_transform_func(self.h5_main[self.data_slice]))

        print('Took {} to compute {}'.format(format_time(time.time() - t1), self.method_name))

        t1 = time.time()
        self.__mean_resp = self._get_mean_response(results.labels_)
        print('Took {} to calculate mean response per cluster'.format(format_time(time.time() - t1)))

        self.__labels = results.labels_
        if rearrange_clusters:
            self.__labels, self.__mean_resp = reorder_clusters(results.labels_, self.__mean_resp,
                                                               self.data_transform_func)

        # TODO: What if test() is called repeatedly?
        labels_mat, success = reshape_to_n_dims(np.expand_dims(np.squeeze(self.__labels), axis=1),
                                                h5_pos=self.h5_main.h5_pos_inds, h5_spec=np.expand_dims([0], axis=0))
        if not success:
            raise ValueError('Could not reshape labels to N-Dimensional dataset! Error:' + success)

        centroid_mat, success = reshape_to_n_dims(self.__mean_resp,
                                                  h5_spec=self.h5_main.h5_spec_inds[:, :self.num_comps],
                                                  h5_pos=np.expand_dims(np.arange(self.__mean_resp.shape[0]), axis=1))

        if not success:
            raise ValueError('Could not reshape mean response to N-Dimensional dataset! Error:' + success)

        return np.squeeze(labels_mat), centroid_mat

    def delete_results(self):
        """
        Deletes results from memory.
        """
        del self.__labels, self.__mean_resp
        self.__labels = None
        self.__mean_resp = None
        self.h5_results_grp = None

    def compute(self, rearrange_clusters=True, override=False):
        """
        Clusters the hdf5 dataset and calculates mean response for each cluster (by calling test() if it has
        not already been called), and writes the labels and mean response back to the h5 file.

        Consider calling test_on_subset() to check results before writing to file. Results are deleted from memory
        upon writing to the HDF5 file

        Parameters
        ----------
        rearrange_clusters : bool, optional. Default = True
            Whether or not the clusters should be re-ordered by relative distances between the mean response
        override : bool, optional. default = False
            Set to true to recompute results if prior results are available. Else, returns existing results

        Returns
        --------
        h5_group : HDF5 Group reference
            Reference to the group that contains the clustering results
        """
        if self.__labels is None and self.__mean_resp is None:
            _ = self.test(rearrange_clusters=rearrange_clusters, override=override)

        if self.h5_results_grp is None:
            h5_group = self._write_results_chunk()
            self.delete_results()
        else:
            h5_group = self.h5_results_grp

        return h5_group

    def _get_mean_response(self, labels):
        """
        Gets the mean response for each cluster

        Parameters
        -------------
        labels : 1D unsigned int array
            Array of cluster labels as obtained from the fit

        Returns
        ---------
        mean_resp : 2D numpy array
            Array of the mean response for each cluster arranged as [cluster number, response]
        """
        print('Calculated the Mean Response of each cluster.')
        num_clusts = len(np.unique(labels))

        def __mean_resp_for_cluster(clust_ind, h5_raw, labels_vec, data_slice, xform_func):
            # get all pixels with this label
            targ_pos = np.argwhere(labels_vec == clust_ind)
            # slice to get the responses for all these pixels, ensure that it's 2d
            data_chunk = np.atleast_2d(h5_raw[:, data_slice[1]][targ_pos, :])
            # transform to real from whatever type it was
            avg_data = np.mean(xform_func(data_chunk), axis=0, keepdims=True)
            # transform back to the source data type and insert into the mean response
            return np.squeeze(stack_real_to_target_dtype(avg_data, h5_raw.dtype))

        # TODO: Force usage of multiple threads. This should not take 3 cores
        mean_resp = np.array(parallel_compute(np.arange(num_clusts), __mean_resp_for_cluster,
                                              func_args=[self.h5_main, labels, self.data_slice,
                                                         self.data_transform_func], lengthy_computation=True,
                                              verbose=self.verbose))

        return mean_resp

    def _write_results_chunk(self):
        """
        Writes the labels and mean response to the h5 file

        Returns
        ---------
        h5_group : HDF5 Group reference
            Reference to the group that contains the clustering results
        """
        print('Writing clustering results to file.')
        num_clusters = self.__mean_resp.shape[0]

        h5_cluster_group = create_results_group(self.h5_main, self.process_name)

        write_simple_attrs(h5_cluster_group, self.parms_dict)
        h5_cluster_group.attrs['last_pixel'] = self.h5_main.shape[0]

        h5_labels = write_main_dataset(h5_cluster_group, np.uint32(self.__labels.reshape([-1, 1])), 'Labels',
                                       'Cluster ID', 'a. u.', None, Dimension('Cluster', 'ID', 1),
                                       h5_pos_inds=self.h5_main.h5_pos_inds, h5_pos_vals=self.h5_main.h5_pos_vals,
                                       aux_spec_prefix='Cluster_', dtype=np.uint32)

        if self.num_comps != self.h5_main.shape[1]:
            '''
            Setup the Spectroscopic Indices and Values for the Mean Response if we didn't use all components
            Note that a sliced spectroscopic matrix may not be contiguous. Let's just lose the spectroscopic data
            for now until a better method is figured out
            '''
            """
            if isinstance(self.data_slice[1], np.ndarray):
                centroid_vals_mat = h5_centroids.h5_spec_vals[self.data_slice[1].tolist()]

            else:
                centroid_vals_mat = h5_centroids.h5_spec_vals[self.data_slice[1]]

            ds_centroid_values.data[0, :] = centroid_vals_mat
            """
            if isinstance(self.data_slice[1], np.ndarray):
                vals_slice = self.data_slice[1].tolist()
            else:
                vals_slice = self.data_slice[1]
            vals = self.h5_main.h5_spec_vals[:, vals_slice].squeeze()
            new_spec = Dimension('Original_Spectral_Index', 'a.u.', vals)
            h5_inds, h5_vals = write_ind_val_dsets(h5_cluster_group, new_spec, is_spectral=True)

        else:
            h5_inds = self.h5_main.h5_spec_inds
            h5_vals = self.h5_main.h5_spec_vals

        # For now, link centroids with default spectroscopic indices and values.
        h5_centroids = write_main_dataset(h5_cluster_group, self.__mean_resp, 'Mean_Response',
                                          get_attr(self.h5_main, 'quantity')[0], get_attr(self.h5_main, 'units')[0],
                                          Dimension('Cluster', 'a. u.', np.arange(num_clusters)), None,
                                          h5_spec_inds=h5_inds, aux_pos_prefix='Mean_Resp_Pos_',
                                          h5_spec_vals=h5_vals)

        return h5_cluster_group


def reorder_clusters(labels, mean_response, transform_function=None):
    """
    Reorders clusters by the distances between the clusters

    Parameters
    ----------
    labels : 1D unsigned int numpy array
        Labels for the clusters
    mean_response : 2D numpy array
        Mean response of each cluster arranged as [cluster , features]
    transform_function : callable, optional
        Function that will convert the mean_response into real values

    Returns
    -------
    new_labels : 1D unsigned int numpy array
        Labels for the clusters arranged by distances
    new_mean_response : 2D numpy array
        Mean response of each cluster arranged as [cluster , features]
    """

    num_clusters = mean_response.shape[0]
    # Get the distance between cluster means
    if transform_function is not None:
        distance_mat = pdist(transform_function(mean_response))
    else:
        distance_mat = pdist(mean_response)

    # get hierarchical pairings of clusters
    linkage_pairing = linkage(distance_mat, 'weighted')

    # get the new order - this has been checked to be OK
    new_cluster_order = []
    for row in range(linkage_pairing.shape[0]):
        for col in range(2):
            if linkage_pairing[row, col] < num_clusters:
                new_cluster_order.append(int(linkage_pairing[row, col]))

    # Now that we know the order, rearrange the clusters and labels:
    new_labels = np.zeros(shape=labels.shape, dtype=labels.dtype)
    new_mean_response = np.zeros(shape=mean_response.shape, dtype=mean_response.dtype)

    # Reorder clusters
    for old_clust_ind, new_clust_ind in enumerate(new_cluster_order):
        new_labels[np.where(labels == new_clust_ind)[0]] = old_clust_ind
        new_mean_response[old_clust_ind] = mean_response[new_clust_ind]

    return new_labels, new_mean_response
