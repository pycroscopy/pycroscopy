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
from ..core.processing.process import Process
from ..core.io.hdf_utils import get_h5_obj_refs, check_and_link_ancillary, copy_main_attributes
from ..core.io.write_utils import build_ind_val_dsets, AuxillaryDescriptor
from ..core.io.hdf_writer import HDFwriter
from ..core.io.dtype_utils import check_dtype, stack_real_to_target_dtype
from ..core.io.virtual_data import VirtualGroup, VirtualDataset


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
        self.duplicate_h5_groups = self._check_for_duplicates()

        # figure out the operation that needs need to be performed to convert to real scalar
        (self.data_transform_func, self.data_is_complex, self.data_is_compound,
         self.data_n_features, self.data_n_samples, self.data_type_mult) = check_dtype(h5_main)

        self.__labels = None
        self.__mean_resp = None

    def test(self, rearrange_clusters=True):
        """
        Clusters the hdf5 dataset and calculates mean response for each cluster. This function does NOT write results to
        the hdf5 file. Call compute() to  write to the file. Handles complex, compound datasets such that the
        mean response vector for each cluster matrix is of the same data-type as the input matrix.

        Parameters
        ----------
        rearrange_clusters : bool, optional. Default = True
            Whether or not the clusters should be re-ordered by relative distances between the mean response

        Returns
        -------
        labels : 1D unsigned int array
            Array of cluster labels as obtained from the fit
        mean_response : 2D numpy array
            Array of the mean response for each cluster arranged as [cluster number, response]
        """
        t1 = time.time()

        print('Performing clustering on {}.'.format(self.h5_main.name))
        # perform fit on the real dataset
        results = self.estimator.fit(self.data_transform_func(self.h5_main[self.data_slice]))

        print('Took {} seconds to compute {}'.format(round(time.time() - t1, 2), self.method_name))

        t1 = time.time()
        self.__mean_resp = self._get_mean_response(results.labels_)
        print('Took {} seconds to calculate mean response per cluster'.format(round(time.time() - t1, 2)))

        self.__labels = results.labels_
        if rearrange_clusters:
            self.__labels, self.__mean_resp = reorder_clusters(results.labels_, self.__mean_resp)

        # TODO: Return N dimensional form instead of 2D!
        return self.__labels, self.__mean_resp

    def delete_results(self):
        """
        Deletes results from memory.
        """
        del self.__labels, self.__mean_resp
        self.__labels = None
        self.__mean_resp = None

    def compute(self, rearrange_clusters=True):
        """
        Clusters the hdf5 dataset and calculates mean response for each cluster (by calling test() if it has
        not already been called), and writes the labels and mean response back to the h5 file.

        Consider calling test_on_subset() to check results before writing to file. Results are deleted from memory
        upon writing to the HDF5 file

        Parameters
        ----------
        rearrange_clusters : bool, optional. Default = True
            Whether or not the clusters should be re-ordered by relative distances between the mean response

        Returns
        --------
        h5_group : HDF5 Group reference
            Reference to the group that contains the clustering results
        """
        if self.__labels is None and self.__mean_resp is None:
            self.test(rearrange_clusters=rearrange_clusters)

        h5_group = self._write_results_chunk()
        self.delete_results()

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
        mean_resp = np.zeros(shape=(num_clusts, self.num_comps), dtype=self.h5_main.dtype)
        # TODO: Oppurtunity to do this in parallel
        for clust_ind in range(num_clusts):
            # get all pixels with this label
            targ_pos = np.argwhere(labels == clust_ind)
            # slice to get the responses for all these pixels, ensure that it's 2d
            data_chunk = np.atleast_2d(self.h5_main[:, self.data_slice[1]][targ_pos, :])
            # transform to real from whatever type it was
            avg_data = np.mean(self.data_transform_func(data_chunk), axis=0, keepdims=True)
            # transform back to the source data type and insert into the mean response
            mean_resp[clust_ind] = stack_real_to_target_dtype(avg_data, self.h5_main.dtype)
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
        ds_labels = VirtualDataset('Labels', np.uint32(self.__labels.reshape([-1, 1])), dtype=np.uint32)
        ds_labels.attrs['quantity'] = 'Cluster ID'
        ds_labels.attrs['units'] = 'a. u.'

        clust_desc = AuxillaryDescriptor([num_clusters], ['Cluster'], ['a. u.'])
        ds_centroid_inds, ds_centroid_vals = build_ind_val_dsets(clust_desc, is_spectral=False, base_name='Cluster_')

        ds_centroids = VirtualDataset('Mean_Response', self.__mean_resp, dtype=self.__mean_resp.dtype)
        # Main attributes will be copied from h5_main after writing
        lab_desc = AuxillaryDescriptor([1], ['Cluster'], ['ID'])
        ds_label_inds, ds_label_vals = build_ind_val_dsets(lab_desc, is_spectral=True, base_name='Label_Spectroscopic_')

        cluster_grp = VirtualGroup(self.h5_main.name.split('/')[-1] + '-' + self.process_name + '_',
                                   self.h5_main.parent.name[1:])
        cluster_grp.add_children([ds_labels, ds_centroids, ds_centroid_inds, ds_centroid_vals, ds_label_inds,
                                  ds_label_vals])

        # Write out all the parameters including those from the estimator
        cluster_grp.attrs.update(self.parms_dict)

        h5_spec_inds = self.h5_main.file[self.h5_main.attrs['Spectroscopic_Indices']]
        h5_spec_vals = self.h5_main.file[self.h5_main.attrs['Spectroscopic_Values']]

        '''
        Setup the Spectroscopic Indices and Values for the Mean Response if we didn't use all components
        '''
        if self.num_comps != self.h5_main.shape[1]:
            comp_desc = AuxillaryDescriptor([self.num_comps], ['Spectroscopic_Component'], ['a.u.'])
            ds_centroid_indices, ds_centroid_values = build_ind_val_dsets(comp_desc, is_spectral=True,
                                                                          base_name='Mean_Response_')

            if isinstance(self.data_slice[1], np.ndarray):
                centroid_vals_mat = h5_spec_vals[self.data_slice[1].tolist()]

            else:
                centroid_vals_mat = h5_spec_vals[self.data_slice[1]]

            ds_centroid_values.data[0, :] = centroid_vals_mat

            cluster_grp.add_children([ds_centroid_indices, ds_centroid_values])

        hdf = HDFwriter(self.h5_main.file)
        h5_clust_refs = hdf.write(cluster_grp)

        h5_labels = get_h5_obj_refs(['Labels'], h5_clust_refs)[0]
        h5_centroids = get_h5_obj_refs(['Mean_Response'], h5_clust_refs)[0]
        h5_clust_inds = get_h5_obj_refs(['Cluster_Indices'], h5_clust_refs)[0]
        h5_clust_vals = get_h5_obj_refs(['Cluster_Values'], h5_clust_refs)[0]
        h5_label_inds = get_h5_obj_refs(['Label_Spectroscopic_Indices'], h5_clust_refs)[0]
        h5_label_vals = get_h5_obj_refs(['Label_Spectroscopic_Values'], h5_clust_refs)[0]

        copy_main_attributes(self.h5_main, h5_centroids)

        if isinstance(self.data_slice[1], np.ndarray):
            h5_mean_resp_inds = get_h5_obj_refs(['Mean_Response_Indices'], h5_clust_refs)[0]
            h5_mean_resp_vals = get_h5_obj_refs(['Mean_Response_Values'], h5_clust_refs)[0]
        else:
            h5_mean_resp_inds = h5_spec_inds
            h5_mean_resp_vals = h5_spec_vals

        check_and_link_ancillary(h5_labels,
                              ['Position_Indices', 'Position_Values'],
                              h5_main=self.h5_main)
        check_and_link_ancillary(h5_labels,
                              ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                              anc_refs=[h5_label_inds, h5_label_vals])

        check_and_link_ancillary(h5_centroids,
                              ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                              anc_refs=[h5_mean_resp_inds, h5_mean_resp_vals])

        check_and_link_ancillary(h5_centroids,
                              ['Position_Indices', 'Position_Values'],
                              anc_refs=[h5_clust_inds, h5_clust_vals])

        # return the h5 group object
        return h5_labels.parent


def reorder_clusters(labels, mean_response):
    """
    Reorders clusters by the distances between the clusters

    Parameters
    ----------
    labels : 1D unsigned int numpy array
        Labels for the clusters
    mean_response : 2D numpy array
        Mean response of each cluster arranged as [cluster , features]

    Returns
    -------
    new_labels : 1D unsigned int numpy array
        Labels for the clusters arranged by distances
    new_mean_response : 2D numpy array
        Mean response of each cluster arranged as [cluster , features]
    """

    num_clusters = mean_response.shape[0]
    # Get the distance between cluster means
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
