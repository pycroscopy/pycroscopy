# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import sklearn.cluster as cls
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from .process import Process
from ..io.hdf_utils import getH5DsetRefs, checkAndLinkAncillary, copy_main_attributes
from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import check_dtype, transformToTargetType, to_ranges
from ..io.microdata import MicroDataGroup, MicroDataset


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

        comp_slice, num_comps = self._get_component_slice(num_comps)

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

    def compute(self, rearrange_clusters=True):
        """
        Clusters the hdf5 dataset, calculates mean response for each cluster, and writes the labels and mean response
        back to the h5 file

        Parameters
        ----------
        rearrange_clusters : (Optional) Boolean. Default = True
            Whether or not the clusters should be re-ordered by relative distances between the mean response

        Returns
        --------
        h5_group : HDF5 Group reference
            Reference to the group that contains the clustering results
        """
        self._fit()
        new_mean_response = self._get_mean_response(self.results.labels_)
        new_labels = self.results.labels_
        if rearrange_clusters:
            new_labels, new_mean_response = reorder_clusters(self.results.labels_, new_mean_response)
        return self._write_results_chunk(new_labels, new_mean_response)

    def _fit(self):
        """
        Fits the provided dataset
        """
        print('Performing clustering on {}.'.format(self.h5_main.name))
        # perform fit on the real dataset
        self.results = self.estimator.fit(self.data_transform_func(self.h5_main[self.data_slice]))

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
        for clust_ind in range(num_clusts):
            # get all pixels with this label
            targ_pos = np.argwhere(labels == clust_ind)
            # slice to get the responses for all these pixels, ensure that it's 2d
            data_chunk = np.atleast_2d(self.h5_main[:, self.data_slice[1]][targ_pos, :])
            # transform to real from whatever type it was
            avg_data = np.mean(self.data_transform_func(data_chunk), axis=0, keepdims=True)
            # transform back to the source data type and insert into the mean response
            mean_resp[clust_ind] = transformToTargetType(avg_data, self.h5_main.dtype)
        return mean_resp

    def _get_component_slice(self, components):
        """
        Check the components object to determine how to use it to slice the dataset

        Parameters
        ----------
        components : {int, array-like of ints, slice, or None}
            Input Options
            integer: Components less than the input will be kept
            length 2 iterable of integers: Integers define start and stop of component slice to retain
            other iterable of integers or slice: Selection of component indices to retain
            None: All components will be used

        Returns
        -------
        comp_slice : slice or numpy.ndarray of uints
            Slice or array specifying which components should be kept
        """

        if components is None:
            num_comps = self.h5_main.shape[1]
            comp_slice = slice(0, num_comps)
        elif isinstance(components, int):
            # Component is integer
            num_comps = int(np.min([components, self.h5_main.shape[1]]))
            comp_slice = slice(0, num_comps)
        elif hasattr(components, '__iter__') and not isinstance(components, dict):
            # Component is array, list, or tuple
            if len(components) == 2:
                # If only 2 numbers are given, use them as the start and stop of a slice
                comp_slice = slice(int(components[0]), int(components[1]))
                num_comps = abs(comp_slice.stop - comp_slice.start)
            else:
                # Convert components to an unsigned integer array
                comp_slice = np.uint(components)
                # sort and take unique values only
                comp_slice.sort()
                comp_slice = np.unique(comp_slice)
                num_comps = len(comp_slice)
                # check to see if this giant list of integers is just a simple range
                list_of_ranges = list(to_ranges(comp_slice))
                if len(list_of_ranges) == 1:
                    # increment the second index by 1 to be consistent with python
                    comp_slice = slice(int(list_of_ranges[0][0]), int(list_of_ranges[0][1] + 1))

        elif isinstance(components, slice):
            # Components is already a slice
            comp_slice = components
            num_comps = abs(comp_slice.stop - comp_slice.start)
        else:
            raise TypeError('Unsupported component type supplied to clean_and_build.  '
                            'Allowed types are integer, numpy array, list, tuple, and slice.')

        return comp_slice, num_comps

    def _write_results_chunk(self, labels, mean_response):
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
        print('Writing clustering results to file.')
        num_clusters = mean_response.shape[0]
        ds_label_mat = MicroDataset('Labels', np.uint32(labels.reshape([-1, 1])), dtype=np.uint32)
        ds_label_mat.attrs['quantity'] = 'Cluster ID'
        ds_label_mat.attrs['units'] = 'a. u.'

        clust_ind_mat = np.transpose(np.atleast_2d(np.arange(num_clusters)))

        ds_cluster_inds = MicroDataset('Cluster_Indices', np.uint32(clust_ind_mat))
        ds_cluster_vals = MicroDataset('Cluster_Values', np.float32(clust_ind_mat))
        ds_cluster_centroids = MicroDataset('Mean_Response', mean_response, dtype=mean_response.dtype)
        # Main attributes will be copied from h5_main after writing
        ds_label_inds = MicroDataset('Label_Spectroscopic_Indices', np.atleast_2d([0]), dtype=np.uint32)
        ds_label_vals = MicroDataset('Label_Spectroscopic_Values', np.atleast_2d([0]), dtype=np.float32)

        # write the labels and the mean response to h5
        clust_slices = {'Cluster': (slice(None), slice(0, 1))}
        ds_cluster_inds.attrs['labels'] = clust_slices
        ds_cluster_inds.attrs['units'] = ['']
        ds_cluster_vals.attrs['labels'] = clust_slices
        ds_cluster_vals.attrs['units'] = ['']

        cluster_grp = MicroDataGroup(self.h5_main.name.split('/')[-1] + '-' + self.process_name + '_', self.h5_main.parent.name[1:])
        cluster_grp.addChildren([ds_label_mat, ds_cluster_centroids, ds_cluster_inds, ds_cluster_vals, ds_label_inds,
                                 ds_label_vals])

        # Write out all the parameters including those from the estimator
        cluster_grp.attrs.update(self.parms_dict)

        h5_spec_inds = self.h5_main.file[self.h5_main.attrs['Spectroscopic_Indices']]
        h5_spec_vals = self.h5_main.file[self.h5_main.attrs['Spectroscopic_Values']]

        '''
        Setup the Spectroscopic Indices and Values for the Mean Response if we didn't use all components
        '''
        if self.num_comps != self.h5_main.shape[1]:
            ds_centroid_indices = MicroDataset('Mean_Response_Indices', np.arange(self.num_comps, dtype=np.uint32))

            if isinstance(self.data_slice[1], np.ndarray):
                centroid_vals_mat = h5_spec_vals[self.data_slice[1].tolist()]

            else:
                centroid_vals_mat = h5_spec_vals[self.data_slice[1]]

            ds_centroid_values = MicroDataset('Mean_Response_Values', centroid_vals_mat)

            cluster_grp.addChildren([ds_centroid_indices, ds_centroid_values])

        hdf = ioHDF5(self.h5_main.file)
        h5_clust_refs = hdf.writeData(cluster_grp)

        # Get the h5 objects we just created
        h5_labels = getH5DsetRefs(['Labels'], h5_clust_refs)[0]
        h5_centroids = getH5DsetRefs(['Mean_Response'], h5_clust_refs)[0]
        h5_clust_inds = getH5DsetRefs(['Cluster_Indices'], h5_clust_refs)[0]
        h5_clust_vals = getH5DsetRefs(['Cluster_Values'], h5_clust_refs)[0]
        h5_label_inds = getH5DsetRefs(['Label_Spectroscopic_Indices'], h5_clust_refs)[0]
        h5_label_vals = getH5DsetRefs(['Label_Spectroscopic_Values'], h5_clust_refs)[0]

        # Add the attributes to label spectroscopic datasets
        h5_label_inds.attrs['labels'] = np.array([''], dtype='S')
        h5_label_inds.attrs['units'] = np.array([''], dtype='S')
        h5_label_vals.attrs['labels'] = np.array([''], dtype='S')
        h5_label_vals.attrs['units'] = np.array([''], dtype='S')

        copy_main_attributes(self.h5_main, h5_centroids)

        if isinstance(self.data_slice[1], np.ndarray):
            h5_mean_resp_inds = getH5DsetRefs(['Mean_Response_Indices'], h5_clust_refs)[0]
            h5_mean_resp_vals = getH5DsetRefs(['Mean_Response_Values'], h5_clust_refs)[0]
            h5_mean_resp_inds.attrs['labels'] = np.array([''], dtype='S')
            h5_mean_resp_inds.attrs['units'] = np.array([''], dtype='S')
            h5_mean_resp_vals.attrs['labels'] = np.array([''], dtype='S')
            h5_mean_resp_vals.attrs['units'] = np.array([''], dtype='S')
        else:
            h5_mean_resp_inds = h5_spec_inds
            h5_mean_resp_vals = h5_spec_vals

        checkAndLinkAncillary(h5_labels,
                              ['Position_Indices', 'Position_Values'],
                              h5_main=self.h5_main)
        checkAndLinkAncillary(h5_labels,
                              ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                              anc_refs=[h5_label_inds, h5_label_vals])

        checkAndLinkAncillary(h5_centroids,
                              ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                              anc_refs=[h5_mean_resp_inds, h5_mean_resp_vals])

        checkAndLinkAncillary(h5_centroids,
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
