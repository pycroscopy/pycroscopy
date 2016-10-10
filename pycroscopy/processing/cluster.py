# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith
"""
import numpy as np
import sklearn.cluster as cls

from ..io.hdf_utils import checkIfMain
from ..io.hdf_utils import getH5DsetRefs, checkAndLinkAncillary
from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import check_dtype, transformToTargetType
from ..io.microdata import MicroDataGroup, MicroDataset


class Cluster(object):
    """
    Pycroscopy wrapper around the sklearn.cluster classes.
    """

    def __init__(self, h5_main, method_name, num_comps=None, *args, **kwargs):
        """
        Constructs the Cluster object

        Parameters
        ------------
        h5_main : HDF5 dataset object
            Main dataset with ancillary spectroscopic, position indices and values datasets
        method_name : string / unicode
            Name of the sklearn.cluster estimator
        num_comps : (optional) unsigned int
            Number of features / spectroscopic indices to be used to cluster the data. Default = all
        *args and **kwargs : arguments to be passed to the estimator
        """

        allowed_methods = ['AgglomerativeClustering','Birch','KMeans',
                           'MiniBatchKMeans','SpectralClustering']
        
        # check if h5_main is a valid object - is it a hub?
        if not checkIfMain(h5_main):
            raise TypeError('Supplied dataset is not a pycroscopy main dataset')

        if method_name not in allowed_methods:
            raise TypeError('Cannot work with {} just yet'.format(method_name))

        self.h5_main = h5_main

        # Instantiate the clustering object
        self.estimator = cls.__dict__[method_name].__call__(*args, **kwargs)
        self.method_name = method_name

        if num_comps is None:
            self.num_comps = self.h5_main.shape[1]
        else:
            self.num_comps = np.min([num_comps, self.h5_main.shape[1]])
        self.data_slice = (slice(None), slice(0, num_comps))

        # figure out the operation that needs need to be performed to convert to real scalar
        retval = check_dtype(h5_main)
        self.data_transform_func, self.data_is_complex, self.data_is_compound, \
        self.data_n_features, self.data_n_samples, self.data_type_mult = retval


    def doCluster(self):
        """
        Clusters the hdf5 dataset, calculates mean response for each cluster, and writes the labels and mean response back to the h5 file

        Returns
        --------
        h5_group : HDF5 Group reference
            Reference to the group that contains the clustering results
        """
        self._fit()
        mean_response = self._getMeanResponse(self.results.labels_)
        return self._writeToHDF5(self.results.labels_, mean_response)

    def _fit(self):
        """
        Fits the provided dataset

        Returns
        ------
        None
        """
        # perform fit on the real dataset
        self.results = self.estimator.fit(self.data_transform_func(self.h5_main[self.data_slice]))
        

    def _getMeanResponse(self, labels):
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
        num_clusts = len(np.unique(labels))
        mean_resp = np.zeros(shape=(num_clusts, self.num_comps), dtype=self.h5_main.dtype)
        for clust_ind in xrange(num_clusts):
            # get all pixels with this label
            targ_pos = np.where(labels == clust_ind)[0]
            # slice to get the responses for all these pixels, ensure that it's 2d
            data_chunk = np.atleast_2d(self.h5_main[targ_pos, self.data_slice[1]])
            #transform to real from whatever type it was
            avg_data = np.mean(self.data_transform_func(data_chunk), axis=0, keepdims=True)
            # transform back to the source data type and insert into the mean response
            mean_resp[clust_ind] = transformToTargetType(avg_data, self.h5_main.dtype)
        return mean_resp

    def _writeToHDF5(self, labels, mean_response):
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
        num_clusters = mean_response.shape[0]
        ds_label_mat = MicroDataset('Labels', np.float32(labels), dtype=np.float32)
        clust_ind_mat = np.transpose(np.atleast_2d(np.arange(num_clusters)))

        ds_cluster_inds = MicroDataset('Cluster_Indices', np.uint32(clust_ind_mat))
        ds_cluster_vals = MicroDataset('Cluster_Values', np.float32(clust_ind_mat))
        ds_cluster_centroids = MicroDataset('Mean_Response', mean_response, dtype=mean_response.dtype)

        # write the labels and the mean response to h5
        clust_slices = {'Cluster': (slice(None), slice(0, 1))}
        ds_cluster_inds.attrs['labels'] = clust_slices
        ds_cluster_inds.attrs['units'] = ['']
        ds_cluster_vals.attrs['labels'] = clust_slices
        ds_cluster_vals.attrs['units'] = ['']

        cluster_grp = MicroDataGroup(self.h5_main.name.split('/')[-1] + '-Cluster_', self.h5_main.parent.name[1:])
        cluster_grp.addChildren([ds_label_mat, ds_cluster_centroids, ds_cluster_inds, ds_cluster_vals])

        cluster_grp.attrs['num_clusters'] = num_clusters
        cluster_grp.attrs['num_samples'] = self.h5_main.shape[0]
        cluster_grp.attrs['cluster_algorithm'] = self.method_name
        if self.num_comps is not None:
            cluster_grp.attrs['components_used'] = self.num_comps

        '''
        Get the parameters of the estimator used and write them
        as attributes of the group
        '''
        for parm, val in self.estimator.get_params().iteritems():
            cluster_grp.attrs[parm] = val

        hdf = ioHDF5(self.h5_main.file)
        h5_clust_refs = hdf.writeData(cluster_grp)

        h5_labels = getH5DsetRefs(['Labels'], h5_clust_refs)[0]
        h5_centroids = getH5DsetRefs(['Mean_Response'], h5_clust_refs)[0]
        h5_clust_inds = getH5DsetRefs(['Cluster_Indices'], h5_clust_refs)[0]
        h5_clust_vals = getH5DsetRefs(['Cluster_Values'], h5_clust_refs)[0]

        checkAndLinkAncillary(h5_labels,
                              ['Position_Indices', 'Position_Values'],
                              h5_main=self.h5_main)

        checkAndLinkAncillary(h5_centroids,
                              ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                              h5_main=self.h5_main)

        checkAndLinkAncillary(h5_centroids,
                              ['Position_Indices', 'Position_Values'],
                              anc_refs=[h5_clust_inds, h5_clust_vals])

        # return the h5 group object
        return h5_labels.parent