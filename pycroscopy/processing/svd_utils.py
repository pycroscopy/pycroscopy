# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:45:08 2016

@author: Suhas Somnath, Chris Smith
"""

import time
from warnings import warn

import numpy as np
from sklearn.utils.extmath import randomized_svd

from ..io.hdf_utils import getH5DsetRefs, checkAndLinkAncillary, \
    getH5RegRefIndices, createRefFromIndices, checkIfMain, calc_chunks
from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import check_dtype
from ..io.microdata import MicroDataset, MicroDataGroup


###############################################################################

def doSVD(h5_main, num_comps=None):
    """
    Does SVD on the provided dataset and writes the result. File is not closed

    Parameters
    ----------
    h5_main : h5py.Dataset reference
        Reference to the dataset on which SVD will be performed
    num_comps : Unsigned integer (Optional)
        Number of principal components of interest

    Returns
    -------
    h5_pca : h5py.Datagroup reference
        Reference to the group containing the PCA results
    """

    if not checkIfMain(h5_main):
        warn('Dataset does not meet requirements for performing PCA.')
        return

    dset_name = h5_main.name.split('/')[-1]

    t1 = time.time()

    '''
    Calculate the size of the main data in memory and compare to max_mem
    We use the minimum of the actual dtype's itemsize and float32 since we
    don't want to read it in yet and do the proper type conversions.
    '''
    func, is_complex, is_compound, n_features, n_samples, type_mult = check_dtype(h5_main)

    if num_comps is None:
        num_comps = min(n_samples, n_features)
    else:
        num_comps = min(n_samples, n_features, num_comps)

    '''
    Check if a number of compnents has been set and ensure that the number is less than
    the minimum axis length of the data.  If both conditions are met, use fsvd.  If not
    use the regular svd.

    C.Smith -- We might need to put a lower limit on num_comps in the future.  I don't
               know enough about svd to be sure.
    '''
    print 'Performing SVD decomposition'


    U, S, V = randomized_svd(func(h5_main), num_comps, n_iter=3)

    svd_type = 'sklearn-randomized'

    print 'SVD took {} seconds.  Writing results to file.'.format((time.time() - t1))

    '''
    Create datasets for V and S, deleting original arrays afterward to save
    memory.
    '''
    ds_S = MicroDataset('S', data=np.float32(S))
    ds_inds = MicroDataset('Component_Indices', data=np.uint32(np.arange(len(S))))
    del S

    u_chunks = calc_chunks(U.shape, np.float32(0).itemsize, unit_chunks=[1, num_comps])
    ds_U = MicroDataset('U', data=np.float32(U), chunking=u_chunks)
    del U

    if is_complex:
        # Put the real and imaginary sections together to make complex V
        V = V[:, :int(0.5 * V.shape[1])] + 1j * V[:, int(0.5 * V.shape[1]):]
        v_chunks = calc_chunks(V.shape, h5_main.dtype.itemsize, unit_chunks=(num_comps, 1))
        ds_V = MicroDataset('V', data=np.complex64(V), chunking=v_chunks)
    elif is_compound:
        V2 = np.empty([V.shape[0], h5_main.shape[1]], dtype=h5_main.dtype)
        for iname, name in enumerate(h5_main.dtype.names):
            istart = iname * V2.shape[1]
            iend = (iname + 1) * V2.shape[1]
            V2[name] = V[:, istart:iend]
        v_chunks = calc_chunks(V2.shape, h5_main.dtype.itemsize, unit_chunks=(num_comps, 1))
        ds_V = MicroDataset('V', data=V2, chunking=v_chunks)
        del V2
    else:
        v_chunks = calc_chunks(V.shape, h5_main.dtype.itemsize, unit_chunks=(num_comps, 1))
        ds_V = MicroDataset('V', data=np.float32(V), chunking=v_chunks)
    del V

    '''
    Create the Group to hold the results and add the existing datasets as
    children
    '''
    grp_name = dset_name + '-SVD_'
    svd_grp = MicroDataGroup(grp_name, h5_main.parent.name[1:])
    svd_grp.addChildren([ds_V, ds_S, ds_U, ds_inds])

    '''
    Write the attributes to the group
    '''
    svd_grp.attrs['num_components'] = num_comps
    svd_grp.attrs['svd_method'] = svd_type

    '''
    Write the data and retrieve the HDF5 objects then delete the Microdatasets
    '''
    hdf = ioHDF5(h5_main.file)
    h5_svd_refs = hdf.writeData(svd_grp)

    h5_U = getH5DsetRefs(['U'], h5_svd_refs)[0]
    h5_S = getH5DsetRefs(['S'], h5_svd_refs)[0]
    h5_V = getH5DsetRefs(['V'], h5_svd_refs)[0]
    h5_svd_inds = getH5DsetRefs(['Component_Indices'], h5_svd_refs)[0]
    h5_svd_grp = h5_S.parent

    del ds_S, ds_V, ds_U, svd_grp

    # Will attempt to see if there is anything linked to this dataset.
    # Since I was meticulous about the translators that I wrote, I know I will find something here
    checkAndLinkAncillary(h5_U,
                          ['Position_Indices', 'Position_Values'],
                          h5_main=h5_main)

    checkAndLinkAncillary(h5_V,
                          ['Position_Indices', 'Position_Values'],
                          anc_refs=[h5_svd_inds, h5_S])

    checkAndLinkAncillary(h5_U,
                          ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                          anc_refs=[h5_svd_inds, h5_S])

    checkAndLinkAncillary(h5_V,
                          ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                          h5_main=h5_main)

    '''
    Check h5_main for plot group references.
    Copy them into V if they exist
    '''
    for key, ref in h5_main.attrs.iteritems():
        if '_Plot_Group' not in key:
            continue

        ref_inds = getH5RegRefIndices(ref, h5_main, return_method='corners')
        ref_inds = ref_inds.reshape([-1, 2, 2])
        ref_inds[:, 1, 0] = h5_V.shape[0] - 1

        svd_ref = createRefFromIndices(h5_V, ref_inds)

        h5_V.attrs[key] = svd_ref

    return h5_svd_grp


###############################################################################

def simplifiedKPCA(kpca, source_data):
    """
    Performs kernel PCA on the provided dataset and returns the familiar
    eigenvector, eigenvalue, and scree matrices.

    Note that the positions in the eigenvalues may need to be transposed

    Parameters
    ----------
    kpca : KernelPCA object
        configured Kernel PCA object ready to perform analysis
    source_data : 2D numpy array
        Data arranged as [iteration, features] example - [position, time]

    Returns
    -------
    eigenvalues : 2D numpy array
        Eigenvalues in the original space arranged as [component,iteration]
    scree : 1D numpy array
        S component
    eigenvector : 2D numpy array
        Eigenvectors in the original space arranged as [component,features]

    """
    X_kpca = kpca.fit(source_data.T)
    eigenvectors = X_kpca.alphas_.T
    eigenvalues = X_kpca.fit_transform(source_data)
    # kpca_explained_variance = np.var(kpca.fit_transform(source_data), axis=0)
    # information_content = kpca_explained_variance / np.sum(kpca_explained_variance)
    scree = kpca.lambdas_
    return eigenvalues, scree, eigenvectors