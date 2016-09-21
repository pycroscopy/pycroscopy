# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:45:08 2016

@author: Suhas Somnath, Chris Smith
"""

import time
import numpy as np
from warnings import warn
from sklearn.utils.extmath import randomized_svd
from ..io.io_hdf5 import ioHDF5
from ..io.hdf_utils import getH5DsetRefs, checkAndLinkAncillary, \
    getH5RegRefIndices, createRefFromIndices, checkIfMain
from ..io.io_utils import check_dtype
from ..io.microdata import MicroDataset, MicroDataGroup


###############################################################################

def doSVD(h5_main, num_comps=None):
    '''
    Does SVD on the provided dataset and writes the result. File is not closed

    Parameters:
    ---------
    h5_main : h5py.Dataset reference
        Reference to the dataset on which SVD will be performed
    num_comps : Unsigned integer (Optional)
        Number of principal components of interest
    max_mem : integer (Optional)
        Maximum amount of memory, in Mb, that the code is allowed to use

    * Note : If ancillary datasets are not provided, thus function will attempt
    to find any appropriately named datasets linked to the main dataset. Linking
    is recommended to make it easier for data agnostic visualization

    Returns:
    ----------
    h5_pca : h5py.Datagroup reference
        Reference to the group containing the PCA results
    '''

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
        num_comps = n_samples
    else:
        num_comps = min(n_samples, num_comps)

    '''
    Check if a number of compnents has been set and ensure that the number is less than
    the minimum axis length of the data.  If both conditions are met, use fsvd.  If not
    use the regular svd.

    C.Smith -- We might need to put a lower limit on num_comps in the future.  I don't
               know enough about svd to be sure.
    '''
    print 'Performing SVD decomposition'


    U, S, V = randomized_svd(func(h5_main), num_comps, n_iter=3)

    pca_type = 'sklearn-randomized'

    print 'PCA took {} seconds.  Writing results to file.'.format((time.time() - t1))

    '''
    Create datasets for V and S, deleting original arrays afterward to save
    memory.
    '''
    ds_S = MicroDataset('S', data=np.float32(S))
    ds_inds = MicroDataset('Component_Indices', data=np.uint32(np.arange(len(S))))
    del S

    ds_U = MicroDataset('U', data=np.float32(U))
    del U

    if is_complex:
        # Put the real and imaginary sections together to make complex V
        V = V[:, :int(0.5 * V.shape[1])] + 1j * V[:, int(0.5 * V.shape[1]):]
        ds_V = MicroDataset('V', data=np.complex64(V))
    elif is_compound:
        V2 = np.empty([V.shape[0], h5_main.shape[1]], dtype=h5_main.dtype)
        for iname, name in enumerate(h5_main.dtype.names):
            istart = iname * V2.shape[1]
            iend = (iname + 1) * V2.shape[1]
            V2[name] = V[:, istart:iend]

        ds_V = MicroDataset('V', data=V2)
        del V2
    else:
        ds_V = MicroDataset('V', data=np.float32(V))
    del V

    '''
    Create the Group to hold the results and add the existing datasets as
    children
    '''
    grp_name = dset_name + '-PCA_'
    pca_grp = MicroDataGroup(grp_name, h5_main.parent.name[1:])
    pca_grp.addChildren([ds_V, ds_S, ds_U, ds_inds])

    '''
    Write the attributes to the group
    '''
    pca_grp.attrs['num_components'] = num_comps
    pca_grp.attrs['pca_method'] = pca_type

    '''
    Write the data and retrieve the HDF5 objects then delete the Microdatasets
    '''
    hdf = ioHDF5(h5_main.file)
    h5_pca_refs = hdf.writeData(pca_grp)

    h5_U = getH5DsetRefs(['U'], h5_pca_refs)[0]
    h5_S = getH5DsetRefs(['S'], h5_pca_refs)[0]
    h5_V = getH5DsetRefs(['V'], h5_pca_refs)[0]
    h5_pca_inds = getH5DsetRefs(['Component_Indices'], h5_pca_refs)[0]
    h5_pca_grp = h5_S.parent

    del ds_S, ds_V, ds_U, pca_grp

    # Will attempt to see if there is anything linked to this dataset.
    # Since I was meticulous about the translators that I wrote, I know I will find something here
    checkAndLinkAncillary(h5_U,
                          ['Position_Indices', 'Position_Values'],
                          h5_main=h5_main)

    checkAndLinkAncillary(h5_V,
                          ['Position_Indices', 'Position_Values'])

    checkAndLinkAncillary(h5_U,
                          ['Spectroscopic_Indices', 'Spectroscopic_Values'])

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

        pca_ref = createRefFromIndices(h5_V, ref_inds)

        h5_V.attrs[key] = pca_ref

    return h5_pca_grp