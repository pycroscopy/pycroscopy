# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:45:08 2016

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import
import time
from warnings import warn
from multiprocessing import cpu_count
import numpy as np
from sklearn.utils import gen_batches
from sklearn.utils.extmath import randomized_svd

from ..core.processing.process import Process
from .proc_utils import get_component_slice
from ..core.io.hdf_utils import get_h5_obj_refs, check_and_link_ancillary, find_results_groups, \
    get_indices_for_region_ref, create_region_reference, calc_chunks, copy_main_attributes, copy_attributes
from pycroscopy.core.io.hdf_writer import HDFwriter
from ..core.io.io_utils import get_available_memory
from ..core.io.dtype_utils import check_dtype, transform_to_target_dtype
from ..core.io.microdata import MicroDataset, MicroDataGroup


class SVD(Process):

    def __init__(self, h5_main, num_components=None):

        super(SVD, self).__init__(h5_main)
        self.process_name = 'SVD'

        '''
        Calculate the size of the main data in memory and compare to max_mem
        We use the minimum of the actual dtype's itemsize and float32 since we
        don't want to read it in yet and do the proper type conversions.
        '''
        self.data_transform_func, is_complex, is_compound, n_features, n_samples, type_mult = check_dtype(h5_main)

        if num_components is None:
            num_components = min(n_samples, n_features)
        else:
            num_components = min(n_samples, n_features, num_components)
        self.num_components = num_components
        self.parms_dict = {'num_components': num_components}
        self.duplicate_h5_groups = self._check_for_duplicates()

    def compute(self):
        """
        Computes SVD and writes results to file

        Returns
        -------
         h5_results_grp : h5py.Datagroup object
            Datagroup containing all the results
        """


        '''
        Check if a number of compnents has been set and ensure that the number is less than
        the minimum axis length of the data.  If both conditions are met, use fsvd.  If not
        use the regular svd.

        C.Smith -- We might need to put a lower limit on num_comps in the future.  I don't
                   know enough about svd to be sure.
        '''
        print('Performing SVD')

        t1 = time.time()

        U, S, V = randomized_svd(self.data_transform_func(self.h5_main), self.num_components, n_iter=3)

        print('SVD took {} seconds.  Writing results to file.'.format(round(time.time() - t1, 2)))

        self._write_results_chunk(U, S, V)
        del U, S, V

        return self.h5_results_grp

    def _write_results_chunk(self, U, S, V):
        """
        Writes the provided SVD results to file

        Parameters
        ----------
        U : array-like
            Abundance matrix
        S : array-like
            variance vector
        V : array-like
            eigenvector matrix
        """

        ds_S = MicroDataset('S', data=np.float32(S))
        ds_S.attrs['labels'] = {'Principal Component': [slice(0, None)]}
        ds_S.attrs['units'] = ['']
        ds_inds = MicroDataset('Component_Indices', data=np.uint32(np.arange(len(S))))
        ds_inds.attrs['labels'] = {'Principal Component': [slice(0, None)]}
        ds_inds.attrs['units'] = ['']
        del S

        u_chunks = calc_chunks(U.shape, np.float32(0).itemsize)
        ds_U = MicroDataset('U', data=np.float32(U), chunking=u_chunks)
        del U

        V = transform_to_target_dtype(V, self.h5_main.dtype)
        v_chunks = calc_chunks(V.shape, self.h5_main.dtype.itemsize)
        ds_V = MicroDataset('V', data=V, chunking=v_chunks)
        del V

        '''
        Create the Group to hold the results and add the existing datasets as
        children
        '''
        grp_name = self.h5_main.name.split('/')[-1] + '-' + self.process_name + '_'
        svd_grp = MicroDataGroup(grp_name, self.h5_main.parent.name[1:])
        svd_grp.add_children([ds_V, ds_S, ds_U, ds_inds])

        '''
        Write the attributes to the group
        '''
        svd_grp.attrs = self.parms_dict
        svd_grp.attrs.update({'svd_method': 'sklearn-randomized', 'last_pixel': self.h5_main.shape[0] - 1})

        '''
        Write the data and retrieve the HDF5 objects then delete the Microdatasets
        '''
        hdf = HDFwriter(self.h5_main.file)
        h5_svd_refs = hdf.write(svd_grp)

        h5_U = get_h5_obj_refs(['U'], h5_svd_refs)[0]
        h5_S = get_h5_obj_refs(['S'], h5_svd_refs)[0]
        h5_V = get_h5_obj_refs(['V'], h5_svd_refs)[0]
        h5_svd_inds = get_h5_obj_refs(['Component_Indices'], h5_svd_refs)[0]
        self.h5_results_grp = h5_S.parent

        # copy attributes
        copy_main_attributes(self.h5_main, h5_V)
        h5_V.attrs['units'] = np.array(['a. u.'], dtype='S')

        del ds_S, ds_V, ds_U, svd_grp

        # Will attempt to see if there is anything linked to this dataset.
        # Since I was meticulous about the translators that I wrote, I know I will find something here
        check_and_link_ancillary(h5_U,
                                 ['Position_Indices', 'Position_Values'],
                                 h5_main=self.h5_main)

        check_and_link_ancillary(h5_V,
                                 ['Position_Indices', 'Position_Values'],
                                 anc_refs=[h5_svd_inds, h5_S])

        check_and_link_ancillary(h5_U,
                                 ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                                 anc_refs=[h5_svd_inds, h5_S])

        check_and_link_ancillary(h5_V,
                                 ['Spectroscopic_Indices', 'Spectroscopic_Values'],
                                 h5_main=self.h5_main)

        '''
        Check h5_main for plot group references.
        Copy them into V if they exist
        '''
        for key in self.h5_main.attrs.keys():
            if '_Plot_Group' not in key:
                continue

            ref_inds = get_indices_for_region_ref(self.h5_main.attrs[key], self.h5_main, return_method='corners')
            ref_inds = ref_inds.reshape([-1, 2, 2])
            ref_inds[:, 1, 0] = h5_V.shape[0] - 1

            svd_ref = create_region_reference(h5_V, ref_inds)

            h5_V.attrs[key] = svd_ref

###############################################################################


def simplified_kpca(kpca, source_data):
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


def rebuild_svd(h5_main, components=None, cores=None, max_RAM_mb=1024):
    """
    Rebuild the Image from the SVD results on the windows
    Optionally, only use components less than n_comp.

    Parameters
    ----------
    h5_main : hdf5 Dataset
        dataset which SVD was performed on
    components : {int, iterable of int, slice} optional
        Defines which components to keep
        Default - None, all components kept

        Input Types
        integer : Components less than the input will be kept
        length 2 iterable of integers : Integers define start and stop of component slice to retain
        other iterable of integers or slice : Selection of component indices to retain
    cores : int, optional
        How many cores should be used to rebuild
        Default - None, all but 2 cores will be used, min 1
    max_RAM_mb : int, optional
        Maximum ammount of memory to use when rebuilding, in Mb.
        Default - 1024Mb

    Returns
    -------
    rebuilt_data : HDF5 Dataset
        the rebuilt dataset

    """

    hdf = HDFwriter(h5_main.file)
    comp_slice, num_comps = get_component_slice(components, total_components=h5_main.shape[1])
    dset_name = h5_main.name.split('/')[-1]

    # Ensuring that at least one core is available for use / 2 cores are available for other use
    max_cores = max(1, cpu_count() - 2)
    #         print('max_cores',max_cores)
    if cores is not None:
        cores = min(round(abs(cores)), max_cores)
    else:
        cores = max_cores

    max_memory = min(max_RAM_mb * 1024 ** 2, 0.75 * get_available_memory())
    if cores != 1:
        max_memory = int(max_memory / 2)

    '''
    Get the handles for the SVD results
    '''
    try:
        h5_svd = find_results_groups(h5_main, 'SVD')[-1]

        h5_S = h5_svd['S']
        h5_U = h5_svd['U']
        h5_V = h5_svd['V']

    except KeyError:
        warnstring = 'SVD Results for {dset} were not found.'.format(dset=dset_name)
        warn(warnstring)
        return
    except:
        raise

    func, is_complex, is_compound, n_features, n_samples, type_mult = check_dtype(h5_V)

    '''
    Calculate the size of a single batch that will fit in the available memory
    '''
    n_comps = h5_S[comp_slice].size
    mem_per_pix = (h5_U.dtype.itemsize + h5_V.dtype.itemsize * h5_V.shape[1]) * n_comps
    fixed_mem = h5_main.size * h5_main.dtype.itemsize

    if cores is None:
        free_mem = max_memory - fixed_mem
    else:
        free_mem = max_memory * 2 - fixed_mem

    batch_size = int(round(float(free_mem) / mem_per_pix))
    batch_slices = gen_batches(h5_U.shape[0], batch_size)

    print('Reconstructing in batches of {} positions.'.format(batch_size))
    print('Batchs should be {} Mb each.'.format(mem_per_pix * batch_size / 1024.0 ** 2))

    '''
    Loop over all batches.
    '''
    ds_V = np.dot(np.diag(h5_S[comp_slice]), func(h5_V[comp_slice, :]))
    rebuild = np.zeros((h5_main.shape[0], ds_V.shape[1]))
    for ibatch, batch in enumerate(batch_slices):
        rebuild[batch, :] += np.dot(h5_U[batch, comp_slice], ds_V)

    rebuild = transform_to_target_dtype(rebuild, h5_V.dtype)

    print('Completed reconstruction of data from SVD results.  Writing to file.')
    '''
    Create the Group and dataset to hold the rebuild data
    '''
    rebuilt_grp = MicroDataGroup('Rebuilt_Data_', h5_svd.name[1:])

    ds_rebuilt = MicroDataset('Rebuilt_Data', rebuild,
                              chunking=h5_main.chunks,
                              compression=h5_main.compression)
    rebuilt_grp.add_children([ds_rebuilt])

    if isinstance(comp_slice, slice):
        rebuilt_grp.attrs['components_used'] = '{}-{}'.format(comp_slice.start, comp_slice.stop)
    else:
        rebuilt_grp.attrs['components_used'] = components

    h5_refs = hdf.write(rebuilt_grp)

    h5_rebuilt = get_h5_obj_refs(['Rebuilt_Data'], h5_refs)[0]
    copy_attributes(h5_main, h5_rebuilt, skip_refs=False)

    hdf.flush()

    print('Done writing reconstructed data to file.')

    return h5_rebuilt
