# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:45:08 2016

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import
import time
from multiprocessing import cpu_count
import numpy as np
from sklearn.utils import gen_batches
from sklearn.utils.extmath import randomized_svd

from pyUSID.processing.process import Process
from .proc_utils import get_component_slice
from pyUSID.io.hdf_utils import find_results_groups, get_indices_for_region_ref, \
    create_region_reference, copy_attributes, reshape_to_n_dims, get_attr, write_main_dataset, \
    create_results_group, write_simple_attrs, create_indexed_group
from pyUSID.io.io_utils import get_available_memory, format_time
from pyUSID.io.dtype_utils import check_dtype, stack_real_to_target_dtype
from pyUSID.io.write_utils import Dimension, calc_chunks
from pyUSID import USIDataset


class SVD(Process):

    def __init__(self, h5_main, num_components=None):

        super(SVD, self).__init__(h5_main)
        self.process_name = 'SVD'

        '''
        Calculate the size of the main data in memory and compare to max_mem
        We use the minimum of the actual dtype's itemsize and float32 since we
        don't want to read it in yet and do the proper type conversions.
        '''
        n_samples, n_features = h5_main.shape
        self.data_transform_func, is_complex, is_compound, n_features, type_mult = check_dtype(h5_main)

        if num_components is None:
            num_components = min(n_samples, n_features)
        else:
            num_components = min(n_samples, n_features, num_components)
        self.num_components = num_components
        self.parms_dict = {'num_components': num_components}
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        # supercharge h5_main!
        self.h5_main = USIDataset(self.h5_main)

        self.__u = None
        self.__v = None
        self.__s = None

    def test(self, override=False):
        """
        Applies randomised VD to the dataset. This function does NOT write results to the hdf5 file. Call compute() to
        write to the file. Handles complex, compound datasets such that the V matrix is of the same data-type as the
        input matrix.

        Parameters
        ----------
        override : bool, optional. default = False
            Set to true to recompute results if prior results are available. Else, returns existing results

        Returns
        -------
        U : numpy.ndarray
            Abundance matrix
        S : numpy.ndarray
            variance vector
        V : numpy.ndarray
            eigenvector matrix
        """
        '''
        Check if a number of compnents has been set and ensure that the number is less than
        the minimum axis length of the data.  If both conditions are met, use fsvd.  If not
        use the regular svd.

        C.Smith -- We might need to put a lower limit on num_comps in the future.  I don't
                   know enough about svd to be sure.
        '''
        if not override:
            if isinstance(self.duplicate_h5_groups, list) and len(self.duplicate_h5_groups) > 0:
                self.h5_results_grp = self.duplicate_h5_groups[-1]
                print('Returning previously computed results from: {}'.format(self.h5_results_grp.name))
                print('set the "override" flag to True to recompute results')
                return reshape_to_n_dims(self.h5_results_grp['U'])[0], self.h5_results_grp['S'][()], \
                       reshape_to_n_dims(self.h5_results_grp['V'])[0]

        self.h5_results_grp = None

        t1 = time.time()

        self.__u, self.__s, self.__v = randomized_svd(self.data_transform_func(self.h5_main), self.num_components,
                                                      n_iter=3)
        self.__v = stack_real_to_target_dtype(self.__v, self.h5_main.dtype)

        print('Took {} to compute randomized SVD'.format(format_time(time.time() - t1)))

        u_mat, success = reshape_to_n_dims(self.__u, h5_pos=self.h5_main.h5_pos_inds,
                                           h5_spec=np.expand_dims(np.arange(self.__u.shape[1]), axis=0))
        if not success:
            raise ValueError('Could not reshape U to N-Dimensional dataset! Error:' + success)

        v_mat, success = reshape_to_n_dims(self.__v, h5_pos=np.expand_dims(np.arange(self.__u.shape[1]), axis=1),
                                           h5_spec=self.h5_main.h5_spec_inds)
        if not success:
            raise ValueError('Could not reshape V to N-Dimensional dataset! Error:' + success)

        return u_mat, self.__s, v_mat

    def compute(self, override=False):
        """
        Computes SVD (by calling test_on_subset() if it has not already been called) and writes results to file.
        Consider calling test() to check results before writing to file. Results are deleted from memory
        upon writing to the HDF5 file

        Parameters
        ----------
        override : bool, optional. default = False
            Set to true to recompute results if prior results are available. Else, returns existing results

        Returns
        -------
         h5_results_grp : h5py.Datagroup object
            Datagroup containing all the results
        """
        if self.__u is None and self.__v is None and self.__s is None:
            self.test(override=override)

        if self.h5_results_grp is None:
            self._write_results_chunk()
            self.delete_results()

        h5_group = self.h5_results_grp

        return h5_group

    def delete_results(self):
        """
        Deletes results from memory.
        """
        del self.__u, self.__s, self.__v
        self.__u = None
        self.__v = None
        self.__s = None

    def _write_results_chunk(self):
        """
        Writes the provided SVD results to file

        Parameters
        ----------
        """
        comp_dim = Dimension('Principal Component', 'a. u.', len(self.__s))

        h5_svd_group = create_results_group(self.h5_main, self.process_name)
        self.h5_results_grp = h5_svd_group

        write_simple_attrs(h5_svd_group, self.parms_dict)
        write_simple_attrs(h5_svd_group, {'svd_method': 'sklearn-randomized', 'last_pixel': self.h5_main.shape[0]})

        h5_u = write_main_dataset(h5_svd_group, np.float32(self.__u), 'U', 'Abundance', 'a.u.', None, comp_dim,
                                  h5_pos_inds=self.h5_main.h5_pos_inds, h5_pos_vals=self.h5_main.h5_pos_vals,
                                  dtype=np.float32, chunks=calc_chunks(self.__u.shape, np.float32(0).itemsize))
        # print(get_attr(self.h5_main, 'quantity')[0])
        h5_v = write_main_dataset(h5_svd_group, self.__v, 'V', get_attr(self.h5_main, 'quantity')[0],
                                  'a.u.', comp_dim, None, h5_spec_inds=self.h5_main.h5_spec_inds,
                                  h5_spec_vals=self.h5_main.h5_spec_vals,
                                  chunks=calc_chunks(self.__v.shape, self.h5_main.dtype.itemsize))

        # No point making this 1D dataset a main dataset
        h5_s = h5_svd_group.create_dataset('S', data=np.float32(self.__s))

        '''
        Check h5_main for plot group references.
        Copy them into V if they exist
        '''
        for key in self.h5_main.attrs.keys():
            if '_Plot_Group' not in key:
                continue

            ref_inds = get_indices_for_region_ref(self.h5_main, self.h5_main.attrs[key], return_method='corners')
            ref_inds = ref_inds.reshape([-1, 2, 2])
            ref_inds[:, 1, 0] = h5_v.shape[0] - 1

            svd_ref = create_region_reference(h5_v, ref_inds)

            h5_v.attrs[key] = svd_ref


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
    comp_slice, num_comps = get_component_slice(components, total_components=h5_main.shape[1])
    if isinstance(comp_slice, np.ndarray):
        comp_slice = list(comp_slice)
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
        h5_svd_group = find_results_groups(h5_main, 'SVD')[-1]

        h5_S = h5_svd_group['S']
        h5_U = h5_svd_group['U']
        h5_V = h5_svd_group['V']

    except KeyError:
        raise KeyError('SVD Results for {dset} were not found.'.format(dset=dset_name))
    except:
        raise

    func, is_complex, is_compound, n_features, type_mult = check_dtype(h5_V)

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

    rebuild = stack_real_to_target_dtype(rebuild, h5_V.dtype)

    print('Completed reconstruction of data from SVD results.  Writing to file.')
    '''
    Create the Group and dataset to hold the rebuild data
    '''
    rebuilt_grp = create_indexed_group(h5_svd_group, 'Rebuilt_Data')
    h5_rebuilt = write_main_dataset(rebuilt_grp, rebuild, 'Rebuilt_Data',
                                    get_attr(h5_main, 'quantity'), get_attr(h5_main, 'units'),
                                    None, None,
                                    h5_pos_inds=h5_main.h5_pos_inds, h5_pos_vals=h5_main.h5_pos_vals,
                                    h5_spec_inds=h5_main.h5_spec_inds, h5_spec_vals=h5_main.h5_spec_vals,
                                    chunks=h5_main.chunks, compression=h5_main.compression)

    if isinstance(comp_slice, slice):
        rebuilt_grp.attrs['components_used'] = '{}-{}'.format(comp_slice.start, comp_slice.stop)
    else:
        rebuilt_grp.attrs['components_used'] = components

    copy_attributes(h5_main, h5_rebuilt, skip_refs=False)

    h5_main.file.flush()

    print('Done writing reconstructed data to file.')

    return h5_rebuilt
