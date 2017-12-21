"""
Created on 7/17/16 10:08 AM
@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import psutil
import joblib

from ..io.hdf_utils import checkIfMain, check_for_old, get_attributes
from ..io.io_hdf5 import ioHDF5
from ..io.io_utils import recommendCores, getAvailableMem


def calc_func(some_func, args):
    """

    Parameters
    ----------
    some_func
    args

    Returns
    -------

    """

    return some_func(*args)


def setup_func(args):
    """

    Parameters
    ----------
    args

    Returns
    -------

    """

    return calc_func(*args)


class Process(object):
    """
    Encapsulates the typical steps performed when applying a processing function to  a dataset.
    """

    def __init__(self, h5_main, h5_results_grp=None, cores=None, max_mem_mb=4*1024, verbose=False):
        """
        Parameters
        ----------
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.
        h5_results_grp : h5py.Datagroup object, optional
            Datagroup containing partially computed results
        cores : uint, optional
            Default - all available cores - 2
            How many cores to use for the computation
        max_mem_mb : uint, optional
            How much memory to use for the computation.  Default 1024 Mb
        verbose : Boolean, (Optional, default = False)
            Whether or not to print debugging statements
        """

        if h5_main.file.mode != 'r+':
            raise TypeError('Need to ensure that the file is in r+ mode to write results back to the file')

        # Checking if dataset is "Main"
        if checkIfMain(h5_main):
            self.h5_main = h5_main
            self.hdf = ioHDF5(self.h5_main.file)
        else:
            raise ValueError('Provided dataset is not a "Main" dataset with necessary ancillary datasets')

        self.verbose = verbose
        self._max_pos_per_read = None
        self._max_mem_mb = None

        self._start_pos = 0
        self._end_pos = self.h5_main.shape[0]

        # Determining the max size of the data that can be put into memory
        self._set_memory_and_cores(cores=cores, mem=max_mem_mb)
        self.duplicate_h5_groups = []
        self.process_name = None  # Reset this in the extended classes
        self.parms_dict = None

        self._results = None
        self.h5_results_grp = h5_results_grp
        if self.h5_results_grp is not None:
            self._extract_params(h5_results_grp)

        # DON'T check for duplicates since parms_dict has not yet been initialized.
        # Sub classes will check by themselves if they are interested.

    def _check_for_duplicates(self):
        """
        Checks for instances where the process was applied to the same dataset with the same parameters

        Returns
        -------
        duplicate_h5_groups : list of h5py.Datagroup objects
            List of groups satisfying the above conditions
        """
        duplicate_h5_groups = check_for_old(self.h5_main, self.process_name, new_parms=self.parms_dict)
        if self.verbose:
            print('Checking for duplicates:')
        if duplicate_h5_groups is not None:
            print('WARNING! ' + self.process_name + ' has already been performed with the same parameters before. '
                                                    'Consider reusing results')
            print(duplicate_h5_groups)
        return duplicate_h5_groups

    def _extract_params(self, h5_partial_group):
        """
        Extracts the necessary parameters from the provided h5 group to resume computation

        Parameters
        ----------
        h5_partial_group : h5py.Datagroup object
            Datagroup containing partially computed results

        """
        self.parms_dict = get_attributes(h5_partial_group)
        self._start_pos = self.parms_dict.pop('last_pixel')
        if self._start_pos == self.h5_main.shape[0] - 1:
            raise ValueError('The last computed pixel shows that the computation was already complete')

    def _set_memory_and_cores(self, cores=1, mem=1024):
        """
        Checks hardware limitations such as memory, # cpus and sets the recommended datachunk sizes and the
        number of cores to be used by analysis methods.

        Parameters
        ----------
        cores : uint, optional
            Default - 1
            How many cores to use for the computation
        mem : uint, optional
            Default - 1024
            The amount a memory in Mb to use in the computation
        """

        if cores is None:
            self._cores = psutil.cpu_count() - 2
        else:
            cores = int(abs(cores))
            self._cores = min(psutil.cpu_count(), cores)

        self._cores = max(1, self._cores)
        _max_mem_mb = getAvailableMem() / 1E6  # in MB

        self._max_mem_mb = min(_max_mem_mb, mem)

        max_data_chunk = self._max_mem_mb / self._cores

        # Now calculate the number of positions that can be stored in memory in one go.
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1] / 1e6
        self._max_pos_per_read = int(np.floor(max_data_chunk / mb_per_position))

        if self.verbose:
            print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))
            print('Allowed to use up to', str(self._cores), 'cores and', str(self._max_mem_mb), 'MB of memory')

    @staticmethod
    def _unit_function(*args):
        raise NotImplementedError('Please override the _unit_function specific to your process')

    def _read_data_chunk(self):
        """
        Reads a chunk of data for the intended computation into memory

       Parameters
        -----
        verbose : bool, optional
            Whether or not to print log statements
        """
        if self._start_pos < self.h5_main.shape[0]:
            self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
            self.data = self.h5_main[self._start_pos:self._end_pos, :]
            if self.verbose:
                print('Reading pixels {} to {} of {}'.format(self._start_pos, self._end_pos, self.h5_main.shape[0]))

            # DON'T update the start position

        else:
            if self.verbose:
                print('Finished reading all data!')
            self.data = None

    def _write_results_chunk(self):
        """
        Writes the computed results into appropriate datasets.

        This needs to be rewritten since the processed data is expected to be at least as large as the dataset
        """
        # Now update the start position
        self._start_pos = self._end_pos
        raise NotImplementedError('Please override the _set_results specific to your process')

    def _create_results_datasets(self):
        """
        Process specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.
        """
        raise NotImplementedError('Please override the _create_results_datasets specific to your process')

    def _get_existing_datasets(self):
        """
        The purpose of this function is to allow processes to resume from partly computed results

        Start with self.h5_results_grp
        """
        raise NotImplementedError('Please override the _get_existing_datasets specific to your process')

    def compute(self, *args, **kwargs):
        """

        Parameters
        ----------
        kwargs:
            processors: int
                number of processors to use. Default all processors on the system except for 1.

        Returns
        -------

        """
        if self._start_pos == 0:
            # starting fresh
            self._create_results_datasets()
        else:
            # resuming from previous checkpoint
            self._get_existing_datasets()

        self._read_data_chunk()
        while self.data is not None:
            self._results = parallel_compute(self.data, self._unit_function(), cores=self._cores,
                                             lengthy_computation=False,
                                             func_args=args, func_kwargs=kwargs)
            self._write_results_chunk()
            self._read_data_chunk()

        print('Completed computation on chunk. Writing to file.')

        return self.h5_results_grp


def parallel_compute(data, func, cores=1, lengthy_computation=False, func_args=list(), func_kwargs=dict()):
    """
    Computes the guess function using multiple cores

    Parameters
    ----------
    data : numpy.ndarray
        Data to map function to. Function will be mapped to the first axis of data
    func : callable
        Function to map to data
    cores : uint, optional
        Number of logical cores to use to compute
        Default - 1 (serial computation)
    lengthy_computation : bool, optional
        Whether or not each computation is expected to take substantial time.
        Sometimes the time for adding more cores can outweigh the time per core
        Default - False
    func_args : list, optional
        arguments to be passed to the function
    func_kwargs : dict, optional
        keyword arguments to be passed onto function

    Returns
    -------
    results : list
        List of computational results
    """

    if not callable(func):
        raise TypeError('Function argument is not callable')

    cores = recommendCores(data.shape[0], requested_cores=cores,
                           lengthy_computation=lengthy_computation)

    if cores > 1:
        values = [joblib.delayed(func)(x, *func_args, **func_kwargs) for x in data]
        results = joblib.Parallel(n_jobs=cores)(values)

        # Finished reading the entire data set
        print('Finished parallel computation')

    else:
        print("Computing serially ...")
        results = [func(vector, *func_args, **func_kwargs) for vector in data]

    return results
