"""
Created on 7/17/16 10:08 AM
@author: Suhas Somnath, Chris Smith
"""

from warnings import warn

import numpy as np
import psutil
import scipy

from ..io.hdf_utils import checkIfMain
from ..io.io_hdf5 import ioHDF5
import multiprocessing as mp


class Process(object):
    """
    Encapsulates the typical steps performed when applying a processing function to  a dataset.
    """

    def __init__(self, h5_main):
        """


        Parameters:
        ----
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.
        """
        # Checking if dataset is "Main"
        if checkIfMain(h5_main):
            self.h5_main = h5_main
            self.hdf = ioHDF5(self.h5_main.file)
        else:
            warn('Provided dataset is not a "Main" dataset with necessary ancillary datasets')
            return

        # Determining the max size of the data that can be put into memory
        self._setMemoryAndCPUs()

        self._start_pos = 0
        self._end_pos = self.h5_main.shape[0]

    def _setMemoryAndCPUs(self):
        """
        Checks hardware limitations such as memory, # cpus and sets the recommended datachunk sizes and the
        number of cores to be used by analysis methods.

        Returns
        -------
        None
        """

        if self._parallel:
            self._maxCpus = psutil.cpu_count() - 2
        else:
            self._maxCpus = 1
        self._maxMemoryMB = psutil.virtual_memory().available / 1e6 # in MB

        self._maxDataChunk = self._maxMemoryMB / self._maxCpus

        # Now calculate the number of positions that can be stored in memory in one go.
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1]/1e6
        self._max_pos_per_read = int(np.floor(self._maxDataChunk / mb_per_position))
        print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))

    def _get_data_chunk(self):
        """
        Returns a chunk of data for the guess or the fit

        Parameters:
        -----
        None

        Returns:
        --------
        """
        if self._start_pos < self.h5_main.shape[0]:
            self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
            self.data = self.h5_main[self._start_pos:self._end_pos, :]
            print('Reading pixels {} to {} of {}'.format(self._start_pos, self._end_pos, self.h5_main.shape[0]))
        else:
            print('Finished reading all data!')
            self.data = None

    def _set_results(self):
        """
        Writes the provided guess or fit results into appropriate datasets.
        Given that the guess and fit datasets are relatively small, we should be able to hold them in memory just fine

        """
        # Now update the start position
        self._start_pos = self._end_pos
        warn('Please override the _set_results specific to your model')
        pass

    def _create_results_datasets(self):
        """
        Model specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.
        """
        warn('Please override the _create_results_datasets specific to your model')
        pass

    def compute(self, func, **kwargs):
        """

        Parameters
        ----------
        data
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. For updated list, run GuessMethods.methods
        options: dict
            Default {"peaks_widths": np.array([10,200])}}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.

        kwargs:
            processors: int
                number of processors to use. Default all processors on the system except for 1.

        Returns
        -------

        """

        self._create_results_datasets()
        self._start_pos = 0

        processors = kwargs.get("processors", self._maxCpus)
        results = list()
        if self._parallel:
            # start pool of workers
            print('Computing in parallel ... launching %i kernels...' % processors)
            pool = mp.Pool(processors)
            self._get_data_chunk()
            while self.data is not None:  # as long as we have not reached the end of this data set:
                # apply guess to this data chunk:
                tasks = [vector for vector in self.data]
                chunk = int(self.data.shape[0] / processors)
                jobs = pool.imap(func, tasks, chunksize=chunk)
                # get Results from different processes
                print('Extracting Results...')
                temp = [j for j in jobs]
                # Reformat the data to the appropriate type and or do additional computation now

                # Write to file
                self._set_results(is_guess=True)
                # read the next chunk
                self._get_data_chunk()

            # Finished reading the entire data set
            print('closing %i kernels...' % processors)
            pool.close()
        else:
            print("Computing Guesses In Serial ...")
            self._get_data_chunk()
            while self.data is not None:  # as long as we have not reached the end of this data set:
                temp = [func(vector) for vector in self.data]
                results.append(self._reformatResults(temp, strategy))
                # read the next chunk
                self._get_data_chunk()

        # reorder to get one numpy array out
        self.guess = np.hstack(tuple(results))
        print('Completed computing guess. Writing to file.')

