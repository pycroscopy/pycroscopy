"""
Created on 7/17/16 10:08 AM
@author: Numan Laanait, Suhas Somnath
"""

from warnings import warn

import numpy as np
import psutil
import scipy
from .guess_methods import GuessMethods
from .fit_methods import Fit_Methods
from ..io.hdf_utils import checkIfMain, getAuxData
from ..io.io_hdf5 import ioHDF5
from .optimize import Optimize


class Model(object):
    """
    Encapsulates the typical routines performed during model-dependent analysis of data.
    This abstract class should be extended to cover different types of imaging modalities.

    Parameters
    ----------
    h5_main : h5py.Dataset instance
        The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
        indices and values, and position indices and values datasets.
    variables : list(string), Default ['Frequency']
        Lists of attributes that h5_main should possess so that it may be analyzed by Model.
    parallel : bool, optional
        Should the parallel implementation of the fitting be used.  Default True.

    Returns
    -------
    None

    """
    def __init__(self, h5_main, variables=['Frequency'], parallel=True):
        """
        For now, we assume that the guess dataset has not been generated for this dataset but we will relax this requirement
        after testing the basic components.

        """
        # Checking if dataset is "Main"
        if self._is_legal(h5_main, variables):
            self.h5_main = h5_main
            self.hdf = ioHDF5(self.h5_main.file)

        else:
            raise ValueError('Provided dataset is not a "Main" dataset with necessary ancillary datasets')

        # Checking if parallel processing will be used
        self._parallel = parallel

        # Determining the max size of the data that can be put into memory
        self._set_memory_and_cores()

        self._start_pos = 0
        self._end_pos = self.h5_main.shape[0]
        self.h5_guess = None
        self.h5_fit = None

        self.data = None
        self.guess = None
        self.fit = None

    def _set_memory_and_cores(self):
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
        self._maxMemoryMB = psutil.virtual_memory().available / 1e6  # in MB

        self._maxDataChunk = self._maxMemoryMB / self._maxCpus

        # Now calculate the number of positions that can be stored in memory in one go.
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1]/1e6
        self._max_pos_per_read = int(np.floor(self._maxDataChunk / mb_per_position))
        print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))

    def _is_legal(self, h5_main, variables):
        """
        Checks whether or not the provided object can be analyzed by this Model class.
        Classes that extend this class will do additional checks to ensure that the supplied dataset is legal.

        Parameters
        ----
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.

        variables : list(string)
            The dimensions needed to be present in the attributes of h5_main to analyze the data with Model.

        Returns
        -------
        legal : Boolean
            Whether or not this dataset satisfies the necessary conditions for analysis

        """

        # Check if h5_main is a "Main" dataset
        cond_A = checkIfMain(h5_main)

        # Check if variables are in the attributes of spectroscopic indices
        h5_spec_vals = getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0]
        # assert isinstance(h5_spec_vals, list)
        cond_B = set(variables).issubset(set(h5_spec_vals.attrs.keys()))

        if cond_A and cond_B:
            legal = True
        else:
            legal = False

        return legal

    def _get_data_chunk(self):
        """
        Returns the next chunk of data for the guess or the fit

        Parameters
        -----
        None

        Returns
        --------

        """
        if self._start_pos < self.h5_main.shape[0]:
            self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
            self.data = self.h5_main[self._start_pos:self._end_pos, :]
            print('Reading pixels {} to {} of {}'.format(self._start_pos, self._end_pos, self.h5_main.shape[0]))

            # Now update the start position
            self._start_pos = self._end_pos
        else:
            print('Finished reading all data!')
            self.data = None


    def _get_guess_chunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset.
        Should be called BEFORE _get_data_chunk since it relies upon current values of
        `self._start_pos`, `self._end_pos`

        Parameters
        -----
        None

        Returns
        --------

        """
        if self.data is None:
            self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
            self.guess = self.h5_guess[self._start_pos:self._end_pos, :]
        else:
            self.guess = self.h5_guess[self._start_pos:self._end_pos, :]

    def _set_results(self, is_guess=False):
        """
        Writes the provided guess or fit results into appropriate datasets.
        Given that the guess and fit datasets are relatively small, we should be able to hold them in memory just fine

        Parameters
        ---------
        is_guess : Boolean
            Flag that differentiates the guess from the fit

        """
        if is_guess:
            targ_dset = self.h5_guess
            source_dset = self.guess
        else:
            targ_dset = self.h5_fit
            source_dset = self.fit

        """print('Writing data to positions: {} to {}'.format(self.__start_pos, self._end_pos))
        targ_dset[self._start_pos:self._end_pos, :] = source_dset"""
        targ_dset[:, :] = source_dset

        # flush the file
        self.hdf.flush()
        print('Finished writing to file!')

    def _create_guess_datasets(self):
        """
        Model specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.

        The guess dataset will NOT be populated here but will be populated by the __setData function
        The fit dataset should NOT be populated here unless the user calls the optimize function.

        Parameters
        --------
        None

        Returns
        -------
        None

        """
        warn('Please override the _create_guess_datasets specific to your model')
        self.guess = None  # replace with actual h5 dataset
        pass

    def _create_fit_datasets(self):
        """
        Model specific call that will write the h5 group, fit dataset, corresponding spectroscopic datasets and also
        link the fit dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.

        The fit dataset will NOT be populated here but will be populated by the __setData function
        The guess dataset should NOT be populated here unless the user calls the optimize function.

        Parameters
        --------
        None

        Returns
        -------
        None

        """
        warn('Please override the _create_fit_datasets specific to your model')
        self.fit = None # replace with actual h5 dataset
        pass

    def do_guess(self, processors=4, strategy='wavelet_peaks',
                 options={"peak_widths": np.array([10,200]), "peak_step":20}):
        """

        Parameters
        ----------
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. 
            For updated list, run GuessMethods.methods
        options: dict
            Default, options for wavelet_peaks {"peaks_widths": np.array([10,200]), "peak_step":20}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.

        Returns
        -------

        """

        self._start_pos = 0
        self._get_data_chunk()
        processors = min(processors, self._maxCpus)
        gm = GuessMethods()
        results = list()
        if strategy in gm.methods:
            print("Using %s to find guesses...\n" % (strategy))
            while self.data is not None:
                opt = Optimize(data=self.data, parallel=self._parallel)
                temp = opt.computeGuess(processors=processors, strategy=strategy, options=options)
                results.append(self._reformat_results(temp, strategy))
                self._get_data_chunk()

            # reorder to get one numpy array out
            self.guess = np.hstack(tuple(results))
            print('Completed computing guess. Writing to file.')

            # Write to file
            self._set_results(is_guess=True)
        else:
            raise KeyError('Error: %s is not implemented in pycroscopy.analysis.GuessMethods to find guesses' %
                           strategy)

        return self.guess

    def _reformat_results(self, results, strategy='wavelet_peaks'):
        """
        Model specific restructuring / reformatting of the parallel compute results

        Parameters
        ----------
        results : array-like
            Results to be formatted for writing
        strategy : str
            The strategy used in the fit.  Determines how the results will be reformatted.
            Default 'wavelet_peaks'

        Returns
        -------
        results : numpy.ndarray
            Formatted array that is ready to be writen to the HDF5 file 

        """
        return np.array(results)

    def _create_fit_dataset(self):
        """
        Model specific call that will write the HDF5 fit dataset. pycroscopy requires that the h5 group, guess dataset,
        corresponding spectroscopic and position datasets be created and populated at this point.
        This function will create the HDF5 dataset for the fit and link it to same ancillary datasets as the guess.
        The fit dataset will NOT be populated here but will instead be populated using the __setData function
        """
        warn('Please override the _create_fit_dataset specific to your model')
        self.h5_fit = None  # replace with actual h5 dataset
        pass

    def do_fit(self, processors=None, solver_type='least_squares', solver_options={'jac': '2-point'},
               obj_func={'class': 'Fit_Methods', 'obj_func': 'SHO', 'xvals': np.array([])}):
        """
        Generates the fit for the given dataset and writes back to file

        Parameters
        ----------
        processors : int
            Number of cpu cores the user wishes to run on.  The minimum of this and self._maxCpus is used.
        solver_type : str
            The name of the solver in scipy.optimize to use for the fit
        solver_options : dict
            Dictionary of parameters to pass to the solver specified by `solver_type`
        obj_func : dict
            Dictionary defining the class and method containing the function to be fit as well as any 
            additional function parameters.

        Returns
        -------

        """
        if self.h5_guess is None:
            print("You need to guess before fitting\n")
            return None

        if processors is None:
            processors = self._maxCpus
        else:
            processors = min(processors, self._maxCpus)

        self._start_pos = 0
        self._get_guess_chunk()
        self._get_data_chunk()
        results = list()
        legit_solver = solver_type in scipy.optimize.__dict__.keys()
        legit_obj_func = obj_func['obj_func'] in Fit_Methods().methods
        if legit_solver and legit_obj_func:
            print("Using solver %s and objective function %s to fit your data\n" %(solver_type, obj_func['obj_func']))
            while self.data is not None:
                opt = Optimize(data=self.data, guess=self.guess, parallel=self._parallel)
                temp = opt.computeFit(processors=processors, solver_type=solver_type, solver_options=solver_options,
                                      obj_func=obj_func)
                # TODO: need a different .reformatResults to process fitting results
                results.append(self._reformat_results(temp, obj_func['obj_func']))
                self._get_guess_chunk()
                self._get_data_chunk()

            self.fit = np.hstack(tuple(results))
            self._set_results()

        elif legit_obj_func:
            raise KeyError('Error: Solver "%s" does not exist!. For additional info see scipy.optimize\n' % solver_type)

        elif legit_solver:
            raise KeyError('Error: Objective Functions "%s" is not implemented in pycroscopy.analysis.Fit_Methods' %
                           obj_func['obj_func'])

        return results
