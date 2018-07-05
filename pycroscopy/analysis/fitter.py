"""
Created on 7/17/16 10:08 AM
@author: Numan Laanait, Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import psutil
import scipy
import h5py
import time as tm
from .guess_methods import GuessMethods
from .fit_methods import Fit_Methods
from pyUSID import USIDataset
from pyUSID.io.io_utils import get_available_memory, recommend_cpu_cores, format_time
from pyUSID.io.hdf_utils import check_for_old, find_results_groups, check_for_matching_attrs, get_attr
from .optimize import Optimize


class Fitter(object):
    """
    Encapsulates the typical routines performed during model-dependent analysis of data.
    This abstract class should be extended to cover different types of imaging modalities.
    """

    def __init__(self, h5_main, variables=['Frequency'], parallel=True, verbose=False):
        """
        For now, we assume that the guess dataset has not been generated for this dataset but we will relax this
        requirement after testing the basic components.

        Parameters
        ----------
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.
        variables : list(string), Default ['Frequency']
            Lists of attributes that h5_main should possess so that it may be analyzed by Model.
        parallel : bool, optional
            Should the parallel implementation of the fitting be used.  Default True
        verbose : bool, optional. default = False
            Whether or not to print statements that aid in debugging

        """

        if not isinstance(h5_main, USIDataset):
            h5_main = USIDataset(h5_main)

        # Checking if dataset has the proper dimensions for the model to run.
        if self._is_legal(h5_main, variables):
            self.h5_main = h5_main

        else:
            raise ValueError('Provided dataset is not a "Main" dataset with necessary ancillary datasets')

        # Checking if parallel processing will be used
        self._parallel = parallel
        self._verbose = verbose

        # Determining the max size of the data that can be put into memory
        self._set_memory_and_cores()

        self._start_pos = 0
        self._end_pos = self.h5_main.shape[0]
        self.h5_guess = None
        self.h5_fit = None
        self.h5_results_grp = None

        # TODO: do NOT expose a lot of innards. Turn it into private with _var_name
        self.data = None
        self.guess = None
        self.fit = None

        self._fitter_name = None  # Reset this in the extended classes
        self._parms_dict = dict()

    def _set_memory_and_cores(self):
        """
        Checks hardware limitations such as memory, # cpus and sets the recommended datachunk sizes and the
        number of cores to be used by analysis methods.
        """

        if self._parallel:
            self._maxCpus = max(1, psutil.cpu_count() - 2)
        else:
            self._maxCpus = 1

        if self._maxCpus == 1:
            self._parallel = False

        self._maxMemoryMB = get_available_memory() / 1024 ** 2  # in Mb

        self._maxDataChunk = int(self._maxMemoryMB / self._maxCpus)

        # Now calculate the number of positions that can be stored in memory in one go.
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1] / 1024.0 ** 2

        # TODO: The size of the chunk should be determined by BOTH the computation time and memory restrictions
        self._max_pos_per_read = int(np.floor(self._maxDataChunk / mb_per_position))
        if self._verbose:
            print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))

    def _is_legal(self, h5_main, variables):
        """
        Checks whether or not the provided object can be analyzed by this Model class.
        Classes that extend this class will do additional checks to ensure that the supplied dataset is legal.

        Parameters
        ----
        h5_main : USIDataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.

        variables : list(string)
            The dimensions needed to be present in the attributes of h5_main to analyze the data with Model.

        Returns
        -------
        legal : Boolean
            Whether or not this dataset satisfies the necessary conditions for analysis

        """
        return np.all(np.isin(variables, h5_main.spec_dim_labels))

    def _get_data_chunk(self):
        """
        Reads the next chunk of data for the guess or the fit into memory
        """
        if self._start_pos < self.h5_main.shape[0]:
            self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
            self.data = self.h5_main[self._start_pos:self._end_pos, :]
            if self._verbose:
                print('\nReading pixels {} to {} of {}'.format(self._start_pos, self._end_pos, self.h5_main.shape[0]))

        else:
            if self._verbose:
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

        if self._verbose:
            print('Guess of shape: {}'.format(self.guess.shape))

    def _set_results(self, is_guess=False):
        """
        Writes the provided guess or fit results into appropriate datasets.
        Given that the guess and fit datasets are relatively small, we should be able to hold them in memory just fine

        Parameters
        ---------
        is_guess : bool, optional
            Default - False
            Flag that differentiates the guess from the fit
        """
        statement = 'guess'

        if is_guess:
            targ_dset = self.h5_guess
            source_dset = self.guess
        else:
            statement = 'fit'
            targ_dset = self.h5_fit
            source_dset = self.fit

        if self._verbose:
            print('Writing data to positions: {} to {}'.format(self._start_pos, self._end_pos))
        targ_dset[self._start_pos: self._end_pos, :] = source_dset

        # This flag will let us resume the computation if it is aborted
        targ_dset.attrs['last_pixel'] = self._end_pos

        # Now update the start position
        self._start_pos = self._end_pos

        # flush the file
        self.h5_main.file.flush()
        if self._verbose:
            print('Finished writing ' + statement + ' results (chunk) to file!')

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
        self.guess = None  # replace with actual h5 dataset
        raise NotImplementedError('Please override the _create_guess_datasets specific to your model')

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
        self.fit = None  # replace with actual h5 dataset
        raise NotImplementedError('Please override the _create_fit_datasets specific to your model')

    def _check_for_old_guess(self):
        """
        Returns a list of datasets where the same parameters have already been used to compute Guesses for this dataset

        Returns
        -------
        list
            List of datasets with results from do_guess on this dataset
        """
        groups = check_for_old(self.h5_main, self._fitter_name, new_parms=self._parms_dict, target_dset='Guess',
                               verbose=self._verbose)
        datasets = [grp['Guess'] for grp in groups]

        # Now sort these datasets into partial and complete:
        completed_dsets = []
        partial_dsets = []

        for dset in datasets:
            try:
                last_pix = get_attr(dset, 'last_pixel')
            except KeyError:
                last_pix = None
                
            # Skip datasets without last_pixel attribute
            if last_pix is None:
                continue
            elif last_pix < self.h5_main.shape[0]:
                partial_dsets.append(dset)
            else:
                completed_dsets.append(dset)

        return partial_dsets, completed_dsets

    def do_guess(self, processors=None, strategy=None, options=dict(), h5_partial_guess=None, override=False):
        """
        Parameters
        ----------
        strategy: string (optional)
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes'].
            For updated list, run GuessMethods.methods
        processors : int (optional)
            Number of cores to use for computing. Default = all available - 2 cores
        options: dict
            Default, options for wavelet_peaks {"peaks_widths": np.array([10,200]), "peak_step":20}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.
        h5_partial_guess : h5py.group. optional, default = None
            Datagroup containing (partially computed) guess results. do_guess will resume computation if provided.
        override : bool, optional. default = False
            By default, will simply return duplicate results to avoid recomputing or resume computation on a
            group with partial results. Set to True to force fresh computation.

        Returns
        -------
        h5_guess : h5py.Dataset
            Dataset containing guesses that can be passed on to do_fit()
        """
        gm = GuessMethods()
        if strategy not in gm.methods:
            raise KeyError('Error: %s is not implemented in pycroscopy.analysis.GuessMethods to find guesses' %
                           strategy)

        # ################## CHECK FOR DUPLICATES AND RESUME PARTIAL #######################################

        # Prepare the parms dict that will be used for comparison:
        self._parms_dict = options.copy()
        self._parms_dict.update({'strategy': strategy})

        # check for old:
        partial_dsets, completed_dsets = self._check_for_old_guess()

        if len(completed_dsets) == 0 and len(partial_dsets) == 0:
            print('No existing datasets found')
            override = True

        if not override:
            # First try to simply return any completed computation
            if len(completed_dsets) > 0:
                print('Returned previously computed results at ' + completed_dsets[-1].name)
                self.h5_guess = USIDataset(completed_dsets[-1])
                return

            # Next attempt to resume automatically if nothing is provided
            if len(partial_dsets) > 0:
                # attempt to use whatever the user provided (if legal)
                target_partial_dset = partial_dsets[-1]
                if h5_partial_guess is not None:
                    if not isinstance(h5_partial_guess, h5py.Dataset):
                        raise ValueError('Provided parameter is not an h5py.Dataset object')
                    if h5_partial_guess not in partial_dsets:
                        raise ValueError('Provided dataset for partial Guesses is not compatible')
                    if self._verbose:
                        print('Provided partial Guess dataset was acceptable')
                    target_partial_dset = h5_partial_guess

                # Finally resume from this dataset
                print('Resuming computation in group: ' + target_partial_dset.name)
                self.h5_guess = target_partial_dset
                self._start_pos = target_partial_dset.attrs['last_pixel']

        # No completed / partials available or forced via override:
        if self.h5_guess is None:
            if self._verbose:
                print('Starting a fresh computation!')
            self._start_pos = 0
            self._create_guess_datasets()

        # ################## BEGIN THE ACTUAL COMPUTING #######################################

        if processors is None:
            processors = self._maxCpus
        else:
            processors = min(int(processors), self._maxCpus)
        processors = recommend_cpu_cores(self._max_pos_per_read, processors, verbose=self._verbose)

        print("Using %s to find guesses...\n" % strategy)

        time_per_pix = 0
        num_pos = self.h5_main.shape[0] - self._start_pos
        orig_start_pos = self._start_pos

        print('You can abort this computation at any time and resume at a later time!\n'
              '\tIf you are operating in a python console, press Ctrl+C or Cmd+C to abort\n'
              '\tIf you are in a Jupyter notebook, click on "Kernel">>"Interrupt"\n')

        self._get_data_chunk()
        while self.data is not None:

            t_start = tm.time()

            opt = Optimize(data=self.data, parallel=self._parallel)
            temp = opt.computeGuess(processors=processors, strategy=strategy, options=options)

            # reorder to get one numpy array out
            temp = self._reformat_results(temp, strategy)
            self.guess = np.hstack(tuple(temp))

            # Write to file
            self._set_results(is_guess=True)

            # basic timing logs
            tot_time = np.round(tm.time() - t_start, decimals=2)  # in seconds
            if self._verbose:
                print('Done parallel computing in {} or {} per pixel'.format(format_time(tot_time),
                                                                             format_time(tot_time / self.data.shape[0])))
            if self._start_pos == orig_start_pos:
                time_per_pix = tot_time / self._end_pos  # in seconds
            else:
                time_remaining = (num_pos - self._end_pos) * time_per_pix  # in seconds
                print('Time remaining: ' + format_time(time_remaining))

            # get next batch of data
            self._get_data_chunk()

        print('Completed computing guess')
        print()
        return USIDataset(self.h5_guess)

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

    def _check_for_old_fit(self):
        """
        Returns three lists of h5py.Dataset objects where the group contained:
            1. Completed guess only
            2. Partial Fit
            3. Completed Fit

        Returns
        -------

        """
        # First find all groups that match the basic condition of matching tool name
        all_groups = find_results_groups(self.h5_main, self._fitter_name)
        if self._verbose:
            print('Groups that matched the nomenclature: {}'.format(all_groups))

        # Next sort these groups into three categories:
        completed_guess = []
        partial_fits = []
        completed_fits = []

        for h5_group in all_groups:

            if 'Fit' in h5_group.keys():
                # check group for fit attribute

                h5_fit = h5_group['Fit']

                # check Fit dataset against parms_dict
                if not check_for_matching_attrs(h5_fit, new_parms=self._parms_dict, verbose=self._verbose):
                    if self._verbose:
                        print('{} did not match the given parameters'.format(h5_fit.name))
                    continue

                # sort this dataset:
                try:
                    last_pix = get_attr(h5_fit, 'last_pixel')
                except KeyError:
                    last_pix = None

                # For now skip any fits that are missing 'last_pixel'
                if last_pix is None:
                    continue
                elif last_pix < self.h5_main.shape[0]:
                    partial_fits.append(h5_fit.parent)
                else:
                    completed_fits.append(h5_fit)
            else:
                if 'Guess' in h5_group.keys():
                    h5_guess = h5_group['Guess']

                    # sort this dataset:
                    try:
                        last_pix = get_attr(h5_guess, 'last_pixel')
                    except KeyError:
                        last_pix = None

                    # For now skip any fits that are missing 'last_pixel'
                    if last_pix is None:
                        continue
                    elif last_pix == self.h5_main.shape[0]:
                        if self._verbose:
                            print('{} was a completed Guess'.format(h5_guess.name))
                        completed_guess.append(h5_guess)
                    else:
                        if self._verbose:
                            print('{} did not not have completed Guesses'.format(h5_guess.name))
                else:
                    if self._verbose:
                        print('{} did not even have Guess. Categorizing as defective Group'.format(h5_group.name))

        return completed_guess, partial_fits, completed_fits

    def do_fit(self, processors=None, solver_type='least_squares', solver_options=None, obj_func=None,
               h5_partial_fit=None, h5_guess=None, override=False):
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
        h5_partial_fit : h5py.group. optional, default = None
            Datagroup containing (partially computed) fit results. do_fit will resume computation if provided.
        h5_guess : h5py.group. optional, default = None
            Datagroup containing guess results. do_fit will use this if provided.
        override : bool, optional. default = False
            By default, will simply return duplicate results to avoid recomputing or resume computation on a
            group with partial results. Set to True to force fresh computation.

        Returns
        -------
        h5_results : h5py.Dataset object
            Dataset with the fit parameters
        """

        # ################## PREPARE THE SOLVER #######################################

        legit_solver = solver_type in scipy.optimize.__dict__.keys()

        if not legit_solver:
            raise KeyError('Error: Objective Functions "%s" is not implemented in pycroscopy.analysis.Fit_Methods' %
                           obj_func['obj_func'])

        obj_func_name = obj_func['obj_func']
        legit_obj_func = obj_func_name in Fit_Methods().methods

        if not legit_obj_func:
            raise KeyError('Error: Solver "%s" does not exist!. For additional info see scipy.optimize\n' % solver_type)

        # ################## CHECK FOR DUPLICATES AND RESUME PARTIAL #######################################

        def _get_group_to_resume(legal_groups, provided_partial_fit):
            for h5_group in legal_groups:
                if h5_group['Fit'] == provided_partial_fit:
                    return h5_group
            return None

        def _resume_fit(fitter, h5_group):
            fitter.h5_guess = h5_group['Guess']
            fitter.h5_fit = h5_group['Fit']
            fitter._start_pos = fitter.h5_fit.attrs['last_pixel']

        def _start_fresh_fit(fitter, h5_guess_legal):
            fitter.h5_guess = h5_guess_legal
            fitter._create_fit_datasets()
            fitter._start_pos = 0

        # Prepare the parms dict that will be used for comparison:
        self._parms_dict = solver_options.copy()
        self._parms_dict.update({'solver_type': solver_type})
        self._parms_dict.update(obj_func)

        completed_guess, partial_fit_groups, completed_fits = self._check_for_old_fit()

        override = override or (h5_partial_fit is not None or h5_guess is not None)

        if not override:
            # First try to simply return completed results
            if len(completed_fits) > 0:
                print('Returned previously computed results at ' + completed_fits[-1].name)
                self.h5_fit = USIDataset(completed_fits[-1])
                return

            # Next, attempt to resume automatically:
            elif len(partial_fit_groups) > 0:
                print('Will resume fitting in {}. '
                      'You can supply a dataset using the h5_partial_fit argument'.format(partial_fit_groups[-1].name))
                _resume_fit(self, partial_fit_groups[-1])

            # Finally, attempt to do fresh fitting using completed Guess:
            elif len(completed_guess) > 0:
                print('Will use {} for generating new Fit. '
                      'You can supply a dataset using the h5_guess argument'.format(completed_guess[-1].name))
                _start_fresh_fit(self, completed_guess[-1])

            else:
                raise ValueError('Could not find a compatible Guess to use for Fit. Call do_guess() before do_fit()')

        else:
            if h5_partial_fit is not None:
                h5_group = _get_group_to_resume(partial_fit_groups, h5_partial_fit)
                if h5_group is None:
                    raise ValueError('Provided dataset with partial Fit was not found to be compatible')
                _resume_fit(self, h5_group)

            elif h5_guess is not None:
                if h5_guess not in completed_guess:
                    raise ValueError('Provided dataset with completed Guess was not found to be compatible')
                _start_fresh_fit(self, h5_guess)

            else:
                raise ValueError('Please provide a completed guess or partially completed Fit to resume')

        # ################## BEGIN THE ACTUAL FITTING #######################################

        print("Using solver %s and objective function %s to fit your data\n" % (solver_type, obj_func['obj_func']))

        if processors is None:
            processors = self._maxCpus
        else:
            processors = min(processors, self._maxCpus)
        processors = recommend_cpu_cores(self._max_pos_per_read, processors, verbose=self._verbose)

        time_per_pix = 0
        num_pos = self.h5_main.shape[0] - self._start_pos
        orig_start_pos = self._start_pos

        print('You can abort this computation at any time and resume at a later time!\n'
              '\tIf you are operating in a python console, press Ctrl+C or Cmd+C to abort\n'
              '\tIf you are in a Jupyter notebook, click on "Kernel">>"Interrupt"\n')

        self._get_guess_chunk()
        self._get_data_chunk()

        while self.data is not None:

            t_start = tm.time()

            opt = Optimize(data=self.data, guess=self.guess, parallel=self._parallel)
            temp = opt.computeFit(processors=processors, solver_type=solver_type, solver_options=solver_options,
                                  obj_func=obj_func.copy())

            # TODO: need a different .reformatResults to process fitting results
            # reorder to get one numpy array out
            temp = self._reformat_results(temp, obj_func_name)
            self.fit = np.hstack(tuple(temp))

            # Write to file
            self._set_results(is_guess=False)

            # basic timing logs
            tot_time = np.round(tm.time() - t_start, decimals=2)  # in seconds
            if self._verbose:
                print('Done parallel computing in {} or {} per pixel'.format(format_time(tot_time),
                                                                             format_time(
                                                                                 tot_time / self.data.shape[0])))
            if self._start_pos == orig_start_pos:
                time_per_pix = tot_time / self._end_pos  # in seconds
            else:
                time_remaining = (num_pos - self._end_pos) * time_per_pix  # in seconds
                print('Time remaining: ' + format_time(time_remaining))

            # get next batch of data
            self._get_guess_chunk()
            self._get_data_chunk()

        print('Completed computing fit. Writing to file.')

        return USIDataset(self.h5_fit)
