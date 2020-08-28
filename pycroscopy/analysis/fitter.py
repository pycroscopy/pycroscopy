# -*- coding: utf-8 -*-
"""
:class:`~pycroscopy.analysis.fitter.Fitter` - Abstract class that provides the
framework for building application-specific children classes

Created on Thu Aug 15 11:48:53 2019

@author: Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, \
    unicode_literals
import numpy as np
from warnings import warn
import joblib
from scipy.optimize import least_squares

from sidpy.proc.comp_utils import recommend_cpu_cores
from pyUSID.processing.process import Process
from pyUSID.io.usi_data import USIDataset

# TODO: All reading, holding operations should use Dask arrays


class Fitter(Process):

    def __init__(self, h5_main, proc_name, variables=None, **kwargs):
        """
        Creates a new instance of the abstract Fitter class

        Parameters
        ----------
        h5_main : h5py.Dataset or pyUSID.io.USIDataset object
            Main datasets whose one or dimensions will be reduced
        proc_name : str or unicode
            Name of the child process
        variables : str or list, optional
            List of spectroscopic dimension names that will be reduced
        h5_target_group : h5py.Group, optional. Default = None
            Location where to look for existing results and to place newly
            computed results. Use this kwarg if the results need to be written
            to a different HDF5 file. By default, this value is set to the
            parent group containing `h5_main`
        kwargs : dict
            Keyword arguments that will be passed on to
            pyUSID.processing.process.Process
        """

        super(Fitter, self).__init__(h5_main, proc_name, **kwargs)

        # Validate other arguments / kwargs here:
        if variables is not None:
            if isinstance(variables, str):
                variables = [variables]
            if not isinstance(variables, (list, tuple)):
                raise TypeError('variables should be a string / list or tuple'
                                'of strings. Provided object was of type: {}'
                                ''.format(type(variables)))
            if not all([dim in self.h5_main.spec_dim_labels for dim in variables]):
                raise ValueError('Provided dataset does not appear to have the'
                                 ' spectroscopic dimension(s): {} that need '
                                 'to be fitted: {}'
                                 ''.format(self.h5_main.spec_dim_labels,
                                           variables))

        # Variables specific to Fitter
        self._guess = None
        self._fit = None
        self._is_guess = True
        self._h5_guess = None
        self._h5_fit = None
        self.__set_up_called = False

        # Variables from Process:
        self.compute = self.set_up_guess
        self._unit_computation = super(Fitter, self)._unit_computation
        self._create_results_datasets = self._create_guess_datasets
        self._map_function = None

    def _read_guess_chunk(self):
        """
        Returns a chunk of guess dataset corresponding to the same pixels of
        the main dataset.
        """
        curr_pixels = self._get_pixels_in_current_batch()
        self._guess = self._h5_guess[curr_pixels, :]

        if self.verbose and self.mpi_rank == 0:
            print('Guess of shape: {}'.format(self._guess.shape))

    def _write_results_chunk(self):
        """
        Writes the guess or fit results into appropriate HDF5 datasets.
        """
        if self._is_guess:
            targ_dset = self._h5_guess
            source_dset = self._guess
        else:
            targ_dset = self._h5_fit
            source_dset = self._fit

        curr_pixels = self._get_pixels_in_current_batch()

        if self.verbose and self.mpi_rank == 0:
            print('Writing data of shape: {} and dtype: {} to position range: '
                  '{} in HDF5 dataset:{}'.format(source_dset.shape,
                                                 source_dset.dtype,
                                              [curr_pixels[0],curr_pixels[-1]],
                                                 targ_dset))
        targ_dset[curr_pixels, :] = source_dset

    def _create_guess_datasets(self):
        """
        Model specific call that will create the h5 group, empty guess dataset,
        corresponding spectroscopic datasets and also link the guess dataset
        to the spectroscopic datasets.
        """
        raise NotImplementedError('Please override the _create_guess_datasets '
                                  'specific to your model')

    def _create_fit_datasets(self):
        """
        Model specific call that will create the (empty) fit dataset, and
        link the fit dataset to the spectroscopic datasets.
        """
        raise NotImplementedError('Please override the _create_fit_datasets '
                                  'specific to your model')

    def _get_existing_datasets(self):
        """
        Gets existing Guess, Fit, status datasets, from the HDF5 group.

        All other domain-specific datasets should be loaded in the classes that
        extend this class
        """
        self._h5_guess = USIDataset(self.h5_results_grp['Guess'])

        try:
            self._h5_status_dset = self.h5_results_grp[self._status_dset_name]
        except KeyError:
            warn('status dataset not created yet')
            self._h5_status_dset = None

        try:
            self._h5_fit = self.h5_results_grp['Fit']
            self._h5_fit = USIDataset(self._h5_fit)
        except KeyError:
            self._h5_fit = None
            if not self._is_guess:
                self._create_fit_datasets()

    def do_guess(self, *args, override=False, **kwargs):
        """
        Computes the Guess

        Parameters
        ----------
        args : list, optional
            List of arguments
        override : bool, optional
            If True, computes a fresh guess even if existing Guess was found
            Else, returns existing Guess dataset. Default = False
        kwargs : dict, optional
            Keyword arguments

        Returns
        -------
        USIDataset
            HDF5 dataset with the Guesses computed
        """
        if not self.__set_up_called:
            raise ValueError('Please call set_up_guess() before calling '
                             'do_guess()')
        self.h5_results_grp = super(Fitter, self).compute(override=override)
        # to be on the safe side, expect setup again
        self.__set_up_called = False
        return USIDataset(self.h5_results_grp['Guess'])

    def do_fit(self, *args, override=False, **kwargs):
        """
        Computes the Fit

        Parameters
        ----------
        args : list, optional
            List of arguments
        override : bool, optional
            If True, computes a fresh guess even if existing Fit was found
            Else, returns existing Fit dataset. Default = False
        kwargs : dict, optional
            Keyword arguments

        Returns
        -------
        USIDataset
            HDF5 dataset with the Fit computed
        """
        if not self.__set_up_called:
            raise ValueError('Please call set_up_guess() before calling '
                             'do_guess()')
        """
        Either delete or reset 'last_pixel' attribute to 0
        This value will be used for filling in the status dataset.
        """
        self.h5_results_grp.attrs['last_pixel'] = 0
        self.h5_results_grp = super(Fitter, self).compute(override=override)
        # to be on the safe side, expect setup again
        self.__set_up_called = False
        return USIDataset(self.h5_results_grp['Fit'])

    def _reformat_results(self, results, strategy='wavelet_peaks'):
        """
        Model specific restructuring / reformatting of the parallel compute
        results

        Parameters
        ----------
        results : list or array-like
            Results to be formatted for writing
        strategy : str
            The strategy used in the fit.  Determines how the results will be
            reformatted, if multiple strategies for guess / fit are available

        Returns
        -------
        results : numpy.ndarray
            Formatted array that is ready to be writen to the HDF5 file

        """
        return np.array(results)

    def set_up_guess(self, h5_partial_guess=None):
        """
        Performs necessary book-keeping before do_guess can be called

        Parameters
        ----------
        h5_partial_guess: h5py.Dataset or pyUSID.io.USIDataset, optional
            HDF5 dataset containing partial Guess. Not implemented
        """
        # TODO: h5_partial_guess needs to be utilized
        if h5_partial_guess is not None:
            raise NotImplementedError('Provided h5_partial_guess cannot be '
                                      'used yet. Ask developer to implement')

        # Set up the parms dict so everything necessary for checking previous
        # guess / fit is ready
        self._is_guess = True
        self._status_dset_name = 'completed_guess_positions'
        ret_vals = self._check_for_duplicates()
        self.duplicate_h5_groups, self.partial_h5_groups = ret_vals

        if self.verbose and self.mpi_rank == 0:
            print('Groups with Guess in:\nCompleted: {}\nPartial:{}'.format(
                self.duplicate_h5_groups, self.partial_h5_groups))

        self._unit_computation = super(Fitter, self)._unit_computation
        self._create_results_datasets = self._create_guess_datasets
        self.compute = self.do_guess
        self.__set_up_called = True

    def set_up_fit(self, h5_partial_fit=None, h5_guess=None):
        """
        Performs necessary book-keeping before do_fit can be called

        Parameters
        ----------
        h5_partial_fit: h5py.Dataset or pyUSID.io.USIDataset, optional
            HDF5 dataset containing partial Fit. Not implemented
        h5_guess: h5py.Dataset or pyUSID.io.USIDataset, optional
            HDF5 dataset containing completed Guess. Not implemented
        """
        # TODO: h5_partial_guess needs to be utilized
        if h5_partial_fit is not None or h5_guess is not None:
            raise NotImplementedError('Provided h5_partial_fit cannot be '
                                      'used yet. Ask developer to implement')
        self._is_guess = False

        self._map_function = None
        self._unit_computation = None
        self._create_results_datasets = self._create_fit_datasets

        # Case 1: Fit already complete or partially complete.
        # This is similar to a partial process. Leave as is
        self._status_dset_name = 'completed_fit_positions'
        ret_vals = self._check_for_duplicates()
        self.duplicate_h5_groups, self.partial_h5_groups = ret_vals
        if self.verbose and self.mpi_rank == 0:
            print('Checking on partial / completed fit datasets')
            print(
                'Completed results groups:\n{}\nPartial results groups:\n'
                '{}'.format(self.duplicate_h5_groups, self.partial_h5_groups))

        # Case 2: Fit neither partial / completed. Search for guess.
        # Most popular scenario:
        if len(self.duplicate_h5_groups) == 0 and len(
                self.partial_h5_groups) == 0:
            if self.verbose and self.mpi_rank == 0:
                print('No fit datasets found. Looking for Guess datasets')
            # Change status dataset name back to guess to check for status
            # on guesses:
            self._status_dset_name = 'completed_guess_positions'
            # Note that check_for_duplicates() will be against fit's parm_dict.
            # So make a backup of that
            fit_parms = self.parms_dict.copy()
            # Set parms_dict to an empty dict so that we can accept any Guess
            # dataset:
            self.parms_dict = dict()
            ret_vals = self._check_for_duplicates()
            guess_complete_h5_grps, guess_partial_h5_grps = ret_vals
            if self.verbose and self.mpi_rank == 0:
                print(
                    'Guess datasets search resulted in:\nCompleted: {}\n'
                    'Partial:{}'.format(guess_complete_h5_grps,
                                        guess_partial_h5_grps))
            # Now put back the original parms_dict:
            self.parms_dict.update(fit_parms)

            # Case 2.1: At least guess is completed:
            if len(guess_complete_h5_grps) > 0:
                # Just set the last group as the current results group
                self.h5_results_grp = guess_complete_h5_grps[-1]
                if self.verbose and self.mpi_rank == 0:
                    print('Guess found! Using Guess in:\n{}'.format(
                        self.h5_results_grp))
                # It will grab the older status default unless we set the
                # status dataset back to fit
                self._status_dset_name = 'completed_fit_positions'
                # Get handles to the guess dataset. Nothing else will be found
                self._get_existing_datasets()

            elif len(guess_complete_h5_grps) == 0 and len(
                    guess_partial_h5_grps) > 0:
                FileNotFoundError(
                    'Guess not yet completed. Please complete guess first')
                return
            else:
                FileNotFoundError(
                    'No Guess found. Please complete guess first')
                return

        # We want compute to call our own manual unit computation function:
        self._unit_computation = self._unit_compute_fit
        self.compute = self.do_fit
        self.__set_up_called = True

    def _unit_compute_fit(self, obj_func, obj_func_args=[],
                          solver_options={'jac': 'cs'}):
        """
        Performs least-squares fitting on self.data using self.guess for
        initial conditions.

        Results of the computation are captured in self._results

        Parameters
        ----------
        obj_func : callable
            Objective function to minimize on
        obj_func_args : list
            Arguments required by obj_func following the guess parameters
            (which should be the first argument)
        solver_options : dict, optional
            Keyword arguments passed onto scipy.optimize.least_squares
        """

        # At this point data has been read in. Read in the guess as well:
        self._read_guess_chunk()

        if self.verbose and self.mpi_rank == 0:
            print('_unit_compute_fit got:\nobj_func: {}\nobj_func_args: {}\n'
                  'solver_options: {}'.format(obj_func, obj_func_args,
                                              solver_options))

        # TODO: Generalize this bit. Use Parallel compute instead!

        if self.mpi_size > 1:
            if self.verbose:
                print('Rank {}: About to start serial computation'
                      '.'.format(self.mpi_rank))

            self._results = list()
            for pulse_resp, pulse_guess in zip(self.data, self._guess):
                curr_results = least_squares(obj_func, pulse_guess,
                                             args=[pulse_resp] + obj_func_args,
                                             **solver_options)
                self._results.append(curr_results)
        else:
            cores = recommend_cpu_cores(self.data.shape[0],
                                        verbose=self.verbose)
            if self.verbose:
                print('Starting parallel fitting with {} cores'.format(cores))

            values = [joblib.delayed(least_squares)(obj_func, pulse_guess,
                                                    args=[pulse_resp] + obj_func_args,
                                                    **solver_options) for
                      pulse_resp, pulse_guess in zip(self.data, self._guess)]
            self._results = joblib.Parallel(n_jobs=cores)(values)

        if self.verbose and self.mpi_rank == 0:
            print(
                'Finished computing fits on {} objects. Results of length: {}'
                '.'.format(self.data.shape[0], len(self._results)))

        # What least_squares returns is an object that needs to be extracted
        # to get the coefficients. This is handled by the write function
