"""
Created on 7/17/16 10:08 AM
@author: Suhas Somnath, Chris R. Smith, Numan Laanait
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import numpy as np

from .fitter import Fitter
from pyUSID import USIDataset
from pyUSID.io.hdf_utils import copy_region_refs, write_simple_attrs, create_results_group, write_reduced_spec_dsets, \
                                create_empty_dataset, get_auxiliary_datasets, write_main_dataset

'''
Custom dtype for the datasets created during fitting.
'''
field_names = ['Amplitude [V]', 'Frequency [Hz]', 'Quality Factor', 'Phase [rad]', 'R2 Criterion']
sho32 = np.dtype({'names': field_names,
                  'formats': [np.float32 for name in field_names]})


class BESHOfitter(Fitter):

    def __init__(self, h5_main, variables=None, **kwargs):
        """
        Analysis of Band excitation spectra with harmonic oscillator responses.

        Parameters
        ----------
        h5_main : h5py.Dataset instance
           The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
           indices and values, and position indices and values datasets.
        variables : list(string), Default ['Frequency']
           Lists of attributes that h5_main should possess so that it may be analyzed by Model.
        """
        if variables is None:
            variables = ['Frequency']

        super(BESHOfitter, self).__init__(h5_main, variables, **kwargs)

        self.step_start_inds = None
        self.is_reshapable = True
        self.num_udvs_steps = None
        self.freq_vec = None
        # self._maxDataChunk = 1
        # self._max_pos_per_read = 1
        self._fitter_name = "SHO_Fit"
        self._parms_dict = None
        self._fit_dim_name = variables[0]

        # Extract some basic parameters that are necessary for either the guess or fit
        freq_dim_ind = self.h5_main.spec_dim_labels.index(variables[0])
        self.step_start_inds = np.where(self.h5_main.h5_spec_inds[freq_dim_ind] == 0)[0]
        self.num_udvs_steps = len(self.step_start_inds)

        # find the frequency vector and hold in memory
        self._get_frequency_vector()

        self.is_reshapable = is_reshapable(self.h5_main, self.step_start_inds)

        if self._parallel:
            # accounting for memory copies
            self._max_pos_per_read = self._max_pos_per_read // 2

    def _create_guess_datasets(self):
        """
        Creates the h5 group, guess dataset, corresponding spectroscopic datasets and also
        links the guess dataset to the spectroscopic datasets.
        """
        h5_group = create_results_group(self.h5_main, 'SHO_Fit')
        write_simple_attrs(h5_group, {'SHO_guess_method': "pycroscopy BESHO"})

        h5_sho_inds, h5_sho_vals = write_reduced_spec_dsets(h5_group, self.h5_main.h5_spec_inds,
                                                            self.h5_main.h5_spec_vals, self._fit_dim_name)

        self.h5_guess = write_main_dataset(h5_group, (self.h5_main.shape[0], self.num_udvs_steps), 'Guess', 'SHO',
                                           'compound', None, None, h5_pos_inds=self.h5_main.h5_pos_inds,
                                           h5_pos_vals=self.h5_main.h5_pos_vals, h5_spec_inds=h5_sho_inds,
                                           h5_spec_vals=h5_sho_vals, chunks=(1, self.num_udvs_steps), dtype=sho32,
                                           main_dset_attrs=self._parms_dict, verbose=self._verbose)

        write_simple_attrs(self.h5_guess, {'SHO_guess_method': "pycroscopy BESHO", 'last_pixel': 0})

        copy_region_refs(self.h5_main, self.h5_guess)

    def _create_fit_datasets(self):
        """
        Creates the HDF5 fit dataset. pycroscopy requires that the h5 group, guess dataset,
        corresponding spectroscopic and position datasets be created and populated at this point.
        This function will create the HDF5 dataset for the fit and link it to same ancillary datasets as the guess.
        The fit dataset will NOT be populated here but will instead be populated using the __setData function
        """

        if self.h5_guess is None:
            warn('Need to guess before fitting!')
            return

        if self.step_start_inds is None:
            h5_spec_inds = self.h5_main.h5_spec_inds
            self.step_start_inds = np.where(h5_spec_inds[0] == 0)[0]

        if self.num_udvs_steps is None:
            self.num_udvs_steps = len(self.step_start_inds)

        if self.freq_vec is None:
            self._get_frequency_vector()

        h5_sho_grp = self.h5_guess.parent
        write_simple_attrs(h5_sho_grp, {'SHO_fit_method': "pycroscopy BESHO"})

        # Create the fit dataset as an empty dataset of the same size and dtype as the guess.
        # Also automatically links in the ancillary datasets.
        self.h5_fit = USIDataset(create_empty_dataset(self.h5_guess, dtype=sho32, dset_name='Fit'))

        # This is necessary comparing against new runs to avoid re-computation + resuming partial computation
        write_simple_attrs(self.h5_fit, self._parms_dict)
        write_simple_attrs(self.h5_fit, {'SHO_fit_method': "pycroscopy BESHO", 'last_pixel': 0})

        self.h5_fit.file.flush()

    def _get_frequency_vector(self):
        """
        Reads the frequency vector from the Spectroscopic_Values dataset.  
        This assumes that the data is reshape-able.
        
        """
        h5_spec_vals = self.h5_main.h5_spec_vals
        freq_dim = np.argwhere('Frequency' == np.array(self.h5_main.spec_dim_labels)).squeeze()

        if len(self.step_start_inds) == 1:  # BE-Line
            end_ind = h5_spec_vals.shape[1]
        else:  # BEPS
            end_ind = self.step_start_inds[1]

        self.freq_vec = h5_spec_vals[freq_dim, self.step_start_inds[0]:end_ind]

    def _get_data_chunk(self):
        """
        Returns the next chunk of data for the guess or the fit
        """

        # The model class should take care of all the basic reading
        super(BESHOfitter, self)._get_data_chunk()

        # At this point the self.data object is the raw data that needs to be reshaped to a single UDVS step:
        if self.data is not None:
            if self._verbose:
                print('Got raw data of shape {} from super'.format(self.data.shape))
            self.data = reshape_to_one_step(self.data, self.num_udvs_steps)
            if self._verbose:
                print('Reshaped raw data to shape {}'.format(self.data.shape))

    def _get_guess_chunk(self):
        """
        Returns the next chunk of the guess dataset corresponding to the main dataset.
        
        """

        # if self.data is None:
        #     self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
        #     self.guess = self.h5_guess[self._start_pos:self._end_pos, :]
        # else:
        #     self.guess = self.h5_guess[self._start_pos:self._end_pos, :]
        self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
        self.guess = self.h5_guess[self._start_pos:self._end_pos, :]
        # At this point the self.data object is the raw data that needs to be reshaped to a single UDVS step:
        self.guess = reshape_to_one_step(self.guess, self.num_udvs_steps)
        # don't keep the R^2.
        self.guess = np.hstack([self.guess[name] for name in self.guess.dtype.names if name != 'R2 Criterion'])
        # bear in mind that this self.guess is a compound dataset.

    def _set_results(self, is_guess=False):
        """
        Writes the provided chunk of data into the guess or fit datasets. 
        This method is responsible for any and all book-keeping.

        Parameters
        ---------
        is_guess : Boolean
            Flag that differentiates the guess from the fit
        """
        if is_guess:
            # prepare to reshape:
            self.guess = np.transpose(np.atleast_2d(self.guess))
            if self._verbose:
                print('Prepared guess of shape {} before reshaping'.format(self.guess.shape))
            self.guess = reshape_to_n_steps(self.guess, self.num_udvs_steps)
            if self._verbose:
                print('Reshaped guess to shape {}'.format(self.guess.shape))
        else:
            self.fit = np.transpose(np.atleast_2d(self.fit))
            self.fit = reshape_to_n_steps(self.fit, self.num_udvs_steps)

        # ask super to take care of the rest, which is a standardized operation
        super(BESHOfitter, self)._set_results(is_guess)

    def do_guess(self, max_mem=None, processors=None, strategy='complex_gaussian',
                 options={"peak_widths": np.array([10, 200]), "peak_step": 20},
                 h5_partial_guess=None, override=False, **kwargs):
        """

        Parameters
        ----------
        max_mem : uint, optional
            Memory in MB to use for computation
            Default None, available memory from psutil.virtual_memory is used
        processors: int
            Number of processors to use during parallel guess
            Default None, output of psutil.cpu_count - 2 is used
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. For updated list, run
            GuessMethods.methods
        options: dict
            Default Options for wavelet_peaks{"peaks_widths": np.array([10,200]), "peak_step":20}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.
        h5_partial_guess : h5py.group. optional, default = None
            Datagroup containing (partially computed) guess results. do_guess will resume computation if provided.
        override : bool, optional. default = False
            By default, will simply return duplicate results to avoid recomputing or resume computation on a
            group with partial results. Set to True to force fresh computation.

        Returns
        -------
        results : h5py.Dataset object
            Dataset with the SHO guess parameters
            
        """

        """
        if self._parallel:
            self._max_pos_per_read = int(self._max_pos_per_read / 2)
        """
        if strategy == 'complex_gaussian':
            options.update({'frequencies': self.freq_vec})
        super(BESHOfitter, self).do_guess(processors=processors, strategy=strategy, options=options,
                                          h5_partial_guess=h5_partial_guess, override=override)

        return self.h5_guess

    def do_fit(self, max_mem=None, processors=None, solver_type='least_squares', solver_options={'jac': 'cs'},
               obj_func={'class': 'Fit_Methods', 'obj_func': 'SHO', 'xvals': np.array([])},
               h5_partial_fit=None, h5_guess=None, override=False):
        """
        Fits the dataset to the SHO function

        Parameters
        ----------
        max_mem : uint, optional
            Memory in MB to use for computation
            Default None, available memory from psutil.virtual_memory is used
        processors : int
            Number of processors the user requests.  The minimum of this and self._maxCpus is used.
            Default None
        solver_type : string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. 
            For updated list, run GuessMethods.methods
        solver_options : dict
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.
            Default {"peaks_widths": np.array([10,200])}}.
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
        obj_func['xvals'] = self.freq_vec
        super(BESHOfitter, self).do_fit(processors=processors, solver_type=solver_type,
                                        solver_options=solver_options, obj_func=obj_func,
                                        h5_partial_fit=h5_partial_fit, h5_guess=h5_guess, override=override)
        return self.h5_fit

    def _reformat_results(self, results, strategy='wavelet_peaks'):
        """
        Model specific calculation and or reformatting of the raw guess or fit results

        Parameters
        ----------
        results : array-like
            Results to be formatted for writing
        strategy : str
            The strategy used in the fit.  Determines how the results will be reformatted.
            Default 'wavelet_peaks'

        Returns
        -------
        sho_vec : numpy.ndarray
            The reformatted array of parameters.
            
        """
        if self._verbose:
            print('Strategy to use: {}'.format(strategy))
        # Create an empty array to store the guess parameters
        sho_vec = np.zeros(shape=(len(results)), dtype=sho32)
        if self._verbose:
            print('Raw results and compound SHO vector of shape {}'.format(len(results)))

        # Extracting and reshaping the remaining parameters for SHO
        if strategy in ['wavelet_peaks', 'relative_maximum', 'absolute_maximum']:
            # wavelet_peaks sometimes finds 0, 1, 2, or more peaks. Need to handle that:
            # peak_inds = np.array([pixel[0] for pixel in results])
            peak_inds = np.zeros(shape=(len(results)), dtype=np.uint32)
            for pix_ind, pixel in enumerate(results):
                if len(pixel) == 1:  # majority of cases - one peak found
                    peak_inds[pix_ind] = pixel[0]
                elif len(pixel) == 0:  # no peak found
                    peak_inds[pix_ind] = int(0.5*self.data.shape[1])  # set to center of band
                else:  # more than one peak found
                    dist = np.abs(np.array(pixel) - int(0.5*self.data.shape[1]))
                    peak_inds[pix_ind] = pixel[np.argmin(dist)]  # set to peak closest to center of band
            if self._verbose:
                print('Peak positions of shape {}'.format(peak_inds.shape))
            # First get the value (from the raw data) at these positions:
            comp_vals = np.array(
                [self.data[pixel_ind, peak_inds[pixel_ind]] for pixel_ind in np.arange(peak_inds.size)])
            if self._verbose:
                print('Complex values at peak positions of shape {}'.format(comp_vals.shape))
            sho_vec['Amplitude [V]'] = np.abs(comp_vals)  # Amplitude
            sho_vec['Phase [rad]'] = np.angle(comp_vals)  # Phase in radians
            sho_vec['Frequency [Hz]'] = self.freq_vec[peak_inds]  # Frequency
            sho_vec['Quality Factor'] = np.ones_like(comp_vals) * 10  # Quality factor
            # Add something here for the R^2
            sho_vec['R2 Criterion'] = np.array([self.r_square(self.data, self._sho_func, self.freq_vec, sho_parms)
                                                for sho_parms in sho_vec])
        elif strategy in ['complex_gaussian']:
            for iresult, result in enumerate(results):
                sho_vec['Amplitude [V]'][iresult] = result[0]
                sho_vec['Frequency [Hz]'][iresult] = result[1]
                sho_vec['Quality Factor'][iresult] = result[2]
                sho_vec['Phase [rad]'][iresult] = result[3]
                sho_vec['R2 Criterion'][iresult] = result[4]
        elif strategy in ['SHO']:
            for iresult, result in enumerate(results):
                sho_vec['Amplitude [V]'][iresult] = result.x[0]
                sho_vec['Frequency [Hz]'][iresult] = result.x[1]
                sho_vec['Quality Factor'][iresult] = result.x[2]
                sho_vec['Phase [rad]'][iresult] = result.x[3]
                sho_vec['R2 Criterion'][iresult] = 1-result.fun

        return sho_vec


def reshape_to_one_step(raw_mat, num_steps):
    """
    Reshapes provided data from (pos, step * bin) to (pos * step, bin).
    This is useful when unraveling data for parallel processing.

    Parameters
    -------------
    raw_mat : 2D numpy array
        Data organized as (positions, step * bins)
    num_steps : unsigned int
        Number of spectroscopic steps per pixel (eg - UDVS steps)

    Returns
    --------------
    two_d : 2D numpy array
        Data rearranged as (positions * step, bin)
    """
    num_pos = raw_mat.shape[0]
    num_bins = int(raw_mat.shape[1] / num_steps)
    one_d = raw_mat
    one_d = one_d.reshape((num_bins * num_steps * num_pos))
    two_d = one_d.reshape((num_steps * num_pos, num_bins))
    return two_d


def reshape_to_n_steps(raw_mat, num_steps):
    """
    Reshapes provided data from (positions * step, bin) to (positions, step * bin).
    Use this to restructure data back to its original form after parallel computing

    Parameters
    --------------
    raw_mat : 2D numpy array
        Data organized as (positions * step, bin)
    num_steps : unsigned int
         Number of spectroscopic steps per pixel (eg - UDVS steps)

    Returns
    ---------------
    two_d : 2D numpy array
        Data rearranged as (positions, step * bin)
    """
    num_bins = raw_mat.shape[1]
    num_pos = int(raw_mat.shape[0] / num_steps)
    one_d = raw_mat
    one_d = one_d.reshape(num_bins * num_steps * num_pos)
    two_d = one_d.reshape((num_pos, num_steps * num_bins))
    return two_d


def is_reshapable(h5_main, step_start_inds=None):
    """
    A BE dataset is said to be reshape-able if the number of bins per steps is constant. Even if the dataset contains
    multiple excitation waveforms (harmonics), We know that the measurement is always at the resonance peak, so the
    frequency vector should not change.

    Parameters
    ----------
    h5_main : h5py.Dataset object
        Reference to the main dataset
    step_start_inds : list or 1D array
        Indices that correspond to the start of each BE pulse / UDVS step

    Returns
    ---------
    reshapable : Boolean
        Whether or not the number of bins per step are constant in this dataset
    """
    if step_start_inds is None:
        h5_spec_inds = get_auxiliary_datasets(h5_main, aux_dset_name=['Spectroscopic_Indices'])[0]
        step_start_inds = np.where(h5_spec_inds[0] == 0)[0]
    # Adding the size of the main dataset as the last (virtual) step
    step_start_inds = np.hstack((step_start_inds, h5_main.shape[1]))
    num_bins = np.diff(step_start_inds)
    step_types = np.unique(num_bins)
    return len(step_types) == 1
