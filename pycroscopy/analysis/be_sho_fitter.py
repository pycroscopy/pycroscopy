"""
Created on 7/17/16 10:08 AM
@author: Suhas Somnath, Numan Laanait, Chris R. Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import numpy as np
from .fitter import Fitter
from ..io.pycro_data import PycroDataset
from ..io.be_hdf_utils import isReshapable, reshapeToNsteps, reshapeToOneStep
from ..io.hdf_utils import buildReducedSpec, copyRegionRefs, linkRefs, getAuxData, getH5DsetRefs, \
            create_empty_dataset
from ..io.microdata import MicroDataset, MicroDataGroup

'''
Custom dtype for the datasets created during fitting.
'''
# sho32 = np.dtype([('Amplitude [V]', np.float32), ('Frequency [Hz]', np.float32),
#                   ('Quality Factor', np.float32), ('Phase [rad]', np.float32),
#                   ('R2 Criterion', np.float32)])
field_names = ['Amplitude [V]', 'Frequency [Hz]', 'Quality Factor', 'Phase [rad]', 'R2 Criterion']
sho32 = np.dtype({'names': field_names,
                  'formats': [np.float32 for name in field_names]})


class BESHOfitter(Fitter):

    def __init__(self, h5_main, variables=['Frequency'], **kwargs):
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
        super(BESHOfitter, self).__init__(h5_main, variables, **kwargs)
        self.step_start_inds = None
        self.is_reshapable = True
        self.num_udvs_steps = None
        self.freq_vec = None
        # self._maxDataChunk = 1
        # self._max_pos_per_read = 1
        self._fitter_name = "SHO_Fit"
        self._parms_dict = None

        # Extract some basic parameters that are necessary for either the guess or fit
        self.step_start_inds = np.where(self.h5_main.h5_spec_inds[0] == 0)[0]
        self.num_udvs_steps = len(self.step_start_inds)

        # find the frequency vector and hold in memory
        self._get_frequency_vector()

        self.is_reshapable = isReshapable(self.h5_main, self.step_start_inds)

        if self._parallel:
            # accounting for memory copies
            self._max_pos_per_read /= 2

    def _create_guess_datasets(self):
        """
        Creates the h5 group, guess dataset, corresponding spectroscopic datasets and also
        links the guess dataset to the spectroscopic datasets.
        """
        # Create all the ancilliary datasets, allocate space.....
        ds_guess = MicroDataset('Guess', data=[],
                                maxshape=(self.h5_main.shape[0], self.num_udvs_steps),
                                chunking=(1, self.num_udvs_steps), dtype=sho32)
        ds_guess.attrs = self._parms_dict

        not_freq = np.array(self.h5_main.spec_dim_labels) != 'Frequency'

        ds_sho_inds, ds_sho_vals = buildReducedSpec(self.h5_main.h5_spec_inds,
                                                    self.h5_main.h5_spec_vals,
                                                    not_freq, self.step_start_inds)

        dset_name = self.h5_main.name.split('/')[-1]
        sho_grp = MicroDataGroup('-'.join([dset_name,
                                           'SHO_Fit_']),
                                 self.h5_main.parent.name[1:])
        sho_grp.addChildren([ds_guess,
                             ds_sho_inds,
                             ds_sho_vals])
        sho_grp.attrs['SHO_guess_method'] = "pycroscopy BESHO"

        h5_sho_grp_refs = self.hdf.writeData(sho_grp, print_log=self._verbose)

        self.h5_guess = getH5DsetRefs(['Guess'], h5_sho_grp_refs)[0]
        h5_sho_inds = getH5DsetRefs(['Spectroscopic_Indices'],
                                    h5_sho_grp_refs)[0]
        h5_sho_vals = getH5DsetRefs(['Spectroscopic_Values'],
                                    h5_sho_grp_refs)[0]

        # Reference linking before actual fitting
        linkRefs(self.h5_guess, [h5_sho_inds, h5_sho_vals])
        # Linking ancillary position datasets:
        aux_dsets = getAuxData(self.h5_main, auxDataName=['Position_Indices', 'Position_Values'])
        linkRefs(self.h5_guess, aux_dsets)

        copyRegionRefs(self.h5_main, self.h5_guess)

        self.h5_guess = PycroDataset(self.h5_guess)

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
        h5_sho_grp.attrs['SHO_fit_method'] = "pycroscopy BESHO"

        # Create the fit dataset as an empty dataset of the same size and dtype as the guess.
        # Also automatically links in the ancillary datasets.
        self.h5_fit = PycroDataset(create_empty_dataset(self.h5_guess, dtype=sho32, dset_name='Fit'))

        # This is necessary comparing against new runs to avoid re-computation + resuming partial computation
        self.h5_fit.attrs.update(self._parms_dict)

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
            self.data = reshapeToOneStep(self.data, self.num_udvs_steps)
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
        self.guess = reshapeToOneStep(self.guess, self.num_udvs_steps)
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
            self.guess = reshapeToNsteps(self.guess, self.num_udvs_steps)
            if self._verbose:
                print('Reshaped guess to shape {}'.format(self.guess.shape))
        else:
            self.fit = np.transpose(np.atleast_2d(self.fit))
            self.fit = reshapeToNsteps(self.fit, self.num_udvs_steps)

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
            freq_vec = self.freq_vec
            options.update({'frequencies': freq_vec})
        super(BESHOfitter, self).do_guess(processors=processors, strategy=strategy, options=options,
                                          h5_partial_guess=h5_partial_guess, override=override, **kwargs)

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
