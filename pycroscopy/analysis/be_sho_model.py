"""
Created on 7/17/16 10:08 AM
@author: Suhas Somnath, Numan Laanait
"""

from __future__ import division
from warnings import warn
import numpy as np
from .model import Model
from ..io.be_hdf_utils import isReshapable, reshapeToNsteps, reshapeToOneStep
from ..io.hdf_utils import buildReducedSpec, copyRegionRefs, linkRefs, getAuxData, getH5DsetRefs, \
            copyAttributes
from ..io.microdata import MicroDataset, MicroDataGroup
from .guess_methods import r_square
from .utils.be_sho import SHOfunc


sho32 = np.dtype([('Amplitude [V]', np.float32), ('Frequency [Hz]', np.float32),
                  ('Quality Factor', np.float32), ('Phase [rad]', np.float32),
                  ('R2 Criterion', np.float32)])


class BESHOmodel(Model):
    """
    Analysis of Band excitation spectra with harmonic oscillator responses.
    """

    def __init__(self, h5_main, variables=['Frequency'], parallel=True):
        super(BESHOmodel, self).__init__(h5_main, variables, parallel)
        self.step_start_inds = None
        self.is_reshapable = True
        self.num_udvs_steps = None
        self.freq_vec = None

    def _createGuessDatasets(self):
        """
        Creates the h5 group, guess dataset, corresponding spectroscopic datasets and also
        links the guess dataset to the spectroscopic datasets.

        Parameters
        --------
        None

        Returns
        -------
        None
        """
        # Create all the ancilliary datasets, allocate space.....

        h5_spec_inds = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Indices'])[0]
        h5_spec_vals = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Values'])[0]

        self.step_start_inds = np.where(h5_spec_inds[0] == 0)[0]
        self.num_udvs_steps = len(self.step_start_inds)

        # find the frequency vector and hold in memory
        self._getFrequencyVector()

        self.is_reshapable = isReshapable(self.h5_main, self.step_start_inds)

        ds_guess = MicroDataset('Guess', data=[],
                                maxshape=(self.h5_main.shape[0], self.num_udvs_steps),
                                chunking=(1, self.num_udvs_steps), dtype=sho32)

        not_freq = h5_spec_inds.attrs['labels'] != 'Frequency'

        ds_sho_inds, ds_sho_vals = buildReducedSpec(h5_spec_inds, h5_spec_vals, not_freq, self.step_start_inds)

        dset_name = self.h5_main.name.split('/')[-1]
        sho_grp = MicroDataGroup('-'.join([dset_name,
                                           'SHO_Fit_']),
                                 self.h5_main.parent.name[1:])
        sho_grp.addChildren([ds_guess,
                             ds_sho_inds,
                             ds_sho_vals])
        sho_grp.attrs['SHO_guess_method'] = "pycroscopy BESHO"

        h5_sho_grp_refs = self.hdf.writeData(sho_grp)

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

    def _createFitDatasets(self):
        """
        Creates the HDF5 fit dataset. pycroscopy requires that the h5 group, guess dataset,
        corresponding spectroscopic and position datasets be created and populated at this point.
        This function will create the HDF5 dataset for the fit and link it to same ancillary datasets as the guess.
        The fit dataset will NOT be populated here but will instead be populated using the __setData function

        Parameters
        --------
        None

        Returns
        -------
        None
        """

        if self.h5_guess is None:
            warn('Need to guess before fitting!')
            return

        if self.step_start_inds is None:
            h5_spec_inds = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Indices'])[0]
            self.step_start_inds = np.where(h5_spec_inds[0] == 0)[0]

        if self.num_udvs_steps is None:
            self.num_udvs_steps = len(self.step_start_inds)

        if self.freq_vec is None:
            self._getFrequencyVector()

        h5_sho_grp = self.h5_guess.parent

        sho_grp = MicroDataGroup(h5_sho_grp.name.split('/')[-1],
                                 h5_sho_grp.parent.name[1:])

        # dataset size is same as guess size
        ds_result = MicroDataset('Fit', data=[], maxshape=(self.h5_guess.shape[0], self.h5_guess.shape[1]),
                                 chunking=self.h5_guess.chunks, dtype=sho32)
        sho_grp.addChildren([ds_result])
        sho_grp.attrs['SHO_fit_method'] = "pycroscopy BESHO"

        h5_sho_grp_refs = self.hdf.writeData(sho_grp)

        self.h5_fit = getH5DsetRefs(['Fit'], h5_sho_grp_refs)[0]

        '''
        Copy attributes of the fit guess
        '''
        copyAttributes(self.h5_guess, self.h5_fit, skip_refs=False)


    def _getFrequencyVector(self):
        """
        Assumes that the data is reshape-able
        :return:
        """
        h5_spec_vals = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Values'])[0]
        if len(self.step_start_inds) == 1:  # BE-Line
            end_ind = h5_spec_vals.shape[1]
        else:  # BEPS
            end_ind = self.step_start_inds[1]
        self.freq_vec = h5_spec_vals[0, self.step_start_inds[0]:end_ind]

    def _getDataChunk(self, verbose=False):
        """
        Returns a chunk of data for the guess or the fit

        Parameters:
        -----
        None

        Returns:
        --------
        None
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
        # At this point the self.data object is the raw data that needs to be reshaped to a single UDVS step:
        if self.data is not None:
            if verbose: print('Got raw data of shape {} from super'.format(self.data.shape))
            self.data = reshapeToOneStep(self.data, self.num_udvs_steps)
            if verbose: print('Reshaped raw data to shape {}'.format(self.data.shape))

    def _getGuessChunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset

        Parameters:
        -----
        None

        Returns:
        --------
        None
        """
        if self.data is None:
            self._end_pos = int(min(self.h5_main.shape[0], self._start_pos + self._max_pos_per_read))
            self.guess = self.h5_guess[self._start_pos:self._end_pos, :]
        else:
            self.guess = self.h5_guess[self._start_pos:self._end_pos, :]
        # At this point the self.data object is the raw data that needs to be reshaped to a single UDVS step:
        self.guess = reshapeToOneStep(self.guess, self.num_udvs_steps)
        # don't keep the R^2.
        self.guess = np.hstack([self.guess[name] for name in self.guess.dtype.names[:-1]])
        # bear in mind that this self.guess is a compound dataset.

    def _setResults(self, is_guess=False, verbose=False):
        """
        Writes the provided chunk of data into the guess or fit datasets. This method is responsible for any and all book-keeping

        Parameters
        ---------
        data_chunk : nd array
            n dimensional array of the same type as the guess / fit dataset
        is_guess : Boolean
            Flag that differentiates the guess from the fit
        """
        if is_guess:
            # prepare to reshape:
            self.guess = np.transpose(np.atleast_2d(self.guess))
            if verbose:
                print('Prepared guess of shape {} before reshaping'.format(self.guess.shape))
            self.guess = reshapeToNsteps(self.guess, self.num_udvs_steps)
            if verbose:
                print('Reshaped guess to shape {}'.format(self.guess.shape))
        else:
            self.fit = np.transpose(np.atleast_2d(self.fit))
            self.fit = reshapeToNsteps(self.fit, self.num_udvs_steps)

        # ask super to take care of the rest, which is a standardized operation
        super(BESHOmodel, self)._setResults(is_guess)

    def doGuess(self, processors=None, strategy='wavelet_peaks',
                     options={"peak_widths": np.array([10,200]),"peak_step":20}):
        """

        Parameters
        ----------
        data
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. For updated list, run GuessMethods.methods
        options: dict
            Default Options for wavelet_peaks{"peaks_widths": np.array([10,200]), "peak_step":20}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.

        kwargs:
            processors: int
                number of processors to use. Default all processors on the system except for 1.

        Returns
        -------

        """
        if processors is None:
            processors = self._maxCpus
        else:
            processors = min(processors, self._maxCpus)

        self._createGuessDatasets()
        self._start_pos = 0
        if strategy == 'complex_gaussian':
            freq_vec = self.freq_vec
            options = {'frequencies': freq_vec}
        super(BESHOmodel, self).doGuess(processors=processors, strategy=strategy, options=options)


    def doFit(self, processors=None, solver_type='least_squares',solver_options={'jac':'cs'},
              obj_func={'class': 'Fit_Methods', 'obj_func': 'SHO', 'xvals': np.array([])}):
        """

        Parameters
        ----------
        processors: int
            Default is 1.
            Number of processors to use.
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. For updated list, run GuessMethods.methods
        options: dict
            Default {"peaks_widths": np.array([10,200])}}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.


        Returns
        -------

        """
        if processors is None:
            processors = self._maxCpus
        else:
            processors = min(processors, self._maxCpus)

        self._createFitDatasets()
        self._start_pos = 0
        xvals = self.freq_vec
        results = super(BESHOmodel, self).doFit(processors=processors, solver_type=solver_type, solver_options=solver_options,
                          obj_func={'class':'Fit_Methods','obj_func':'SHO', 'xvals':xvals})
        return results

    def _reformatResults(self, results, strategy='wavelet_peaks', verbose=False):
        """
        Model specific calculation and or reformatting of the raw guess or fit results
        :param results:
        :return:
        """
        if verbose:
            print('Strategy to use: {}'.format(strategy))
        # Create an empty array to store the guess parameters
        sho_vec = np.zeros(shape=(len(results)), dtype=sho32)
        if verbose:
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
            if verbose:
                print('Peak positions of shape {}'.format(peak_inds.shape))
            # First get the value (from the raw data) at these positions:
            comp_vals = np.array(
                [self.data[pixel_ind, peak_inds[pixel_ind]] for pixel_ind in np.arange(peak_inds.size)])
            if verbose:
                print('Complex values at peak positions of shape {}'.format(comp_vals.shape))
            sho_vec['Amplitude [V]'] = np.abs(comp_vals)  # Amplitude
            sho_vec['Phase [rad]'] = np.angle(comp_vals)  # Phase in radians
            sho_vec['Frequency [Hz]'] = self.freq_vec[peak_inds]  # Frequency
            sho_vec['Quality Factor'] = np.ones_like(comp_vals) * 10  # Quality factor
            # Add something here for the R^2
            sho_vec['R2 Criterion'] = np.array([self.r_square(self.data, self._sho_func, self.freq_vec, sho_parms) for sho_parms in sho_vec])
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

#####################################
# Guess Functions                   #
#####################################

# def waveletPeaks(vector, peakWidthBounds=[10,300],**kwargs):
#     waveletWidths = np.linspace(peakWidthBounds[0],peakWidthBounds[1],20)
#     def __wpeaks(vector):
#         peakIndices = find_peaks_cwt(vector, waveletWidths,**kwargs)
#         return peakIndices
#     return __wpeaks
#
# def relativeMax(vector, peakWidthBounds=[10,300],**kwargs):
#     waveletWidths = np.linspace(peakWidthBounds[0],peakWidthBounds[1],20)
#     peakIndices = find_peaks_cwt(vector, waveletWidths,**kwargs)
#     return peakIndices




