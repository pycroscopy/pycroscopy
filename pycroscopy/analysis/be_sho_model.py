"""
Created on 7/17/16 10:08 AM
@author: Numan Laanait, Suhas Somnath
"""

from __future__ import division

from warnings import warn

import numpy as np
from scipy.signal import find_peaks_cwt

from .model import Model
from ..io.be_hdf_utils import isReshapable, reshapeToNsteps, reshapeToOneStep
from ..io.hdf_utils import buildReducedSpec, copyRegionRefs, linkRefs
from ..io.hdf_utils import getAuxData, getH5DsetRefs, \
    getH5RegRefIndices, createRefFromIndices
from ..io.microdata import MicroDataset, MicroDataGroup

sho32 = np.dtype([('Amplitude [V]',np.float32),('Frequency [Hz]',np.float32),('Quality Factor',np.float32),('Phase [rad]',np.float32),('R2 Criterion',np.float32)])

class BESHOmodel(Model):
    """
    Analysis of Band excitation spectra with harmonic oscillator responses.
    """

    def __init__(self, h5_main, variables=['Frequency']):
        super(BESHOmodel,self).__init__(h5_main, variables)
        self.h5_main = h5_main
        self.h5_guess = None
        self.h5_fit = None
        self.fit_points = 5
        self.step_start_inds = None
        self.is_reshapable = True


    def __createGuessDatasets(self):
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

        self.is_reshapable = isReshapable(self.h5_main, self.step_start_inds)

        ds_guess = MicroDataset('Guess', data=[],
                                maxshape=(self.h5_main.shape[0], self.num_udvs_steps),
                                chunking=(1, self.num_udvs_steps), dtype=sho32)

        not_freq = h5_spec_inds.attrs['labels'] != 'Frequency'
        if h5_spec_inds.shape[0] > 1:
            # More than just the frequency dimension, eg Vdc etc - makes it a BEPS dataset

            ds_sho_inds, ds_sho_vals = buildReducedSpec(h5_spec_inds, h5_spec_vals, not_freq, self.step_start_inds)

        else:
            '''
            Special case for datasets that only vary by frequency. Example - BE-Line
            '''
            ds_sho_inds = MicroDataset('Spectroscopic_Indices', np.array([[0]], dtype=np.uint32))
            ds_sho_vals = MicroDataset('Spectroscopic_Values', np.array([[0]], dtype=np.float32))

            ds_sho_inds.attrs['labels'] = {'Single_Step': (slice(0, None), slice(None))}
            ds_sho_vals.attrs['labels'] = {'Single_Step': (slice(0, None), slice(None))}
            ds_sho_inds.attrs['units'] = ''
            ds_sho_vals.attrs['units'] = ''

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

    def __createFitDataset(self):
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
        Check the guess dataset for plot groups, copy them if they exist
        '''
        for attr_name, attr_val in self.h5_guess.attrs.iteritems():

            if '_Plot_Group' in attr_name:
                ref_inds = getH5RegRefIndices(attr_val, self.h5_guess, return_method='corners')
                ref_inds = ref_inds.reshape([-1, 2, 2])
                fit_ref = createRefFromIndices(self.h5_fit, ref_inds)

                self.h5_fit.attrs[attr_name] = fit_ref
            else:
                self.h5_fit.attrs[attr_name] = attr_val

        # Reference linking
        self.hdf.linkRefs(self.h5_main, [self.h5_fit])

    def __getDataChunk(self):
        """
        Returns a chunk of data for the guess or the fit

        Parameters:
        -----
        None

        Returns:
        --------
        None
        """
        self.data = reshapeToOneStep(self.h5_main.value, self.num_udvs_steps)

    def __getGuessChunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset

        Parameters:
        -----
        None

        Returns:
        --------
        None
        """
        guess_mat = self.h5_guess['Amplitude [V]', 'Frequency [Hz]', 'Quality Factor', 'Phase [rad]'][:, :]#[st_pix:en_pix, :]
        # don't keep the R^2.
        self.guess = reshapeToOneStep(guess_mat, self.num_udvs_steps)
        # bear in mind that this self.guess is a compound dataset.

    def __setDataChunk(self, data_chunk, is_guess=False):
        """
        Writes the provided chunk of data into the guess or fit datasets. This method is responsible for any and all book-keeping

        Parameters
        ---------
        data_chunk : nd array
            n dimensional array of the same type as the guess / fit dataset
        is_guess : Boolean
            Flag that differentiates the guess from the fit
        """

        reorganized = reshapeToNsteps(data_chunk, self.num_udvs_steps)
        if is_guess:
            self.h5_guess[:, :] = reorganized
        else:
            self.h5_fit[:, :] = reorganized

    def computeGuess(self, strategy='wavelet_peaks', options={"peak_widths": np.array([10,200])}, **kwargs):
        '''

        Parameters
        ----------
        data
        strategy
        kwargs

        Returns
        -------

        '''
        self.__createGuessDatasets()
        self.__getDataChunk()
        self.guess = super(BESHOmodel,self).computeGuess(strategy=strategy,**options)

        # Extracting and reshaping the remaining parameters for SHO
        # TODO: Remove the dummy slice from self.data
        slic = slice(0,400,None)
        if strategy in ['wavelet_peaks','relative_maximum','absolute_maximum']:
            peaks = np.array([g[0] for g in self.guess])
            ampl = np.abs(self.data[slic])
            res_ampl = np.array([ampl[ind,peaks[ind]] for ind in np.arange(peaks.size)])
            # res_freq = getfrequencyarray [peaks]
            q_factor = np.ones_like(self.guess)*10
            phase = np.angle(self.data[slic])
            res_phase = np.array([phase[ind,peaks[ind]] for ind in np.arange(peaks.size)])

        return res_ampl, q_factor, res_phase





#####################################
# Guess Functions                   #
#####################################

def waveletPeaks(vector, peakWidthBounds=[10,300],**kwargs):
    waveletWidths = np.linspace(peakWidthBounds[0],peakWidthBounds[1],20)
    def __wpeaks(vector):
        peakIndices = find_peaks_cwt(vector, waveletWidths,**kwargs)
        return peakIndices
    return __wpeaks

def relativeMax(vector, peakWidthBounds=[10,300],**kwargs):
    waveletWidths = np.linspace(peakWidthBounds[0],peakWidthBounds[1],20)
    peakIndices = find_peaks_cwt(vector, waveletWidths,**kwargs)
    return peakIndices




