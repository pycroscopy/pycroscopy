# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 16:06:39 2016

@author: Suhas Somnath, Chris R. Smith, Numan Laanait
"""

from __future__ import division

from warnings import warn

import numpy as np
from scipy.signal import find_peaks_cwt

from .Model import Model
from ..io.be_hdf_utils import isReshapable
from ..io.hdf_utils import buildReducedSpec, copyRegionRefs
from ..io.hdf_utils import getAuxData, getH5DsetRefs, \
    getH5RegRefIndices, createRefFromIndices
from ..io.io_hdf5 import ioHDF5
from ..io.microdata import MicroDataset, MicroDataGroup

sho32 = np.dtype([('Amplitude [V]',np.float32),('Frequency [Hz]',np.float32),('Quality Factor',np.float32),('Phase [rad]',np.float32),('R2 Criterion',np.float32)])

class BESHOmodel(Model):

    def __init__(self, h5_main):

        if super.__isLegal(h5_main,variables=['Frequency']):
            self.hdf = ioHDF5(self.h5_main.file)
            self.h5_main = h5_main
            self.h5_guess = None
            self.h5_fit = None
            self.fit_points = 5
            self.udvs_step_starts = None
            self.is_reshapable = False
        else:
            warn('Provided dataset is not "Main" dataset and lacks necessary ancillary datasets!')


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

        self.udvs_step_starts = np.where(h5_spec_inds[0] == 0)[0]
        num_udvs_steps = len(self.udvs_step_starts)

        self.is_reshapable = isReshapable(self.h5_main, self.step_start_inds)

        ds_guess = MicroDataset('Guess', data=[],
                                maxshape=(self.h5_main.shape[0], num_udvs_steps),
                                chunking=(1, num_udvs_steps), dtype=sho32)

        not_freq = h5_spec_inds.attrs['labels'] != 'Frequency'
        if h5_spec_inds.shape[0] > 1:
            # More than just the frequency dimension, eg Vdc etc - makes it a BEPS dataset

            ds_sho_inds, ds_sho_vals = buildReducedSpec(h5_spec_inds, h5_spec_vals, not_freq, self.udvs_step_starts)

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
        self.hdf.linkRefs(self.h5_guess, [h5_sho_inds, h5_sho_vals])

        # Linking ancillary position datasets:
        aux_dsets = getAuxData(self.h5_main, auxDataName=['Position_Indices', 'Position_Values'])
        self.hdf.linkRefs(self.h5_guess, aux_dsets)

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
        dset : n dimensional array
            A portion of the main dataset
        """
        pass

    def __getGuessChunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset

        Parameters:
        -----
        None

        Returns:
        --------
        dset : n dimensional array
            A portion of the guess dataset
        """
        pass

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
        pass

    def computeGuess(self, data, strategy='Wavelet_Peaks', **kwargs):
        '''

        Parameters
        ----------
        data
        strategy
        kwargs

        Returns
        -------

        '''

        super.computeGuess()



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




