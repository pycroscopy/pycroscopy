# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:48:53 2016

@author: Suhas Somnath

Pending changes and bug fixes:
1. Fit = guess. Not sure why.
2. Parallel fitting
3. Use SHO fit R2 to leave out points when doing guess
4. Use SHO fit R2 for each point to weight each position when fitting - helps in ignoring bad fits
"""

from __future__ import division

from warnings import warn

import numpy as np

from .model import Model
from ..io.hdf_utils import getH5DsetRefs, getAuxData
from ..io.io_hdf5 import ioHDF5
from ..io.microdata import MicroDataset, MicroDataGroup

crit32 = np.dtype([('AIC_loop', np.float32),
                   ('BIC_loop', np.float32),
                   ('AIC_line', np.float32),
                   ('BIC_line', np.float32)])

loop_fit32 = np.dtype([('a_0', np.float32),
                       ('a_1', np.float32),
                       ('a_2', np.float32),
                       ('a_3', np.float32),
                       ('a_4', np.float32),
                       ('b_0', np.float32),
                       ('b_1', np.float32),
                       ('b_2', np.float32),
                       ('b_3', np.float32)])

class BELoopmodel(Model):

    def __init__(self, h5_main):
        if super.__isLegal(h5_main,variables=['DC Bias (V)']):
            self.hdf = ioHDF5(self.h5_main.file)
            self.h5_main = h5_main
            self.h5_guess = None
            self.h5_fit = None
        else:
            warn('Provided dataset is not "Main" dataset or lacks necessary ancillary datasets!')

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
        h5_loop_metrics = getAuxData(self.h5_main, auxDataName=['Loop_Metrics'])[0]

        # % Now make room for new  datasets:
        ds_fitted_loops = MicroDataset('Fitted_Loops', data=[], dtype=np.float32,
                                       maxshape=self.h5_main.shape, chunking=self.h5_main.chunks,
                                       compression='gzip')
        ds_criteria = MicroDataset('Criteria', data=[], dtype=crit32, maxshape=h5_loop_metrics.shape)
        ds_guess = MicroDataset('Guess', data=[], dtype=loop_fit32, maxshape=h5_loop_metrics.shape)

        # name of the dataset being projected. I know its 'Fit' here
        dset_name = self.h5_main.name.split('/')[-1]

        fit_grp = MicroDataGroup('-'.join([dset_name, 'Loop_Fit_']), self.h5_main.parent.name[1:])
        fit_grp.attrs['Loop_guess_method'] = "pycroscopy BE Loop"
        fit_grp.addChildren([ds_fitted_loops, ds_criteria, ds_guess])

        h5_fit_grp_refs = self.hdf.writeData(fit_grp, print_log=False)

        self.h5_fitted_loops = getH5DsetRefs(['Fitted_Loops'], h5_fit_grp_refs)[0]
        self.h5_criteria = getH5DsetRefs(['Criteria'], h5_fit_grp_refs)[0]
        self.h5_guess = getH5DsetRefs(['Guess'], h5_fit_grp_refs)[0]
        h5_fit_grp = self.h5_guess.parent

        h5_pos_dsets = getAuxData(self.h5_main, auxDataName=['Position_Indices',
                                                              'Position_Values'])
        # do linking here
        # first the positions
        for dset in [self.h5_fitted_loops, self.h5_criteria, self.h5_guess]:
            self.hdf.linkRefs(dset, h5_pos_dsets)

        # then the spectroscopic 
        h5_anc_sp_ind = getAuxData(h5_loop_metrics, auxDataName=['Spectroscopic_Indices'])[0]
        h5_anc_sp_val = getAuxData(h5_loop_metrics, auxDataName=['Spectroscopic_Values'])[0]
        for dset in [self.h5_criteria, self.h5_guess, self.h5_fit]:
            self.hdf.linkRefAsAlias(dset, h5_anc_sp_ind, 'Spectroscopic_Indices')
            self.hdf.linkRefAsAlias(dset, h5_anc_sp_val, 'Spectroscopic_Values')

        # make a new spectroscopic values dataset for the shifted Vdc...
        h5_spec_ind = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Indices'])[0]
        h5_spec_vals = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Values'])[0]
        h5_fit_grp.copy(h5_spec_vals, h5_fit_grp, name='Spectroscopic_Values')
        h5_spec_vals_shifted = h5_fit_grp['Spectroscopic_Values']
        #self.hdf.linkRefs(h5_fitted_loops, [h5_spec_ind, h5_spec_vals_shifted])

        # Link the fitted parms to the ancillary?
        #self.hdf.linkRefs(h5_fit, [h5_guess, h5_fitted_loops, h5_criteria])
        '''
        Check the input dataset for plot groups, copy them if they exist
        Also make references in the Spectroscopic Values and Indices tables
        '''

    def _createFitDataset(self):
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
                                 chunking=self.h5_guess.chunks, dtype=self.h5_guess.dtype)
        sho_grp.addChildren([ds_result])
        sho_grp.attrs['Loop_fit_method'] = "pycroscopy BE Loop"

        h5_sho_grp_refs = self.hdf.writeData(sho_grp)

        self.h5_fit = getH5DsetRefs(['Fit'], h5_sho_grp_refs)[0]

        h5_pos_dsets = getAuxData(self.h5_main, auxDataName=['Position_Indices',
                                                             'Position_Values'])
        '''self.hdf.linkRefs(self.h5_fit, h5_pos_dsets)
        self.hdf.linkRefAsAlias(self.h5_fit, h5_anc_sp_ind, 'Spectroscopic_Indices')
        self.hdf.linkRefAsAlias(self.h5_fit, h5_anc_sp_val, 'Spectroscopic_Values')'''
        '''
        Copy attributes of the fit guess
        Check the guess dataset for plot groups, copy them if they exist
        '''