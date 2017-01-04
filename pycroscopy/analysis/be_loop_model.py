# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:48:53 2016

@author: Suhas Somnath

"""

from __future__ import division

from warnings import warn

import numpy as np

from .model import Model
from .be_sho_model import sho32
from ..io.hdf_utils import getH5DsetRefs, getAuxData, copyRegionRefs, linkRefs, linkRefAsAlias
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


class BELoopModel(Model):
    """
    Analysis of Band excitation loops using functional fits
    """

    def __init__(self, h5_main, variables=['DC_Offset'], parallel=True):
        super(BELoopModel, self).__init__(h5_main, variables, parallel)

    def _isLegal(self, h5_main, variables=['DC_Offset']):
        """
        Checks whether or not the provided object can be analyzed by this class.

        Parameters:
        ----
        h5_main : h5py.Dataset instance
            The dataset containing the SHO Fit (not necessarily the dataset directly resulting from SHO fit)
            over which the loop projection, guess, and fit will be performed.
        variables : list(string)
            The dimensions needed to be present in the attributes of h5_main to analyze the data with Model.

        Returns:
        -------
        legal : Boolean
            Whether or not this dataset satisfies the necessary conditions for analysis
        """
        if h5_main.file.attrs['data_type'] != 'BEPSData':
            warn('Provided dataset does not appear to be a BEPS dataset')
            return False

        if not h5_main.name.startswith('/Measurement_'):
            warn('Provided dataset is not derived from a measurement group')
            return False

        meas_grp_name = h5_main.name.split('/')
        h5_meas_grp = h5_main.file[meas_grp_name[1]]

        if h5_meas_grp.attrs['VS_mode'] not in ['DC modulation mode', 'current mode']:
            warn('Provided dataset is not a DC modulation or current mode BEPS dataset')
            return False

        if h5_meas_grp.attrs['VS_cycle_fraction'] != 'full':
            warn('Provided dataset does not have full cycles')
            return False

        if h5_main.dtype != sho32:
            warn('Provided dataset is not a SHO results dataset.')
            return False

        return super(BELoopModel, self)._isLegal(h5_main, variables)

    def __create_projection_datasets(self):
        # First grab the spectroscopic indices and values
        h5_spec_inds = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Indices'])[0]
        h5_spec_vals = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Values'])[0]

        dc_ind = np.argwhere(h5_spec_vals.attrs['labels'] == 'DC_Offset').flatten()
        not_dc_inds = np.delete(np.arange(h5_spec_vals.shape[0]), dc_ind)

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(h5_spec_inds[dc_ind, :] == 0)
        tot_cycles = cycle_start_inds.size

        if not_dc_inds.size == 0:
            # default case - single cycle: simple 1D in spectroscopic
            metrics_labels = ''
            metrics_units = ''

            met_spec_inds_mat = np.zeros([1, 1], dtype=np.uint8)
            met_spec_vals_mat = np.zeros([1, 1], dtype=np.float32)
        else:
            # typical case - multiple spectroscopic indices
            # Metrics lose the first dimension - Vdc:
            metrics_labels = h5_spec_vals.attrs['labels'][not_dc_inds]
            metrics_units = h5_spec_vals.attrs['units'][not_dc_inds]

            met_spec_inds_mat = h5_spec_inds[not_dc_inds, :][:, cycle_start_inds].squeeze()
            met_spec_vals_mat = h5_spec_vals[not_dc_inds, :][:, cycle_start_inds].squeeze()

        # Prepare containers for the dataets
        ds_projected_loops = MicroDataset('Projected_Loops', data=[], dtype=np.float32,
                                          maxshape=self.h5_main.shape, chunking=self.h5_main.chunks,
                                          compression='gzip')
        ds_loop_metrics = MicroDataset('Loop_Metrics', data=[], dtype=loop_fit32,
                                       maxshape=(self.h5_main.shape[0], tot_cycles))
        ds_loop_met_spec_inds = MicroDataset('Loop_Metrics_Indices', data=met_spec_inds_mat)
        ds_loop_met_spec_vals = MicroDataset('Loop_Metrics_Values', data=met_spec_vals_mat)

        # prepare slices for the metrics. Order is preserved anyway
        metrics_slices = dict()
        for row_ind, row_name in enumerate(metrics_labels):
            metrics_slices[row_name] = (slice(row_ind, row_ind + 1), slice(None))

        # Add necessary labels and units
        ds_loop_met_spec_inds.attrs['labels'] = metrics_slices
        ds_loop_met_spec_vals.attrs['labels'] = metrics_slices
        ds_loop_met_spec_vals.attrs['units'] = metrics_units

        # name of the dataset being projected.
        dset_name = self.h5_main.name.split('/')[-1]

        proj_grp = MicroDataGroup('-'.join([dset_name, 'Loop_Fit_']),
                                  self.h5_main.parent.name[1:])
        proj_grp.attrs['projection_method'] = 'pycroscopy BE loop model'
        proj_grp.addChildren([ds_projected_loops, ds_loop_metrics,
                              ds_loop_met_spec_inds, ds_loop_met_spec_vals])

        h5_proj_grp_refs = self.hdf.writeData(proj_grp)

        h5_projected_loops = getH5DsetRefs(['Projected_Loops'], h5_proj_grp_refs)[0]
        h5_loop_metrics = getH5DsetRefs(['Loop_Metrics'], h5_proj_grp_refs)[0]
        h5_loop_met_spec_inds = getH5DsetRefs(['Loop_Metrics_Indices'], h5_proj_grp_refs)[0]
        h5_loop_met_spec_vals = getH5DsetRefs(['Loop_Metrics_Values'], h5_proj_grp_refs)[0]

        h5_pos_dsets = getAuxData(self.h5_main, auxDataName=['Position_Indices',
                                                             'Position_Values'])
        # do linking here
        # first the positions
        linkRefs(h5_projected_loops, h5_pos_dsets)
        linkRefs(h5_projected_loops, [h5_loop_metrics])
        linkRefs(h5_loop_metrics, h5_pos_dsets)
        # then the spectroscopic
        linkRefs(h5_projected_loops, [h5_spec_inds, h5_spec_vals])
        linkRefAsAlias(h5_loop_metrics, h5_loop_met_spec_inds, 'Spectroscopic_Indices')
        linkRefAsAlias(h5_loop_metrics, h5_loop_met_spec_vals, 'Spectroscopic_Values')

        copyRegionRefs(self.h5_main, h5_projected_loops)
        copyRegionRefs(self.h5_main, h5_loop_metrics)

        self.hdf.flush()

        return

    def __project_loops(self):
        pass

    def _createGuessDatasets(self):
        """
        Creates the h5 group, guess dataset, corresponding spectroscopic datasets and also
        links the guess dataset to the spectroscopic datasets.

        """


        """
        h5_loop_metrics = getAuxData(self.h5_main, auxDataName=['Loop_Metrics'])[0]

        # % Now make room for new  datasets:
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
        """
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