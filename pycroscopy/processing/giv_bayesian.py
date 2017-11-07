# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 11:48:53 2017

@author: Suhas Somnath

"""

from __future__ import division, print_function, absolute_import, unicode_literals

import time as tm
import numpy as np
from .process import Process, parallel_compute
from ..io.microdata import MicroDataset, MicroDataGroup
from ..io.io_utils import realToCompound
from ..io.hdf_utils import getH5DsetRefs, getAuxData, copyAttributes, link_as_main
from ..io.translators.utils import build_ind_val_dsets
from ..io.io_hdf5 import ioHDF5
from .giv_utils import do_bayesian_inference

cap_dtype = np.dtype({'names': ['Forward', 'Reverse'],
                      'formats': [np.float32, np.float32]})


class GIVBayesian(Process):
    def __init__(self, h5_main, ex_freq, gain, num_x_steps=250, r_extra=220, verbose=False,
                 cores=None, max_mem_mb=1024, **kwargs):
        """
        Parameters
        ----------
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.
        cores : uint, optional
            Default - 1
            How many cores to use for the computation
        max_mem_mb : uint, optional
            How much memory to use for the computation.  Default 1024 Mb
        """
        super(GIVBayesian, self).__init__(h5_main, cores=cores, max_mem_mb=max_mem_mb, verbose=verbose)
        self.verbose = verbose
        self.gain = gain
        self.ex_freq = ex_freq
        self.r_extra = r_extra
        self.num_x_steps = int(num_x_steps)
        if self.num_x_steps % 4 == 0:
            self.num_x_steps = ((self.num_x_steps // 2) + 1) * 2
        if verbose:
            print('ensuring that half steps should be odd, num_x_steps is now', self.num_x_steps)

        # take these from kwargs
        self.bayesian_parms = {'gam': 0.03, 'e': 10.0, 'sigma': 10.0, 'sigmaC': 1.0, 'num_samples': 2E3}

        self.parm_dict = {'freq': self.ex_freq, 'num_x_steps': self.num_x_steps}
        self.parm_dict.update(self.bayesian_parms)

        h5_spec_vals = getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0]
        self.single_ao = np.squeeze(h5_spec_vals[()])

        roll_cyc_fract = -0.25
        self.roll_pts = int(self.single_ao.size * roll_cyc_fract)
        self.rolled_bias = np.roll(self.single_ao, self.roll_pts)

    def _set_memory_and_cores(self, cores=1, mem=1024):
        """
        Checks hardware limitations such as memory, # cpus and sets the recommended datachunk sizes and the
        number of cores to be used by analysis methods.

        Parameters
        ----------
        cores : uint, optional
            Default - 1
            How many cores to use for the computation
        mem : uint, optional
            Default - 1024
            The amount a memory in Mb to use in the computation
        """
        super(GIVBayesian, self)._set_memory_and_cores(cores=cores, mem=mem)
        self._max_pos_per_read = 1024  # Need a better way of figuring out a more appropriate estimate

    def _create_results_datasets(self):
        """
        Process specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.
        """
        # create all h5 datasets here:
        num_pos = self.h5_main.shape[0]

        if self.verbose:
            print('Now creating the datasets')

        ds_spec_inds, ds_spec_vals = build_ind_val_dsets([self.num_x_steps], is_spectral=True,
                                                         labels=['Bias'], units=['V'], verbose=self.verbose)

        cap_shape = (num_pos, 1)
        ds_cap = MicroDataset('Capacitance', data=[], maxshape=cap_shape, dtype=cap_dtype, chunking=cap_shape,
                              compression='gzip')
        ds_cap.attrs = {'quantity': 'Capacitance', 'units': 'pF'}
        ds_cap_spec_inds, ds_cap_spec_vals = build_ind_val_dsets([1], is_spectral=True,
                                                                 labels=['Direction'], units=[''], verbose=self.verbose)
        # the names of these datasets will clash with the ones created above. Change names manually:
        ds_cap_spec_inds.name = 'Spectroscopic_Indices_Cap'
        ds_cap_spec_vals.name = 'Spectroscopic_Values_Cap'

        ds_r_var = MicroDataset('R_variance', data=[], maxshape=(num_pos, self.num_x_steps), dtype=np.float32,
                                chunking=(1, self.num_x_steps), compression='gzip')
        ds_r_var.attrs = {'quantity': 'Resistance', 'units': 'GOhms'}
        ds_res = MicroDataset('Resistance', data=[], maxshape=(num_pos, self.num_x_steps), dtype=np.float32,
                              chunking=(1, self.num_x_steps), compression='gzip')
        ds_res.attrs = {'quantity': 'Resistance', 'units': 'GOhms'}
        ds_i_corr = MicroDataset('Corrected_Current', data=[], maxshape=(num_pos, self.single_ao.size),
                                 dtype=np.float32,
                                 chunking=(1, self.single_ao.size), compression='gzip')
        # don't bother adding any other attributes, all this will be taken from h5_main

        bayes_grp = MicroDataGroup(self.h5_main.name.split('/')[-1] + '-Bayesian_Inference_',
                                   parent=self.h5_main.parent.name)
        bayes_grp.addChildren([ds_spec_inds, ds_spec_vals, ds_cap, ds_r_var, ds_res, ds_i_corr,
                               ds_cap_spec_inds, ds_cap_spec_vals])
        bayes_grp.attrs = {'split_directions': True, 'algorithm_author': 'Kody J. Law',
                           'r_extra': self.r_extra, 'last_pixel': 0}
        bayes_grp.attrs.update(self.bayesian_parms)

        if self.verbose:
            bayes_grp.showTree()

        self.hdf = ioHDF5(self.h5_main.file)
        h5_refs = self.hdf.writeData(bayes_grp, print_log=self.verbose)

        self.h5_new_spec_vals = getH5DsetRefs(['Spectroscopic_Values'], h5_refs)[0]
        h5_new_spec_inds = getH5DsetRefs(['Spectroscopic_Indices'], h5_refs)[0]
        h5_cap_spec_vals = getH5DsetRefs(['Spectroscopic_Values_Cap'], h5_refs)[0]
        h5_cap_spec_inds = getH5DsetRefs(['Spectroscopic_Indices_Cap'], h5_refs)[0]
        self.h5_cap = getH5DsetRefs(['Capacitance'], h5_refs)[0]
        self.h5_variance = getH5DsetRefs(['R_variance'], h5_refs)[0]
        self.h5_resistance = getH5DsetRefs(['Resistance'], h5_refs)[0]
        self.h5_i_corrected = getH5DsetRefs(['Corrected_Current'], h5_refs)[0]
        self.h5_results_grp = self.h5_cap.parent

        if self.verbose:
            print('Finished making room for the datasets. Now linking them')

        # Now link the datasets appropriately so that they become hubs:
        h5_pos_vals = getAuxData(self.h5_main, auxDataName=['Position_Values'])[0]
        h5_pos_inds = getAuxData(self.h5_main, auxDataName=['Position_Indices'])[0]

        # Capacitance main dataset:
        link_as_main(self.h5_cap, h5_pos_inds, h5_pos_vals, h5_cap_spec_inds, h5_cap_spec_vals)

        # the corrected current dataset is the same as the main dataset in every way
        copyAttributes(self.h5_main, self.h5_i_corrected, skip_refs=False)

        # The resistance datasets get new spec datasets but reuse the old pos datasets:
        for new_dset in [self.h5_resistance, self.h5_variance]:
            link_as_main(new_dset, h5_pos_inds, h5_pos_vals, h5_new_spec_inds, self.h5_new_spec_vals)

        if self.verbose:
            print('Finished linking all datasets!')

    def _write_results_chunk(self):

        if self.verbose:
            print('Started accumulating all results')
        num_pixels = len(self.forward_results)
        cap_mat = np.zeros((num_pixels, 2), dtype=np.float32)
        r_inf_mat = np.zeros((num_pixels, self.num_x_steps), dtype=np.float32)
        r_var_mat = np.zeros((num_pixels, self.num_x_steps), dtype=np.float32)
        i_cor_sin_mat = np.zeros((num_pixels, self.single_ao.size), dtype=np.float32)

        for pix_ind, i_meas, forw_results, rev_results in zip(range(num_pixels), self.data,
                                                              self.forward_results, self.reverse_results):
            full_results = dict()
            for item in ['Irec', 'cValue']:
                full_results[item] = np.hstack((forw_results[item], rev_results[item]))
                # print(item, full_results[item].shape)

            # Capacitance is always doubled - halve it now:
            full_results['cValue'] *= 0.5
            cap_val = np.mean(full_results['cValue'])

            # Compensating the resistance..
            omega = 2 * np.pi * self.ex_freq
            i_cap = cap_val * omega * self.rolled_bias
            i_extra = self.r_extra * cap_val * self.single_ao
            i_corr_sine = i_meas - i_cap - i_extra

            # Now also correct the inferred resistance
            for res_part in [forw_results, rev_results]:
                res_part['mR'] = res_part['mR'] / (1 - (res_part['mR'] * self.r_extra * cap_val))

            # by default Bayesian inference will sort bias in ascending order
            for item in ['x', 'mR', 'vR']:
                full_results[item] = np.hstack((forw_results[item], np.flipud(rev_results[item])))

            i_cor_sin_mat[pix_ind] = i_corr_sine
            cap_mat[pix_ind] = full_results['cValue'] * 1000  # convert from nF to pF
            r_inf_mat[pix_ind] = full_results['mR']
            r_var_mat[pix_ind] = full_results['vR']

        # Now write to h5 files:
        if self.verbose:
            print('Finished accumulating results. Writing to h5')

        if self._start_pos == 0:
            self.h5_new_spec_vals[0, :] = full_results['x']  # Technically this needs to only be done once

        pos_slice = slice(self._start_pos, self._end_pos)
        self.h5_cap[pos_slice] = np.atleast_2d(realToCompound(cap_mat, cap_dtype)).T
        self.h5_variance[pos_slice] = r_var_mat
        self.h5_resistance[pos_slice] = r_inf_mat
        self.h5_i_corrected[pos_slice] = i_cor_sin_mat

        # Leaving in this provision that will allow restarting of processes
        self.h5_results_grp['last_pixel'] = self._end_pos

        self.hdf.flush()

        # Now update the start position
        self._start_pos = self._end_pos

    @staticmethod
    def _unit_function():
        return do_bayesian_inference

    def compute(self, *args, **kwargs):
        """

        Parameters
        ----------
        kwargs:
            processors: int
                number of processors to use. Default all processors on the system except for 1.

        Returns
        -------

        """
        self._create_results_datasets()

        half_v_steps = self.single_ao.size // 2

        # remove additional parm and halve the x points
        bayes_parms = self.parm_dict.copy()
        bayes_parms['num_x_steps'] = self.num_x_steps // 2
        bayes_parms['econ'] = True
        del(bayes_parms['freq'])

        time_per_pix = 0

        num_pos = self.h5_main.shape[0]

        self._read_data_chunk()
        while self.data is not None:

            t_start = tm.time()

            # first roll the data
            rolled_raw_data = np.roll(self.data, self.roll_pts, axis=1)

            self.reverse_results = parallel_compute(rolled_raw_data[:, :half_v_steps], do_bayesian_inference,
                                                    cores=self._cores,
                                                    func_args=[self.rolled_bias[:half_v_steps], self.ex_freq],
                                                    func_kwargs=bayes_parms)

            if self.verbose:
                print('Finished processing forward sections. Now working on reverse sections....')

            self.forward_results = parallel_compute(rolled_raw_data[:, half_v_steps:], do_bayesian_inference,
                                                    cores=self._cores,
                                                    func_args=[self.rolled_bias[half_v_steps:], self.ex_freq],
                                                    func_kwargs=bayes_parms)
            if self.verbose:
                print('Finished processing reverse loops')

            tot_time = np.round(tm.time() - t_start)

            if self.verbose:
                print('Done parallel computing in {} sec or {} sec per pixel'.format(tot_time,
                                                                                     tot_time / self._max_pos_per_read))
            if self._start_pos == 0:
                time_per_pix = tot_time / self._end_pos  # in seconds
            else:
                print('Time remaining: {} hours'.format(np.round((num_pos - self._end_pos) * time_per_pix / 3600, 2)))

            self._write_results_chunk()
            self._read_data_chunk()

        if self.verbose:
            print('Finished processing the dataset completely')

        # return self.h5_cap.parent
        return self.h5_results_grp
