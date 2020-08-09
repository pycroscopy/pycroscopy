# -*- coding: utf-8 -*-
"""
:class:`~pycroscopy.analysis.giv_bayesian.GIVBayesian` - A class that performs Bayesian inference to decouple spurious
hysteresis from current-voltage spectroscopy signals in General mode I-V data

Created on Thu Nov 02 11:48:53 2017

@author: Suhas Somnath

"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from sidpy.proc.comp_utils import parallel_compute
from sidpy.hdf.dtype_utils import stack_real_to_compound
from sidpy.hdf.hdf_utils import write_simple_attrs, print_tree, get_attributes
from pyUSID.io.write_utils import Dimension
from pyUSID.processing.process import Process
from pyUSID.io.hdf_utils import write_main_dataset, create_results_group, \
    create_empty_dataset
from pyUSID import USIDataset
from .utils.giv_utils import do_bayesian_inference, bayesian_inference_on_period

cap_dtype = np.dtype({'names': ['Forward', 'Reverse'],
                      'formats': [np.float32, np.float32]})
# TODO : Take lesser used bayesian inference params from kwargs if provided


class GIVBayesian(Process):
    """
    A class that performs Bayesian inference to decouple spurious hysteresis signals from current-voltage spectroscopy
    signals in General mode I-V data
    """

    def __init__(self, h5_main, ex_freq, gain, num_x_steps=250, r_extra=110,
                 **kwargs):
        """
        Applies Bayesian Inference to General Mode IV (G-IV) data to extract the true current

        Parameters
        ----------
        h5_main : h5py.Dataset object
            Dataset to process
        ex_freq : float
            Frequency of the excitation waveform
        gain : uint
            Gain setting on current amplifier (typically 7-9)
        num_x_steps : uint (Optional, default = 250)
            Number of steps for the inferred results. Note: this may be end up being slightly different from specified.
        r_extra : float (Optional, default = 110 [Ohms])
            Extra resistance in the RC circuit that will provide correct current and resistance values
        h5_target_group : h5py.Group, optional. Default = None
            Location where to look for existing results and to place newly
            computed results. Use this kwarg if the results need to be written
            to a different HDF5 file. By default, this value is set to the
            parent group containing `h5_main`
        kwargs : dict
            Other parameters specific to the Process class and nuanced bayesian_inference parameters
        """
        super(GIVBayesian, self).__init__(h5_main, 'Bayesian_Inference',
                                          **kwargs)
        self.gain = gain
        self.ex_freq = ex_freq
        self.r_extra = r_extra
        self.num_x_steps = int(num_x_steps)
        if self.num_x_steps % 4 == 0:
            self.num_x_steps = ((self.num_x_steps // 2) + 1) * 2
        if self.verbose and self.mpi_rank == 0:
            print('ensuring that half steps should be odd, num_x_steps is now', self.num_x_steps)

        self.h5_main = USIDataset(self.h5_main)

        # take these from kwargs
        bayesian_parms = {'gam': 0.03, 'e': 10.0, 'sigma': 10.0, 'sigmaC': 1.0, 'num_samples': 2E3}

        self.parms_dict = {'freq': self.ex_freq, 'num_x_steps': self.num_x_steps, 'r_extra': self.r_extra}
        self.parms_dict.update(bayesian_parms)

        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        # Should not be extracting excitation this way!
        h5_spec_vals = self.h5_main.h5_spec_vals[0]
        self.single_ao = np.squeeze(h5_spec_vals[()])

        roll_cyc_fract = -0.25
        self.roll_pts = int(self.single_ao.size * roll_cyc_fract)
        self.rolled_bias = np.roll(self.single_ao, self.roll_pts)

        dt = 1 / (ex_freq * self.single_ao.size)
        self.dvdt = np.diff(self.single_ao) / dt
        self.dvdt = np.append(self.dvdt, self.dvdt[-1])

        self.reverse_results = None
        self.forward_results = None
        self._bayes_parms = None

        self.__first_batch = True

    def test(self, pix_ind=None, show_plots=True):
        """
        Tests the inference on a single pixel (randomly chosen unless manually specified) worth of data.

        Parameters
        ----------
        pix_ind : int, optional. default = random
            Index of the pixel whose data will be used for inference
        show_plots : bool, optional. default = True
            Whether or not to show plots

        Returns
        -------
        fig, axes
        """
        if self.mpi_rank > 0:
            return

        if pix_ind is None:
            pix_ind = np.random.randint(0, high=self.h5_main.shape[0])
        other_params = self.parms_dict.copy()
        # removing duplicates:
        _ = other_params.pop('freq')

        return bayesian_inference_on_period(self.h5_main[pix_ind], self.single_ao, self.parms_dict['freq'],
                                            show_plots=show_plots, **other_params)

    def _set_memory_and_cores(self, cores=None, man_mem_limit=None,
                              mem_multiplier=1.0):
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
        super(GIVBayesian, self)._set_memory_and_cores(cores=cores,
                                                       mem_multiplier=mem_multiplier,
                                                       man_mem_limit=man_mem_limit)
        # Remember that the default number of pixels corresponds to only the raw data that can be held in memory
        # In the case of simplified Bayesian inference, four (roughly) equally sized datasets need to be held in memory:
        # raw, compensated current, resistance, variance
        self._max_pos_per_read = self._max_pos_per_read // 4  # Integer division
        # Since these computations take far longer than functional fitting, do in smaller batches:
        self._max_pos_per_read = min(1000, self._max_pos_per_read)

        if self.verbose and self.mpi_rank == 0:
            print('Max positions per read set to {}'.format(self._max_pos_per_read))

    def _create_results_datasets(self):
        """
        Creates hdf5 datasets and datagroups to hold the resutls
        """
        # create all h5 datasets here:
        num_pos = self.h5_main.shape[0]

        if self.verbose and self.mpi_rank == 0:
            print('Now creating the datasets')

        self.h5_results_grp = create_results_group(self.h5_main,
                                                   self.process_name,
                                                   h5_parent_group=self._h5_target_group)

        write_simple_attrs(self.h5_results_grp, {'algorithm_author': 'Kody J. Law', 'last_pixel': 0})
        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        if self.verbose and self.mpi_rank == 0:
            print('created group: {} with attributes:'.format(self.h5_results_grp.name))
            print(get_attributes(self.h5_results_grp))

        # One of those rare instances when the result is exactly the same as the source
        self.h5_i_corrected = create_empty_dataset(self.h5_main, np.float32, 'Corrected_Current', h5_group=self.h5_results_grp)

        if self.verbose and self.mpi_rank == 0:
            print('Created I Corrected')
            # print_tree(self.h5_results_grp)

        # For some reason, we cannot specify chunks or compression!
        # The resistance dataset requires the creation of a new spectroscopic dimension
        self.h5_resistance = write_main_dataset(self.h5_results_grp, (num_pos, self.num_x_steps), 'Resistance', 'Resistance',
                                                'GOhms', None, Dimension('Bias', 'V', self.num_x_steps),
                                                dtype=np.float32, # chunks=(1, self.num_x_steps), #compression='gzip',
                                                h5_pos_inds=self.h5_main.h5_pos_inds,
                                                h5_pos_vals=self.h5_main.h5_pos_vals)

        if self.verbose and self.mpi_rank == 0:
            print('Created Resistance')
            # print_tree(self.h5_results_grp)

        assert isinstance(self.h5_resistance, USIDataset)  # only here for PyCharm
        self.h5_new_spec_vals = self.h5_resistance.h5_spec_vals

        # The variance is identical to the resistance dataset
        self.h5_variance = create_empty_dataset(self.h5_resistance, np.float32, 'R_variance')

        if self.verbose and self.mpi_rank == 0:
            print('Created Variance')
            # print_tree(self.h5_results_grp)

        # The capacitance dataset requires new spectroscopic dimensions as well
        self.h5_cap = write_main_dataset(self.h5_results_grp, (num_pos, 1), 'Capacitance', 'Capacitance', 'pF', None,
                                         Dimension('Direction', '', [1]),  h5_pos_inds=self.h5_main.h5_pos_inds,
                                         h5_pos_vals=self.h5_main.h5_pos_vals, dtype=cap_dtype, #compression='gzip',
                                         aux_spec_prefix='Cap_Spec_')

        if self.verbose and self.mpi_rank == 0:
            print('Created Capacitance')
            # print_tree(self.h5_results_grp)
            print('Done creating all results datasets!')

        if self.mpi_size > 1:
            self.mpi_comm.Barrier()
        self.h5_main.file.flush()

    def _get_existing_datasets(self):
        """
        Extracts references to the existing datasets that hold the results
        """
        self.h5_new_spec_vals = self.h5_results_grp['Spectroscopic_Values']
        self.h5_cap = self.h5_results_grp['Capacitance']
        self.h5_variance = self.h5_results_grp['R_variance']
        self.h5_resistance = self.h5_results_grp['Resistance']
        self.h5_i_corrected = self.h5_results_grp['Corrected_Current']

    def _write_results_chunk(self):
        """
        Writes data chunks back to the h5 file
        """

        if self.verbose:
            print('Rank {} - Started accumulating results for this chunk'.format(self.mpi_rank))

        num_pixels = len(self.forward_results)
        cap_mat = np.zeros((num_pixels, 2), dtype=np.float32)
        r_inf_mat = np.zeros((num_pixels, self.num_x_steps), dtype=np.float32)
        r_var_mat = np.zeros((num_pixels, self.num_x_steps), dtype=np.float32)
        i_cor_sin_mat = np.zeros((num_pixels, self.single_ao.size), dtype=np.float32)

        for pix_ind, i_meas, forw_results, rev_results in zip(range(num_pixels), self.data,
                                                              self.forward_results, self.reverse_results):
            full_results = dict()
            for item in ['cValue']:
                full_results[item] = np.hstack((forw_results[item], rev_results[item]))
                # print(item, full_results[item].shape)

            # Capacitance is always doubled - halve it now (locally):
            # full_results['cValue'] *= 0.5
            cap_val = np.mean(full_results['cValue']) * 0.5

            # Compensating the resistance..
            """
            omega = 2 * np.pi * self.ex_freq
            i_cap = cap_val * omega * self.rolled_bias
            """
            i_cap = cap_val * self.dvdt
            i_extra = self.r_extra * 2 * cap_val * self.single_ao
            i_corr_sine = i_meas - i_cap - i_extra

            # Equivalent to flipping the X:
            rev_results['x'] *= -1

            # Stacking the results - no flipping required for reverse:
            for item in ['x', 'mR', 'vR']:
                full_results[item] = np.hstack((forw_results[item], rev_results[item]))

            i_cor_sin_mat[pix_ind] = i_corr_sine
            cap_mat[pix_ind] = full_results['cValue'] * 1000  # convert from nF to pF
            r_inf_mat[pix_ind] = full_results['mR']
            r_var_mat[pix_ind] = full_results['vR']

        # Now write to h5 files:
        if self.verbose:
            print('Rank {} - Finished accumulating results. Writing results of chunk to h5'.format(self.mpi_rank))

        if self.__first_batch:
            self.h5_new_spec_vals[0, :] = full_results['x']  # Technically this needs to only be done once
            self.__first_batch = False

        # Get access to the private variable:
        pos_in_batch = self._get_pixels_in_current_batch()

        self.h5_cap[pos_in_batch, :] = np.atleast_2d(stack_real_to_compound(cap_mat, cap_dtype)).T
        self.h5_variance[pos_in_batch, :] = r_var_mat
        self.h5_resistance[pos_in_batch, :] = r_inf_mat
        self.h5_i_corrected[pos_in_batch, :] = i_cor_sin_mat

    def _unit_computation(self, *args, **kwargs):
        """
        Processing per chunk of the dataset

        Parameters
        ----------
        args : list
            Not used
        kwargs : dictionary
            Not used
        """
        half_v_steps = self.single_ao.size // 2

        # first roll the data
        rolled_raw_data = np.roll(self.data, self.roll_pts, axis=1)
        # Ensure that the bias has a positive slope. Multiply current by -1 accordingly
        if self.verbose:
            print('Rank {} beginning parallel compute for Forward'.format(self.mpi_rank))
        self.reverse_results = parallel_compute(rolled_raw_data[:, :half_v_steps] * -1, do_bayesian_inference,
                                                cores=self._cores,
                                                func_args=[self.rolled_bias[:half_v_steps] * -1, self.ex_freq],
                                                func_kwargs=self._bayes_parms, lengthy_computation=False,
                                                verbose=self.verbose)

        if self.verbose:
            print('Rank {} finished processing forward sections. Now working on reverse sections'.format(self.mpi_rank))

        self.forward_results = parallel_compute(rolled_raw_data[:, half_v_steps:], do_bayesian_inference,
                                                cores=self._cores,
                                                func_args=[self.rolled_bias[half_v_steps:], self.ex_freq],
                                                func_kwargs=self._bayes_parms, lengthy_computation=False,
                                                verbose=self.verbose)
        if self.verbose:
            print('Rank {} Finished processing reverse loops (and this chunk)'.format(self.mpi_rank))

    def compute(self, override=False, *args, **kwargs):
        """
        Creates placeholders for the results, applies the inference to the data, and writes the output to the file.
        Consider calling test() before this function to make sure that the parameters are appropriate.

        Parameters
        ----------
        override : bool, optional. default = False
            By default, compute will simply return duplicate results to avoid recomputing or resume computation on a
            group with partial results. Set to True to force fresh computation.
        args : list
            Not used
        kwargs : dictionary
            Not used

        Returns
        -------
        h5_results_grp : h5py.Datagroup object
            Datagroup containing all the results
        """

        # remove additional parm and halve the x points
        self._bayes_parms = self.parms_dict.copy()
        self._bayes_parms['num_x_steps'] = self.num_x_steps // 2
        self._bayes_parms['econ'] = True
        del(self._bayes_parms['freq'])

        return super(GIVBayesian, self).compute(override=override, *args, **kwargs)
