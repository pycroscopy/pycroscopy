# -*- coding: utf-8 -*-
"""
:class:`~pycroscopy.analysis.be_loop_fitter.BELoopFitter` that fits Simple
Harmonic Oscillator model data to a parametric model to describe hysteretic
switching in ferroelectric materials

Created on Thu Nov 20 11:48:53 2019

@author: Suhas Somnath, Chris R. Smith, Rama K. Vasudevan

"""

from __future__ import division, print_function, absolute_import, \
    unicode_literals
import joblib
import dask
import time
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sidpy.hdf.dtype_utils import stack_real_to_compound, \
    flatten_compound_to_real
from sidpy.hdf.hdf_utils import get_attr, write_simple_attrs
from sidpy.proc.comp_utils import get_MPI, recommend_cpu_cores
from pyUSID.io.hdf_utils import get_unit_values, get_sort_order, \
    reshape_to_n_dims, create_empty_dataset, create_results_group, \
    write_reduced_anc_dsets, write_main_dataset
from pyUSID.io.usi_data import USIDataset
from .utils.be_loop import projectLoop, fit_loop, generate_guess, \
    loop_fit_function, calc_switching_coef_vec, switching32
from ..processing.tree import ClusterTree
from .be_sho_fitter import sho32
from .fitter import Fitter

'''
Custom dtypes for the datasets created during fitting.
'''
loop_metrics32 = np.dtype({'names': ['Area', 'Centroid x', 'Centroid y',
                                     'Rotation Angle [rad]', 'Offset'],
                           'formats': [np.float32, np.float32, np.float32,
                                       np.float32, np.float32]})

crit32 = np.dtype({'names': ['AIC_loop', 'BIC_loop', 'AIC_line', 'BIC_line'],
                   'formats': [np.float32, np.float32, np.float32,
                               np.float32]})

__field_names = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'b_0', 'b_1', 'b_2', 'b_3',
                 'R2 Criterion']
loop_fit32 = np.dtype({'names': __field_names,
                       'formats': [np.float32 for name in __field_names]})


class BELoopFitter(Fitter):
    """
    A class that fits Simple Harmonic Oscillator model data to a 9-parameter
    model to describe hysteretic switching in
    ferroelectric materials

    Notes
    -----
    Quantitative mapping of switching behavior in piezoresponse force microscopy, Stephen Jesse, Ho Nyung Lee,
    and Sergei V. Kalinin, Review of Scientific Instruments 77, 073702 (2006); doi: http://dx.doi.org/10.1063/1.2214699

    """

    def __init__(self, h5_main, be_data_type, vs_mode, vs_cycle_frac,
                 **kwargs):
        """

        Parameters
        ----------
        h5_main : h5py.Dataset
            The dataset over which the analysis will be performed. This dataset
            should be linked to the spectroscopic indices and values, and position
            indices and values datasets.
        data_type : str
            Type of data. This is an attribute written to the HDF5 file at the
            root level by either the translator or the acquisition software.
            Accepted values are: 'BEPSData' and 'cKPFMData'
            Default - this function will attempt to extract this metadata from the
            HDF5 file
        vs_mode: str
            Type of measurement. Accepted values are:
             'AC modulation mode with time reversal' or 'DC modulation mode'
             This is an attribute embedded under the "Measurement" group with the
             following key: 'VS_mode'. Default - this function will attempt to
             extract this metadata from the HDF5 file
        vs_cycle_frac : str
            Fraction of the bi-polar triangle waveform for voltage spectroscopy
            used in this experiment
        h5_target_group : h5py.Group, optional. Default = None
            Location where to look for existing results and to place newly
            computed results. Use this kwarg if the results need to be written
            to a different HDF5 file. By default, this value is set to the
            parent group containing `h5_main`
        kwargs : passed onto pyUSID.Process
        """

        super(BELoopFitter, self).__init__(h5_main, "Loop_Fit",
                                           variables=None, **kwargs)

        # This will be reset h5_main to this value before guess / fit
        # Some simple way to guard against failure
        self.__h5_main_orig = USIDataset(h5_main)

        self.parms_dict = None

        self._check_validity(h5_main, be_data_type, vs_mode, vs_cycle_frac)

        # Instead of the variables kwarg to the Fitter. Do check here:
        if 'DC_Offset' in self.h5_main.spec_dim_labels:
            self._fit_dim_name = 'DC_Offset'
        elif 'write_bias' in self.h5_main.spec_dim_labels:
            self._fit_dim_name = 'write_bias'
        else:
            raise ValueError('Neither "DC_Offset", nor "write_bias" were '
                             'spectroscopic dimension in the provided dataset '
                             'which has dimensions: {}'
                             '.'.format(self.h5_main.spec_dim_labels))

        if 'FORC' in self.h5_main.spec_dim_labels:
            self._forc_dim_name = 'FORC'
        else:
            self._forc_dim_name = 'FORC_Cycle'

        # accounting for memory copies
        self._max_raw_pos_per_read = self._max_pos_per_read

        # Declaring attributes here for PEP8 cleanliness
        self.h5_projected_loops = None
        self.h5_loop_metrics = None
        self._met_spec_inds = None
        self._write_results_chunk = None

    @staticmethod
    def _check_validity(h5_main, data_type, vs_mode, vs_cycle_frac):
        """
        Checks whether or not the provided object can be analyzed by this class

        Parameters
        ----------
        h5_main : h5py.Dataset instance
            The dataset containing the SHO Fit (not necessarily the dataset
            directly resulting from SHO fit)
            over which the loop projection, guess, and fit will be performed.
        data_type : str
            Type of data. This is an attribute written to the HDF5 file at the
            root level by either the translator or the acquisition software.
            Accepted values are: 'BEPSData' and 'cKPFMData'
            Default - this function will attempt to extract this metadata from the
            HDF5 file
        vs_mode: str
            Type of measurement. Accepted values are:
             'AC modulation mode with time reversal' or 'DC modulation mode'
             This is an attribute embedded under the "Measurement" group with the
             following key: 'VS_mode'. Default - this function will attempt to
             extract this metadata from the HDF5 file
        vs_cycle_frac : str
            Fraction of the bi-polar triangle waveform for voltage spectroscopy
            used in this experiment
        """
        if h5_main.dtype != sho32:
            raise TypeError('Provided dataset is not a SHO results dataset.')

        if data_type == 'BEPSData':
            if vs_mode not in ['DC modulation mode', 'current mode']:
                raise ValueError('Provided dataset has a mode: "' + vs_mode +
                                 '" is not a "DC modulation" or "current mode"'
                                 ' BEPS dataset')
            elif vs_cycle_frac != 'full':
                raise ValueError('Provided dataset does not have full cycles')

        elif data_type == 'cKPFMData':
            if vs_mode != 'cKPFM':
                raise ValueError('Provided dataset has an unsupported VS_mode:'
                                 ' "' + vs_mode + '"')
        else:
            raise NotImplementedError('Loop fitting not supported for Band '
                                      'Excitation experiment type: {}'
                                      ''.format(data_type))

    def _create_projection_datasets(self):
        """
        Creates the Loop projection and metrics HDF5 dataset & results group
        """

        # Which row in the spec datasets is DC offset?
        _fit_spec_index = self.h5_main.spec_dim_labels.index(
            self._fit_dim_name)

        # TODO: Unkown usage of variable. Waste either way
        # self._fit_offset_index = 1 + _fit_spec_index

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(
            self.h5_main.h5_spec_inds[_fit_spec_index, :] == 0).flatten()
        tot_cycles = cycle_start_inds.size
        if self.verbose and self.mpi_rank == 0:
            print('Found {} cycles starting at indices: {}'.format(tot_cycles,
                                                                   cycle_start_inds))

        # Make the results group
        self.h5_results_grp = create_results_group(self.h5_main,
                                                   self.process_name,
                                                   h5_parent_group=self._h5_target_group)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        # If writing to a new HDF5 file:
        # Add back the data_type attribute - still being used in the visualizer
        if self.h5_results_grp.file != self.h5_main.file:
            write_simple_attrs(self.h5_results_grp.file,
                               {'data_type': get_attr(self.h5_main.file,
                                                      'data_type')})

        # Write datasets
        self.h5_projected_loops = create_empty_dataset(self.h5_main,
                                                       np.float32,
                                                       'Projected_Loops',
                                                       h5_group=self.h5_results_grp)

        h5_loop_met_spec_inds, h5_loop_met_spec_vals = write_reduced_anc_dsets(
            self.h5_results_grp, self.h5_main.h5_spec_inds,
            self.h5_main.h5_spec_vals, self._fit_dim_name,
            basename='Loop_Metrics', verbose=False)

        self.h5_loop_metrics = write_main_dataset(self.h5_results_grp,
                                                  (self.h5_main.shape[0], tot_cycles), 'Loop_Metrics',
                                                  'Metrics', 'compound', None,
                                                  None, dtype=loop_metrics32,
                                                  h5_pos_inds=self.h5_main.h5_pos_inds,
                                                  h5_pos_vals=self.h5_main.h5_pos_vals,
                                                  h5_spec_inds=h5_loop_met_spec_inds,
                                                  h5_spec_vals=h5_loop_met_spec_vals)

        # Copy region reference:
        # copy_region_refs(self.h5_main, self.h5_projected_loops)
        # copy_region_refs(self.h5_main, self.h5_loop_metrics)

        self.h5_main.file.flush()
        self._met_spec_inds = self.h5_loop_metrics.h5_spec_inds

        if self.verbose and self.mpi_rank == 0:
            print('Finished creating Guess dataset')

    def _create_guess_datasets(self):
        """
        Creates the HDF5 Guess dataset
        """
        self._create_projection_datasets()

        self._h5_guess = create_empty_dataset(self.h5_loop_metrics, loop_fit32,
                                             'Guess')

        self._h5_guess = USIDataset(self._h5_guess)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        self.h5_main.file.flush()

    def _create_fit_datasets(self):
        """
        Creates the HDF5 Fit dataset
        """

        if self._h5_guess is None:
            raise ValueError('Need to guess before fitting!')

        self._h5_fit = create_empty_dataset(self._h5_guess, loop_fit32, 'Fit')
        self._h5_fit = USIDataset(self._h5_fit)

        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        self.h5_main.file.flush()

    def _read_data_chunk(self):
        """
        Get the next chunk of SHO data (in the case of Guess) or Projected
        loops (in the case of Fit)

        Notes
        -----
        self.data contains data for N pixels.
        The challenge is that this may contain M FORC cycles
        Each FORC cycle needs its own V DC vector
        So, we can't blindly use the inherited unit_compute.
        Our variables now are Position, Vdc, FORC, all others

        We want M lists of [VDC x all other variables]

        The challenge is that VDC and FORC are inner dimensions -
        neither the fastest nor the slowest (guaranteed)
        """

        # The Process class should take care of all the basic reading
        super(BELoopFitter, self)._read_data_chunk()

        if self.data is None:
            # Nothing we can do at this point
            return

        if self.verbose and self.mpi_rank == 0:
            print('BELoopFitter got data chunk of shape {} from Fitter'
                  '.'.format(self.data.shape))

        spec_dim_order_s2f = get_sort_order(self.h5_main.h5_spec_inds)[::-1]

        self._dim_labels_s2f = list(['Positions']) + list(
            np.array(self.h5_main.spec_dim_labels)[spec_dim_order_s2f])

        self._num_forcs = int(
            any([targ in self.h5_main.spec_dim_labels for targ in
                 ['FORC', 'FORC_Cycle']]))

        order_to_s2f = [0] + list(1 + spec_dim_order_s2f)
        if self.verbose and self.mpi_rank == 0:
            print('Order for reshaping to S2F: {}'.format(order_to_s2f))

        if self.verbose and self.mpi_rank == 0:
            print(self._dim_labels_s2f, order_to_s2f)

        if self._num_forcs:
            forc_pos = self.h5_main.spec_dim_labels.index(self._forc_dim_name)
            self._num_forcs = self.h5_main.spec_dim_sizes[forc_pos]

        if self.verbose and self.mpi_rank == 0:
            print('Num FORCS: {}'.format(self._num_forcs))

        all_but_forc_rows = []
        for ind, dim_name in enumerate(self.h5_main.spec_dim_labels):
            if dim_name not in ['FORC', 'FORC_Cycle', 'FORC_repeat']:
                all_but_forc_rows.append(ind)

        if self.verbose and self.mpi_rank == 0:
            print('All but FORC rows: {}'.format(all_but_forc_rows))

        dc_mats = []

        forc_mats = []

        num_reps = 1 if self._num_forcs == 0 else self._num_forcs
        for forc_ind in range(num_reps):
            if self.verbose and self.mpi_rank == 0:
                print('\nWorking on FORC #{}'.format(forc_ind))

            if self._num_forcs:
                this_forc_spec_inds = \
                    np.where(self.h5_main.h5_spec_inds[forc_pos] == forc_ind)[
                        0]
            else:
                this_forc_spec_inds = np.ones(
                    shape=self.h5_main.h5_spec_inds.shape[1], dtype=np.bool)

            if self._num_forcs:
                this_forc_dc_vec = get_unit_values(
                    self.h5_main.h5_spec_inds[all_but_forc_rows][:,
                    this_forc_spec_inds],
                    self.h5_main.h5_spec_vals[all_but_forc_rows][:,
                    this_forc_spec_inds],
                    all_dim_names=list(np.array(self.h5_main.spec_dim_labels)[
                                           all_but_forc_rows]),
                    dim_names=self._fit_dim_name)
            else:
                this_forc_dc_vec = get_unit_values(self.h5_main.h5_spec_inds,
                                                   self.h5_main.h5_spec_vals,
                                                   dim_names=self._fit_dim_name)
            this_forc_dc_vec = this_forc_dc_vec[self._fit_dim_name]
            dc_mats.append(this_forc_dc_vec)

            this_forc_2d = self.data[:, this_forc_spec_inds]
            if self.verbose and self.mpi_rank == 0:
                print('2D slice shape for this FORC: {}'.format(this_forc_2d.shape))

            this_forc_nd, success = reshape_to_n_dims(this_forc_2d,
                                                      h5_pos=None,
                                                      h5_spec=self.h5_main.h5_spec_inds[
                                                              :,
                                                              this_forc_spec_inds])

            if success != True:
                raise ValueError('Unable to reshape data to N dimensions')

            if self.verbose and self.mpi_rank == 0:
                print(this_forc_nd.shape)

            this_forc_nd_s2f = this_forc_nd.transpose(
                order_to_s2f).squeeze()  # squeeze out FORC
            dim_names_s2f = self._dim_labels_s2f.copy()
            if self._num_forcs > 0:
                dim_names_s2f.remove(
                    self._forc_dim_name)
                # because it was never there in the first place.
            if self.verbose and self.mpi_rank == 0:
                print('Reordered to S2F: {}, {}'.format(this_forc_nd_s2f.shape,
                                                        dim_names_s2f))

            rest_dc_order = list(range(len(dim_names_s2f)))
            _dc_ind = dim_names_s2f.index(self._fit_dim_name)
            rest_dc_order.remove(_dc_ind)
            rest_dc_order = rest_dc_order + [_dc_ind]
            if self.verbose and self.mpi_rank == 0:
                print('Transpose for reordering to rest, DC: {}'
                      ''.format(rest_dc_order))

            rest_dc_nd = this_forc_nd_s2f.transpose(rest_dc_order)
            rest_dc_names = list(np.array(dim_names_s2f)[rest_dc_order])

            self._pre_flattening_shape = list(rest_dc_nd.shape)
            self._pre_flattening_dim_name_order = list(rest_dc_names)

            if self.verbose and self.mpi_rank == 0:
                print('After reodering: {}, {}'.format(rest_dc_nd.shape,
                                                       rest_dc_names))

            dc_rest_2d = rest_dc_nd.reshape(np.prod(rest_dc_nd.shape[:-1]),
                                            np.prod(rest_dc_nd.shape[-1]))

            if self.verbose and self.mpi_rank == 0:
                print('Shape after flattening to 2D: {}'
                      ''.format(dc_rest_2d.shape))

            forc_mats.append(dc_rest_2d)

        self.data = forc_mats, dc_mats

        if self.verbose and self.mpi_rank == 0:
            print('self.data loaded')

    def _read_guess_chunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset.

        Notes
        -----
        Use the same strategy as that used for reading the raw data.
        The technique is slightly simplified since the end result
        per FORC cycle is just a 1D array of loop metrics.
        However, this compound dataset needs to be converted to float
        in order to send to scipy.optimize.least_squares
        """
        # The Fitter class should take care of all the basic reading
        super(BELoopFitter, self)._read_guess_chunk()

        if self.verbose and self.mpi_rank == 0:
            print('_read_guess_chunk got guess of shape: {} from super'
                  '.'.format(self._guess.shape))

        spec_dim_order_s2f = get_sort_order(self._h5_guess.h5_spec_inds)[::-1]

        order_to_s2f = [0] + list(1 + spec_dim_order_s2f)
        if self.verbose and self.mpi_rank == 0:
            print('Order for reshaping to S2F: {}'.format(order_to_s2f))

        dim_labels_s2f = list(['Positions']) + list(
            np.array(self._h5_guess.spec_dim_labels)[spec_dim_order_s2f])

        if self.verbose and self.mpi_rank == 0:
            print(dim_labels_s2f, order_to_s2f)

        num_forcs = int(any([targ in self._h5_guess.spec_dim_labels for targ in
                             ['FORC', 'FORC_Cycle']]))
        if num_forcs:
            forc_pos = self._h5_guess.spec_dim_labels.index(self._forc_dim_name)
            num_forcs = self._h5_guess.spec_dim_sizes[forc_pos]

        if self.verbose and self.mpi_rank == 0:
            print('Num FORCS: {}'.format(num_forcs))

        all_but_forc_rows = []
        for ind, dim_name in enumerate(self._h5_guess.spec_dim_labels):
            if dim_name not in ['FORC', 'FORC_Cycle', 'FORC_repeat']:
                all_but_forc_rows.append(ind)

        if self.verbose and self.mpi_rank == 0:
            print('All but FORC rows: {}'.format(all_but_forc_rows))

        forc_mats = []

        num_reps = 1 if num_forcs == 0 else num_forcs
        for forc_ind in range(num_reps):
            if self.verbose and self.mpi_rank == 0:
                print('\nWorking on FORC #{}'.format(forc_ind))
            if num_forcs:
                this_forc_spec_inds = \
                np.where(self._h5_guess.h5_spec_inds[forc_pos] == forc_ind)[0]
            else:
                this_forc_spec_inds = np.ones(
                    shape=self._h5_guess.h5_spec_inds.shape[1], dtype=np.bool)

            this_forc_2d = self._guess[:, this_forc_spec_inds]
            if self.verbose and self.mpi_rank == 0:
                print('2D slice shape for this FORC: {}'.format(this_forc_2d.shape))

            this_forc_nd, success = reshape_to_n_dims(this_forc_2d,
                                                      h5_pos=None,
                                                      h5_spec=self._h5_guess.h5_spec_inds[
                                                              :,
                                                              this_forc_spec_inds])

            if success != True:
                raise ValueError('Unable to reshape 2D guess to N dimensions')

            if self.verbose and self.mpi_rank == 0:
                print('N dimensional shape for this FORC: {}'.format(this_forc_nd.shape))

            this_forc_nd_s2f = this_forc_nd.transpose(
                order_to_s2f).squeeze()  # squeeze out FORC
            dim_names_s2f = dim_labels_s2f.copy()
            if num_forcs > 0:
                dim_names_s2f.remove(self._forc_dim_name)
                # because it was never there in the first place.
            if self.verbose and self.mpi_rank == 0:
                print('Reordered to S2F: {}, {}'.format(this_forc_nd_s2f.shape,
                                                        dim_names_s2f))

            dc_rest_2d = this_forc_nd_s2f.ravel()
            if self.verbose and self.mpi_rank == 0:
                print('Shape after raveling: {}'.format(dc_rest_2d.shape))

            # Scipy will not understand compound values. Flatten.
            # Ignore the R2 error
            # TODO: avoid memory copies!
            float_mat = np.zeros(shape=list(dc_rest_2d.shape) +
                                       [len(loop_fit32.names)-1],
                                 dtype=np.float32)
            if self.verbose and self.mpi_rank == 0:
                print('Created empty float matrix of shape: {}'
                      '.'.format(float_mat.shape))
            for ind, field_name in enumerate(loop_fit32.names[:-1]):
                float_mat[..., ind] = dc_rest_2d[field_name]

            if self.verbose and self.mpi_rank == 0:
                print('Shape after flattening to float: {}'
                      '.'.format(float_mat.shape))

            forc_mats.append(float_mat)

        self._guess = np.array(forc_mats)
        if self.verbose and self.mpi_rank == 0:
            print('Flattened Guesses to shape: {} and dtype:'
                  '.'.format(self._guess.shape, self._guess.dtype))

    @staticmethod
    def _project_loop(sho_response, dc_offset):
        """
        Projects a provided piezoelectric hysteresis loop

        Parameters
        ----------
        sho_response : numpy.ndarray
            Compound valued array with the SHO response for a single loop
        dc_offset : numpy.ndarray
            DC offset corresponding to the provided loop

        Returns
        -------
        projected_loop : numpy.ndarray
            Projected loop
        ancillary : numpy.ndarray
            Metrics for the loop projection
        """
        # projected_loop = np.zeros(shape=sho_response.shape, dtype=np.float32)
        ancillary = np.zeros(shape=1, dtype=loop_metrics32)

        pix_dict = projectLoop(np.squeeze(dc_offset),
                               sho_response['Amplitude [V]'],
                               sho_response['Phase [rad]'])

        projected_loop = pix_dict['Projected Loop']
        ancillary['Rotation Angle [rad]'] = pix_dict['Rotation Matrix'][0]
        ancillary['Offset'] = pix_dict['Rotation Matrix'][1]
        ancillary['Area'] = pix_dict['Geometric Area']
        ancillary['Centroid x'] = pix_dict['Centroid'][0]
        ancillary['Centroid y'] = pix_dict['Centroid'][1]

        return projected_loop, ancillary

    @staticmethod
    def __compute_batches(data_mat_list, ref_vec_list, map_func, req_cores,
                          verbose=False):
        """
        Maps the provided function onto the sets of data and their
        corresponding reference vector. This function is almost identical and
        is based on pyUSID.processing.comp_utils.parallel_compute. Except,
        this function allows the data and reference vectors to be specified
        as a list of arrays as opposed to limiting to a single reference vector
        as in the case of parallel_compute()

        Parameters
        ----------
        data_mat_list : list
            List of numpy.ndarray objects
        ref_vec_list : list
            List of numpy.ndarray objects
        map_func : callable
            Function that the data matrices will be mapped to
        req_cores : uint
            Number of CPU cores to use for the computation
        verbose : bool, optional. Default = False
            Whether or not to print logs for debugging

        Returns
        -------
        list
            List of values returned by map_func when applied to the provided
            data
        """
        MPI = get_MPI()
        if MPI is not None:
            rank = MPI.COMM_WORLD.Get_rank()
            cores = 1
        else:
            rank = 0
            cores = req_cores

        if verbose:
            print(
                'Rank {} starting computation on {} cores (requested {} '
                'cores)'.format(rank, cores, req_cores))

        if cores > 1:
            values = []
            for loops_2d, curr_vdc in zip(data_mat_list, ref_vec_list):
                values += [joblib.delayed(map_func)(x, [curr_vdc])
                           for x
                           in loops_2d]
            results = joblib.Parallel(n_jobs=cores)(values)

            # Finished reading the entire data set
            if verbose:
                print('Rank {} finished parallel computation'.format(rank))

        else:
            if verbose:
                print("Rank {} computing serially ...".format(rank))
            # List comprehension vs map vs for loop?
            # https://stackoverflow.com/questions/1247486/python-list-comprehension-vs-map
            results = []
            for loops_2d, curr_vdc in zip(data_mat_list, ref_vec_list):
                results += [map_func(vector, curr_vdc) for vector in
                            loops_2d]

            if verbose:
                print('Rank {} finished serial computation'.format(rank))

        return results

    def _unit_compute_guess(self):
        """
        Performs loop projection followed by clustering-based guess for
        the self.data loaded into memory.

        In the end self._results is a tuple containing the projected loops,
        loop metrics, and the guess parameters ready to be written to HDF5
        """
        if self.verbose and self.mpi_rank == 0:
            print("Rank {} at _unit_compute_guess".format(self.mpi_rank))

        resp_2d_list, dc_vec_list = self.data

        if self.verbose and self.mpi_rank == 0:
            print('Unit computation found {} FORC datasets with {} '
                  'corresponding DC vectors'.format(len(resp_2d_list),
                                                    len(dc_vec_list)))
            print('First dataset of shape: {}'.format(resp_2d_list[0].shape))

        results = self.__compute_batches(resp_2d_list, dc_vec_list,
                                         self._project_loop, self._cores,
                                         verbose=self.verbose)

        # Step 1: unzip the two components in results into separate arrays
        if self.verbose and self.mpi_rank == 0:
            print('Unzipping loop projection results')
        loop_mets = np.zeros(shape=len(results), dtype=loop_metrics32)
        proj_loops = np.zeros(shape=(len(results), self.data[0][0].shape[1]),
                              dtype=np.float32)

        if self.verbose and self.mpi_rank == 0:
            print(
                'Prepared empty arrays for loop metrics of shape: {} and '
                'projected loops of shape: {}.'
                ''.format(loop_mets.shape, proj_loops.shape))

        for ind in range(len(results)):
            proj_loops[ind] = results[ind][0]
            loop_mets[ind] = results[ind][1]

        # NOW do the guess:
        proj_forc = proj_loops.reshape((len(dc_vec_list),
                                        len(results) // len(dc_vec_list),
                                        proj_loops.shape[-1]))

        if self.verbose and self.mpi_rank == 0:
            print('Reshaped projected loops from {} to: {}'.format(
                proj_loops.shape, proj_forc.shape))

        # Convert forc dimension to a list
        if self.verbose and self.mpi_rank == 0:
            print('Going to compute guesses now')

        all_guesses = []

        for proj_loops_this_forc, curr_vdc in zip(proj_forc, dc_vec_list):
            # this works on batches and not individual loops
            # Cannot be done in parallel
            this_guesses = guess_loops_hierarchically(curr_vdc,
                                                      proj_loops_this_forc)
            all_guesses.append(this_guesses)

        self._results = proj_loops, loop_mets, np.array(all_guesses)

    def set_up_guess(self, h5_partial_guess=None):
        """
        Performs necessary book-keeping before do_guess can be called.
        Also remaps data reading, computation, writing functions to those
        specific to Guess

        Parameters
        ----------
        h5_partial_guess: h5py.Dataset or pyUSID.io.USIDataset, optional
            HDF5 dataset containing partial Guess. Not implemented
        """
        self.h5_main = self.__h5_main_orig
        self.parms_dict = {'projection_method': 'pycroscopy BE loop model',
                           'guess_method': "pycroscopy Cluster Tree"}

        # ask super to take care of the rest, which is a standardized operation
        super(BELoopFitter, self).set_up_guess(h5_partial_guess=h5_partial_guess)

        self._max_pos_per_read = self._max_raw_pos_per_read // 1.5

        self._unit_computation = self._unit_compute_guess
        self.compute = self.do_guess
        self._write_results_chunk = self._write_guess_chunk

    def set_up_fit(self, h5_partial_fit=None, h5_guess=None, ):
        """
        Performs necessary book-keeping before do_fit can be called.
        Also remaps data reading, computation, writing functions to those
        specific to Fit

        Parameters
        ----------
        h5_partial_fit: h5py.Dataset or pyUSID.io.USIDataset, optional
            HDF5 dataset containing partial Fit. Not implemented
        h5_guess: h5py.Dataset or pyUSID.io.USIDataset, optional
            HDF5 dataset containing completed Guess. Not implemented
        """
        self.h5_main = self.__h5_main_orig
        self.parms_dict = {'fit_method': 'pycroscopy functional'}

        # ask super to take care of the rest, which is a standardized operation
        super(BELoopFitter, self).set_up_fit(h5_partial_fit=h5_partial_fit,
                                             h5_guess=h5_guess)

        self._max_pos_per_read = self._max_raw_pos_per_read // 1.5

        self._unit_computation = self._unit_compute_fit
        self.compute = self.do_fit
        self._write_results_chunk = self._write_fit_chunk

    def _get_existing_datasets(self):
        """
        The purpose of this function is to allow processes to resume from partly computed results
        Start with self.h5_results_grp
        """
        super(BELoopFitter, self)._get_existing_datasets()
        self.h5_projected_loops = self.h5_results_grp['Projected_Loops']
        self.h5_loop_metrics = self.h5_results_grp['Loop_Metrics']
        try:
            _ = self.h5_results_grp['Guess_Loop_Parameters']
        except KeyError:
            _ = self.extract_loop_parameters(self._h5_guess)
        try:
            # This has already been done by super
            _ = self.h5_results_grp['Fit']
            try:
                _ = self.h5_results_grp['Fit_Loop_Parameters']
            except KeyError:
                _ = self.extract_loop_parameters(self._h5_fit)
        except KeyError:
            pass

    def do_fit(self, override=False,):
        """
        Computes the Fit

        Parameters
        ----------
        override : bool, optional
            If True, computes a fresh guess even if existing Fit was found
            Else, returns existing Fit dataset. Default = False

        Returns
        -------
        USIDataset
            HDF5 dataset with the Fit computed
        """

        """
        This is REALLY ugly but needs to be done because projection, guess,
        and fit work in such a unique manner. At the same time, this complexity
        needs to be invisible to the end-user
        """
        # Manually setting this variable because this is not set if resuming
        # an older computation
        self.h5_projected_loops = USIDataset(self.h5_results_grp['Projected_Loops'])

        # raw data is actually projected loops not raw SHO data
        self.h5_main = self.h5_projected_loops

        # TODO: h5_main swap is not resilient against failure of do_fit()
        temp = super(BELoopFitter, self).do_fit(override=override)

        # Reset h5_main so that this swap is invisible to the user
        self.h5_main = self.__h5_main_orig

        # Extract material properties from loop coefficients
        _ = self.extract_loop_parameters(temp)

        return temp

    def _unit_compute_fit(self):
        """
        Performs least-squares fitting on self.data using self.guess for
        initial conditions.
        Results of the computation are captured in self._results
        """

        obj_func = _be_loop_err
        opt_func = least_squares
        solver_options = {'jac': 'cs'}

        resp_2d_list, dc_vec_list = self.data

        # At this point data has been read in. Read in the guess as well:
        self._read_guess_chunk()

        if self.mpi_size == 1:
            if self.verbose:
                print('Using Dask for parallel computation')
            opt_func = dask.delayed(opt_func)
        else:
            if self.verbose:
                print('Rank {} using serial computation'.format(self.mpi_rank))

        t0 = time.time()

        self._results = list()
        for dc_vec, loops_2d, guess_parms in zip(dc_vec_list, resp_2d_list,
                                                 self._guess):
            '''
            Shift the loops and vdc vector
            '''
            shift_ind, vdc_shifted = shift_vdc(dc_vec)
            loops_2d_shifted = np.roll(loops_2d, shift_ind, axis=1)

            if self.verbose and self.mpi_rank == 0:
                print('Computing on set: DC: {}<{}>, loops: {}<{}>, Guess: {}<{}>'.format(
                        vdc_shifted.shape, vdc_shifted.dtype,
                        loops_2d_shifted.shape, loops_2d_shifted.dtype,
                        guess_parms.shape, guess_parms.dtype))

            for loop_resp, loop_guess in zip(loops_2d_shifted, guess_parms):
                curr_results = opt_func(obj_func, loop_guess,
                                             args=[loop_resp, vdc_shifted],
                                             **solver_options)
                self._results.append(curr_results)

        t1 = time.time()

        if self.mpi_size == 1:
            if self.verbose and self.mpi_rank == 0:
                print('Now computing delayed tasks:')

            self._results = dask.compute(self._results, scheduler='processes')[0]

            t2 = time.time()

            if self.verbose and self.mpi_rank == 0:
                print('Dask Setup time: {} sec. Compute time: {} sec'.format(t1- t0, t2 - t1))
        else:
            if self.verbose:
                print('Rank {}: Serial compute time: {} sec'.format(self.mpi_rank, t1 - t0))

    @staticmethod
    def extract_loop_parameters(h5_loop_fit, nuc_threshold=0.03):
        """
        Method to extract a set of physical loop parameters from a dataset of fit parameters
        Parameters
        ----------
        h5_loop_fit : h5py.Dataset
            Dataset of loop fit parameters
        nuc_threshold : float
            Nucleation threshold to use in calculation physical parameters
        Returns
        -------
        h5_loop_parm : h5py.Dataset
            Dataset of physical parameters
        """
        dset_name = h5_loop_fit.name.split('/')[-1] + '_Loop_Parameters'
        h5_loop_parameters = create_empty_dataset(h5_loop_fit,
                                                  dtype=switching32,
                                                  dset_name=dset_name,
                                                  new_attrs={
                                                      'nuc_threshold': nuc_threshold})

        loop_coef_vec = flatten_compound_to_real(
            np.reshape(h5_loop_fit, [-1, 1]))
        switching_coef_vec = calc_switching_coef_vec(loop_coef_vec,
                                                     nuc_threshold)

        h5_loop_parameters[:, :] = switching_coef_vec.reshape(
            h5_loop_fit.shape)

        h5_loop_fit.file.flush()

        return h5_loop_parameters

    def _unit_compute_fit_jl_broken(self):
        """
        JobLib version of the unit computation function that unforunately
        did not work.
        """

        # 1 - r_squared = _sho_error(guess, data_vec, freq_vector)

        obj_func = _be_loop_err
        solver_options = {'jac': 'cs', 'max_nfev': 2}

        resp_2d_list, dc_vec_list = self.data

        # At this point data has been read in. Read in the guess as well:
        self._read_guess_chunk()

        if self.verbose and self.mpi_rank == 0:
            print('_unit_compute_fit got:\nobj_func: {}\n'
                  'solver_options: {}'.format(obj_func, solver_options))

        # TODO: Generalize this bit. Use Parallel compute instead!
        if self.mpi_size > 1:
            if self.verbose:
                print('Rank {}: About to start serial computation'
                      '.'.format(self.mpi_rank))

            self._results = list()
            for dc_vec, loops_2d, guess_parms in zip(dc_vec_list, resp_2d_list, self._guess):
                if self.verbose:
                    print('Setting up delayed joblib based on DC: {}<{}>, loops: {}<{}>, Guess: {}<{}>'.format(dc_vec.shape, dc_vec.dtype, loops_2d.shape, loops_2d.dtype, guess_parms.shape, guess_parms.dtype))

                '''
                Shift the loops and vdc vector
                '''
                shift_ind, vdc_shifted = shift_vdc(dc_vec)
                loops_2d_shifted = np.roll(loops_2d, shift_ind, axis=1)

                if self.verbose:
                    print('Setting up delayed joblib based on DC: {}<{}>, loops: {}<{}>, Guess: {}<{}>'.format(vdc_shifted.shape, vdc_shifted.dtype, loops_2d_shifted.shape, loops_2d_shifted.dtype, guess_parms.shape, guess_parms.dtype))

                for loop_resp, loop_guess in zip(loops_2d_shifted, guess_parms):
                    curr_results = least_squares(obj_func, loop_guess,
                                                 args=[loop_resp, vdc_shifted],
                                                 **solver_options)
                    self._results.append(curr_results)
        else:
            cores = recommend_cpu_cores(len(resp_2d_list) * resp_2d_list[0].shape[0],
                                        verbose=self.verbose)
            if self.verbose:
                print('Starting parallel fitting with {} cores'.format(cores))

            values = list()
            for dc_vec, loops_2d, guess_parms in zip(dc_vec_list, resp_2d_list, self._guess):
                if self.verbose:
                    print('Setting up delayed joblib based on DC: {} loops: {}, Guess: {}'.format(dc_vec.shape, loops_2d.shape, guess_parms.shape))
                '''
                Shift the loops and vdc vector
                '''
                shift_ind, vdc_shifted = shift_vdc(dc_vec)
                loops_2d_shifted = np.roll(loops_2d, shift_ind, axis=1)

                if self.verbose:
                    print('Setting up delayed joblib based on DC: {}<{}>, loops: {}<{}>, Guess: {}<{}>'.format(vdc_shifted.shape, vdc_shifted.dtype, loops_2d_shifted.shape, loops_2d_shifted.dtype, guess_parms.shape, guess_parms.dtype))

                temp = [joblib.delayed(least_squares)(obj_func, loop_guess, args=[loop_resp, vdc_shifted], **solver_options) for loop_resp, loop_guess in zip(loops_2d_shifted, guess_parms)]


                values.append(temp)
            if self.verbose:
                print('Finished setting up delayed computations. Starting parallel compute')
                print(temp[0])
                from pickle import dumps
                for item in temp[0]:
                    print(dumps(item))
            self._results = joblib.Parallel(n_jobs=cores)(values)

        if self.verbose and self.mpi_rank == 0:
            print(
                'Finished computing fits on {} objects'
                ''.format(len(self._results)))

        # What least_squares returns is an object that needs to be extracted
        # to get the coefficients. This is handled by the write function

    @staticmethod
    def _reformat_results_chunk(num_forcs, raw_results, first_n_dim_shape,
                                first_n_dim_names, dim_labels_s2f,
                                forc_dim_name, verbose=False):
        """
        Reshapes the provided flattened 2D results back to correct 2D form
        that can be written back to the HDF5 dataset via a few reshape and
        transpose operations

        Parameters
        ----------
        num_forcs : uint
            Number of FORC cycles in this data chunk / HDF5 dataset
        raw_results : numpy.ndarray
            Numpy array of the results (projected loops, guess, fit,
            loop metrics, etc.
        first_n_dim_shape : list
            Shape of the N-dimensional raw data chunk before it was flattened
            to the 2D or 1D (guess) shape
        first_n_dim_names : list
            Corresponding names of the dimensions for first_n_dim_shape
        dim_labels_s2f : list
            Names of the dimensions arranged from slowest to fastest
        forc_dim_name : str
            Name of the FORC dimension if present.
        verbose : bool, optional. Default = False
            Whether or not to print logs for debugging

        Returns
        -------
        results_2d : numpy.ndarray
            2D array that is ready to be written to the HDF5 file

        Notes
        -----
        Step 1 will fold back the flattened 1 / 2D array into the N-dim form
        Step 2 will reverse all transposes
        Step 3 will flatten back to its original 2D form
        """

        # What we need to do is put the forc back as the slowest dimension before the pre_flattening shape:
        if num_forcs > 1:
            first_n_dim_shape = [num_forcs] + first_n_dim_shape
            first_n_dim_names = [forc_dim_name] + first_n_dim_names
        if verbose:
            print('Dimension sizes & order: {} and names: {} that flattened '
                  'results will be reshaped to'
                  '.'.format(first_n_dim_shape, first_n_dim_names))

        # Now, reshape the flattened 2D results to its N-dim form before flattening (now FORC included):
        first_n_dim_results = raw_results.reshape(first_n_dim_shape)

        # Need to put data back to slowest >> fastest dim
        map_to_s2f = [first_n_dim_names.index(dim_name) for dim_name in
                      dim_labels_s2f]
        if verbose:
            print('Will permute as: {} to arrange dimensions from slowest to '
                  'fastest varying'.format(map_to_s2f))

        results_nd_s2f = first_n_dim_results.transpose(map_to_s2f)

        if verbose:
            print('Shape: {} and dimension labels: {} of results arranged from'
                  ' slowest to fastest varying'
                  '.'.format(results_nd_s2f.shape, dim_labels_s2f))

        pos_size = np.prod(results_nd_s2f.shape[:1])
        spec_size = np.prod(results_nd_s2f.shape[1:])

        if verbose:
            print('Results will be flattend to: {}'
                  '.'.format((pos_size, spec_size)))

        results_2d = results_nd_s2f.reshape(pos_size, spec_size)

        return results_2d

    def _write_guess_chunk(self):
        """
        Writes the results present in self._results to appropriate HDF5
        results datasets after appropriate manipulations
        """
        proj_loops, loop_mets, all_guesses = self._results

        if self.verbose:
            print('Unzipped results into Projected loops and Metrics arrays')

        # Step 2: Fold to N-D before reversing transposes:
        loops_2d = self._reformat_results_chunk(self._num_forcs, proj_loops,
                                                self._pre_flattening_shape,
                                                self._pre_flattening_dim_name_order,
                                                self._dim_labels_s2f,
                                                self._forc_dim_name,
                                                verbose=self.verbose)

        met_labels_s2f = self._dim_labels_s2f.copy()
        met_labels_s2f.remove(self._fit_dim_name)

        mets_2d = self._reformat_results_chunk(self._num_forcs, loop_mets,
                                               self._pre_flattening_shape[:-1],
                                               self._pre_flattening_dim_name_order[:-1],
                                               met_labels_s2f,
                                               self._forc_dim_name,
                                               verbose=self.verbose)

        guess_2d = self._reformat_results_chunk(self._num_forcs, all_guesses,
                                               self._pre_flattening_shape[:-1],
                                               self._pre_flattening_dim_name_order[:-1],
                                               met_labels_s2f,
                                               self._forc_dim_name,
                                               verbose=self.verbose)

        # Which pixels are we working on?
        curr_pixels = self._get_pixels_in_current_batch()

        if self.verbose and self.mpi_rank == 0:
            print(
                'Writing projected loops of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    loops_2d.shape, loops_2d.dtype,
                    self.h5_projected_loops.shape,
                    self.h5_projected_loops.dtype))
            print(
                'Writing loop metrics of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    mets_2d.shape, mets_2d.dtype, self.h5_loop_metrics.shape,
                    self.h5_loop_metrics.dtype))

            print(
                'Writing Guesses of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    guess_2d.shape, guess_2d.dtype, self._h5_guess.shape,
                    self._h5_guess.dtype))

        self.h5_projected_loops[curr_pixels, :] = loops_2d
        self.h5_loop_metrics[curr_pixels, :] = mets_2d
        self._h5_guess[curr_pixels, :] = guess_2d

        self._h5_guess.file.flush()

    def _write_fit_chunk(self):
        """
        Writes the results present in self._results to appropriate HDF5
        results datasets after appropriate manipulations
        """
        # TODO: To compound dataset: Note that this is a memory duplication!
        temp = np.array(
            [np.hstack([result.x, result.fun]) for result in self._results])
        self._results = stack_real_to_compound(temp, loop_fit32)

        all_fits = np.array(self._results)

        if self.verbose and self.mpi_rank == 0:
            print('Results of shape: {} and dtype: {}'.format(all_fits.shape, all_fits.dtype))

        met_labels_s2f = self._dim_labels_s2f.copy()
        met_labels_s2f.remove(self._fit_dim_name)

        fits_2d = self._reformat_results_chunk(self._num_forcs, all_fits,
                                               self._pre_flattening_shape[:-1],
                                               self._pre_flattening_dim_name_order[:-1],
                                               met_labels_s2f,
                                               self._forc_dim_name,
                                               verbose=self.verbose)

        # Which pixels are we working on?
        curr_pixels = self._get_pixels_in_current_batch()

        if self.verbose and self.mpi_rank == 0:
            print(
                'Writing Fits of shape: {} and data type: {} to a dataset of shape: {} and data type {}'.format(
                    fits_2d.shape, fits_2d.dtype, self._h5_fit.shape,
                    self._h5_fit.dtype))

        self._h5_fit[curr_pixels, :] = fits_2d

        self._h5_fit.file.flush()


def _be_loop_err(coef_vec, data_vec, dc_vec, *args):
    """

    Parameters
    ----------
    coef_vec : numpy.ndarray
    data_vec : numpy.ndarray
    dc_vec : numpy.ndarray
        The DC offset vector
    args : list

    Returns
    -------
    fitness : float
        The 1-r^2 value for the current set of loop coefficients

    """
    if coef_vec.size < 9:
        raise ValueError(
            'Error: The Loop Fit requires 9 parameter guesses!')

    data_mean = np.mean(data_vec)

    func = loop_fit_function(dc_vec, coef_vec)

    ss_tot = sum(abs(data_vec - data_mean) ** 2)
    ss_res = sum(abs(data_vec - func) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return 1 - r_squared


def guess_loops_hierarchically(vdc_vec, projected_loops_2d):
    """
    Provides loop parameter guesses for a given set of loops

    Parameters
    ----------
    vdc_vec : 1D numpy float numpy array
        DC voltage offsets for the loops
    projected_loops_2d : 2D numpy float array
        Projected loops arranged as [instance or position x dc voltage steps]

    Returns
    -------
    guess_parms : 1D compound numpy array
        Loop parameter guesses for the provided projected loops

    """

    def _loop_fit_tree(tree, guess_mat, fit_results, vdc_shifted,
                       shift_ind):
        """
        Recursive function that fits a tree object describing the cluster results

        Parameters
        ----------
        tree : ClusterTree object
            Tree describing the clustering results
        guess_mat : 1D numpy float array
            Loop parameters that serve as guesses for the loops in the tree
        fit_results : 1D numpy float array
            Loop parameters that serve as fits for the loops in the tree
        vdc_shifted : 1D numpy float array
            DC voltages shifted be 1/4 cycle
        shift_ind : unsigned int
            Number of units to shift loops by

        Returns
        -------
        guess_mat : 1D numpy float array
            Loop parameters that serve as guesses for the loops in the tree
        fit_results : 1D numpy float array
            Loop parameters that serve as fits for the loops in the tree

        """
        # print('Now fitting cluster #{}'.format(tree.name))
        # I already have a guess. Now fit myself
        curr_fit_results = fit_loop(vdc_shifted,
                                    np.roll(tree.value, shift_ind),
                                    guess_mat[tree.name])
        # keep all the fit results
        fit_results[tree.name] = curr_fit_results
        for child in tree.children:
            # Use my fit as a guess for the lower layers:
            guess_mat[child.name] = curr_fit_results[0].x
            # Fit this child:
            guess_mat, fit_mat = _loop_fit_tree(child, guess_mat,
                                                fit_results, vdc_shifted,
                                                shift_ind)
        return guess_mat, fit_results

    num_clusters = max(2, int(projected_loops_2d.shape[
                                  0] ** 0.5))  # change this to 0.6 if necessary
    estimators = KMeans(num_clusters)
    results = estimators.fit(projected_loops_2d)
    centroids = results.cluster_centers_
    labels = results.labels_

    # Get the distance between cluster means
    distance_mat = pdist(centroids)
    # get hierarchical pairings of clusters
    linkage_pairing = linkage(distance_mat, 'weighted')
    # Normalize the pairwise distance with the maximum distance
    linkage_pairing[:, 2] = linkage_pairing[:, 2] / max(
        linkage_pairing[:, 2])

    # Now use the tree class:
    cluster_tree = ClusterTree(linkage_pairing[:, :2], labels,
                               distances=linkage_pairing[:, 2],
                               centroids=centroids)
    num_nodes = len(cluster_tree.nodes)

    # prepare the guess and fit matrices
    loop_guess_mat = np.zeros(shape=(num_nodes, 9), dtype=np.float32)
    # loop_fit_mat = np.zeros(shape=loop_guess_mat.shape, dtype=loop_guess_mat.dtype)
    loop_fit_results = list(
        np.arange(num_nodes, dtype=np.uint16))  # temporary placeholder

    shift_ind, vdc_shifted = shift_vdc(vdc_vec)

    # guess the top (or last) node
    loop_guess_mat[-1] = generate_guess(vdc_vec, cluster_tree.tree.value)

    # Now guess the rest of the tree
    loop_guess_mat, loop_fit_results = _loop_fit_tree(cluster_tree.tree,
                                                      loop_guess_mat,
                                                      loop_fit_results,
                                                      vdc_shifted,
                                                      shift_ind)

    # Prepare guesses for each pixel using the fit of the cluster it belongs to:
    guess_parms = np.zeros(shape=projected_loops_2d.shape[0],
                           dtype=loop_fit32)
    for clust_id in range(num_clusters):
        pix_inds = np.where(labels == clust_id)[0]
        temp = np.atleast_2d(loop_fit_results[clust_id][0].x)
        # convert to the appropriate dtype as well:
        r2 = 1 - np.sum(np.abs(loop_fit_results[clust_id][0].fun ** 2))
        guess_parms[pix_inds] = stack_real_to_compound(
            np.hstack([temp, np.atleast_2d(r2)]), loop_fit32)

    return guess_parms


def shift_vdc(vdc_vec):
    """
    Rolls the Vdc vector by a quarter cycle

    Parameters
    ----------
    vdc_vec : 1D numpy array
        DC offset vector

    Returns
    -------
    shift_ind : int
        Number of indices by which the vector was rolled
    vdc_shifted : 1D numpy array
        Vdc vector rolled by a quarter cycle

    """
    shift_ind = int(
        -1 * len(vdc_vec) / 4)  # should NOT be hardcoded like this!
    vdc_shifted = np.roll(vdc_vec, shift_ind)
    return shift_ind, vdc_shifted