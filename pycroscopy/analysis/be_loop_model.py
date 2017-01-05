# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:48:53 2016

@author: Suhas Somnath

"""

from __future__ import division

from warnings import warn

import numpy as np

from .model import Model
from .utils.be_loop import projectLoop
from .be_sho_model import sho32
from ..io.hdf_utils import getH5DsetRefs, getAuxData, copyRegionRefs, linkRefs, linkRefAsAlias, \
    get_sort_order, get_dimensionality, reshape_to_Ndims, reshape_from_Ndims, create_empty_dataset
from ..io.microdata import MicroDataset, MicroDataGroup

loop_metrics32 = np.dtype([('Area', np.float32),
                           ('Centroid x', np.float32),
                           ('Centroid y', np.float32),
                           ('Rotation Angle [rad]', np.float32),
                           ('Offset', np.float32)])

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
        self.__h5_group = None
        self.__sho_spec_inds = None
        self.__sho_spec_vals = None  # used only at one location. can remove if deemed unnecessary
        self.__met_spec_inds = None
        self.__num_forcs = None
        self.__h5_pos_inds = None
        self.__current_pos_slice = None
        self.__current_sho_spec_slice = None
        self.__current_met_spec_slice = None
        self.__dc_offset_index = None
        self.__sho_all_but_forc_inds = None
        self.__sho_all_but_dc_forc_inds = None
        self.__met_all_but_forc_inds = None
        

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

    def simulate_script(self):

        self.__create_projection_datasets()
        max_pos, sho_spec_inds_per_forc, metrics_spec_inds_per_forc = self.__get_sho_chunk_sizes(10, verbose=True)
        dc_vec = self.__get_dc_offset(verbose=True)

        # turn this into a loop
        forc_chunk_index = 0
        pos_chunk_index = 0

        self.__current_pos_slice = slice(pos_chunk_index * max_pos, (pos_chunk_index + 1) * max_pos)
        self.__current_sho_spec_slice = slice(sho_spec_inds_per_forc * forc_chunk_index,
                                              sho_spec_inds_per_forc * (forc_chunk_index + 1))
        self.__current_met_spec_slice = slice(metrics_spec_inds_per_forc * forc_chunk_index,
                                              metrics_spec_inds_per_forc * (forc_chunk_index + 1))
        # read the data here
        raw_2d = self.h5_main[self.__current_pos_slice, self.__current_sho_spec_slice]
        loops_2d, order_dc_offset_reverse, nd_mat_shape_dc_first = self.__reshape_sho_matrix(raw_2d, verbose=True)

        # step 8: perform loop unfolding
        projected_loops_2d, loop_metrics_1d = self._project_loop_batch(dc_vec, np.transpose(loops_2d))
        print('Finished projecting all loops')
        print 'Projected loops of shape:', projected_loops_2d.shape, ', need to bring to:', nd_mat_shape_dc_first
        print 'Loop metrics of shape:', loop_metrics_1d.shape, ', need to bring to:', nd_mat_shape_dc_first[1:]

        # test the reshapes back

    def __create_projection_datasets(self):
        # First grab the spectroscopic indices and values
        self.__sho_spec_inds = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Indices'])[0]
        self.__sho_spec_vals = getAuxData(self.h5_main, auxDataName=['Spectroscopic_Values'])[0]

        dc_ind = np.argwhere(self.__sho_spec_vals.attrs['labels'] == 'DC_Offset').flatten()
        not_dc_inds = np.delete(np.arange(self.__sho_spec_vals.shape[0]), dc_ind)

        # Calculate the number of loops per position
        cycle_start_inds = np.argwhere(self.__sho_spec_inds[dc_ind, :] == 0)
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
            metrics_labels = self.__sho_spec_vals.attrs['labels'][not_dc_inds]
            metrics_units = self.__sho_spec_vals.attrs['units'][not_dc_inds]

            met_spec_inds_mat = self.__sho_spec_inds[not_dc_inds, :][:, cycle_start_inds].squeeze()
            met_spec_vals_mat = self.__sho_spec_vals[not_dc_inds, :][:, cycle_start_inds].squeeze()

        # Prepare containers for the dataets
        ds_projected_loops = MicroDataset('Projected_Loops', data=[], dtype=np.float32,
                                          maxshape=self.h5_main.shape, chunking=self.h5_main.chunks,
                                          compression='gzip')
        ds_loop_metrics = MicroDataset('Loop_Metrics', data=[], dtype=loop_metrics32,
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
        self.h5_projected_loops = getH5DsetRefs(['Projected_Loops'], h5_proj_grp_refs)[0]
        self.h5_loop_metrics = getH5DsetRefs(['Loop_Metrics'], h5_proj_grp_refs)[0]
        self.__met_spec_inds = getH5DsetRefs(['Loop_Metrics_Indices'], h5_proj_grp_refs)[0]
        h5_loop_met_spec_vals = getH5DsetRefs(['Loop_Metrics_Values'], h5_proj_grp_refs)[0]
        self.__h5_group = h5_loop_met_spec_vals.parent

        h5_pos_dsets = getAuxData(self.h5_main, auxDataName=['Position_Indices',
                                                             'Position_Values'])
        self.__h5_pos_inds = getAuxData(self.h5_main, auxDataName=['Position_Indices'])[0]
        # do linking here
        # first the positions
        linkRefs(self.h5_projected_loops, h5_pos_dsets)
        linkRefs(self.h5_projected_loops, [self.h5_loop_metrics])
        linkRefs(self.h5_loop_metrics, h5_pos_dsets)
        # then the spectroscopic
        linkRefs(self.h5_projected_loops, [self.__sho_spec_inds, self.__sho_spec_vals])
        linkRefAsAlias(self.h5_loop_metrics, self.__met_spec_inds, 'Spectroscopic_Indices')
        linkRefAsAlias(self.h5_loop_metrics, h5_loop_met_spec_vals, 'Spectroscopic_Values')

        copyRegionRefs(self.h5_main, self.h5_projected_loops)
        copyRegionRefs(self.h5_main, self.h5_loop_metrics)

        self.hdf.flush()

        return
    
    def __get_sho_chunk_sizes(self, max_mem_MB, verbose=False):
        # Step 1: Find number of FORC cycles (if any), DC steps, and number of loops
        dc_offset_index = np.argwhere(self.__sho_spec_inds.attrs['labels'] == 'DC_Offset')[0][0]
        num_dc_steps = np.unique(self.__sho_spec_inds[dc_offset_index]).size
        all_spec_dims = range(self.__sho_spec_inds.shape[0])
        all_spec_dims.remove(dc_offset_index)
        self.__num_forcs = 1
        if 'FORC' in self.__sho_spec_inds.attrs['labels']:
            forc_pos = np.argwhere(self.__sho_spec_inds.attrs['labels'] == 'FORC')[0][0]
            self.__num_forcs = np.unique(self.__sho_spec_inds[forc_pos]).size
            all_spec_dims.remove(forc_pos)
        # calculate number of loops:
        loop_dims = get_dimensionality(np.transpose(self.__h5_pos_inds), all_spec_dims)
        loops_per_forc = np.product(loop_dims)

        # Step 2: Calculate the largest number of FORCS and positions that can be read given memory limits:
        size_per_forc = num_dc_steps * loops_per_forc * len(self.h5_main.dtype) * self.h5_main.dtype[0].itemsize
        """
        How we arrive at the number for the overhead (how many times the size of the data-chunk we will use in memory)
        1 for the original data, 1 for data copied to all children processes, 1 for results, 0.5 for fit, guess, misc
        """
        mem_overhead = 3.5  
        max_pos = int(max_mem_MB * 1024 ** 2 / (size_per_forc * mem_overhead))
        if verbose:
            print('Can read {} of {} pixels given a {} MB memory limit'.format(max_pos, 
                                                                               self.__h5_pos_inds.shape[0], max_mem_MB))
        max_pos = min(self.__h5_pos_inds.shape[0], max_pos)
        sho_spec_inds_per_forc = self.__sho_spec_inds.shape[1] / self.__num_forcs
        metrics_spec_inds_per_forc = self.__met_spec_inds.shape[1] / self.__num_forcs

        # Step 3: Read allowed chunk
        self.__sho_all_but_forc_inds = range(self.__sho_spec_inds.shape[0])
        self.__met_all_but_forc_inds = range(self.__met_spec_inds.shape[0])
        if self.__num_forcs > 1:
            self.__sho_all_but_forc_inds.remove(forc_pos)
            met_forc_pos = np.argwhere(self.__met_spec_inds.attrs['labels'] == 'FORC')[0][0]
            self.__met_all_but_forc_inds.remove(met_forc_pos)

        return max_pos, sho_spec_inds_per_forc, metrics_spec_inds_per_forc
        
    
    def __reshape_sho_matrix(self, raw_2d, verbose=False):
        # step 4: reshape to N dimensions
        fit_nd, success = reshape_to_Ndims(raw_2d,
                                           h5_pos=self.__h5_pos_inds[self.__current_pos_slice],
                                           h5_spec=self.__sho_spec_inds[self.__sho_all_but_forc_inds,
                                                                       self.__current_sho_spec_slice])
        dim_names_orig = np.hstack((self.__h5_pos_inds.attrs['labels'],
                                    self.__sho_spec_inds.attrs['labels'][self.__sho_all_but_forc_inds]))
        
        if not success:
            warn('Error - could not reshape provided raw data chunk...')
            return None
        if verbose:
            print 'Shape of N dimensional dataset:', fit_nd.shape
            print 'Dimensions of order:', dim_names_orig

        # step 5: Move the voltage dimension to the first dim
        self.__dc_offset_index += len(self.__h5_pos_inds.attrs['labels'])
        order_dc_outside_nd = [self.__dc_offset_index] + range(self.__dc_offset_index) + \
                                                         range(self.__dc_offset_index + 1, len(fit_nd.shape))
        order_dc_offset_reverse = range(1, self.__dc_offset_index + 1) + [0] + range(self.__dc_offset_index + 1,
                                                                                            len(fit_nd.shape))
        fit_Nd2 = np.transpose(fit_nd, tuple(order_dc_outside_nd))
        dim_names_dc_out = dim_names_orig[order_dc_outside_nd]
        if verbose:
            print 'originally:', fit_nd.shape, ', after moving DC offset outside:', fit_Nd2.shape
            print 'new dim names:', dim_names_dc_out

        # step 6: reshape the ND data to 2D arrays
        loops_2d = np.reshape(fit_Nd2, (fit_Nd2.shape[0], -1))
        if verbose:
            print 'Loops ready to be projected of shape (Vdc, all other dims besides FORC):', loops_2d.shape

        return loops_2d, order_dc_offset_reverse, fit_Nd2.shape

    def __reshape_projected_loops_for_h5(self, projected_loops_2d, order_dc_offset_reverse,
                                         nd_mat_shape_dc_first, verbose=False):
        if verbose:
            print 'Projected loops of shape:', projected_loops_2d.shape, ', need to bring to:', nd_mat_shape_dc_first
        # Step 9: Reshape back to same shape as fit_Nd2:
        projected_loops_nd = np.reshape(projected_loops_2d, nd_mat_shape_dc_first)
        if verbose:
            print 'Projected loops reshaped to N dimensions :', projected_loops_nd.shape
        # Step 10: Move Vdc back inwards. Only for projected loop
        projected_loops_nd_2 = np.transpose(projected_loops_nd, order_dc_offset_reverse)
        if verbose:
            print 'Projected loops after moving DC offset inwards:', projected_loops_nd_2.shape
        # step 11: reshape back to 2D
        proj_loops_2d, success = reshape_from_Ndims(projected_loops_nd_2,
                                                    h5_pos=self.__h5_pos_inds[self.__current_pos_slice],
                                                    h5_spec=self.__sho_spec_inds[self.__sho_all_but_forc_inds,
                                                                                self.__current_sho_spec_slice])
        if not success:
            warn('unable to reshape projected loops')
            return None
        if verbose:
            print 'loops shape after collapsing dimensions:', proj_loops_2d.shape

        return proj_loops_2d

    def __reshape_results_for_h5(self, raw_results, nd_mat_shape_dc_first, verbose=False):

        if verbose:
            print 'Loop metrics of shape:', raw_results.shape
        # Step 9: Reshape back to same shape as fit_Nd2:
        loop_metrics_nd = np.reshape(raw_results, nd_mat_shape_dc_first[1:])
        if verbose:
            print 'Loop metrics reshaped to N dimensions :', loop_metrics_nd.shape
        # step 11: reshape back to 2D
        metrics_2d, success = reshape_from_Ndims(loop_metrics_nd,
                                                 h5_pos=self.__h5_pos_inds[self.__current_pos_slice],
                                                 h5_spec=self.__met_spec_inds)
        if not success:
            warn('unable to reshape ND results back to 2D')
            return None
        if verbose:
            print 'metrics shape after collapsing dimensions:', metrics_2d.shape

        return metrics_2d

    def __get_dc_offset(self, verbose=False):
        """
        Gets the DC offset for the current FORC step

        Parameters
        ----------
        verbose : boolean (optional)
            Whether or not to print debugging statements

        Returns
        -------
        dc_vec : 1D float numpy array
            DC offsets for the current FORC step
        """
        spec_sort = get_sort_order(self.__sho_spec_inds[self.__sho_all_but_forc_inds, self.__current_sho_spec_slice])
        # get the size for each of these dimensions
        spec_dims = get_dimensionality(self.__sho_spec_inds[self.__sho_all_but_forc_inds,
                                                            self.__current_sho_spec_slice], spec_sort)
        # apply this knowledge to reshape the spectroscopic values
        # remember to reshape such that the dimensions are arranged in reverse order (slow to fast)
        spec_vals_nd = np.reshape(self.__sho_spec_vals[self.__sho_all_but_forc_inds, self.__current_sho_spec_slice],
                                  [-1] + spec_dims[::-1])
        # This should result in a N+1 dimensional matrix where the first index contains the actual data
        # the other dimensions are present to easily slice the data
        spec_labels_sorted = np.hstack(('Dim', self.__sho_spec_inds.attrs['labels'][spec_sort[::-1]]))
        if verbose:
            print('Spectroscopic dimensions sorted by rate of change:')
            print(spec_labels_sorted)
        # slice the N dimensional dataset such that we only get the DC offset for default values of other dims
        dc_pos = np.argwhere(spec_labels_sorted == 'DC_Offset')[0][0]
        dc_slice = list()
        for dim_ind in range(spec_labels_sorted.size):
            if dim_ind == dc_pos:
                dc_slice.append(slice(None))
            else:
                dc_slice.append(slice(0, 1))
        if verbose:
            print('slice to extract Vdc:')
            print(dc_slice)
        return np.squeeze(spec_vals_nd[tuple(dc_slice)])

    @staticmethod
    def _project_loop_batch(dc_offset, sho_mat):
        """
        This function projects loops given a matrix of the amplitude and phase.
        These matrices (and the Vdc vector) must have a single cycle's worth of
        points on the second dimension

        Parameters
        ------------
        dc_offset : 1D list or numpy array
            DC voltages. vector of length N
        sho_mat : 2D compound numpy array of type - sho32
            SHO response matrix of size MxN - [pixel, dc voltage]

        Returns
        ----------
        results : tuple
            Results from projecting the provided matrices with following components

            projected_loop_mat : MxN numpy array
                Array of Projected loops
            ancillary_mat : M, compound numpy array
                This matrix contains the ancillary information extracted when projecting the loop.
                It contains the following components per loop:
                    'Area' : geometric area of the loop

                    'Centroid x': x positions of centroids for each projected loop

                    'Centroid y': y positions of centroids for each projected loop

                    'Rotation Angle': Angle by which loop was rotated [rad]

                    'Offset': Offset removed from loop
        Note
        -----
        This is the function that can be made parallel if need be.
        However, it is fast enough as is
        """
        num_pixels = int(sho_mat.shape[0])
        projected_loop_mat = np.zeros(shape=sho_mat.shape, dtype=np.float32)
        ancillary_mat = np.zeros(shape=num_pixels, dtype=loop_fit32)

        for pixel in range(num_pixels):
            if pixel % 50 == 0:
                print("Projecting Loop {} of {}".format(pixel, num_pixels))

            pix_dict = projectLoop(np.squeeze(dc_offset),
                                   sho_mat[pixel]['Amplitude [V]'],
                                   sho_mat[pixel]['Phase [rad]'])

            projected_loop_mat[pixel, :] = pix_dict['Projected Loop']
            ancillary_mat[pixel]['Rotation Angle [rad]'] = pix_dict['Rotation Matrix'][0]
            ancillary_mat[pixel]['Offset'] = pix_dict['Rotation Matrix'][1]
            ancillary_mat[pixel]['Area'] = pix_dict['Geometric Area']
            ancillary_mat[pixel]['Centroid x'] = pix_dict['Centroid'][0]
            ancillary_mat[pixel]['Centroid y'] = pix_dict['Centroid'][1]

        return projected_loop_mat, ancillary_mat

    def __project_loops(self):
        pass

    def _createGuessDatasets(self):
        """
        Creates the HDF5 Guess dataset and links the it to the ancillary datasets.
        """
        self.h5_guess = create_empty_dataset(self.h5_loop_metrics, loop_fit32, 'Guess')
        self.__h5_group.attrs['guess method'] = 'pycroscopy statistical'

    def _createFitDataset(self):
        """
        Creates the HDF5 Fit dataset and links the it to the ancillary datasets.
        """

        if self.h5_guess is None:
            warn('Need to guess before fitting!')
            return

        self.h5_fit = create_empty_dataset(self.h5_guess, loop_fit32, 'Fit')
        self.__h5_group.attrs['fit method'] = 'pycroscopy functional'
