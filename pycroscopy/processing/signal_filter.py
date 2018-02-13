# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 11:48:53 2017

@author: Suhas Somnath

"""

from __future__ import division, print_function, absolute_import, unicode_literals

import time as tm
import numpy as np
from collections import Iterable
from .process import Process, parallel_compute
from ..io.microdata import MicroDataset, MicroDataGroup
from ..io.hdf_utils import getH5DsetRefs, getAuxData, copyAttributes, link_as_main, linkRefs, check_for_old
from ..io.translators.utils import build_ind_val_dsets
from ..io.io_hdf5 import ioHDF5
from .fft import get_noise_floor, are_compatible_filters, build_composite_freq_filter
# TODO: implement phase compensation
# TODO: correct implementation of num_pix


class SignalFilter(Process):
    def __init__(self, h5_main, frequency_filters=None, noise_threshold=None, write_filtered=True,
                 write_condensed=False, num_pix=1, phase_rad=0,  **kwargs):
        """
        Filters the entire h5 dataset with the given filtering parameters.

        Parameters
        ----------
        h5_main : h5py.Dataset object
            Dataset to process
        frequency_filters : (Optional) single or list of pycroscopy.fft.FrequencyFilter objects
            Frequency (vertical) filters to apply to signal
        noise_threshold : (Optional) float. Default - None
            Noise tolerance to apply to data. Value must be within (0, 1)
        write_filtered : (Optional) bool. Default - True
            Whether or not to write the filtered data to file
        write_condensed : Optional) bool. Default - False
            Whether or not to write the condensed data in frequency space to file. Use this for datasets that are very
            large but sparse in frequency space.
        num_pix : (Optional) uint. Default - 1
            Number of pixels to use for filtering. More pixels means a lower noise floor and the ability to pick up
            weaker signals. Use only if absolutely necessary. This value must be a divisor of the number of pixels in
            the dataset
        phase_rad : (Optional). float
            Degrees by which the output is rotated with respect to the input to compensate for phase lag.
            This feature has NOT yet been implemented.
        kwargs : (Optional). dictionary
            Please see Process class for additional inputs
        """

        super(SignalFilter, self).__init__(h5_main, **kwargs)

        if frequency_filters is None and noise_threshold is None:
            raise ValueError('Need to specify at least some noise thresholding / frequency filter')

        if noise_threshold is not None:
            if noise_threshold >= 1 or noise_threshold <= 0:
                raise ValueError('Noise threshold must be within (0 1)')

        self.composite_filter = 1
        if frequency_filters is not None:
            if not isinstance(frequency_filters, Iterable):
                frequency_filters = [frequency_filters]
            if not are_compatible_filters(frequency_filters):
                raise ValueError('frequency filters must be a single or list of FrequencyFilter objects')
            self.composite_filter = build_composite_freq_filter(frequency_filters)
        else:
            write_condensed = False

        if write_filtered is False and write_condensed is False:
            raise ValueError('You need to write the filtered and/or the condensed dataset to the file')

        num_effective_pix = h5_main.shape[0] * 1.0 / num_pix
        if num_effective_pix % 1 > 0:
            raise ValueError('Number of pixels not divisible by the number of pixels to use for FFT filter')

        self.num_effective_pix = int(num_effective_pix)
        self.phase_rad = phase_rad

        self.noise_threshold = noise_threshold
        self.frequency_filters = frequency_filters

        self.write_filtered = write_filtered
        self.write_condensed = write_condensed

        """
        Remember that the default number of pixels corresponds to only the raw data that can be held in memory
        In the case of signal filtering, the datasets that will occupy space are:
        1. Raw, 2. filtered (real + freq space copies), 3. Condensed (substantially lesser space)
        The actual scaling of memory depends on options:
        """
        scaling_factor = 1 + 2 * self.write_filtered + 0.25 * self.write_condensed
        self._max_pos_per_read = int(self._max_pos_per_read / scaling_factor)

        if self.verbose:
            print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))

        self.parms_dict = dict()
        if self.frequency_filters is not None:
            for filter in self.frequency_filters:
                self.parms_dict.update(filter.get_parms())
        if self.noise_threshold is not None:
            self.parms_dict['noise_threshold'] = self.noise_threshold
        self.parms_dict['num_pix'] = self.num_effective_pix

        self.process_name = 'FFT_Filtering'
        self.duplicate_h5_groups = self._check_for_duplicates()

        self.data = None
        self.filtered_data = None
        self.condensed_data = None
        self.noise_floors = None
        self.h5_filtered = None
        self.h5_condensed = None
        self.h5_noise_floors = None

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
        verbose = self.verbose
        self.verbose = False

        super(SignalFilter, self)._set_memory_and_cores(cores, mem)

        self.verbose = verbose

        max_data_chunk = self._max_mem_mb / self._cores

        # Now calculate the number of positions that can be stored in memory in one go.
        # Mem for initial data and final data before writing to file
        mb_per_position = self.h5_main.dtype.itemsize * self.h5_main.shape[1] / 1e6 * 2
        # Mem for fft of data
        mb_per_position += np.complex128.dtype.__sizeof__() * self.h5_main.shape[1] / 1e6

        self._max_pos_per_read = max_data_chunk // mb_per_position

        if self.verbose:
            print('Allowed to read {} pixels per chunk'.format(self._max_pos_per_read))
            print('Allowed to use up to', str(self._cores), 'cores and', str(self._max_mem_mb), 'MB of memory')

    def _create_results_datasets(self):
        """
        Creates all the datasets necessary for holding all parameters + data.
        """

        grp_name = self.h5_main.name.split('/')[-1] + '-' + self.process_name + '_'
        grp_filt = MicroDataGroup(grp_name, self.h5_main.parent.name)

        self.parms_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_SignalFilter'})
        grp_filt.attrs = self.parms_dict

        if isinstance(self.composite_filter, np.ndarray):
            ds_comp_filt = MicroDataset('Composite_Filter', np.float32(self.composite_filter))
            grp_filt.addChildren([ds_comp_filt])

        if self.noise_threshold is not None:
            ds_noise_floors = MicroDataset('Noise_Floors',
                                           data=np.zeros(shape=(self.num_effective_pix, 1), dtype=np.float32))
            ds_noise_spec_inds, ds_noise_spec_vals = build_ind_val_dsets([1], is_spectral=True,
                                                                         labels=['arb'], units=[''],
                                                                         verbose=self.verbose)
            ds_noise_spec_inds.name = 'Noise_Spectral_Indices'
            ds_noise_spec_vals.name = 'Noise_Spectral_Values'
            grp_filt.addChildren([ds_noise_floors, ds_noise_spec_inds, ds_noise_spec_vals])

        if self.write_filtered:
            ds_filt_data = MicroDataset('Filtered_Data', data=[], maxshape=self.h5_main.maxshape,
                                        dtype=np.float32, chunking=self.h5_main.chunks, compression='gzip')
            grp_filt.addChildren([ds_filt_data])

        self.hot_inds = None

        h5_pos_inds = getAuxData(self.h5_main, auxDataName=['Position_Indices'])[0]
        h5_pos_vals = getAuxData(self.h5_main, auxDataName=['Position_Values'])[0]

        if self.write_condensed:
            self.hot_inds = np.where(self.composite_filter > 0)[0]
            self.hot_inds = np.uint(self.hot_inds[int(0.5 * len(self.hot_inds)):])  # only need to keep half the data
            ds_spec_inds, ds_spec_vals = build_ind_val_dsets([int(0.5 * len(self.hot_inds))], is_spectral=True,
                                                             labels=['hot_frequencies'], units=[''],
                                                             verbose=self.verbose)
            ds_spec_vals.data = np.atleast_2d(self.hot_inds)  # The data generated above varies linearly. Override.
            ds_cond_data = MicroDataset('Condensed_Data', data=[],
                                        maxshape=(self.num_effective_pix, len(self.hot_inds)),
                                        dtype=np.complex, chunking=(1, len(self.hot_inds)), compression='gzip')
            grp_filt.addChildren([ds_spec_inds, ds_spec_vals, ds_cond_data])
            if self.num_effective_pix > 1:
                # need to make new position datasets by taking every n'th index / value:
                new_pos_vals = np.atleast_2d(h5_pos_vals[slice(0, None, self.num_effective_pix), :])
                ds_pos_inds, ds_pos_vals = build_ind_val_dsets([int(np.unique(h5_pos_inds[:, dim_ind]).size /
                                                                    self.num_effective_pix)
                                                                for dim_ind in range(h5_pos_inds.shape[1])],
                                                               is_spectral=False,
                                                               labels=h5_pos_inds.attrs['labels'],
                                                               units=h5_pos_inds.attrs['units'], verbose=self.verbose)
                h5_pos_vals.data = np.atleast_2d(new_pos_vals)  # The data generated above varies linearly. Override.
                grp_filt.addChildren([ds_pos_inds, ds_pos_vals])

        if self.verbose:
            grp_filt.showTree()
        hdf = ioHDF5(self.h5_main.file)
        h5_filt_refs = hdf.writeData(grp_filt, print_log=self.verbose)

        if isinstance(self.composite_filter, np.ndarray):
            h5_comp_filt = getH5DsetRefs(['Composite_Filter'], h5_filt_refs)[0]

        if self.noise_threshold is not None:
            self.h5_noise_floors = getH5DsetRefs(['Noise_Floors'], h5_filt_refs)[0]
            self.h5_results_grp = self.h5_noise_floors.parent
            link_as_main(self.h5_noise_floors, h5_pos_inds, h5_pos_vals,
                         getH5DsetRefs(['Noise_Spectral_Indices'], h5_filt_refs)[0],
                         getH5DsetRefs(['Noise_Spectral_Values'], h5_filt_refs)[0])

        # Now need to link appropriately:
        if self.write_filtered:
            self.h5_filtered = getH5DsetRefs(['Filtered_Data'], h5_filt_refs)[0]
            self.h5_results_grp = self.h5_filtered.parent
            copyAttributes(self.h5_main, self.h5_filtered, skip_refs=False)
            if isinstance(self.composite_filter, np.ndarray):
                linkRefs(self.h5_filtered, [h5_comp_filt])

            """link_as_main(self.h5_filtered, h5_pos_inds, h5_pos_vals,
                         getAuxData(h5_main, auxDataName=['Spectroscopic_Indices'])[0],
                         getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0])"""

        if self.write_condensed:
            self.h5_condensed = getH5DsetRefs(['Condensed_Data'], h5_filt_refs)[0]
            self.h5_results_grp = self.h5_condensed.parent
            if isinstance(self.composite_filter, np.ndarray):
                linkRefs(self.h5_condensed, [h5_comp_filt])
            if self.noise_threshold is not None:
                linkRefs(self.h5_condensed, [self.h5_noise_floors])

            if self.num_effective_pix > 1:
                h5_pos_inds = getH5DsetRefs(['Position_Indices'], h5_filt_refs)[0]
                h5_pos_vals = getH5DsetRefs(['Position_Values'], h5_filt_refs)[0]

            link_as_main(self.h5_condensed, h5_pos_inds, h5_pos_vals,
                         getH5DsetRefs(['Spectroscopic_Indices'], h5_filt_refs)[0],
                         getH5DsetRefs(['Spectroscopic_Values'], h5_filt_refs)[0])

    def _write_results_chunk(self):
        """
        Writes data chunks back to the file
        """

        pos_slice = slice(self._start_pos, self._end_pos)

        if self.write_condensed:
            self.h5_condensed[pos_slice] = self.condensed_data
        if self.noise_threshold is not None:
            self.h5_noise_floors[pos_slice] = np.atleast_2d(self.noise_floors)
        if self.write_filtered:
            self.h5_filtered[pos_slice] = self.filtered_data

        # Leaving in this provision that will allow restarting of processes
        self.h5_results_grp.attrs['last_pixel'] = self._end_pos

        self.hdf.flush()

        print('Finished processing upto pixel ' + str(self._end_pos) + ' of ' + str(self.h5_main.shape[0]))

        # Now update the start position
        self._start_pos = self._end_pos

    @staticmethod
    def _unit_function():
        return get_noise_floor

    def compute(self, *args, **kwargs):
        """
        Creates placeholders for the results, applies the filers to the data, and writes the output to the file.

        Parameters
        ----------

        Returns
        -------
        h5_results_grp : h5py.Datagroup object
            Datagroup containing all the results

        """
        self._create_results_datasets()

        time_per_pix = 0

        num_pos = self.h5_main.shape[0]

        self._read_data_chunk()
        while self.data is not None:

            t_start = tm.time()

            # get FFT of the entire data chunk
            self.data = np.fft.fftshift(np.fft.fft(self.data, axis=1), axes=1)

            if self.noise_threshold is not None:
                self.noise_floors = parallel_compute(self.data, get_noise_floor, cores=self._cores,
                                                     func_args=[self.noise_threshold])

            if isinstance(self.composite_filter, np.ndarray):
                # multiple fft of data with composite filter
                self.data *= self.composite_filter

            if self.noise_threshold is not None:
                # apply thresholding
                self.data[np.abs(self.data) < np.tile(np.atleast_2d(self.noise_floors), self.data.shape[1])] = 1E-16

            if self.write_condensed:
                # set self.condensed_data here
                self.condensed_data = self.data[:, self.hot_inds]

            if self.write_filtered:
                # take inverse FFT
                self.filtered_data = np.real(np.fft.ifft(np.fft.ifftshift(self.data, axes=1), axis=1))
                if self.phase_rad > 0:
                    # do np.roll on data
                    # self.data = np.roll(self.data, 0, axis=1)
                    pass

            tot_time = np.round(tm.time() - t_start, decimals=2)

            if self.verbose:
                print('Done parallel computing in {} sec or {} sec per pixel'.format(tot_time,
                                                                                     tot_time / self.data.shape[0]))
            if self._start_pos == 0:
                time_per_pix = tot_time / self._end_pos  # in seconds
            else:
                print('Time remaining: {} mins'.format(np.round((num_pos - self._end_pos) * time_per_pix / 60, 2)))

            self._write_results_chunk()
            self._read_data_chunk()

        if self.verbose:
            print('Finished processing the dataset completely')

        return self.h5_results_grp
