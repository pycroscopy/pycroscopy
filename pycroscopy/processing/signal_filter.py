# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 11:48:53 2017

@author: Suhas Somnath

"""

from __future__ import division, print_function, absolute_import, unicode_literals
import h5py
import numpy as np
from collections import Iterable
from pyUSID.processing.process import Process, parallel_compute
from pyUSID.io.hdf_utils import create_results_group, write_main_dataset, write_simple_attrs, create_empty_dataset, \
    write_ind_val_dsets
from pyUSID.io.write_utils import Dimension
from .fft import get_noise_floor, are_compatible_filters, build_composite_freq_filter
from .gmode_utils import test_filter

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
        self.duplicate_h5_groups, self.partial_h5_groups = self._check_for_duplicates()

        self.data = None
        self.filtered_data = None
        self.condensed_data = None
        self.noise_floors = None
        self.h5_filtered = None
        self.h5_condensed = None
        self.h5_noise_floors = None

    def test(self, pix_ind=None, excit_wfm=None, **kwargs):
        """
        Tests the signal filter on a single pixel (randomly chosen unless manually specified) worth of data.

        Parameters
        ----------
        pix_ind : int, optional. default = random
            Index of the pixel whose data will be used for inference
        excit_wfm : array-like, optional. default = None
            Waveform against which the raw and filtered signals will be plotted. This waveform can be a fraction of the
            length of a single pixel's data. For example, in the case of G-mode, where a single scan line is yet to be
            broken down into pixels, the excitation waveform for a single pixel can br provided to automatically
            break the raw and filtered responses also into chunks of the same size.

        Returns
        -------
        fig, axes
        """
        if pix_ind is None:
            pix_ind = np.random.randint(0, high=self.h5_main.shape[0])
        return test_filter(self.h5_main[pix_ind], frequency_filters=self.frequency_filters, excit_wfm=excit_wfm,
                           noise_threshold=self.noise_threshold, plot_title='Pos #' + str(pix_ind), show_plots=True,
                           **kwargs)

    def _create_results_datasets(self):
        """
        Creates all the datasets necessary for holding all parameters + data.
        """

        self.h5_results_grp = create_results_group(self.h5_main, self.process_name)

        self.parms_dict.update({'last_pixel': 0, 'algorithm': 'pycroscopy_SignalFilter'})
        write_simple_attrs(self.h5_results_grp, self.parms_dict)

        assert isinstance(self.h5_results_grp, h5py.Group)

        if isinstance(self.composite_filter, np.ndarray):
            h5_comp_filt = self.h5_results_grp.create_dataset('Composite_Filter',
                                                              data=np.float32(self.composite_filter))

        # First create the position datsets if the new indices are smaller...
        if self.num_effective_pix != self.h5_main.shape[0]:
            # TODO: Do this part correctly. See past solution:
            """
            # need to make new position datasets by taking every n'th index / value:
                new_pos_vals = np.atleast_2d(h5_pos_vals[slice(0, None, self.num_effective_pix), :])
                pos_descriptor = []
                for name, units, leng in zip(h5_pos_inds.attrs['labels'], h5_pos_inds.attrs['units'],
                                             [int(np.unique(h5_pos_inds[:, dim_ind]).size / self.num_effective_pix)
                                              for dim_ind in range(h5_pos_inds.shape[1])]):
                    pos_descriptor.append(Dimension(name, units, np.arange(leng)))
                ds_pos_inds, ds_pos_vals = build_ind_val_dsets(pos_descriptor, is_spectral=False, verbose=self.verbose)
                h5_pos_vals.data = np.atleast_2d(new_pos_vals)  # The data generated above varies linearly. Override.
                
            """
            h5_pos_inds_new, h5_pos_vals_new = write_ind_val_dsets(self.h5_results_grp,
                                                                   Dimension('pixel', 'a.u.', self.num_effective_pix),
                                                                   is_spectral=False, verbose=self.verbose)
        else:
            h5_pos_inds_new = self.h5_main.h5_pos_inds
            h5_pos_vals_new = self.h5_main.h5_pos_vals

        if self.noise_threshold is not None:
            self.h5_noise_floors = write_main_dataset(self.h5_results_grp, (self.num_effective_pix, 1), 'Noise_Floors',
                                                      'Noise', 'a.u.', None, Dimension('arb', '', [1]),
                                                      dtype=np.float32, aux_spec_prefix='Noise_Spec_',
                                                      h5_pos_inds=h5_pos_inds_new, h5_pos_vals=h5_pos_vals_new,
                                                      verbose=self.verbose)

        if self.write_filtered:
            # Filtered data is identical to Main_Data in every way - just a duplicate
            self.h5_filtered = create_empty_dataset(self.h5_main, self.h5_main.dtype, 'Filtered_Data',
                                                    h5_group=self.h5_results_grp)

        self.hot_inds = None

        if self.write_condensed:
            self.hot_inds = np.where(self.composite_filter > 0)[0]
            self.hot_inds = np.uint(self.hot_inds[int(0.5 * len(self.hot_inds)):])  # only need to keep half the data
            condensed_spec = Dimension('hot_frequencies', '', int(0.5 * len(self.hot_inds)))
            self.h5_condensed = write_main_dataset(self.h5_results_grp, (self.num_effective_pix, len(self.hot_inds)),
                                                   'Condensed_Data', 'Complex', 'a. u.', None, condensed_spec,
                                                   h5_pos_inds=h5_pos_inds_new, h5_pos_vals=h5_pos_vals_new,
                                                   dtype=np.complex, verbose=self.verbose)

    def _get_existing_datasets(self):
        """
        Extracts references to the existing datasets that hold the results
        """
        if self.write_filtered:
            self.h5_filtered = self.h5_results_grp['Filtered_Data']
        if self.write_condensed:
            self.h5_condensed = self.h5_results_grp['Condensed_Data']
        if self.noise_threshold is not None:
            self.h5_noise_floors = self.h5_results_grp['Noise_Floors']

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

        self.h5_main.file.flush()

        print('Finished processing upto pixel ' + str(self._end_pos) + ' of ' + str(self.h5_main.shape[0]))

        # Now update the start position
        self._start_pos = self._end_pos

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
        # get FFT of the entire data chunk
        self.data = np.fft.fftshift(np.fft.fft(self.data, axis=1), axes=1)

        if self.noise_threshold is not None:
            self.noise_floors = parallel_compute(self.data, get_noise_floor, cores=self._cores,
                                                 func_args=[self.noise_threshold],
                                                 verbose=self.verbose)

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
                # TODO: implement phase compensation
                # do np.roll on data
                # self.data = np.roll(self.data, 0, axis=1)
                pass
