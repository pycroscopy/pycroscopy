# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath
"""

from __future__ import division

import abc

import numpy as np

from .utils import makePositionMat, getPositionSlicing, getSpectralSlicing
from ..io_utils import getAvailableMem
from ..microdata import MicroDataset


class Translator(object):
    """
    Abstract class that defines the most basic functionality of a data format translator.
    A translator converts experimental data from binary / proprietary
    data formats to a single standardized HDF5 data file
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_mem_mb=1024):
        """
        Parameters
        -----------
        max_ram_mb : unsigned integer
            Maximum system memory (in megabytes) that the translator can use
            
        Returns
        -------
        Translator object
        """
        self.max_ram = min(max_mem_mb*1024**2, 0.75*getAvailableMem())

    @abc.abstractmethod
    def translate(self, filepath):
        """
        Abstract method.
        To be implemented by extensions of this class. God I miss Java!
        """
        pass

    @abc.abstractmethod
    def _parsefilepath(self, input_path):
        """
        Abstract method
        Parses the `input_path` to determine the `basename` and find
        the appropriate data files

        """
        pass

    @staticmethod
    def _build_ind_val_dsets(dimensions, is_spectral=True, steps=None, initial_values=None, labels=None,
                             units=None, verbose=False):
        """
        Builds the MicroDatasets for the position OR spectroscopic indices and values
        of the data

        Parameters
        ----------
        is_spectral : Boolean
            Spectroscopic (True) or Position (False)
        dimensions : array_like of numpy.uint
            Integer values for the length of each dimension
        steps : array_like of float, optional
            Floating point values for the step-size in each dimension.  One
            if not specified.
        initial_values : array_like of float, optional
            Floating point for the zeroth value in each dimension.  Zero if
            not specified.
        labels : array_like of str, optional
            The names of each dimension.  Empty strings will be used if not
            specified.
        units : array_like of str, optional
            The units of each dimension.  Empty strings will be used if not
            specified.
        verbose : Boolean, optional
            Whether or not to print statements for debugging purposes

        Returns
        -------
        ds_spec_inds : Microdataset of numpy.uint
            Dataset containing the position indices
        ds_spec_vals : Microdataset of float
            Dataset containing the value at each position

        Notes
        -----
        `steps`, `initial_values`, `labels`, and 'units' must be the same length as
        `dimensions` when they are specified.

        Dimensions should be in the order from fastest varying to slowest.
        """

        if steps is None:
            steps = np.ones_like(dimensions)
        elif len(steps) != len(dimensions):
            raise ValueError('The arrays for step sizes and dimension sizes must be the same.')
        if is_spectral:
            steps = np.atleast_2d(steps).transpose()
        if verbose:
            print(steps.shape)
            print(steps)

        if initial_values is None:
            initial_values = np.zeros_like(dimensions)
        elif len(initial_values) != len(dimensions):
            raise ValueError('The arrays for initial values and dimension sizes must be the same.')
        if is_spectral:
            initial_values = np.atleast_2d(initial_values).transpose()
        if verbose:
            print(initial_values.shape)
            print(initial_values)

        if labels is None:
            labels = ['' for _ in len(dimensions)]
        elif len(labels) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')

        # Get the indices for all dimensions
        indices = makePositionMat(dimensions).transpose()
        if verbose:
            print(indices.shape)
            print(indices)

        # Convert the indices to values
        spec_vals = initial_values + np.float32(indices)*steps

        # Create the slices that will define the labels
        if is_spectral:
            mode = 'Spectroscopic'
            region_slices = getSpectralSlicing(labels)
        else:
            mode = 'Position'
            region_slices = getPositionSlicing(labels)

        # Create the MicroDatasets for both Indices and Values
        ds_indices = MicroDataset(mode + '_Indices', indices, dtype=np.uint32)
        ds_indices.attrs['labels'] = region_slices

        ds_values = MicroDataset(mode + 'Values', spec_vals, dtype=np.float32)
        ds_values.attrs['labels'] = region_slices

        if units is None:
            pass
        elif len(units) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')
        else:
            ds_indices.attrs['units'] = units
            ds_values.attrs['units'] = units

        return ds_indices, ds_values

    @abc.abstractmethod
    def _read_data(self):
        """
        Abstract method
        Reads the data into the hdf5 datasets.
        """


