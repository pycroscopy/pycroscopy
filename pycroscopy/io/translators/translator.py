# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2015

@author: Suhas Somnath
"""

from __future__ import division
import abc
import numpy as np
from ..io_utils import getAvailableMem
from ..hdf_utils import linkRefAsAlias
from .utils import makePositionMat, getPositionSlicing, getSpectralSlicing
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
    def _buildpositiondatasets(dimensions, steps=None, initial_values=None, labels=None,
                               units=None):
        """
        Builds the MicroDatasets for the position indices and values
        of the data

        Parameters
        ----------
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

        Returns
        -------
        ds_pos_inds : Microdataset of numpy.uint
            Dataset containing the position indices
        ds_pos_vals : Microdataset of float
            Dataset containing the value at each position

        Notes
        -----
        `steps`, `initial_values`, `labels`, and 'units' must be the same length as
        `dimensions` whenthey are specified.

        Dimensions should be in the order from fastest varying to slowest.
        """

        if steps is None:
            steps = np.ones_like(dimensions)
        elif len(steps) != len(dimensions):
            raise ValueError('The arrays for step sizes and dimension sizes must be the same.')

        if initial_values is None:
            initial_values = np.zeros_like(dimensions)
        elif len(initial_values) != len(dimensions):
            raise ValueError('The arrays for initial values and dimension sizes must be the same.')

        if labels is None:
            labels = ['' for _ in len(dimensions)]
        elif len(labels) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')

        # Get the indices for each dimension
        pos_inds = makePositionMat(dimensions)

        # Convert the indices to values
        pos_vals = initial_values + np.float32(pos_inds)*steps

        # Create the slices that will define the labels
        pos_slices = getPositionSlicing(labels)

        # Create the MicroDatasets for both Indices and Values
        ds_pos_inds = MicroDataset('Position_Indices', pos_inds, dtype=np.uint32)
        ds_pos_inds.attrs['labels'] = pos_slices

        ds_pos_vals = MicroDataset('Position_Values', pos_vals, dtype=np.float32)
        ds_pos_vals.attrs['labels'] = pos_slices

        if units is None:
            pass
        elif len(units) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')
        else:
            ds_pos_inds.attrs['units'] = units
            ds_pos_vals.attrs['units'] = units

        return ds_pos_inds, ds_pos_vals

    @staticmethod
    def _buildspectroscopicdatasets(dimensions, steps=None, initial_values=None, labels=None,
                                    units=None):
        """
        Builds the MicroDatasets for the position indices and values
        of the data

        Parameters
        ----------
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

        Returns
        -------
        ds_spec_inds : Microdataset of numpy.uint
            Dataset containing the position indices
        ds_spec_vals : Microdataset of float
            Dataset containing the value at each position

        Notes
        -----
        `steps`, `initial_values`, `labels`, and 'units' must be the same length as
        `dimensions` whenthey are specified.

        Dimensions should be in the order from fastest varying to slowest.
        """

        if steps is None:
            steps = np.ones_like(dimensions)
        elif len(steps) != len(dimensions):
            raise ValueError('The arrays for step sizes and dimension sizes must be the same.')
        steps = np.atleast_2d(steps).transpose()

        if initial_values is None:
            initial_values = np.zeros_like(dimensions)
        elif len(initial_values) != len(dimensions):
            raise ValueError('The arrays for initial values and dimension sizes must be the same.')
        initial_values = np.atleast_2d(initial_values).transpose()

        if labels is None:
            labels = ['' for _ in len(dimensions)]
        elif len(labels) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')



        # Get the indices for each dimension
        spec_inds = makePositionMat(dimensions).transpose()

        # Convert the indices to values
        spec_vals = initial_values + np.float32(spec_inds)*steps

        # Create the slices that will define the labels
        spec_slices = getSpectralSlicing(labels)

        # Create the MicroDatasets for both Indices and Values
        ds_spec_inds = MicroDataset('Spectroscopic_Indices', spec_inds, dtype=np.uint32)
        ds_spec_inds.attrs['labels'] = spec_slices

        ds_spec_vals = MicroDataset('Spectroscopic_Values', spec_vals, dtype=np.float32)
        ds_spec_vals.attrs['labels'] = spec_slices

        if units is None:
            pass
        elif len(units) != len(dimensions):
            raise ValueError('The arrays for labels and dimension sizes must be the same.')
        else:
            ds_spec_inds.attrs['units'] = units
            ds_spec_vals.attrs['units'] = units

        return ds_spec_inds, ds_spec_vals

    @staticmethod
    def _linkformain(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals):
        """
        Links the object references to the four position and spectrosocpic datasets as
        attributes of `h5_main`

        Parameters
        ----------
        h5_main : h5py.Dataset
        2D Dataset which will have the references added as attributes
        h5_pos_inds : h5py.Dataset
        Dataset that will be linked with the name 'Position_Indices'
        h5_pos_vals : h5py.Dataset
        Dataset that will be linked with the name 'Position_Values'
        h5_spec_inds : h5py.Dataset
        Dataset that will be linked with the name 'Spectroscopic_Indices'
        h5_spec_vals : h5py.Dataset
        Dataset that will be linked with the name 'Spectroscopic_Values'
        """
        linkRefAsAlias(h5_main, h5_pos_inds, 'Position_Indices')
        linkRefAsAlias(h5_main, h5_pos_vals, 'Position_Values')
        linkRefAsAlias(h5_main, h5_spec_inds, 'Spectroscopic_Indices')
        linkRefAsAlias(h5_main, h5_spec_vals, 'Spectroscopic_Values')

    @abc.abstractmethod
    def _read_data(self):
        """
        Abstract method
        Reads the data into the hdf5 datasets.
        """