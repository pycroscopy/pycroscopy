# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Chris Smith, Suhas Somnath

"""

from __future__ import division, print_function, absolute_import
import itertools
import numpy as np


def get_component_slice(components, total_components=None):
    """
    Check the components object to determine how to use it to slice the dataset

    Parameters
    ----------
    components : {int, array-like of ints, slice, or None}
        Input Options
        integer: Components less than the input will be kept
        length 2 iterable of integers: Integers define start and stop of component slice to retain
        other iterable of integers or slice: Selection of component indices to retain
        None: All components will be used
    total_components : uint, optional. Default = None
        Total number of spectral components in the dataset

    Returns
    -------
    comp_slice : slice or numpy.ndarray of uints
        Slice or array specifying which components should be kept
    num_comps : uint
        Number of selected components
    """
    num_comps = None

    if components is None:
        num_comps = total_components
        comp_slice = slice(0, num_comps)

    elif isinstance(components, int):
        # Component is integer
        if total_components is not None:
            num_comps = int(np.min([components, total_components]))
        else:
            num_comps = components

        comp_slice = slice(0, num_comps)

    elif hasattr(components, '__iter__') and not isinstance(components, dict):
        # Component is array, list, or tuple
        if len(components) == 2:
            # If only 2 numbers are given, use them as the start and stop of a slice
            comp_slice = slice(int(components[0]), int(components[1]))
            num_comps = abs(comp_slice.stop - comp_slice.start)

        else:
            # Convert components to an unsigned integer array
            comp_slice = np.uint(components)
            # sort and take unique values only
            comp_slice.sort()
            comp_slice = np.unique(comp_slice).tolist()
            num_comps = len(comp_slice)
            # check to see if this giant list of integers is just a simple range
            list_of_ranges = list(to_ranges(comp_slice))

            if len(list_of_ranges) == 1:
                # increment the second index by 1 to be consistent with python
                comp_slice = slice(int(list_of_ranges[0][0]), int(list_of_ranges[0][1] + 1))

    elif isinstance(components, slice):
        # Components is already a slice
        comp_slice = components
        num_comps = np.arange(components.stop+1)[comp_slice].size

        if total_components is not None:
            num_comps = np.min(num_comps, total_components)

    else:
        raise TypeError('Unsupported component type supplied to get_component_slice.  '
                        'Allowed types are integer, numpy array, list, tuple, and slice.')

    return comp_slice, num_comps


def to_ranges(iterable):
    """
    Converts a sequence of iterables to range tuples

    From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python

    Credits: @juanchopanza and @luca

    Parameters
    ----------
    iterable : collections.Iterable object
        iterable object like a list

    Returns
    -------
    iterable : generator object
        Cast to list or similar to use
    """
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]
