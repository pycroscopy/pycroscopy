"""
Created on Nov 8, 2016

@author: Chris Smith -- csmith55@utk.edu
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import array
import os

import numpy as np
from skimage.io import imread


def read_image(image_path, *args, **kwargs):
    """
    Read the image file at `image_path` into a numpy array

    Parameters
    ----------
    image_path : str
        Path to the image file

    Returns
    -------
    image : numpy.ndarray
        Array containing the image from the file `image_path`
    image_parms : dict
        Dictionary containing image parameters.  If image type does not have
        parameters then an empty dictionary is returned.

    """
    ext = os.path.splitext(image_path)[1]
    if ext == '.txt':
        return read_txt(image_path, *args, **kwargs), dict()
    else:
        # Set the as_grey argument to True is not already provided.
        kwargs['as_grey'] = (kwargs.pop('as_grey', True))
        return imread(image_path, *args, **kwargs), dict()


def unnest_parm_dicts(image_parms, prefix=''):
    """
    Parses the nested image parameter dictionary and converts it to a single
    level dictionary, prepending the name of inner dictionaries to their
    keys to denote level.

    Parameters
    ----------
    image_parms : dict
    prefix : str

    Returns
    -------

    """
    new_parms = dict()
    for name in image_parms.keys():
        val = image_parms[name]
        # print 'name',name,'val',val
        name = '-'.join([prefix] + name.split()).strip('-')
        if isinstance(val, dict):
            new_parms.update(unnest_parm_dicts(val, name))
        elif isinstance(val, list) and isinstance(val[0], dict):
            for thing in val:
                new_parms.update(unnest_parm_dicts(thing, name))
        else:
            new_parms[name] = try_tag_to_string(val)

    return new_parms


def try_tag_to_string(tag_data):
    """
    Attempt to convert array of integers into a string

    Parameters
    ----------
    tag_data : array.array
        Array of 16-bit integers

    Returns
    -------
    tag_data : str
        Decoded string from the integer tag

    """
    if not isinstance(tag_data, array.array):
        return tag_data

    if tag_data.typecode == 'H':
        try:
            tag_data = str(tag_data.tostring().decode('utf-16'))
        except UnicodeDecodeError:
            pass
        except UnicodeEncodeError:
            pass
        except:
            raise

    return tag_data


def read_txt(image_path, header_lines=0, delimiter=None, *args, **kwargs):
    """

    Parameters
    ----------
    image_path : str
        Path to the image file
    header_lines : int
        Number of lines to skip as the header
    delimiter : str
        Separator between the columns of data
    args
    kwargs

    Returns
    -------
    image : numpy.ndarray
        Image array read from the plaintext file

    """
    image = np.loadtxt(image_path, *args,
                       skiprows=header_lines,
                       delimiter=delimiter, **kwargs)

    return image


def no_bin(image, *args, **kwargs):
    """
    Does absolutely nothing to the image.  Exists so that we can have
    a bin function to call whether we actually rebin the image or not.

    Parameters
    ----------
    image : ndarray
        Image
    args:
        Argument list
    kwargs:
        Keyword argument list

    Returns
    -------
    image : ndarray
        The input image
    """
    return image
