"""
Created on Nov 8, 2016

@author: Chris Smith -- csmith55@utk.edu
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import array


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
        # print('name',name,'val',val)
        name = '-'.join([prefix] + name.split()).strip('-')
        if isinstance(val, dict):
            new_parms.update(unnest_parm_dicts(val, name))
        elif isinstance(val, list):
            if len(val)==0:
                continue
            elif isinstance(val[0], dict):
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
