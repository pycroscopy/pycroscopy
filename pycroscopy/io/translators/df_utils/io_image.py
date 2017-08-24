"""
Created on Nov 8, 2016

@author: Chris Smith -- csmith55@utk.edu
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import array
import os

import numpy as np
from skimage.io import imread

from . import dm4reader
from .dm3_image_utils import parse_dm_header, imagedatadict_to_ndarray


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
    if ext == '.dm3':
        kwargs.pop('as_grey', None)
        return read_dm3(image_path, *args, **kwargs)
    elif ext == '.dm4':
        kwargs.pop('as_grey', None)
        return read_dm4(image_path, *args, **kwargs)
    elif ext == '.txt':
        return read_txt(image_path, *args, **kwargs), dict()
    else:
        # Set the as_grey argument to True is not already provided.
        kwargs['as_grey'] = (kwargs.pop('as_grey', True))
        return imread(image_path, *args, **kwargs), dict()


def read_dm3(image_path, get_parms=True):
    """
    Read an image from a dm3 file into a numpy array

    image_path : str
        Path to the image file
    get_parms : Boolean, optional
        Should the parameters from the dm3 file be returned
        Default True

    Returns
    -------
    image : numpy.ndarray
        Array containing the image from the file `image_path`

    """
    image_file = open(image_path, 'rb')
    dmtag = parse_dm_header(image_file)
    img_index = -1
    image = imagedatadict_to_ndarray(dmtag['ImageList'][img_index]['ImageData'])
    image_parms = dmtag['ImageList'][img_index]['ImageTags']

    if get_parms:
        image_parms = unnest_parm_dicts(image_parms)
    else:
        image_parms = dict()

    return image, image_parms


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


def read_dm4(file_path, *args, **kwargs):
    """
    Read dm4 file

    Parameters
    ----------
    file_path : str
        Path to the file to be read

    Returns
    -------
    image_array : numpy.ndarray
        Image data from the file located at `file_path`
    file_parms : dict
        Dictionary of parameters read from the dm4 file

    """
    get_parms = kwargs.pop('get_parms', True)
    header = kwargs.pop('header', None)

    file_parms = dict()
    dm4_file = dm4reader.DM4File.open(file_path)
    if header is None:
        tags = dm4_file.read_directory()
        header = tags.named_subdirs['ImageList'].dm4_tag
        image_list = tags.named_subdirs['ImageList'].unnamed_subdirs
    else:
        dm4_file.hfile.seek(header.offset)
        image_list = dm4_file.read_directory(header)

    for image_dir in image_list:
        image_data_tag = image_dir.named_subdirs['ImageData']
        image_tag = image_data_tag.named_tags['Data']

        x_dim = dm4_file.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
        y_dim = dm4_file.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])

        image_array = np.array(dm4_file.read_tag_data(image_tag), dtype=np.float32)
        image_array = np.reshape(image_array, (y_dim, x_dim))

    if get_parms:
        file_parms = parse_dm4_parms(dm4_file, tags, '')
        file_parms['Image_Tag'] = header

    return image_array, file_parms


def parse_dm4_parms(dm4_file, tag_dir, base_name=''):
    """
    Recursive function to trace the dictionary tree of the Image Data
    and build a single dictionary of all parameters

    Parameters
    ----------
    dm4_file : DM4File
        File object of the dm4 file to be parsed.

    tag_dir : dict
        Dictionary to be traced.  Has the following attributes:
            tag_dir.name : str
                Name of the directory
            tag_dir.dm4_tag : str
                Contents of the directory

    base_name : str
        Base name of parameters.  Tag and subdirectory names will be appended
        for named tags and subdirectories.  Unnamed ones will recieve a number.
        Default ''.  'Root' is automatically prepended to the name.

    Returns
    -------
    parm_dict : dict()
        Dictionary containing the name:value pairs of all parameters `dm4_file`

    """
    parm_dict = dict()

    '''
    Loop over named tags
    '''
    for name in tag_dir.named_tags.keys():
        '''
        Skip Data tags.  These will be handled elseware.
        '''
        if name == 'Data':
            continue
        tag_name = '_'.join([base_name, name.replace(' ', '_')])
        if base_name == '':
            tag_name = 'Root' + tag_name
        tag_data = dm4_file.read_tag_data(tag_dir.named_tags[name])

        '''
        See if we can convert the array into a string
        '''
        tag_data = try_tag_to_string(tag_data)
        parm_dict[tag_name] = tag_data

    '''
    Loop over unnamed tags
    '''
    for itag, tag in enumerate(tag_dir.unnamed_tags):
        tag_name = '_'.join([base_name, 'Tag_{:03d}'.format(itag)])
        if base_name == '':
            tag_name = 'Root' + tag_name

        tag_data = dm4_file.read_tag_data(tag)

        '''
        See if we can convert the array into a string
        '''
        tag_data = try_tag_to_string(tag_data)
        parm_dict[tag_name] = tag_data

    '''
    Loop over named subdirectories
    '''
    for name in tag_dir.named_subdirs.keys():
        dir_name = '_'.join([base_name, name.replace(' ', '_')])
        sub_dir = tag_dir.named_subdirs[name]
        if base_name == '':
            dir_name = 'Root' + dir_name
        sub_parms = parse_dm4_parms(dm4_file, sub_dir, dir_name)
        parm_dict.update(sub_parms)

    '''
    Loop over unnamed subdirectories
    '''
    for idir, sub_dir in enumerate(tag_dir.unnamed_subdirs):
        dir_name = '_'.join([base_name, 'SubDir_{:03d}'.format(idir)])
        if base_name == '':
            dir_name = 'Root' + dir_name
        sub_parms = parse_dm4_parms(dm4_file, sub_dir, dir_name)
        parm_dict.update(sub_parms)

    return parm_dict


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
