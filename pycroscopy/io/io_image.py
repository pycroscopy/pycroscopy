"""
Created on Nov 8, 2016

@author: Chris Smith -- csmith55@utk.edu
"""
import os
import numpy as np
import array
from skimage.io import imread
from .dm3_image_utils import parse_dm_header, imagedatadict_to_ndarray
from . import dm4reader

def read_image(image_path, *args, **kwargs):
    """
    Read the image file at `image_path` into a numpy array

    Parameters
    ----------
    image_path : str
        Path to the image file
    greyscale : Boolean, optional
        Should the image be converted to greyscale after reading.
        Default True

    Returns
    -------
    image : numpy.ndarray
        Array containing the image from the file `image_path`
    """
    ext = os.path.splitext(image_path)[1]
    if ext == '.dm3':
        try:
            image, extra = read_dm3(image_path, *args, **kwargs)
            return image, extra
        except:
            raise
    elif ext == '.dm4':
        try:
            image, extra = read_dm4(image_path, *args, **kwargs)
            return image, extra
        except:
            raise
    else:
        try:
            image = imread(image_path, *args, **kwargs)
            return image, dict()
        except:
            raise


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
        image_parms = _parse_dm3_parms(image_parms)
    else:
        image_parms = dict()

    return image, image_parms

def _parse_dm3_parms(image_parms, prefix=''):
    """
    Parses the nested image parameter dictionary and converts it to a single
    level dictionary, prepending the name of inner dictionaries to their
    keys to denote level.

    Parameters
    ----------
    image_parms : dict

    Returns
    -------

    """
    new_parms = dict()
    for name, val in image_parms.iteritems():
        # print 'name',name,'val',val
        name = '-'.join([prefix]+name.split()).strip('-')
        if isinstance(val, dict):
            new_parms.update(_parse_dm3_parms(val, name))
        elif isinstance(val, list) and isinstance(val[0], dict):
            for thing in val:
                new_parms.update(_parse_dm3_parms(thing, name))
        else:
            new_parms[name] = try_tag_to_string(val)

    return new_parms

def read_dm4(file_path, get_parms=True):
    """
    Read dm4 file

    :param file_path:
    :param get_parms:
    :return:
    """
    dm4_file = dm4reader.DM4File.open(file_path)
    tags = dm4_file.read_directory()
    image_list = tags.named_subdirs['ImageList'].unnamed_subdirs
    for image_dir in image_list:
        image_parm_dict = dict()
        image_data_tag = image_dir.named_subdirs['ImageData']
        image_tag = image_data_tag.named_tags['Data']
        image_tags_dir = image_dir.named_subdirs['ImageTags']

        XDim = dm4_file.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
        YDim = dm4_file.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])

        image_array = np.array(dm4_file.read_tag_data(image_tag), dtype=np.uint16)
        image_array = np.reshape(image_array, (YDim, XDim))

    if get_parms:
        file_parms = parse_dm4_parms(dm4_file, tags, '')

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
            tag_name = 'Root'+tag_name
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
            tag_name = 'Root'+tag_name

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
            dir_name = 'Root'+dir_name
        sub_parms = parse_dm4_parms(dm4_file, sub_dir, dir_name)
        parm_dict.update(sub_parms)

    '''
    Loop over unnamed subdirectories
    '''
    for idir, sub_dir in enumerate(tag_dir.unnamed_subdirs):
        dir_name = '_'.join([base_name, 'SubDir_{:03d}'.format(idir)])
        if base_name == '':
            dir_name = 'Root'+dir_name
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