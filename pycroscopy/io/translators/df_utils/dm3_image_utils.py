# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 07:55:56 2016

@author: Chris Smith
"""
# ParseDM3File reads in a DM3 file and translates it into a dictionary
# this module treats that dictionary as an image-file and extracts the
# appropriate image data as numpy arrays.
# It also tries to create files from numpy arrays that DM can read.
#
# Some notes:
# Only complex64 and complex128 types are converted to structarrays,
# ie they're arrays of structs. Everything else, (including RGB) are
# standard arrays.
# There is a seperate DatatType and PixelDepth stored for images different
# from the tag file datatype. I think these are used more than the tag
# datratypes in describing the data.

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

from .parse_dm3 import *

structarray_to_np_map = {
    ('d', 'd'): np.complex128,
    ('f', 'f'): np.complex64}

np_to_structarray_map = {v: k for k, v in structarray_to_np_map.items()}

# we want to amp any image type to a single np array type
# but a sinlge np array type could map to more than one dm type.
# For the moment, we won't be strict about, eg, discriminating
# int8 from bool, or even unit32 from RGB. In the future we could
# convert np bool type eg to DM bool and treat y,x,3 int8 images
# as RGB.

# note uint8 here returns the same data type as int8 - could be that the
# only way they're differentiated is via this type, not the raw type
# in the tag file? Might be same for PackedComplex images too
# And 8 is missing!
dm_image_dtypes = {
    1: ("int16", np.int16),
    2: ("float32", np.float32),
    3: ("Complex64", np.complex64),
    6: ("uint8", np.int8),
    7: ("int32", np.int32),
    9: ("int8", np.int8),
    10: ("uint16", np.uint16),
    11: ("uint32", np.uint32),
    12: ("float64", np.float64),
    13: ("Complex128", np.complex128),
    14: ("Bool", np.int8),
    23: ("RGB", np.int32),
    27: ("PackedComplex64", np.float32),
    28: ("PackedComplex128", np.float64)
}


def imagedatadict_to_ndarray(imdict):
    """
    Converts the ImageData dictionary, imdict, to an nd image.
    """
    arr = imdict['Data']
    im = None
    if isinstance(arr, array.array):
        im = np.asarray(arr, dtype=arr.typecode)
    elif isinstance(arr, structarray):
        t = tuple(arr.typecodes)
        im = np.frombuffer(
            arr.raw_data,
            dtype=structarray_to_np_map[t])
    # print "Image has dmimagetype", imdict["DataType"], "numpy type is", im.dtype
    assert dm_image_dtypes[imdict["DataType"]][1] == im.dtype
    assert imdict['PixelDepth'] == im.dtype.itemsize
    return im.reshape(imdict['Dimensions'][::-1])


def ndarray_to_imagedatadict(nparr):
    """
    Convert the numpy array nparr into a suitable ImageList entry dictionary.
    Returns a dictionary with the appropriate Data, DataType, PixelDepth
    to be inserted into a dm3 tag dictionary and written to a file.
    """
    ret = {}
    dm_type = (k for k, v in dm_image_dtypes.items() if v[1] == nparr.dtype.type).next()
    ret["DataType"] = dm_type
    ret["PixelDepth"] = nparr.dtype.itemsize
    ret["Dimensions"] = list(nparr.shape[::-1])
    if nparr.dtype.type in np_to_structarray_map:
        types = np_to_structarray_map[nparr.dtype.type]
        ret["Data"] = structarray(types)
        ret["Data"].raw_data = str(nparr.data)
    else:
        ret["Data"] = array.array(nparr.dtype.char, nparr.flatten())
    return ret


def load_image(file):
    """
    Loads the image from the file-like object or string file.
    If file is a string, the file is opened and then read.
    Returns a numpy ndarray of our best guess for the most important image
    in the file.
    """
    if isinstance(file, str):
        with open(file, "rb") as f:
            return load_image(f)
    dmtag = parse_dm_header(file)
    img_index = -1
    return imagedatadict_to_ndarray(dmtag['ImageList'][img_index]['ImageData'])


def save_image(image, file):
    """
    Saves the nparray image to the file-like object (or string) file.
    If file is a string the file is created and written to
    """
    if isinstance(file, str):
        with open(file, "wb") as f:
            return save_image(image, f)
    # we need to create a basic DM tree suitable for an imge
    # we'll try the minimum: just an image list
    # doesn't work. Do we need a ImageSourceList too?
    # and a DocumentObjectList?
    image = ndarray_to_imagedatadict(image)
    ret = dict()
    ret["ImageList"] = [{"ImageData": image}]
    # I think ImageSource list creates a mapping between ImageSourceIds and Images
    ret["ImageSourceList"] = [{"ClassName": "ImageSource:Simple", "Id": [0], "ImageRef": 0}]
    # I think this lists the sources for the DocumentObjectlist. The source number is not
    # the indxe in the imagelist but is either the index in the ImageSourceList or the Id
    # from that list. We also need to set the annotation type to identify it as an image
    ret["DocumentObjectList"] = [{"ImageSource": 0, "AnnotationType": 20}]
    # finally some display options
    ret["Image Behavior"] = {"ViewDisplayID": 8}
    ret["InImageMode"] = 1
    parse_dm_header(file, ret)
