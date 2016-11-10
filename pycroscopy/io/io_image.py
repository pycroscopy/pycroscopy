"""
Created on Nov 8, 2016

@author: Chris Smith -- csmith55@utk.edu
"""
import os
from skimage.io import imread
from .dm3_image_utils import parse_dm_header, imagedatadict_to_ndarray

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
    if os.path.split(image_path) == '.dm3':
        try:
            image, extra = read_dm3(image_path, *args, **kwargs)
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

    def parse_image_parms(image_parms, prefix=''):
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
            print 'name',name,'val',val
            name = '-'.join([prefix]+name.split()).strip('-')
            if isinstance(val, dict):
                new_parms.update(parse_image_parms(val, name))
            else:
                new_parms[name] = val

        return new_parms

    if get_parms:
        image_parms = parse_image_parms(image_parms)
    else:
        image_parms = dict()

    return image, image_parms
