import numpy as np

from . import dm4reader
from .dm3_image_utils import imagedatadict_to_ndarray
from .image_utils import try_tag_to_string, unnest_parm_dicts
from .parse_dm3 import parse_dm_header


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
