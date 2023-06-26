import numpy as np
import sidpy

def crop_image(dataset: sidpy.Dataset, corners: np.ndarray) -> sidpy.Dataset:
    """
    Crops an image according to the corners given in the format of matplotlib.widget.RectangleSelector.

    Parameters
    ----------
    dataset: sidpy.Dataset
        An instance of sidpy.Dataset representing the image to be cropped.
    corners: np.ndarray
        A 1D array of length 4 containing the corners of the rectangular region to be cropped.
        The order of the corners should be (x1, y1, x2, y2).

    Returns
    -------
    sidpy.Dataset
        A new instance of sidpy.Dataset representing the cropped image.

    Raises
    ------
    ValueError
        If dataset is not an instance of sidpy.Dataset or if dataset is not an image dataset.
        If corners parameter is not of correct shape or size.

    """
    if not isinstance(dataset, sidpy.Dataset):
        raise ValueError('Input dataset is not an instance of sidpy.Dataset')
    if not dataset.data_type.name == 'IMAGE':
        raise ValueError('Only image datasets are supported at this point')
    
    if corners.shape != (4,):
        raise ValueError('Input corners parameter should have shape (4,) but got shape {0}'.format(corners.shape))
    if corners[2]-corners[0] <= 0 or corners[3]-corners[1] <= 0:
        raise ValueError('Invalid input corners parameter')
    
    pixel_size = np.array([dataset.x[1]-dataset.x[0], dataset.y[1]-dataset.y[0]])
    corners /= pixel_size
    
    selection = np.stack([np.min(corners[:2])+0.5, np.max(corners[2:])+0.5]).astype(int)

    cropped_dset = dataset.like_data(dataset[selection[0, 0]:selection[1, 0], selection[0, 1]:selection[1, 1]])
    cropped_dset.title = 'cropped_' + dataset.title
    cropped_dset.source = dataset.title
    cropped_dset.metadata = {'crop_dimension': selection, 'original_dimensions': dataset.shape}
    
    return cropped_dset

def flatten_image(sid_dset, order=1, flatten_axis = 'row', method = 'line_fit'):
    """
    Flattens an image according to the method chosen. Used heavily for AFM/STM images

    Parameters
    ----------
    dataset: sidpy.Dataset
        An instance of sidpy.Dataset representing the image to be flattened.
    order: integer, 
        Optional, default = 1. Ordfor the polynomial fit.
    flatten_axis: string, 
        Optional, default = 'row'. Axis along which to flatten the image.
    method: string, 
        Optional, default = 'line_fit'. Method to use for flattening the image.

    Returns
    -------
    sidpy.Dataset
        A new instance of sidpy.Dataset representing the flattened image.
    """
    #TODO: lots of cleanup in this function required...
    new_sid_dset = sid_dset.copy()
    assert len(new_sid_dset._axes) == 2, "Dataset must be 2-D for this function"
    assert new_sid_dset.data_type == sidpy.DataType.IMAGE, "Dataset must IMAGE for this function"
    #check the spatial dimensions, flatten along each row
    if flatten_axis == 'row':
        num_pts = sid_dset.shape[0] #this is hard coded, it shouldn't be
    elif flatten_axis == 'col':
        num_pts = sid_dset.shape[1] #this is hard coded, but it shouldn't be
    else:
        raise ValueError("Gave flatten axis of {} but only 'row', 'col' are allowed".format(flatten_axis))
    
    data_flat = np.zeros(sid_dset.shape) #again this should be the spatial (2 dimensional) part only
    print(sid_dset.shape, num_pts)
    if method == 'line_fit':
        for line in range(num_pts):
            if flatten_axis=='row':
                line_data = np.array(sid_dset[:])[line,:]
            elif flatten_axis=='col':
                line_data = np.array(sid_dset[:])[:,line]
            p = np.polyfit(np.arange(len(line_data)), line_data,order)
            lin_est = np.polyval(p,np.arange(len(line_data)))
            new_line = line_data - lin_est
            data_flat[line] = new_line
    elif method == 'plane_fit':
        #TODO: implement plane fit
        pass
    else:
        raise ValueError("Gave method of {} but only 'line_fit', 'plane_fit' are allowed".format(method))
   
    new_sid_dset[:] = data_flat 
    
    return new_sid_dset