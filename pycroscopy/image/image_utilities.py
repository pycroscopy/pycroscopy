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
