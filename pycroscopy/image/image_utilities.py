import numpy as np
import sidpy


def crop_image(dataset, corners):
    """
    crops an image according to the corners given (in the format of matplotlib.widget.RectangleSelector)
    
    Parameters
    ----------
    dataset: sidpy.Dataset
        image to be selected from
    
    """
    if not isinstance(dataset, sidpy.Dataset):
        raise ValueError('Need a sidpy dataset')
    if not dataset.data_type.name == 'IMAGE':
        raise ValueError('Only Images are supported at this point')

    corners = np.array(corners)
    if corners.ndim <2 or corners.flatten().shape[0] < 4:
        raise ValueError('Not enough corners given')
    
    pixel_size =  np.array([[dataset.x[1]-dataset.x[0]]*corners.shape[1], [dataset.y[1]-dataset.y[0]]*corners.shape[1]])
    corners /= pixel_size
    
    selection =np.stack([np.min(corners, axis=1)+0.5,np.max(corners, axis=1)+0.5]).astype(int)

    selected_dset = dataset.like_data(dataset[selection[0,0]:selection[1,0],selection[0,1]:selection[1,1]])
    selected_dset.title = 'cropped_' + dataset.title
    selected_dset.source = dataset.title
    selected_dset.metadata ={'crop_dimension': selection, 'original_dimensions': dataset.shape}
    return selected_dset