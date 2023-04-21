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

def flatten_image(sid_dset, order=1, flatten_axis = 'row', method = 'line_fit'):
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
