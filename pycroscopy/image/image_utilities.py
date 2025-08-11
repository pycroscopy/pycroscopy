"""
image_utilities part of image package of pycroscopy

"""
import numpy as np
import scipy
from skimage.restoration import inpaint

import sidpy

def crop_image(dataset: sidpy.Dataset, corners: np.ndarray) -> sidpy.Dataset:
    """
    Crops an image according to the corners given in the format of 
    matplotlib.widget.RectangleSelector.

    Parameters
    ----------
    dataset: sidpy.Dataset
        An instance of sidpy.Dataset representing the image to be cropped.
    corners: np.ndarray
        A 1D array of length 4 containing the corners of the rectangular region 
        to be cropped. The order of the corners should be (x1, y1, x2, y2).

    Returns
    -------
    sidpy.Dataset
        A new instance of sidpy.Dataset representing the cropped image.

    Raises
    ------
    ValueError
        If dataset is not an instance of sidpy.Dataset or if dataset is not an image 
        dataset. If corners parameter is not of correct shape or size.

    """
    if not isinstance(dataset, sidpy.Dataset):
        raise ValueError('Input dataset is not an instance of sidpy.Dataset')
    if not dataset.data_type.name == 'IMAGE':
        raise ValueError('Only image datasets are supported at this point')

    if corners.shape != (4,):
        raise ValueError(f'Input corners parameter should have shape (4,) but got shape {corners.shape}')
    if corners[2]-corners[0] <= 0 or corners[3]-corners[1] <= 0:
        raise ValueError('Invalid input corners parameter')

    pixel_size = np.array([dataset.x[1]-dataset.x[0], dataset.y[1]-dataset.y[0]])
    corners /= pixel_size

    selection = np.stack([np.min(corners[:2])+0.5, np.max(corners[2:])+0.5]).astype(int)

    cropped_dset = dataset.like_data(dataset[selection[0, 0]:selection[1, 0],
                                             selection[0, 1]:selection[1, 1]])
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
        num_pts = sid_dset.shape[0]  # this is hard coded, it shouldn't be
    elif flatten_axis == 'col':
        num_pts = sid_dset.shape[1]  # this is hard coded, but it shouldn't be
    else:
        raise ValueError(f"Gave flatten axis of {flatten_axis} but only 'row', 'col' are allowed")

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
        raise ValueError("Gave method of {method} but only 'line_fit', 'plane_fit' are allowed")

    new_sid_dset[:] = data_flat


def rebin(im, binning=2):
    """
    rebin an image by the number of pixels in x and y direction given by binning

    Parameter
    ---------
    image: numpy array in 2 dimensions or sidpy.Dataset of data_type 'Image'

    Returns
    -------
    binned image as numpy array or sidpy.Dataset
    """
    if len(im.shape) == 2:
        rebinned_image = np.array(im).reshape((im.shape[0]//binning,
                                               binning, im.shape[1]//binning,
                                               binning)).mean(axis=3).mean(1)
        if isinstance(im, sidpy.Dataset):
            rebinned_image = im.like_data(rebinned_image)
            rebinned_image.title = 'rebinned_' + im.title
            rebinned_image.data_type = 'image'
            im_dims = im.get_image_dims(return_axis=True)

            rebinned_image.set_dimension(0, sidpy.Dimension(np.arange(rebinned_image.shape[0])/im_dims[0].slope,
                                                            name='x', units=im_dims[0].units,
                                                            dimension_type=im_dims[0].dimension_type,
                                                            quantity=im_dims[0].quantity))
            rebinned_image.set_dimension(1, sidpy.Dimension(np.arange(rebinned_image.shape[1])/im_dims[1].slope,
                                                            name='y', units=im_dims[1].units,
                                                            dimension_type=im_dims[1].dimension_type,
                                                            quantity=im_dims[1].quantity))
            return rebinned_image
    else:
        raise TypeError('not a 2D image')


def cart2pol(points):
    """Cartesian to polar coordinate conversion

    Parameters
    ---------
    points: float or numpy array
        points to be converted (Nx2)

    Returns
    -------
    rho: float or numpy array
        distance
    phi: float or numpy array
        angle
    """

    rho = np.linalg.norm(points[:, 0:2], axis=1)
    phi = np.arctan2(points[:, 1], points[:, 0])

    return rho, phi


def pol2cart(rho, phi):
    """Polar to Cartesian coordinate conversion

    Parameters
    ----------
    rho: float or numpy array
        distance
    phi: float or numpy array
        angle

    Returns
    -------
    x: float or numpy array
        x coordinates of converted points(Nx2)
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def xy2polar(points, rounding=1e-3):
    """ Conversion from carthesian to polar coordinates

    the angles and distances are sorted by r and then phi
    The indices of this sort is also returned

    Parameters
    ----------
    points: numpy array
        number of points in axis 0 first two elements in axis 1 are x and y
    rounding: int
        optional rounding in significant digits

    Returns
    -------
    r, phi, sorted_indices
    """

    r, phi = cart2pol(points)

    r = (np.floor(r/rounding))*rounding  # Remove rounding error differences

    sorted_indices = np.lexsort((phi, r))  # sort first by r and then by phi
    r = r[sorted_indices]
    phi = phi[sorted_indices]

    return r, phi, sorted_indices


def cartesian2polar(x, y, grid, r, t, order=3):
    """Transform cartesian grid to polar grid

    Used by warp
    """

    rr, tt = np.meshgrid(r, t)

    new_x = rr*np.cos(tt)
    new_y = rr*np.sin(tt)

    ix = scipy.interpolate.interp1d(x, np.arange(len(x)))
    iy = scipy.interpolate.interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    return scipy.ndimage.map_coordinates(grid, np.array([new_ix, new_iy]),
                                         order=order).reshape(new_x.shape)


def warp(diff, center):
    """Takes a diffraction pattern (as a sidpy dataset)and warps it to a polar grid"""

    # Define original polar grid
    nx = np.shape(diff)[0]
    ny = np.shape(diff)[1]

    x = np.linspace(1, nx, nx, endpoint=True)-center[0]
    y = np.linspace(1, ny, ny, endpoint=True)-center[1]
    z = diff

    # Define new polar grid
    nr = int(min([center[0], center[1], diff.shape[0]-center[0], diff.shape[1]-center[1]])-1)
    nt = 360 * 3

    r = np.linspace(1, nr, nr)
    t = np.linspace(0., np.pi, nt, endpoint=False)

    return cartesian2polar(x, y, z, r, t, order=3).T


def inpaint_image(sid_dset, mask = None, channel = None):
    """Inpaints a sparse image, given a mask.

    Args:
        sid_dset (_type_): sidpy Dataset 
            with two dimensions being of spatial or reciprocal type
        mask (np.ndarry) : mask [0,1] same shape as sid_dset. 
            If providing a sidpy dataset and mask is in the metadata dict, 
            then this entry is optional
        channel (int): (optional) for multi-channel datasets, 
            provide the channel to in-paint
    """
    if len(sid_dset.shape)==2:
        image_data = np.array(sid_dset).squeeze()
    elif len(sid_dset.shape)==3:
        image_dims = []
        selection = []
        for dim, axis in sid_dset._axes.items():
            if axis.dimension_type in [sidpy.DimensionType.SPATIAL, sidpy.DimensionType.RECIPROCAL]:
                selection.append(slice(None))
                image_dims.append(dim)
            else:
                if channel is None:
                    channel=0
                selection.append(slice(channel, channel+1))

        image_data = np.array(sid_dset[tuple(selection)]).squeeze()
    if mask is None:
        mask_data = sid_dset.metadata["mask"]
        mask = np.copy(mask_data)
        mask[mask==1] = -1
        mask[mask==0] = 1
        mask[mask==-1] = 0

    inpainted_data = inpaint.inpaint_biharmonic(image_data, mask)

    #convert this into a sidpy dataset
    data_set = sidpy.Dataset.from_array(inpainted_data, name='inpainted_image')
    data_set.data_type = 'image'  # supported

    data_set.units = sid_dset.units
    data_set.quantity = sid_dset.quantity

    data_set.set_dimension(0, sid_dset.get_dimension_by_number(image_dims[0])[0])
    data_set.set_dimension(1, sid_dset.get_dimension_by_number(image_dims[1])[0])

    data_set.metadata["mask"] = mask
    return data_set
