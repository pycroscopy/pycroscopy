"""
Image_fft
part of pycroscopy

author: Gerd Duscher, UTK

"""

import numpy as np
import scipy
import itertools
import sidpy
import sklearn


def fourier_transform(dset: sidpy.Dataset) -> sidpy.Dataset:
    """
        Reads information into dictionary 'tags', performs 'FFT', and provides a smoothed FT and reciprocal
        and intensity limits for visualization.

        Parameters
        ----------
        dset: sidpy.Dataset
            image

        Returns
        -------
        fft_dset: sidpy.Dataset
            Fourier transform with correct dimensions

        Example
        -------
        >>> fft_dataset = fourier_transform(sidpy_dataset)
        >>> fft_dataset.plot()
    """

    assert isinstance(dset, sidpy.Dataset), 'Expected a sidpy Dataset'

    selection = []
    image_dims = dset.get_image_dims(return_axis=True)
    if dset.data_type.name == 'IMAGE_STACK':
        stack_dim = dset.get_dimensions_by_type('TEMPORAL')

        if len(image_dims) != 2:
            raise ValueError('need at least two SPATIAL dimension for an image stack')

        for i in range(dset.ndim):
            if i in image_dims:
                selection.append(slice(None))
            if len(stack_dim) == 0:
                stack_dims = i
                selection.append(slice(None))
            elif i in stack_dim:
                stack_dims = i
                selection.append(slice(None))
            else:
                selection.append(slice(0, 1))

        image_stack = np.squeeze(np.array(dset)[selection])
        new_image = np.sum(np.array(image_stack), axis=stack_dim)
    elif dset.data_type.name == 'IMAGE':
        new_image = np.array(dset)
    else:
        return

    new_image = new_image - new_image.min()
    
    fft_transform = (np.fft.fftshift(np.fft.fft2(np.array(new_image))))

    image_dims = dset.get_image_dims(return_axis=True)

    units_x = '1/' + image_dims[0].units
    units_y = '1/' + image_dims[1].units

    fft_dset = sidpy.Dataset.from_array(fft_transform)
    fft_dset.quantity = dset.quantity
    fft_dset.units = 'a.u.'
    fft_dset.data_type = 'IMAGE'
    fft_dset.source = dset.title
    fft_dset.modality = 'fft'

    fft_dset.set_dimension(0, sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(new_image.shape[0],
                                                                             d=dset.x[1]-dset.x[0])),
                                              name='u', units=units_x, dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))
    fft_dset.set_dimension(1, sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(new_image.shape[1],
                                                                             d=dset.y[1]- dset.y[0])),
                                              name='v', units=units_y, dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))
    return fft_dset


def power_spectrum(dset: sidpy.Dataset, smoothing: int=3) -> sidpy.Dataset:
    """
    Calculate power spectrum

    Parameters
    ----------
    dset: sidpy.Dataset
        image
    smoothing: int
        Gaussian smoothing

    Returns
    -------
    power_spec: sidpy.Dataset
        power spectrum with correct dimensions

    """

    fft_transform = fourier_transform(dset)  # dset.fft()
    fft_mag = np.abs(fft_transform)
    fft_mag2 = scipy.ndimage.gaussian_filter(fft_mag, sigma=(smoothing, smoothing), order=0)

    power_spec = fft_transform.like_data(np.log(1.+fft_mag2))

    # prepare mask
    x, y = np.meshgrid(power_spec.v.values, power_spec.u.values)
    mask = np.zeros(power_spec.shape)

    mask_spot = x ** 2 + y ** 2 > 1 ** 2
    mask = mask + mask_spot
    mask_spot = x ** 2 + y ** 2 < 11 ** 2
    mask = mask + mask_spot

    mask[np.where(mask == 1)] = 0  # just in case of overlapping disks

    minimum_intensity = np.array(power_spec)[np.where(mask == 2)].min() * 0.95
    maximum_intensity = np.array(power_spec)[np.where(mask == 2)].max() * 1.05
    power_spec.metadata = {'fft': {'smoothing': smoothing,
                                   'minimum_intensity': minimum_intensity, 'maximum_intensity': maximum_intensity}}
    power_spec.title = 'power spectrum ' + power_spec.source

    return power_spec


def diffractogram_spots(dset: sidpy.Dataset,
                        spot_threshold: float,
                        return_center: bool = True,
                        eps: float=0.1) -> tuple[np.ndarray, np.ndarray | list[float]]:
    """Find spots in diffractogram and sort them by distance from center

    Uses blob_log from scipy.spatial

    Parameters
    ----------
    dset: sidpy.Dataset
        diffractogram
    spot_threshold: float
        threshold for blob finder
    return_center: bool, optional
        return center of image if true
    eps: float, optional
        threshold for blob finder

    Returns
    -------
    spots: numpy array
        sorted position (x,y) and radius (r) of all spots
    """

    # spot detection (for future reference there is no symmetry assumed here)
    data = np.array(np.log(1+np.abs(dset)))
    data = data - data.min()
    data = data/data.max()
    # some images are strange and blob_log does not work on the power spectrum
    try:
        spots_random = scipy.features.blob_log(data, max_sigma=5, threshold=spot_threshold)
    except ValueError:
        spots_random = scipy.features.peak_local_max(np.array(data.T), min_distance=3, threshold_rel=spot_threshold)
        spots_random = np.hstack(spots_random, np.zeros((spots_random.shape[0], 1)))
            
    print(f'Found {spots_random.shape[0]} reflections')

    # Needed for conversion from pixel to Reciprocal space
    image_dims = dset.get_image_dims(return_axis=True)
    rec_scale = np.array([image_dims[0].slope, image_dims[1].slope])
    
    spots_random[:, :2] = spots_random[:, :2]*rec_scale+[dset.u.values[0], dset.v.values[0]]
    # sort reflections
    spots_random[:, 2] = np.linalg.norm(spots_random[:, 0:2], axis=1)
    spots_index = np.argsort(spots_random[:, 2])
    spots = spots_random[spots_index]
    # third row is angles
    spots[:, 2] = np.arctan2(spots[:, 0], spots[:, 1])

    center = [0, 0]

    if return_center:
        points = spots[:, 0:2]

        # Calculate the midpoints between all points
        reshaped_points = points[:, np.newaxis, :]
        midpoints = (reshaped_points + reshaped_points.transpose(1, 0, 2)) / 2.0
        midpoints = midpoints.reshape(-1, 2)

        # Find the most dense cluster of midpoints
        dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(midpoints)
        cluster_counter = Counter(labels)
        largest_cluster_label = max(cluster_counter, key=cluster_counter.get)
        largest_cluster_points = midpoints[labels == largest_cluster_label]

        # Average of these midpoints must be the center
        center = np.mean(largest_cluster_points, axis=0)

    return spots, center


def adaptive_fourier_filter(dset: sidpy.Dataset,
                            spots: np.ndarray,
                            low_pass: float = 3,
                            reflection_radius: float = 0.3) -> sidpy.Dataset:
    """
    Use spots in diffractogram for a Fourier Filter

    Parameters:
    -----------
    dset: sidpy.Dataset
        image to be filtered
    spots: np.ndarray(N,2)
        sorted spots in diffractogram in 1/nm
    low_pass:  float
        low pass filter in center of diffractogram in 1/nm
    reflection_radius:  float
        radius of masked reflections in 1/nm

    Output:
    -------
            Fourier filtered image
    """

    fft_transform = fourier_transform(dset)

    # prepare mask
    x, y = np.meshgrid(fft_transform.v.values, fft_transform.u.values)
    mask = np.zeros(dset.shape)

    # mask reflections
    for spot in spots:
        mask_spot = (x - spot[1]) ** 2 + (y - spot[0]) ** 2 < reflection_radius ** 2  # make a spot
        mask = mask + mask_spot  # add spot to mask

    # mask zero region larger (low-pass filter = intensity variations)
    mask_spot = x ** 2 + y ** 2 < low_pass ** 2
    mask = mask + mask_spot
    mask[np.where(mask > 1)] = 1
    fft_filtered = np.array(fft_transform * mask)

    filtered_image = dset.like_data(np.fft.ifft2(np.fft.fftshift(fft_filtered)).real)
    filtered_image.title = 'Fourier filtered ' + dset.title
    filtered_image.source = dset.title
    filtered_image.metadata = {'analysis': 'adaptive fourier filtered', 'spots': spots,
                               'low_pass': low_pass, 'reflection_radius': reflection_radius}
    return filtered_image


def rotational_symmetry_diffractogram(spots: np.ndarray) -> list[int]:
    """ Test rotational symmetry of diffraction spots"""

    rotation_symmetry = []
    for n in [2, 3, 4, 6]:
        cc = np.array(
            [[np.cos(2 * np.pi / n), np.sin(2 * np.pi / n), 0], [-np.sin(2 * np.pi / n), np.cos(2 * np.pi / n), 0],
             [0, 0, 1]])
        sym_spots = np.dot(spots, cc)
        dif = []
        for p0, p1 in itertools.product(sym_spots[:, 0:2], spots[:, 0:2]):
            dif.append(np.linalg.norm(p0 - p1))
        dif = np.array(sorted(dif))

        if dif[int(spots.shape[0] * .7)] < 0.2:
            rotation_symmetry.append(n)
    return rotation_symmetry
