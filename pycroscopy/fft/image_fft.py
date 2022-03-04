"""
Image_fft
part of pycroscopy

author: Gerd Duscher, UTK

"""

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max, blob_log
import sidpy
from pyTEMlib.file_tools import get_slope


def power_spectrum(dset, smoothing=3):
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
    assert isinstance(dset, sidpy.Dataset)
    smoothing = int(smoothing)
    fft_transform = dset.fft()
    fft_mag = np.abs(fft_transform)
    fft_mag2 = ndimage.gaussian_filter(fft_mag, sigma=(smoothing, smoothing), order=0)

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


def diffractogram_spots(dset, spot_threshold):
    """Find spots in diffractogram and sort them by distance from center

    Uses blob_log from scipy.spatial

    Parameters
    ----------
    dset: sidpy.Dataset
        diffractogram
    spot_threshold: float
        threshold for blob finder

    Returns
    -------
    spots: numpy array
        sorted position (x,y) and radius (r) of all spots
    """

    # spot detection (for future reference there is no symmetry assumed here)
    data = np.array(np.log(1+np.abs(dset)))
    data = (data - data.min())
    data = data/data.max()
    # some images are strange and blob_log does not work on the power spectrum
    try:
        spots_random = blob_log(data, max_sigma=5, threshold=spot_threshold)
    except ValueError:
        spots_random = peak_local_max(np.array(data.T), min_distance=3, threshold_rel=spot_threshold)
        spots_random = np.hstack(spots_random, np.zeros((spots_random.shape[0], 1)))

    print(f'Found {spots_random.shape[0]} reflections')

    # Needed for conversion from pixel to Reciprocal space
    rec_scale = np.array([ft.get_slope(dset.u.values), ft.get_slope(dset.v.values)])
    spots_random[:, :2] = spots_random[:, :2]*rec_scale+[dset.u.values[0], dset.v.values[0]]
    # sort reflections
    spots_random[:, 2] = np.linalg.norm(spots_random[:, 0:2], axis=1)
    spots_index = np.argsort(spots_random[:, 2])
    spots = spots_random[spots_index]
    # third row is angles
    spots[:, 2] = np.arctan2(spots[:, 0], spots[:, 1])
    return spots
