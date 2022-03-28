"""
Voronoi analysis of atom positions

author Gerd and Rama

part of pycroscopy
"""

import numpy as np
import sys

from skimage.feature import blob_log
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import scipy.optimize as optimization

import sidpy
from tqdm import trange
import sidpy

get_slope = sidpy.base.num_utils.get_slope


def make_gauss(size_x, size_y, width=1.0, x0=0.0, y0=0.0, intensity=1.0):
    """Make a Gaussian shaped probe """
    size_x = size_x/2
    size_y = size_y/2
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = np.exp(-((x-x0)**2 + (y-y0)**2) / 2.0 / width**2)
    probe = g / g.sum() * intensity

    return probe


def find_atoms(image, atom_size=0.1, threshold=-1.):
    """ Find atoms is a simple wrapper for blob_log in skimage.feature

    Parameters
    ----------
    image: sidpy.Dataset
        the image to find atoms
    atom_size: float
        visible size of atom blob diameter in nm gives minimal distance between found blobs
    threshold: float
        threshold for blob finder; (usually between 0.001 and 1.0) for threshold <= 0 we use the RMS contrast

    Returns
    -------
    atoms: numpy array(Nx3)
        atoms positions and radius of blob
    """

    if not isinstance(image, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if image.data_type.name != 'IMAGE':
        raise TypeError('We need sidpy.Dataset of sidpy.Datatype: IMAGE')
    if not isinstance(atom_size, (float, int)):
        raise TypeError('atom_size parameter has to be a number')
    if not isinstance(threshold, float):
        raise TypeError('threshold parameter has to be a float number')


    scale_x = get_slope(image.dim_0)
    im = np.array(image-image.min())
    im = im/im.max()
    if threshold < 0.:
        threshold = np.std(im)
    atoms = blob_log(im, max_sigma=atom_size/scale_x, threshold=threshold)

    return atoms


def atoms_clustering(atoms, mid_atoms, number_of_clusters=3, nearest_neighbours=7):
    """ A wrapper for sklearn.cluster kmeans clustering of atoms.

    Parameters
    ----------
    atoms: list or np.array (Nx2)
        list of all atoms
    mid_atoms: list or np.array (Nx2)
        atoms to be evaluated
    number_of_clusters: int
        number of clusters to sort (ini=3)
    nearest_neighbours: int
        number of nearest neighbours evaluated

    Returns
    -------
    clusters, distances, indices: numpy arrays
    """

    # get distances
    nn_tree = cKDTree(np.array(atoms)[:, 0:2])

    distances, indices = nn_tree.query(np.array(mid_atoms)[:, 0:2], nearest_neighbours)

    # Clustering
    k_means = KMeans(n_clusters=number_of_clusters, random_state=0)  # Fixing the RNG in kmeans
    k_means.fit(distances)
    clusters = k_means.predict(distances)

    return clusters, distances, indices


def gauss_difference(params, area):
    """
    Difference between part of an image and a Gaussian
    This function is used in atom refine function of pycroscopy

    Parameters
    ----------
    params: list
        list of Gaussian parameters [width, position_x, position_y, intensity]
    area:  numpy array
        2D matrix = part of an image

    Returns
    -------
    numpy array: flattened array of difference

    """
    gauss = make_gauss(area.shape[0], area.shape[1], width=params[0], x0=params[1], y0=params[2], intensity=params[3])
    return (area - gauss).flatten()


def atom_refine(image, atoms, radius, max_int=0, min_int=0, max_dist=4):
    """Fits a Gaussian in a blob of an image

    Parameters
    ----------
    image: np.array or sidpy Dataset
    atoms: list or np.array
        positions of atoms
    radius: float
        radius of circular mask to define fitting of Gaussian
    max_int: float
        optional - maximum intensity to be considered for fitting (to exclude contaminated areas for example)
    min_int: float
        optional - minimum intensity to be considered for fitting (to exclude contaminated holes for example)
    max_dist: float
        optional - maximum distance of movement of Gaussian during fitting

    Returns
    -------
    sym: dict
        dictionary containing new atom positions and other output such as intensity of the fitted Gaussian
    """
    rr = int(radius + 0.5)  # atom radius
    print('using radius ', rr, 'pixels')

    pixels = np.linspace(0, 2 * rr, 2 * rr + 1) - rr
    x, y = np.meshgrid(pixels, pixels)
    mask = (x ** 2 + y ** 2) < rr ** 2

    guess = [rr * 2, 0.0, 0.0, 1]

    sym = {'number_of_atoms': len(atoms)}

    volume = []
    position = []
    intensities = []
    maximum_area = []
    new_atoms = []
    gauss_width = []
    gauss_amplitude = []
    gauss_intensity = []

    for i in trange(len(atoms)):
        x, y = atoms[i][0:2]
        x = int(x)
        y = int(y)

        area = image[x - rr:x + rr + 1, y - rr:y + rr + 1]

        append = False

        if (x - rr) < 0 or y - rr < 0 or x + rr + 1 > image.shape[0] or y + rr + 1 > image.shape[1]:
            position.append(-1)
            intensities.append(-1.)
            maximum_area.append(-1.)
        else:
            position.append(1)
            intensities.append((area * mask).sum())
            maximum_area.append((area * mask).max())

        if max_int > 0:
            if area.sum() < max_int:
                if area.sum() > min_int:
                    append = True
        elif area.sum() > min_int:
            append = True

        pout = [0, 0, 0, 0]
        if append:
            if (x - rr) < 0 or y - rr < 0 or x + rr + 1 > image.shape[0] or y + rr + 1 > image.shape[1]:
                pass
            else:
                [pout, _] = optimization.leastsq(gauss_difference, guess, args=area)

            if (abs(pout[1]) > max_dist) or (abs(pout[2]) > max_dist):
                pout = [0, 0, 0, 0]

        volume.append(2 * np.pi * pout[3] * pout[0] * pout[0])

        new_atoms.append([x + pout[1], y + pout[2]])  # ,pout[0],  volume)) #,pout[3]))
        if all(v == 0 for v in pout):
            gauss_intensity.append(0.)
        else:
            gauss = make_gauss(area.shape[0], area.shape[1], width=pout[0], x0=pout[1], y0=pout[2], intensity=pout[3])
            gauss_intensity.append((gauss * mask).sum())
        gauss_width.append(pout[0])
        gauss_amplitude.append(pout[3])

    sym['inside'] = position
    sym['intensity_area'] = intensities
    sym['maximum_area'] = maximum_area
    sym['atoms'] = new_atoms
    sym['gauss_width'] = gauss_width
    sym['gauss_amplitude'] = gauss_amplitude
    sym['gauss_intensity'] = gauss_intensity
    sym['gauss_volume'] = volume

    return sym


def intensity_area(image, atoms, radius):
    """
    integrated intensity of atoms in an image with a mask around each atom of radius radius
    """
    rr = int(radius + 0.5)  # atom radius
    print('using radius ', rr, 'pixels')

    pixels = np.linspace(0, 2 * rr, 2 * rr + 1) - rr
    x, y = np.meshgrid(pixels, pixels)
    mask = np.array((x ** 2 + y ** 2) < rr ** 2)
    intensities = []
    for i in range(len(atoms)):
        x = int(atoms[i][1])
        y = int(atoms[i][0])
        area = image[x - rr:x + rr + 1, y - rr:y + rr + 1]
        if area.shape == mask.shape:
            intensities.append((area * mask).sum())
        else:
            intensities.append(-1)
    return intensities
