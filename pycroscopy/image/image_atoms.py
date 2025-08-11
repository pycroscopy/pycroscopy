""" Atom detection

All atom detection is done here
Everything is in unit of pixel!!

Author: Gerd Duscher

part of pycroscopy.image

the core pycroscopy package
"""
import numpy as np

import skimage
import sklearn
import scipy

import sidpy
from tqdm.auto import trange


def make_gauss(size_x: [int, float], size_y: [int, float], width: float = 1.0, x0: float = 0.0,
               y0: float = 0.0, intensity: float = 1.0) -> np.ndarray:
    """
    Generates a 2D Gaussian-shaped probe array.
    Parameters
    ----------
    size_x : int or float
        The size of the probe along the x-axis.
    size_y : int or float
        The size of the probe along the y-axis.
    width : float, optional
        The standard deviation (spread) of the Gaussian (default is 1.0).
    x0 : float, optional
        The x-coordinate of the Gaussian center (default is 0.0).
    y0 : float, optional
        The y-coordinate of the Gaussian center (default is 0.0).
    intensity : float, optional
        The total intensity (sum) of the probe (default is 1.0).
    Returns
    -------
    probe : numpy.ndarray
        A 2D array representing the normalized Gaussian probe.
    """
    size_x = size_x/2
    size_y = size_y/2
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = np.exp(-((x-x0)**2 + (y-y0)**2) / 2.0 / width**2)
    probe = g / g.sum() * intensity

    return probe


def find_atoms(image: sidpy.Dataset, atom_size: float = 0.1, threshold: float = 0.) -> np.ndarray:
    """ Find atoms is a simple wrapper for blob_log in skimage.feature

    threshold for blob finder is usually between 0.001 and 1.0, 
    note: for threshold <= 0 we use the RMS contrast

    Parameters
    ----------
    image: sidpy.Dataset
        the image to find atoms
    atom_size: float
        visible size of atom blob diameter in nm gives minimal distance between found blobs
    threshold: float
        threshold for blob finder; 

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

    scale_x = np.unique(np.gradient(image.dim_0.values))[0]
    im = np.array(image-image.min())
    im = im/im.max()
    if threshold <= 0.:
        threshold = np.std(im)
    atoms = skimage.feature.blob_log(im, max_sigma=atom_size/scale_x, threshold=threshold)

    return atoms


def atoms_clustering(atoms: np.ndarray, mid_atoms: np.ndarray,
                     number_of_clusters: int = 3, nearest_neighbours: int = 7) -> tuple:
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
    nn_tree = scipy.spatial.KDTree(np.array(atoms)[:, 0:2])

    distances, indices = nn_tree.query(np.array(mid_atoms)[:, 0:2], nearest_neighbours)

    # Clustering
    k_means = sklearn.cluster.KMeans(n_clusters=number_of_clusters, random_state=0)
    k_means.fit(distances)
    clusters = k_means.predict(distances)

    return clusters, distances, indices


def gauss_difference(params: list[float], area: np.ndarray) -> np.ndarray:
    """
    Difference between part of an image and a Gaussian
    This function is used int the atom refine function of pyTEMlib

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
    gauss = make_gauss(area.shape[0], area.shape[1], width=params[0], x0=params[1],
                       y0=params[2], intensity=params[3])
    return (area - gauss).flatten()


def atom_refine(image: [np.ndarray, sidpy.Dataset], atoms: [np.ndarray, list], radius: float,
                max_int: float = 0, min_int: float = 0, max_dist: float = 4) -> dict:
    """Fits a Gaussian in a blob of an image

    Parameters
    ----------
    image: np.array or sidpy Dataset
    atoms: list or np.array
        positions of atoms
    radius: float
        radius of circular mask to define fitting of Gaussian
    max_int: float
        optional - maximum intensity to be considered for fitting (to exclude contaminated areas)
    min_int: float
        optional - minimum intensity to be considered for fitting (to exclude contaminated holes )
    max_dist: float
        optional - maximum distance of movement of Gaussian during fitting

    Returns
    -------
    sym: dict
        dictionary containing new atom positions and other output such as intensity of Gaussian
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

        if x-rr<0 or y-rr < 0 or x+rr+1 > image.shape[0] or y+rr+1 > image.shape[1]:
            position.append(-1)
            intensities.append(-1.)
            maximum_area.append(-1.)
        else:  # atom found
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
            if x-rr < 0 or y-rr < 0 or x+rr+1 > image.shape[0] or y+rr+1 > image.shape[1]:
                pass
            else:
                [pout, _] = scipy.optimize.leastsq(gauss_difference, guess, args=area)

            if (abs(pout[1]) > max_dist) or (abs(pout[2]) > max_dist):
                pout = [0, 0, 0, 0]

        volume.append(2 * np.pi * pout[3] * pout[0] * pout[0])

        new_atoms.append([x + pout[1], y + pout[2]])  # ,pout[0],  volume)) #,pout[3]))
        if all(v == 0 for v in pout):
            gauss_intensity.append(0.)
        else:
            gauss = make_gauss(area.shape[0], area.shape[1], width=pout[0], x0=pout[1], y0=pout[2],
                                           intensity=pout[3])
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


def intensity_area(image: np.ndarray, atoms: np.ndarray, radius: float) -> list[float]:
    """
    integrated intensity of atoms in an image with a mask around each atom of radius radius
    """
    rr = int(radius + 0.5)  # atom radius
    print('using radius ', rr, 'pixels')

    pixels = np.linspace(0, 2 * rr, 2 * rr + 1) - rr
    x, y = np.meshgrid(pixels, pixels)
    mask = np.array((x ** 2 + y ** 2) < rr ** 2)
    intensities = []
    for atom in atoms:
        x = int(atom[1])
        y = int(atom[0])
        area = image[x - rr:x + rr + 1, y - rr:y + rr + 1]
        if area.shape == mask.shape:
            intensities.append((area * mask).sum())
        else:
            intensities.append(-1)
    return intensities
