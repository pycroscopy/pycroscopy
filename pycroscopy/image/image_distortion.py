"""
image_distortion part of pycroscopy

Author: Gerd Duscher

Distortions of scanned images are determined by comparing experimentally 
obtained unit cells with ideal ones.

"""
import numpy as np
import scipy
import skimage

import simpleitk

from tqdm.auto import trange

####################
# Distortion Matrix
####################
def get_distortion_matrix(atoms: np.ndarray, ideal_lattice: np.ndarray) -> np.ndarray:
    """    Calculates distortion matrix

    Calculates the distortion matrix by comparing ideal and distorted Voronoi tiles
    Parameters
    ----------
    atoms: numpy array (Nx2)
        atomic positions
    ideal_lattice: numpy array (Mx2)
        ideal lattice positions

    Returns
    -------
    numpy array (Nx4)
        distortion matrix
    """
    vor = scipy.spatial.Voronoi(atoms)
    # determine a middle Voronoi tile
    ideal_vor = scipy.spatial.Voronoi(ideal_lattice)
    near_center = np.average(ideal_lattice, axis=0)
    index = np.argmin(np.linalg.norm(ideal_lattice - near_center, axis=0))

    # the ideal vertices fo such an Voronoi tile (are there crystals with more than one voronoi?)
    ideal_vertices = ideal_vor.vertices[ideal_vor.regions[ideal_vor.point_region[index]]]
    ideal_vertices = get_significant_vertices(ideal_vertices - np.average(ideal_vertices, axis=0))

    distortion_matrix = []
    for index in trange(vor.points.shape[0]):

        # determine vertices of Voronoi polygons of an atom with number index
        poly_point = vor.points[index]
        vertices = vor.vertices[vor.regions[vor.point_region[index]]]
        poly_vertices = get_significant_vertices(vertices - poly_point)

        # where ATOM has to be moved (not pixel)
        ideal_point = ideal_lattice[index]

        # transform voronoi to ideal one and keep transformation matrix A
        uncorrected, corrected, _ = transform_voronoi(poly_vertices, ideal_vertices)

        # pixel positions
        corrected = corrected + ideal_point + (np.rint(poly_point) - poly_point)
        for i in range(len(corrected)):
            # original image pixels
            x, y = uncorrected[i] + np.rint(poly_point)
            # collect the two origin and target coordinates and store
            distortion_matrix.append([x, y, corrected[i, 0], corrected[i, 1]])
    print()
    return np.array(distortion_matrix)


def undistort(distortion_matrix, image_data):
    """ Undistort image according to distortion matrix
    
    Uses the griddata interpolation of scipy to apply distortion matrix to image.
    The distortion matrix contains in origin and target pixel coordinates
    target is where the pixel has to be moved (floats)
    
    Parameters
    ----------
    distortion_matrix: numpy array (Nx2)
        distortion matrix (format N x 2)
    image_data: numpy array or sidpy.Dataset
        image 
        
    Returns
    -------
    interpolated: numpy array
        undistorted image
    """

    intensity_values = image_data[(distortion_matrix[:, 0].astype(int),
                                   distortion_matrix[:, 1].astype(int))]

    corrected = distortion_matrix[:, 2:4]

    size_x, size_y = 2 ** np.round(np.log2(image_data.shape[0:2]))  # nearest power of 2
    size_x = int(size_x)
    size_y = int(size_y)
    grid_x, grid_y = np.mgrid[0:size_x - 1:size_x * 1j, 0:size_y - 1:size_y * 1j]
    print('interpolate')

    interpolated = scipy.interpolate.griddata(np.array(corrected),
                                              np.array(intensity_values),
                                              (grid_x, grid_y), method='linear')
    return interpolated


def transform_voronoi(vertices, ideal_voronoi):
    """ find transformation matrix A between a distorted polygon and a 
        perfect reference one

    Returns
    -------
    uncorrected: list of points: 
        all points on a grid within original polygon
    corrected: list of points: 
        coordinates of these points where pixel have to move to
    aa: 2x2 matrix A:  
        transformation matrix
    """
    # Find Transformation Matrix, note polygons have to be ordered first.
    sort_vert = []
    for vert in ideal_voronoi:
        sort_vert.append(np.argmin(np.linalg.norm(vertices - vert, axis=1)))
    vertices = np.array(vertices)[sort_vert]

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix aa = A
    aa, _, _, _ = np.linalg.lstsq(vertices, ideal_voronoi, rcond=None)

    # expand polygon to include more points in distortion matrix
    vertices2 = vertices + np.sign(vertices)  # +np.sign(vertices)

    ext_v = int(np.abs(vertices2).max() + 1)

    polygon_grid = np.mgrid[0:ext_v * 2 + 1, :ext_v * 2 + 1] - ext_v
    polygon_grid = np.swapaxes(polygon_grid, 0, 2)
    polygon_array = polygon_grid.reshape(-1, polygon_grid.shape[-1])

    p = skimage.measure.points_in_poly(polygon_array, vertices2)
    uncorrected = polygon_array[p]
    corrected = np.dot(uncorrected, aa)
    return uncorrected, corrected, aa


def get_maximum_view(distortion_matrix: np.ndarray):
    """
    Determines the largest rectangular view within a distorted image matrix 
    ----------
    distortion_matrix : np.ndarray
        A 3D numpy array representing the distortion matrix of the image, 
        where invalid pixels are marked with -1000.
    Returns
    -------
    np.ndarray
        A 1D numpy array of four integers [row_start, row_end, col_start, col_end] 
        representing the coordinates of the maximal valid view within the 
        distortion matrix.
    """
    distortion_matrix_extent = np.ones(distortion_matrix.shape[1:], dtype=int)
    distortion_matrix_extent[distortion_matrix[0] == -1000.] = 0

    area = distortion_matrix_extent
    view_square = np.array([0, distortion_matrix.shape[1] - 1, 0,
                            distortion_matrix.shape[2] - 1], dtype=int)
    while np.array(np.where(area == 0)).shape[1] > 0:
        view_square = view_square + [1, -1, 1, -1]
        area = distortion_matrix_extent[view_square[0]:view_square[1],
                                        view_square[2]:view_square[3]]

    change = [-int(np.sum(np.min(distortion_matrix_extent[:view_square[0], view_square[2]:view_square[3]], axis=1))),
              int(np.sum(np.min(distortion_matrix_extent[view_square[1]:, view_square[2]:view_square[3]], axis=1))),
              -int(np.sum(np.min(distortion_matrix_extent[view_square[0]:view_square[1], :view_square[2]], axis=0))),
              int(np.sum(np.min(distortion_matrix_extent[view_square[0]:view_square[1], view_square[3]:], axis=0)))]

    return np.array(view_square) + change


def get_significant_vertices(vertices, distance=3):
    """Calculate average for  all points that are closer than distance apart, 
    otherwise leave the points alone
        
        Parameters
        ----------
        vertices: numpy array (n,2)
            list of points
        distance: float
            (in same scale as points )
        
        Returns
        -------
        ideal_vertices: list of floats
            list of points that are all a minimum of 3 apart.
    """
    tt = scipy.spatial.KDTree(np.array(vertices))
    near = tt.query_ball_point(vertices, distance)
    ideal_vertices = []
    for indices in near:
        if len(indices) == 1:
            ideal_vertices.append(vertices[indices][0])
        else:
            ideal_vertices.append(np.average(vertices[indices], axis=0))
    ideal_vertices = np.unique(np.array(ideal_vertices), axis=0)
    angles = np.arctan2(ideal_vertices[:, 1], ideal_vertices[:, 0])
    ang_sort = np.argsort(angles)
    ideal_vertices = ideal_vertices[ang_sort]

    return ideal_vertices


def undistort_sitk(image_data, distortion_matrix):
    """    use simple ITK to undistort image
    
    Parameters
    ----------
    image_data: numpy array with size NxM 
    distortion_matrix: sidpy.Dataset or numpy array with size 2 x P x Q
    with P, Q >= M, N
    
    Returns
    -------
    image: numpy array MXN
      
    """
    resampler = simpleitk.ResampleImageFilter()
    resampler.SetReferenceImage(simpleitk.GetImageFromArray(image_data))
    resampler.SetInterpolator(simpleitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    distortion_matrix2 = distortion_matrix[:, :image_data.shape[0], :image_data.shape[1]]

    displ2 = simpleitk.Compose([simpleitk.GetImageFromArray(-distortion_matrix2[1]),
                           simpleitk.GetImageFromArray(-distortion_matrix2[0])])
    out_tx = simpleitk.DisplacementFieldTransform(displ2)
    resampler.SetTransform(out_tx)
    out = resampler.Execute(simpleitk.GetImageFromArray(image_data))
    return simpleitk.GetArrayFromImage(out)


def undistort_stack_sitk(distortion_matrix, image_stack):
    """
    use simple ITK to undistort stack of image
    input:
    image: numpy array with size NxM 
    distortion_matrix: h5 Dataset or numpy array with size 2 x P x Q
    with P, Q >= M, N
    output:
    image M, N
    """
    resampler = simpleitk.ResampleImageFilter()
    resampler.SetReferenceImage(simpleitk.GetImageFromArray(image_stack[0]))
    resampler.SetInterpolator(simpleitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    displ2 = simpleitk.Compose([simpleitk.GetImageFromArray(-distortion_matrix[1]),
                           simpleitk.GetImageFromArray(-distortion_matrix[0])])
    out_tx = simpleitk.DisplacementFieldTransform(displ2)
    resampler.SetTransform(out_tx)
    interpolated = np.zeros(image_stack.shape)
    nimages = image_stack.shape[0]
    for i in trange(nimages):
        out = resampler.Execute(simpleitk.GetImageFromArray(image_stack[i]))
        interpolated[i] = simpleitk.GetArrayFromImage(out)
    return interpolated


def undistort_stack(distortion_matrix, data):
    """ Undistort stack with distortion matrix
    
    Use the griddata interpolation of scipy to apply distortion matrix to image
    The distortion matrix contains in each pixel where the pixel has to be 
    moved (floats)

    Parameters
    ----------
    distortion_matrix: numpy array
        distortion matrix to undistort image (format image.shape[0], 
                                            image.shape[2], 2)
    data: numpy array or sidpy.Dataset
        image
    """
    corrected = distortion_matrix[:, 2:4]
    intensity_values = data[:, distortion_matrix[:, 0].astype(int),
                            distortion_matrix[:, 1].astype(int)]
    size_x, size_y = 2 ** np.round(np.log2(data.shape[1:]))  # nearest power of 2
    size_x = int(size_x)
    size_y = int(size_y)

    grid_x, grid_y = np.mgrid[0:size_x - 1:size_x * 1j, 0:size_y - 1:size_y * 1j]
    interpolated = np.zeros([data.shape[0], size_x, size_y])
    nimages = data.shape[0]
    for i in trange(nimages):
        interpolated[i, :, :] = scipy.interpolate.griddata(corrected,
                                                           intensity_values[i, :],
                                                           (grid_x, grid_y),
                                                           method='linear')
    print(':-) \n You have successfully completed undistortion of image stack')
    return interpolated
