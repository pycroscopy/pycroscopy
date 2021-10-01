import numpy as np
import scipy.spatial as spatial

from skimage.measure import grid_points_in_poly  # , points_in_poly
from scipy.spatial import Voronoi, cKDTree  # , KDTree
import matplotlib.patches as patches

#####################
# Polygon Functions #
#####################
def turning_function(corners, points):
    """turning function of polygon"""

    # sort corners in counter-clockwise direction
    # calculate centroid of the polygon
    corners1 = np.array(points[corners])
    corners2 = np.roll(corners1, 1)
    corners0 = np.roll(corners1, -1)

    v = corners1 - corners0
    _ = (np.arctan2(v[:, 0], v[:, 1]) + 2.0 * np.pi) % (2.0 * np.pi) / np.pi * 180
    print(corners1)
    angles = []
    for i in range(len(corners1)):
        a = corners1[i] - corners0[i]
        b = corners1[i] - corners2[i]
        num = np.dot(a, b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        angles.append(np.arccos(num / denominator) * 180 / np.pi)

    return angles


def polygon_sort2(corners, points):
    """sort corners in counter-clockwise direction

    Parameters
    ----------
    corners: list of int
        indices in points array
    points: numpy array
        list of points/atoms

    Returns
    -------
        corners_with_angles: list
    """

    # calculate centroid of the polygon
    n = len(corners)  # of corners
    cx = float(sum(x for x, y in points[corners])) / n
    cy = float(sum(y for x, y in points[corners])) / n

    # create a new list of corners which includes angles
    # angles from the positive x axis
    corners_with_angles = []
    for i in corners:
        x, y = points[i]
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi) % (2. * np.pi)
        corners_with_angles.append([i, np.degrees(an)])

    # sort it using the angles
    corners_with_angles.sort(key=lambda tup: tup[1])

    return corners_with_angles


def polygons_inner(indices, points):
    """Inner angles of polygon"""
    pp = np.array(points)[indices, :]
    # Determine inner angle of polygon
    # Generate second array which is shifted by one
    pp2 = np.roll(pp, 1, axis=0)
    # and subtract it from former: this is now a list of vectors
    p_vectors = pp - pp2

    # angles of vectors with respect to positive x-axis
    ang = np.arctan2(p_vectors[:, 1], p_vectors[:, 0]) / np.pi * 180 + 360 % 360
    # shift angles array by one
    ang2 = np.roll(ang, -1, axis=0)

    # difference of angles is outer angle but we want the inner (inner + outer = 180)
    inner__angles = (180 - (ang2 - ang) + 360) % 360

    return inner__angles


def inner_angles(vertices, rotational_invariant=False, verbose=False):
    """Inner angles of a polygon"""

    a = np.roll(vertices, -1, axis=0) - vertices
    b = np.roll(a, -1, axis=0)
    inner__angles = []
    for i in range(len(vertices)):
        inner__angles.append(np.arctan2(np.cross(a[i], b[i]), np.dot(a[i], b[i])))
    if verbose:
        print(np.degrees(inner__angles))
    if rotational_invariant:
        return np.sort(inner__angles)
    else:
        return np.array(inner__angles)


# sort corners in counter-clockwise direction
def polygon_sort(corners):
    """sort polygon vertices according to angle"""
    # calculate centroid of the polygon
    n = len(corners)  # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n

    # create a new list of corners which includes angles
    corners_with_angles = []
    for x, y in corners:
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        corners_with_angles.append((x, y, np.degrees(an)))

    # sort it using the angles
    corners_with_angles.sort(key=lambda tup: tup[2])

    return corners_with_angles


def polygon_area(corners):
    """ Area of Polygon using Shoelace formula

    http://en.wikipedia.org/wiki/Shoelace_formula
    FB - 20120218
    corners must be ordered in clockwise or counter-clockwise direction
    """

    n = len(corners)  # of corners
    area = 0.0
    c_x = 0
    c_y = 0
    for i in range(n):
        j = (i + 1) % n
        nn = corners[i][0] * corners[j][1] - corners[j][0] * corners[i][1]
        area += nn
        c_x += (corners[i][0] + corners[j][0]) * nn
        c_y += (corners[i][1] + corners[j][1]) * nn

    area = abs(area) / 2.0

    # centroid or arithmetic mean
    c_x = c_x / (6 * area)
    c_y = c_y / (6 * area)

    return area, c_x, c_y


def polygon_angles(corners):
    """angles of polygon"""

    angles = []
    # calculate centroid of the polygon
    n = len(corners)  # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of angles
    # print (cx, cy)
    for x, y in corners:
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        angles.append((np.degrees(an)))

    return angles


def polygon_similarity(vertices, vertices_reference, rotational_invariant=False, verbose=False):
    """Similarity between two polygons"""

    angles_poly2 = inner_angles(vertices, rotational_invariant, verbose)
    angles_ideal = inner_angles(vertices_reference, rotational_invariant, verbose)
    return np.absolute(angles_poly2 - angles_ideal).sum()


##########################
# New Graph Stuff
##########################

def get_graph(atoms, extent, smallest_lattice_parameter=0.3):
    """ Make graph from atom positions

    Parameters
    ----------
    atoms: numpy array (nx2)
        positions of atoms to be evaluated for graph
    extent: list of float (4x1)
        extent of image
    smallest_lattice_parameter: float
        determines how far the Voronoi vertices have to be apart to be considered a distortion

    Returns
    -------
    tags: dictionary
        information of graph
    """

    vor = Voronoi(atoms)
    point_tree = cKDTree(vor.points)

    # vertices of Voronoi of atom are centers of ring
    # distorted structures lead to vertices too close together and need to be averaged
    new_voronoi = make_new_vertices(vor.vertices, extent, smallest_lattice_parameter)

    # Voronoi cells of centers are rings of graph
    vor2 = Voronoi(new_voronoi)

    # rings need to be gathered
    rings = []
    centers = []
    _inner_angles = []
    cyclicities = []
    cells = []
    for i in range(len(vor2.points)):
        region_indices = vor2.regions[vor2.point_region[i]]
        if -1 in region_indices:  # outside indices indicate rim structure
            pass
        else:
            # get nearest atoms to vertices of ring
            ring = vor2.vertices[region_indices]
            dist, nn = point_tree.query(ring, k=1, p=2)

            corners = vor.points[nn[dist < 0.3]]
            if len(corners) > 2:
                ring = np.unique(corners, axis=0)  # cannot do unique on empty array
                if len(ring) > 2:  # ring needs to be at least a triangle
                    cyclicities.append(len(ring))  # length of ring or cyclicity will be stored

                    center = np.average(ring, axis=0)  # center of ring will be stored
                    centers.append(center)

                    # calculate inner angle use sorting to rearrange ring vertices clockwise
                    angles = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
                    ang_sort = np.argsort(angles)
                    angles = (angles[ang_sort] - angles[np.roll(ang_sort, 1)]) % np.pi
                    _inner_angles.append(angles)  # inner angles in radians

                    ring = ring[ang_sort]  # clocks=wise sorted ring vertices will be stored
                    rings.append(ring)
                    cells.append(patches.Polygon(ring, closed=True, fill=True, edgecolor='red', linewidth=2))

    max_ring_size = max(cyclicities)
    tags = {'unit_cells': cells, 'centers': np.array(centers), 'cyclicity': np.array(cyclicities)}

    number_of_rings = len(rings)
    tags['vertices'] = np.zeros((number_of_rings, max_ring_size, 2))
    tags['inner_angles'] = np.zeros((number_of_rings, max_ring_size))

    # a slow way to make a sparse matrix, which we need for h5_file
    for i in range(number_of_rings):
        ring = rings[i]
        angles = _inner_angles[i]
        tags['vertices'][i, :len(ring), :] = ring
        tags['inner_angles'][i, :len(ring)] = angles

    return tags


def make_new_vertices(vertices, extent, smallest_lattice_parameter):
    """ Determine whether vertices are too close and have to be replaced by median

    Part of get_graph function
    Parameters
    ----------
    vertices: numpy array (nx2)
        vertices of Voronoi tiles to be evaluated
    extent: list of float (4x1)
        extent of image
    smallest_lattice_parameter: float
        determines how far the Voronoi vertices have to be apart to be considered caused by distortion

    Returns
    -------
    new_voronoi: numpy array
        vertices of new Voronoi tiling
    """

    vertices_tree = cKDTree(vertices)

    dis = vertices_tree.query_ball_point(vertices, r=smallest_lattice_parameter * .7, p=2)  # , return_length=True)
    nn = vertices_tree.query_ball_point(vertices, r=smallest_lattice_parameter * .7, p=2, return_length=True)

    # handle nn > 2 differently Gerd

    new_voronoi = []
    for near in dis:
        if len(near) > 1:
            new = np.average(vertices[near], axis=0)
        elif len(near) > 0:
            new = vertices[near][0]
        else:
            new = [-1, -1]

        if (new > 0).all() and (new[0] < extent[1]) and (new[1] < extent[2]):
            new_voronoi.append(new)

    ver_sort = np.argsort(nn)
    nn_now = nn[ver_sort[-1]]
    done_list = []
    i = 1
    while nn_now > 2:
        close_vertices = dis[ver_sort[-i]]
        new_vert = []

        for vert in close_vertices:
            if vert not in done_list:
                new_vert.append(vert)

        done_list.extend(new_vert)
        # check whether necessary big_vertex = np.average(vertices[new_vert], axis=0)
        if len(new_vert) > 1:
            big_vertex = np.average(vertices[new_vert], axis=0)
            if (big_vertex[0] > 0) and (big_vertex[1] > 0):
                new_voronoi.append([big_vertex[0], big_vertex[1]])
        elif len(new_vert) > 0:
            new_voronoi.append([(vertices[new_vert[0]])[0], (vertices[new_vert[0]])[1]])
        i += 1
        nn_now = nn[ver_sort[-i]]

    # print(len(new_voronoi))
    new_voronoi = np.unique(new_voronoi, axis=0)
    return new_voronoi

    