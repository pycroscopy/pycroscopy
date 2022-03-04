"""
image_graph part of pycroscopy

Author: Gerd Duscher

The atomic positions are viewed as a graph; more specifically a ring graph.
The projections of the unit_cell and defect structural units are the rings.
The center of the ring is the interstitial location. 
The vertices are the atom positions and the edges are the bonds.

This is a modification of the method of Banadaki and Patala
    http://dx.doi.org/10.1038/s41524-017-0016-0
for 2-D (works in 3D as well)

Starting from the atom positions we make a Delaunay tesselation and determine the size of the intersitital (circumscribed circle radius minus the atom radius).
If neighbouring interstitials overlap we merge those triangles (in 3D the tetrhedra). This will give an unanbiguous tesselation or graph for a given atomic size.

The main functions are:
>import pycroscopy as px
>
>structural_units  = px.image.find_structural_units(atoms[:,:2], .4/np.sqrt(2)/4, lr_dset)
>
>graph_dictionary = px.image.get_polygons(structural_units)
>
>fig = plt.figure()
>plt.imshow(lr_dset.T, extent=[-0.5,dataset.shape[0]-1.5, dataset.shape[1]-1.5,-0.5], cmap = 'gray', vmax= 7)

>px.image.add_graph(graph_dictionary, 'cyclicity', min_q=2.5, max_q=12.5, fig=fig, cmap=plt.cm.tab10)
"""
import numpy as np
import scipy.spatial 
from skimage.measure import grid_points_in_poly  # , points_in_poly
from scipy.spatial import Voronoi, cKDTree  # , KDTree
import matplotlib
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from tqdm import trange

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

def circum_center(vertex_pos, tol=1e-3):
    """
    Function finds the center and the radius of the circumsphere of every tetrahedron.
    Reference:
    Fiedler, Miroslav. Matrices and graphs in geometry. No. 139. Cambridge University Press, 2011.
    (p.29 bottom: example 2.1.11)
    Code (slightly modified) from https://github.com/spatala/gbpy

    Parameters
    -----------------
    vertex_pos : numpy array
        The position of vertices of a tetrahedron
    tol : float
        Tolerance defined  to identify co-planar tetrahedrons
    Returns
    ----------
    circum_center : numpy array
        The center of the circumsphere
    circum_radius : float
        The radius of the circumsphere
    """
    
    if vertex_pos.shape[1] < 3:
        ax = vertex_pos[0, 0]
        ay = vertex_pos[0, 1]
        bx = vertex_pos[1, 0]
        by = vertex_pos[1, 1]
        cx = vertex_pos[2, 0]
        cy = vertex_pos[2, 1]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d

        circum_center =np.array([ux, uy]) 
        circum_radius = np.linalg.norm(circum_center-vertex_pos[0])
        
        return np.array(circum_center), circum_radius
    dis_ij = scipy.spatial.distance.pdist(np.array(vertex_pos), 'euclidean')
    sq_12, sq_13, sq_14, sq_23, sq_24, sq_34 = np.power(dis_ij, 2)

    matrix_c = np.array([[0, 1, 1, 1, 1], [1, 0, sq_12, sq_13, sq_14], [1, sq_12, 0, sq_23, sq_24],
                         [1, sq_13, sq_23, 0, sq_34], [1, sq_14, sq_24, sq_34, 0]])

    det_matrix_c = (np.linalg.det(matrix_c))

    if det_matrix_c < tol:
        return np.array([0, 0, 0]), 0
    else:
        matrix = -2 * np.linalg.inv(matrix_c)
        circum_center = (matrix[0, 1] * vertex_pos[0, :] + matrix[0, 2] * vertex_pos[1, :] +
                         matrix[0, 3] * vertex_pos[2, :] +
                         matrix[0, 4] * vertex_pos[3, :]) / (matrix[0, 1] + matrix[0, 2] + matrix[0, 3] + matrix[0, 4])
        circum_radius = np.sqrt(matrix[0, 0]) / 2

    return np.array(circum_center), circum_radius

def voronoi_volumes(points):
    """
    Volumes of voronoi  cells from
    https://stackoverflow.com/questions/19634993/volume-of-voronoi-cell-python


    """
    v = scipy.spatial.Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            try:
                hull = scipy.spatial.ConvexHull(v.vertices[indices])
                vol[i] = hull.volume
            except:
                vol[i] = 0.
    return vol


def get_voronoi(tetrahedra, atoms, r_a, extent):
    """
    Find Voronoi vertices and keep track of associated tetrahedrons and interstitial radii
    
    Used in find_polyhedra function
    
    Parameters
    ----------
    tetrahedra: scipy.spatial.Delaunay object
        Delaunay tesselation
    atoms: ase.Atoms object
        the structural information
    r_a: float
        the atomic radius

    Returns
    -------
    voronoi_vertices: list
        list of positions of voronoi vertices
    voronoi_tetrahedra:
        list of indices of associated vertices of tetrahedra
    r_vv: list of float
        list of all interstitial sizes
    """

    voronoi_vertices = []
    voronoi_tetrahedrons = []
    r_vv = []
    for vertices in tetrahedra.vertices:
        voronoi, radius = circum_center(atoms[vertices])
        
        if (voronoi >= 0).all() and (extent - voronoi > 0).all() and radius > 0.01:
            voronoi_vertices.append(voronoi)
            voronoi_tetrahedrons.append(vertices)
            r_vv.append(radius - r_a)
    return voronoi_vertices, voronoi_tetrahedrons, r_vv


def find_overlapping_interstitials(voronoi_vertices, r_vv, r_a, cheat=1.):
    """Find overlapping spheres"""
    
    vertex_tree = scipy.spatial.cKDTree(np.array(voronoi_vertices)[:,:2])

    pairs = vertex_tree.query_pairs(r=r_a * 2)

    overlapping_pairs = []
    for (i, j) in pairs:
        if np.linalg.norm(voronoi_vertices[i] - voronoi_vertices[j]) < (r_vv[i] + r_vv[j])*cheat:
            overlapping_pairs.append([i, j])

    return np.array(sorted(overlapping_pairs))


def find_clusters(overlapping_pairs):
    """Make cluste
    We are using a breadth first to go through the list of overlapping spheres to determine clusters
    """
    visited_all = []
    clusters = []
    for initial in overlapping_pairs[:, 0]:
        if initial not in visited_all:
            # breadth first search
            visited = []  # the atoms we visited
            queue = [initial]
            while queue:
                node = queue.pop(0)
                if node not in visited_all:
                    visited.append(node)
                    visited_all.append(node)
                    # neighbors = overlapping_pairs[overlapping_pairs[:,0]==node,1]
                    neighbors = np.append(overlapping_pairs[overlapping_pairs[:, 1] == node, 0],
                                          overlapping_pairs[overlapping_pairs[:, 0] == node, 1])

                    for i, neighbour in enumerate(neighbors):
                        if neighbour not in visited:
                            queue.append(neighbour)
            clusters.append(visited)
    return clusters, visited_all


def make_structural_units(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all):
    """collect output data  and make dictionary"""

    polyhedra = {}
    for index in trange(len(clusters)):
        cluster = clusters[index]
        cc = []
        for c in cluster:
            cc = cc + list(voronoi_tetrahedrons[c])
        hull = scipy.spatial.ConvexHull(atoms[list(set(cc))])
        faces = []
        triangles = []
        for s in hull.simplices:
            faces.append(atoms[list(set(cc))][s])
            triangles.append(list(s))
        polyhedra[index] = {'vertices': atoms[list(set(cc))], 'indices': list(set(cc)),
                            'faces': faces, 'triangles': triangles,
                            'length': len(list(set(cc))),
                            'combined_vertices': cluster,
                            'interstitial_index': index,
                            'interstitial_site': np.array(voronoi_tetrahedrons)[cluster].mean(axis=0),
                            'volume': hull.volume}
        if False:  # isinstance(atoms, ase.Atoms):
                polyhedra[index]['atomic_numbers'] = atoms.get_atomic_numbers()[voronoi_tetrahedrons],

        # 'coplanar': hull.coplanar}

    running_number = index + 0
    for index in trange(len(voronoi_vertices)):
        if index not in visited_all:
            vertices = voronoi_tetrahedrons[index]
            hull = scipy.spatial.ConvexHull(atoms[vertices])
            faces = []
            triangles = []
            for s in hull.simplices:
                faces.append(atoms[vertices][s])
                triangles.append(list(s))

            polyhedra[running_number] = {'vertices': atoms[vertices], 'indices': vertices,
                                         'faces': faces, 'triangles': triangles,
                                         'length': len(vertices),
                                         'combined_vertices': index,
                                         'interstitial_index': running_number,
                                         'interstitial_site': np.array(voronoi_tetrahedrons)[index],
                                         'volume': hull.volume}
            if False:  # isinstance(atoms, ase.Atoms):
                polyhedra[running_number]['atomic_numbers'] = atoms.get_atomic_numbers()[vertices],


            running_number += 1

    return polyhedra

##################################################################
# polyhedra functions
##################################################################


def find_structural_units(atoms, r_a, dataset, cheat=1.0):
    """ get polyhedra information from an ase.Atoms object

    This is following the method of Banadaki and Patala
    http://dx.doi.org/10.1038/s41524-017-0016-0

    Parameter
    ---------
    atoms: ase.Atoms object
        the structural information
    r_a: float
        the atomic radius

    Returns
    -------
    polyhedra: dict
        dictionary with all information of polyhedra
    """
    
    if not isinstance(r_a, (int, float)):
        raise TypeError('Atomic radius must be a real number')
    
    if not (0.05 < r_a < .2):
        print('Strange atomic radius, are you sure you know what you are doing?')
    extent=[dataset.shape[0], dataset.shape[1]]
    r_a = r_a/(dataset.x[1]-dataset.x[0])
    tesselation = scipy.spatial.Delaunay(atoms)

    voronoi_vertices, voronoi_tetrahedrons, r_vv = get_voronoi(tesselation, atoms, r_a, extent)

    overlapping_pairs = find_overlapping_interstitials(voronoi_vertices, r_vv, r_a, cheat=cheat)

    clusters, visited_all = find_clusters(overlapping_pairs)

    polyhedra = make_structural_units(atoms, voronoi_vertices, voronoi_tetrahedrons, clusters, visited_all)

    return polyhedra


def sort_polyhedra_by_vertices(polyhedra, visible=range(4, 100), z_lim=[0, 100], verbose=False):
    indices = []

    for key, polyhedron in polyhedra.items():
        if 'length' not in polyhedron:
            polyhedron['length'] = len(polyhedron['vertices'])

        if polyhedron['length'] in visible:
            center = polyhedron['vertices'].mean(axis=0)
            if z_lim[0] < center[2] < z_lim[1]:
                indices.append(key)
                if verbose:
                    print(key, polyhedron['length'], center)
    return indices


# color_scheme = ['lightyellow', 'silver', 'rosybrown', 'lightsteelblue', 'orange', 'cyan', 'blue', 'magenta',
#                'firebrick', 'forestgreen']

def get_polygons(polyhedra):
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

    rings = []
    centers = []
    _inner_angles = []
    cyclicities = []
    cells = []
    areas = []
    for key, poly in polyhedra.items():
        corners = poly['vertices']
        if len(corners) > 2:
            cyclicities.append(len(corners))  # length of ring or cyclicity will be stored
            center = np.average(corners, axis=0)  # center of ring will be stored
            centers.append(center)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            ang_sort = np.argsort(angles)
            angles = (angles[ang_sort] - angles[np.roll(ang_sort, 1)]) % np.pi
            _inner_angles.append(angles)  # inner angles in radians

            ring = corners[ang_sort]  # clocks=wise sorted ring vertices will be stored
            rings.append(ring)
            areas.append(poly['volume'])
            cells.append(patches.Polygon(ring, closed=True, fill=True, edgecolor='red', linewidth=2))

    max_ring_size = max(cyclicities)
    tags = {'unit_cells': cells, 'centers': np.array(centers), 'cyclicity': np.array(cyclicities), 'areas': np.array(areas)}

    number_of_rings = len(rings)
    tags['vertices'] = np.zeros((number_of_rings, max_ring_size, 2))
    tags['inner_angles'] = np.zeros((number_of_rings, max_ring_size))
    tags['areas'] = areas

    # a slow way to make a sparse matrix, which we need for h5_file
    for i in range(number_of_rings):
        ring = rings[i]
        angles = _inner_angles[i]
        tags['vertices'][i, :len(ring), :] = ring
        tags['inner_angles'][i, :len(ring)] = angles

    return tags


def add_graph(graph_dictionary, quantity='cyclicity', min_q=None, max_q=None, fig=None, cmap=matplotlib.cm.viridis):
    if fig is None:
        fig = matplotlib.figure()
    
    values = np.array(graph_dictionary[quantity])
    if min_q is None:
        min_q = values.min()
    if max_q is None:
        max_q = values.max()
        
    unit_cells = PatchCollection(graph_dictionary['unit_cells'], alpha=.3, cmap=cmap, clim =(min_q, max_q) , edgecolor='black')
    unit_cells.set_array(values)
    
    fig.gca().add_collection(unit_cells)
    #plt.scatter(centers[:,0],centers[:,1],color='blue',alpha=0.5, s = 3)
    cbar = fig.colorbar(unit_cells, label=quantity)
