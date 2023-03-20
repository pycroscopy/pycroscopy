# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np


import sys
# sys.path.insert(0, "../../../sidpy/")
import sidpy
sys.path.insert(0, "../../")

from pycroscopy.image import image_graph

if sys.version_info.major == 3:
    unicode = str


class TestUtilityFunctions(unittest.TestCase):

    def test_polygon_area(self):
        corners = np.asarray([[5., 7.], [6., 6.], [2., -2.]])
        area, c_x, c_y = image_graph.polygon_area(corners)
        self.assertTrue(np.allclose([area, c_x, c_y], (6.0, -4.333, -3.667), atol=1e-3))

    def test_polygon_angles(self):
        triangle = np.asarray([[5., 7.], [6., 6.], [2., -2.]])
        angles = image_graph.polygon_angles(triangle)
        self.assertTrue(np.allclose(angles, (78.69, 54.462, 247.619), atol=1e-3))

    def test_circum_center_tetrahedron(self):
        # tetrahedron = np.random.random([4, 3])
        tetrahedron = np.asarray([[0, -1 / np.sqrt(3), 0], [0.5, 1 / (2 * np.sqrt(3)), 0],
                                  [-0.5, 1 / (2 * np.sqrt(3)), 0], [0, 0, np.sqrt(2 / 3)]])
        center, radius = image_graph.circum_center(tetrahedron)
        self.assertEqual(center.shape, (3,))
        self.assertTrue(np.allclose(center, (0., 0., 0.204), atol=1e-3))

    def test_circum_center_weighted(self):
        # tetrahedron = np.random.random([4, 3])
        tetrahedron = np.asarray([[0, -1 / np.sqrt(3), 0], [0.5, 1 / (2 * np.sqrt(3)), 0],
                                  [-0.5, 1 / (2 * np.sqrt(3)), 0], [0, 0, np.sqrt(2 / 3)]])
        atom_radii = np.array([0.1, 0.1, 0.1, 0.2])
        center, radius = image_graph.interstitial_sphere_center(tetrahedron, atom_radii)
        self.assertEqual(center.shape, (3,))
        self.assertTrue(np.allclose(center, (0., 0., 0.1256), atol=1e-3))
        self.assertTrue(np.allclose(np.linalg.norm(tetrahedron-center, axis=1), atom_radii+radius, atol=1e-3))

    def test_circum_center_radius(self):
        # tetrahedron = np.random.random([4, 3])
        tetrahedron = np.asarray([[0, -1 / np.sqrt(3), 0], [0.5, 1 / (2 * np.sqrt(3)), 0],
                                  [-0.5, 1 / (2 * np.sqrt(3)), 0], [0, 0, np.sqrt(2 / 3)]])
        atom_radii = np.array([0.1, 0.1, 0.1, 0.1])
        center, radius = image_graph.interstitial_sphere_center(tetrahedron, atom_radii)
        center2, radius2 = image_graph.circum_center(tetrahedron)

        self.assertEqual(radius, radius2-atom_radii[0])
        self.assertTrue(np.allclose(center, center2, atol=1e-3))

    def test_circum_center_triangle(self):
        triangle = np.asarray([[5., 7.], [6., 6.], [2., -2.]])

        center, radius = image_graph.circum_center(triangle)
        self.assertTrue(np.allclose(center, (2., 3.)))

    def test_circum_center_triangle_weighted(self):
        triangle = np.asarray([[5., 7.], [6., 6.], [2., -2.]])

        center, radius = image_graph.interstitial_sphere_center(triangle, [0.3, 0.3, 0.4])
        self.assertTrue(np.allclose(center, (2.04, 3.04), atol=1e-2))

    def test_circum_center_line(self):
        triangle = np.asarray([[1., 0.], [2., 0.], [3., 0.]])

        center, radius = image_graph.circum_center(triangle)
        self.assertTrue(np.allclose(center, (0., 0.)))

    