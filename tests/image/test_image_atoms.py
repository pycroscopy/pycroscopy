# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np

import sidpy
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, "../../")

from pycroscopy.image import image_atoms

if sys.version_info.major == 3:
    unicode = str


def make_test_data():
    im = np.zeros([64, 64])
    im[4::8, 4::8] = 1

    image = sidpy.Dataset.from_array(gaussian_filter(im, sigma=2))
    image.data_type = 'Image'
    image.dim_0.dimension_type = 'spatial'
    image.dim_1.dimension_type = 'spatial'

    atoms = []
    for i in range(8):
        for j in range(8):
            atoms.append([8 * i + 4, 8 * j + 4])

    return image, atoms

class TestUtilityFunctions(unittest.TestCase):
    def test_find_atoms(self):
        image, atoms_placed = make_test_data()

        with self.assertRaises(TypeError):
            image_atoms.find_atoms(np.array(image))
        with self.assertRaises(TypeError):
            image.data_type = 'spectrum'
            image_atoms.find_atoms(image)
        image.data_type = 'image'
        with self.assertRaises(TypeError):
            image_atoms.find_atoms(image, atom_size='large')
        with self.assertRaises(TypeError):
            image_atoms.find_atoms(image, threshold='large')

        found_atoms = image_atoms.find_atoms(image)

        matches = 0
        for i, pos in enumerate(atoms_placed):
            if list(found_atoms[i, :2]) in atoms_placed:
                matches += 1
        self.assertEqual(64, matches)
