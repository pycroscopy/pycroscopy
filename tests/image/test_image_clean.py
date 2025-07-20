# -*- coding: utf-8 -*-
"""
Created on January, 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np

import sidpy
from scipy.ndimage import gaussian_filter

from pycroscopy.image import image_clean
import sys

if sys.version_info.major == 3:
    unicode = str


def make_test_data():
    im = np.zeros([64, 64])
    im[4::8, 4::8] = 1

    image = sidpy.Dataset.from_array(gaussian_filter(im, sigma=2))
    image.data_type = 'Image'
    image.set_dimension(0, sidpy.Dimension(np.arange(64), name='x', units='nm', dimension_type='SPATIAL', quantity='length'))
    image.set_dimension(1, sidpy.Dimension(np.arange(64), name='y', units='nm', dimension_type='SPATIAL', quantity='length'))


    atoms = []
    for i in range(8):
        for j in range(8):
            atoms.append([8 * i + 4, 8 * j + 4])

    return image, atoms


class TestUtilityFunctions(unittest.TestCase):
    def test_clean_svd(self):
        image, atoms_placed = make_test_data()

        with self.assertRaises(TypeError):
            image_clean.clean_svd(np.array(image))
        with self.assertRaises(TypeError):
            image.data_type = 'spectrum'
            image_clean.clean_svd(image)
        image.data_type = 'image'

        clean_image = image_clean.clean_svd(image)

        self.assertIsInstance(clean_image, sidpy.Dataset)

    def test_decon_lr(self):
        im = np.random.random([256, 256])

        image = sidpy.Dataset.from_array(gaussian_filter(im, sigma=2))
        image.data_type = 'Image'
        image.dim_0.dimension_type = 'spatial'
        image.dim_1.dimension_type = 'spatial'
        image.x = image.dim_0
        image.y = image.dim_1

        clean_image = image_clean.decon_lr(image, verbose=True)

        self.assertIsInstance(clean_image, sidpy.Dataset)


if __name__ == '__main__':
    unittest.main()
