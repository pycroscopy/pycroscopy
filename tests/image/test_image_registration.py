# -*- coding: utf-8 -*-
"""
Created on January, 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np

import sidpy

from pycroscopy.image import image_registration


def make_test_data():
    im = np.zeros([64, 64, 5])
    im[3:5, 4:6, 0] = 1
    im[4:6, 4:6, 1] = 1
    im[3:5, 3:5, 2] = 1
    im[4:6, 3:5, 3] = 1
    im[3:5, 3:5, 4] = 1

    image = sidpy.Dataset.from_array(im)
    image.data_type = 'Image_stack'
    image.dim_0.dimension_type = 'spatial'
    image.dim_1.dimension_type = 'spatial'
    image.dim_2.dimension_type = 'temporal'

    return image


class TestUtilityFunctions(unittest.TestCase):
    def test_rigid_registration(self):
        image_stack = make_test_data()

        with self.assertRaises(TypeError):
            image_registration.rigid_registration(image_stack)
        with self.assertRaises(TypeError):
            image_stack.data_type = 'image'
            image_registration.rigid_registration(image_stack)
        image_stack.data_type = 'image_stack'

        registered = image_registration.rigid_registration(image_stack)
        
        self.assertIsInstance(registered, sidpy.Dataset)
        self.assertIsInstance(registered.metadata, dict)
        self.assertTrue('drift' in registered.metadata)
        self.assertTrue(np.allclose(registered.metadata['drift'], [[0., -1.], [0., -1.], [-1., -1.], [0.,  0.],
                                                                   [-1., 0.], [0., 0.]]))

        self.assertTrue(np.allclose(registered.metadata['input_crop'], [1, 64, 1, 64]))
        self.assertTrue(registered.shape[0] == 5)
        self.assertTrue(registered.shape[1] == 63)
        self.assertTrue(registered._axes[0].dimension_type.name == 'TEMPORAL')

        self.assertTrue(registered.data_type.name == 'IMAGE_STACK')

    def test_demon_registration(self):
        image_stack = make_test_data()

        registered = image_registration.rigid_registration(image_stack)
        demon_registered = image_registration.demon_registration(registered)
        self.assertIsInstance(demon_registered, sidpy.Dataset)
        self.assertTrue(demon_registered.shape[0] == 5)
        self.assertTrue(demon_registered.shape[1] == 63)
        self.assertTrue(demon_registered.data_type.name == 'IMAGE_STACK')

    def test_complete_registration(self):
        image_stack = make_test_data()
        non_rigid_registered, rigid_registered_dataset = image_registration.complete_registration(image_stack)
        self.assertIsInstance(non_rigid_registered, sidpy.Dataset)


if __name__ == '__main__':
    unittest.main()
