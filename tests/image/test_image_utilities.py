import unittest
import numpy as np

import sidpy
from pycroscopy.image import crop_image


class MyTestCase(unittest.TestCase):
    def test_crop_image(self):
        image = sidpy.Dataset.from_array(np.zeros((256, 256)))
        image.data_type = 'image'
        image.dim_0.dimension_type = 'spatial'
        image.dim_1.dimension_type = 'spatial'

        cropped_image = crop_image(image, np.array([[1, 1], [250,250]]))
        self.assertEqual(cropped_image.shape[0], 249)  # add assertion here


if __name__ == '__main__':
    unittest.main()
