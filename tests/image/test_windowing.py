from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
import numpy as np
from pycroscopy.image import ImageWindowing
import sidpy as sid

if sys.version_info.major == 3:
    unicode = str
    xrange = range

def gen_sidpy_dset(data_mat):
    #Helper function to generate a test sidpy dataset object
    #Just provide the data as a numpy array (must be 2 or three dimensions)
    #Shape expected is (x,y,t) where t is the spectral dimension

    if data_mat.ndim<2 or data_mat.ndim>3:
        raise ValueError("Must provide numpy array with 2 or 3 dimensions")

    # Specify dimensions
    x_dim = np.linspace(0, 0.01,
                        data_mat.shape[0])
    y_dim = np.linspace(0, 11.21,
                        data_mat.shape[1])
    if data_mat.ndim==3:
        z_dim = np.linspace(0, 3.25, data_mat.shape[-1])

    # Make a sidpy dataset
    data_set = sid.Dataset.from_array(data_mat, name='Current_spectral_map')

    # Set the data type
    if data_mat.ndim==3:
        data_set.data_type = sid.DataType.SPECTRAL_IMAGE
    elif data_mat.ndim==2:
        data_set.data_type == sid.DataType.IMAGE

    # Add quantity and units
    data_set.units = 'nA'
    data_set.quantity = 'Current'

    # Add dimension info
    data_set.set_dimension(0, sid.Dimension(x_dim,
                                            name='x',
                                            units='m', quantity='x',
                                            dimension_type='spatial'))
    data_set.set_dimension(1, sid.Dimension(y_dim,
                                            name='y',
                                            units='m', quantity='y',
                                            dimension_type='spatial'))
    if data_mat.ndim==3:
        data_set.set_dimension(2, sid.Dimension(z_dim,
                                                name='Voltage',
                                                units='V', quantity='Voltage',
                                                dimension_type='spectral'))
    return data_set

class TestImageWindowing(unittest.TestCase):

    def test_two_dim_case(self):
        #test two dimensional windowing works
        sidpy_dset_image = gen_sidpy_dset(np.random.uniform(size=(10,7)))
        parms_dict = {}
        parms_dict['window_step_x'] = 2
        parms_dict['window_step_y'] = 2
        parms_dict['window_size_x'] = 3
        parms_dict['window_size_y'] = 7
        iw = ImageWindowing(parms_dict)
        windows = iw.MakeWindows(sidpy_dset_image)
        assert windows.shape == (1, 5, 3, 7)

    def test_three_dim_case(self):
        #test that spectral image works
        sidpy_dset_image = gen_sidpy_dset(np.random.uniform(size=(11, 3, 7)))
        parms_dict = {}
        parms_dict['window_step_x'] = 2
        parms_dict['window_step_y'] = 2
        parms_dict['window_size_x'] = 3
        parms_dict['window_size_y'] = 3
        iw = ImageWindowing(parms_dict)
        windows = iw.MakeWindows(sidpy_dset_image, dim_slice=2)
        assert windows.shape == (1, 5, 3, 3)

    def test_fft_works(self):
        #test that we can fft
        data_mat = np.random.uniform(size=(10, 10, 4))
        sidpy_dset_image = gen_sidpy_dset(data_mat)
        parms_dict = {}
        parms_dict['window_step_x'] = 2
        parms_dict['window_step_y'] = 2
        parms_dict['window_size_x'] = 4
        parms_dict['window_size_y'] = 4
        parms_dict['fft_mode'] = 'abs'

        #check that both modes work
        for mode in ['fft', 'image']:
            parms_dict['mode'] = mode
            for fft_mode in ['abs', 'phase']:
                parms_dict['fft_mode'] = fft_mode
                for parms_dict['filter'] in ['hamming', 'blackman']:
                    iw = ImageWindowing(parms_dict)
                    windows = iw.MakeWindows(sidpy_dset_image, dim_slice=2)
                    assert windows.shape == (4, 4, 4, 4)
                    assert windows.metadata['fft_mode'] == fft_mode
                    assert windows.metadata['mode'] == mode




