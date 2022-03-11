from __future__ import division, print_function, absolute_import
import numpy as np  # for all array, data operations
import sidpy
import dask
# from scipy.special import erf
# from scipy import signal as sps
# from collections import Iterable
# from warnings import warn


get_slope = sidpy.base.num_utils.get_slope


def fourier_transform(dset, dimension_type=None):
    """
        Gets the 2D FFT for a single or stack of images by applying a blackman window

    Parameters
    ----------
    dset : 1D, 2D, or 3D sidpy.Dataset
        Either a 2D matrix [x, y] or a stack of 2D images; needs exactly two spatial dimensions
    dimension_type: None, str, or sidpy.DimensionType - optional
        dimension_type over which fourier transform is performed, if None an educated guess will determine
        that from dimensions of sidpy.Dataset

    Returns
    -------
    fft_dset: 2D or 3D complex sidpy.Dataset (not tested for higher dimensions)
        2 or 3 dimensional matrix arranged in the same way as input

    Example
    -------
    >> fft_dataset = fourier_transform(sidpy_dataset)
    >> fft_dataset.plot()
"""

    if not isinstance(dset, sidpy.Dataset):
        raise TypeError('Expected a sidpy Dataset')

    if dimension_type is None:
        # test for data_type of Dataset
        if dset.data_type.name in ['IMAGE_MAP', 'IMAGE_STACK', 'SPECTRAL_IMAGE', 'IMAGE_4D']:
            dimension_type = dset.dim_2.dimension_type
        else:
            dimension_type = dset.dim_0.dimension_type

    if isinstance(dimension_type, str):
        dimension_type = sidpy.DimensionType[dimension_type.upper()]

    if not isinstance(dimension_type, sidpy.DimensionType):
        raise TypeError('Could not identify a dimension_type to perform Fourier transform on')

    if dimension_type.name == 'SPATIAL':
        if len(dset.get_dimensions_by_type(dimension_type)) != 2:
            raise TypeError('sidpy dataset of type', dset.data_type,
                            ' has no obvious dimension over which to perform fourier transform, '
                            'please specify')
    elif dimension_type.name == 'RECIPROCAL':
        if len(dset.get_dimensions_by_type(dimension_type)) != 2:
            raise TypeError('sidpy dataset of type', dset.data_type,
                            ' has no obvious dimension over which to perform fourier transform, '
                            'please specify')
    elif dimension_type.name == 'SPECTRAL':
        if len(dset.get_dimensions_by_type(dimension_type)) != 1:
            raise TypeError('sidpy dataset of type', dset.data_type,
                            ' has no obvious dimension over which to perform fourier transform, '
                            'please specify')

    new_dset = dset-dset.min()
    if dimension_type == sidpy.DimensionType.SPECTRAL:
        dd = dset.get_dimensions_by_type('spectral')
        print(dd)
        fft_transform = np.fft.fftshift(dask.array.fft.fftn(new_dset, axes=dset.get_dimensions_by_type('spectral')))
    elif dimension_type == sidpy.DimensionType.SPATIAL:
        fft_transform = np.fft.fftshift(dask.array.fft.fft2(new_dset, axes=dset.get_dimensions_by_type('spatial')))
    elif dimension_type == sidpy.DimensionType.RECIPROCAL:
        fft_transform = np.fft.fftshift(dask.array.fft.ifft2(new_dset, axes=dset.get_dimensions_by_type('reciprocal')))
    else:
        raise NotImplementedError('fourier transform not implemented for dimension_type ', dimension_type.name)

    # old code
    #
    # if image_stack.ndim == 2:
    # single image
    #    image_stack = np.expand_dims(image_stack, axis=0)
    # blackman_2d = np.atleast_2d(np.blackman(image_stack.shape[2])) * \
    #   np.atleast_2d(np.blackman(image_stack.shape[1])).T
    # blackman_3d = np.expand_dims(blackman_2d, axis=0)
    # fft_stack = blackman_3d * image_stack
    # fft_stack = np.abs(np.fft.fftshift(np.fft.fft2(fft_stack, axes=(1, 2)), axes=(1, 2)))
    # return np.squeeze(fft_stack)

    fft_dset = sidpy.Dataset.from_array(fft_transform)
    fft_dset.quantity = dset.quantity
    fft_dset.units = 'a.u.'
    fft_dset.data_type = dset.data_type
    fft_dset.source = dset.title
    fft_dset.modality = 'fft'

    if dimension_type == sidpy.DimensionType.SPATIAL:
        image_dims = dset.get_dimensions_by_type(dimension_type)
        units_x = '1/' + dset._axes[image_dims[0]].units
        units_y = '1/' + dset._axes[image_dims[1]].units
        fft_dset.set_dimension(image_dims[0], sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(dset.shape[image_dims[0]],
                                                                                             d=get_slope(dset.x.values)
                                                                                             )),
                                                              name='u', units=units_x, dimension_type='RECIPROCAL',
                                                              quantity='reciprocal_length'))
        fft_dset.set_dimension(image_dims[1], sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(dset.shape[image_dims[1]],
                                                                                             d=get_slope(dset.y.values)
                                                                                             )),
                                                              name='v', units=units_y, dimension_type='RECIPROCAL',
                                                              quantity='reciprocal_length'))
        for i in range(len(dset.shape)):
            if i not in image_dims:
                fft_dset.set_dimension(i, dset._axes[i])

    elif dimension_type == sidpy.DimensionType.RECIPROCAL:
        image_dims = dset.get_dimensions_by_type(dimension_type)
        units_x = '1/' + dset._axes[image_dims[0]].units
        units_y = '1/' + dset._axes[image_dims[1]].units
        fft_dset.set_dimension(image_dims[0], sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(dset.shape[image_dims[0]],
                                                                                             d=get_slope(dset.x.values)
                                                                                             )),
                                                              name='u', units=units_x, dimension_type='SPATIAL',
                                                              quantity='length'))
        fft_dset.set_dimension(image_dims[1], sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(dset.shape[image_dims[1]],
                                                                                             d=get_slope(dset.y.values)
                                                                                             )),
                                                              name='v', units=units_y, dimension_type='SPATIAL',
                                                              quantity='length'))
        for i in range(len(dset.shape)):
            if i not in image_dims:
                fft_dset.set_dimension(i, dset._axes[i])

    elif dimension_type == sidpy.DimensionType.SPECTRAL:
        spec_dim = dset.get_dimensions_by_type(dimension_type)
        units = '1/' + dset._axes[spec_dim[0]].units
        fft_dset.set_dimension(spec_dim[0], sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(dset.shape[spec_dim[0]],
                                                                                           d=get_slope(dset.x.values))),
                                                            name='u', units=units, dimension_type='SPECTRAL',
                                                            quantity='frequency'))

        for i in range(len(dset.shape)):
            if i not in spec_dim:
                fft_dset.set_dimension(i, dset._axes[i])

    return fft_dset
