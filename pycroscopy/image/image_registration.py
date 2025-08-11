"""
image_registration.py
by Gerd Duscher, UTK
part of pycroscopy.image
MIT license except where stated differently
"""

import typing

import numpy as np
import skimage
import scipy

from tqdm.auto import trange

import sidpy
_SIMPLEITK_PRESENT = True
try:
    import SimpleITK
except ModuleNotFoundError:
    _SIMPLEITK_PRESENT = False
if not _SIMPLEITK_PRESENT:
    print('SimpleITK not installed; Registration Functions for Image Stacks not available')

#####################################################
# Registration Functions
#####################################################

def complete_registration(main_dataset: sidpy.Dataset) -> typing.Tuple[sidpy.Dataset,
                                                                       sidpy.Dataset]:
    """Rigid and then non-rigid (demon) registration

    Performs rigid and then non-rigid registration, please see individual functions:
    - rigid_registration
    - demon_registration

    Parameters
    ----------
    main_dataset: sidpy.Dataset
        dataset of data_type 'IMAGE_STACK' to be registered

    Returns
    -------
    non_rigid_registered: sidpy.Dataset
    rigid_registered_dataset: sidpy.Dataset

    """

    if main_dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    rigid_registered_dataset = rigid_registration(main_dataset)

    rigid_registered_dataset.data_type = 'IMAGE_STACK'

    non_rigid_registered = demon_registration(rigid_registered_dataset)
    return non_rigid_registered, rigid_registered_dataset


def demon_registration(dataset: sidpy.Dataset, verbose: bool=False) -> sidpy.Dataset:
    """
    Diffeomorphic Demon Non-Rigid Registration

    Depends on:
        simpleITK and numpy
    Please Cite: http://www.simpleitk.org/SimpleITK/project/parti.html
    and T. Vercauteren, X. Pennec, A. Perchant and N. Ayache
    Diffeomorphic Demons Using ITK\'s Finite Difference Solver Hierarchy
    The Insight Journal, http://hdl.handle.net/1926/510 2007

    Parameters
    ----------
    dataset: sidpy.Dataset
        stack of image after rigid registration and cropping
    verbose: boolean
        optional for increased output
    Returns
    -------
        dem_reg: stack of images with non-rigid registration

    Example
    -------
    dem_reg = demon_reg(stack_dataset, verbose=False)
    """

    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    dem_reg = np.zeros(dataset.shape)
    nimages = dataset.shape[0]
    if verbose:
        print(nimages)
    # create fixed image by summing over rigid registration

    fixed_np = np.average(np.array(dataset), axis=0)

    if not _SIMPLEITK_PRESENT:
        print('This feature is not available:')
        print('Please install simpleITK with: conda install simpleitk -c simpleitk')

    fixed = SimpleITK.GetImageFromArray(fixed_np)
    fixed = SimpleITK.DiscreteGaussian(fixed, 2.0)

    demons = SimpleITK.DiffeomorphicDemonsRegistrationFilter()

    demons.SetNumberOfIterations(200)
    demons.SetStandardDeviations(1.0)

    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(SimpleITK.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    for i in trange(nimages):
        moving = SimpleITK.GetImageFromArray(dataset[i])
        moving_f = SimpleITK.DiscreteGaussian(moving, 2.0)
        displacement_field = demons.Execute(fixed, moving_f)
        out_tx = SimpleITK.DisplacementFieldTransform(displacement_field)
        resampler.SetTransform(out_tx)
        out = resampler.Execute(moving)
        dem_reg[i, :, :] = SimpleITK.GetArrayFromImage(out)

    print(':-)')
    print('You have successfully completed Diffeomorphic Demons Registration')

    demon_registered = dataset.like_data(dem_reg)
    demon_registered.title = 'Non-Rigid Registration'
    demon_registered.source = dataset.title

    demon_registered.metadata =dataset.metadata.copy()
    if 'analysis' not in demon_registered.metadata:
        demon_registered.metadata['analysis'] = {}
    demon_registered.metadata['analysis']['non_rigid_demon_registration'] = {'package': 'simpleITK',
                                                                             'method': 'DiscreteGaussian',
                                                                             'variance': 2,
                                                                             'input_dataset': dataset.source}
    demon_registered.data_type = 'IMAGE_STACK'
    return demon_registered


# ##############################
# Rigid Registration New 05/09/2024
# ##############################
def rigid_registration(dataset: sidpy.Dataset, normalization: typing.Optional[str] = None) -> sidpy.Dataset:
    """
    Rigid registration of image stack with pixel accuracy

    Uses simple cross_correlation
    (we determine drift from one image to next)

    Parameters
    ----------
    dataset: sidpy.Dataset
        sidpy dataset with image_stack dataset
    normalization: str or None
        if 'phase' then phase cross correlation is used, otherwise
        normalized cross correlation is used

    Returns
    -------
    rigid_registered: sidpy.Dataset
        Registered Stack and drift (with respect to center image)
    """

    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    if isinstance (normalization, str):
        if normalization.lower() != 'phase':
            normalization = None
    else:
        normalization = None
    image_dimensions = dataset.get_image_dims(return_axis=True)
    if dataset.get_dimensions_by_type('TEMPORAL')[0] != 0:
        x = image_dimensions[0]
        y = image_dimensions[1]
        z = dataset.get_dimensions_by_type('TEMPORAL', return_axis=True)[0]
        metadata = dataset.metadata.copy()
        original_metadata = dataset.original_metadata.copy()
        arr = np.rollaxis(np.array(dataset), 2, 0)
        dataset = sidpy.Dataset.from_array(arr, title=dataset.title, data_type='IMAGE_STACK',
                                           quantity=dataset.quantity, units=dataset.units)
        dataset.set_dimension(0, sidpy.Dimension(z.values, name='frame', units='frame', quantity='time',
                                                  dimension_type='temporal'))
        dataset.set_dimension(1, x)
        dataset.set_dimension(2, y)
        dataset.metadata = metadata
        dataset.original_metadata = original_metadata

    stack_dim = dataset.get_dimensions_by_type('TEMPORAL', return_axis=True)[0]
    image_dim = dataset.get_image_dims(return_axis=True)
    if len(image_dim) != 2:
        raise ValueError('need at least two SPATIAL dimension for an image stack')

    relative_drift = [[0., 0.]]
    im1 = np.fft.fft2(np.array(dataset[0]))
    for i in range(1, len(stack_dim)):
        im2 = np.fft.fft2(np.array(dataset[i]))
        shift, error, _ = skimage.registration.phase_cross_correlation(im1, im2,
                                                                       normalization=normalization,
                                                                       space='fourier')
        im1 = im2.copy()
        relative_drift.append(shift)

    rig_reg, drift = rig_reg_drift(dataset, relative_drift)
    crop_reg, input_crop = crop_image_stack(rig_reg, drift)

    rigid_registered = sidpy.Dataset.from_array(crop_reg,
                                                title='Rigid Registration',
                                                data_type='IMAGE_STACK',
                                                quantity=dataset.quantity,
                                                units=dataset.units)
    rigid_registered.title = 'Rigid_Registration'
    rigid_registered.source = dataset.title
    rigid_registered.metadata['analysis'] = {'rigid_registration': {'drift': drift,
                                 'input_crop': input_crop, 'input_shape': dataset.shape[1:]}}

    if 'experiment' in dataset.metadata:
        rigid_registered.metadata['experiment'] = dataset.metadata['experiment'].copy()
    rigid_registered.set_dimension(0, sidpy.Dimension(np.arange(rigid_registered.shape[0]),
                                          name='frame', units='frame', quantity='time',
                                          dimension_type='temporal'))

    array_x = image_dim[0].values[input_crop[0]:input_crop[1]]
    rigid_registered.set_dimension(1, sidpy.Dimension(array_x, name='x',
                                                      units='nm', quantity='Length',
                                                      dimension_type='spatial'))
    array_y =image_dim[1].values[input_crop[2]:input_crop[3]]
    rigid_registered.set_dimension(2, sidpy.Dimension(array_y, name='y',
                                                      units='nm', quantity='Length',
                                                      dimension_type='spatial'))
    rigid_registered.data_type = 'IMAGE_STACK'
    return rigid_registered.rechunk({0: 'auto', 1: -1, 2: -1})


def rig_reg_drift(dset: sidpy.Dataset,
                  rel_drift: typing.Union[typing.List[typing.List[float]], np.ndarray]
                  ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """ Shifting images on top of each other

    Uses relative drift to shift images on top of each other,
    with center image as reference.
    Shifting is done with shift routine of ndimage from scipy.
    This function is used by rigid_registration routine

    Parameters
    ----------
    dset: sidpy.Dataset
        dataset with image_stack
    rel_drift:
        relative_drift from image to image as list of [shiftx, shifty]

    Returns
    -------
    stack: numpy array
    drift: list of drift in pixel
    """

    frame_dim = []
    spatial_dim = []
    selection = []

    for i, axis in dset._axes.items():
        if axis.dimension_type.name == 'SPATIAL':
            spatial_dim.append(i)
            selection.append(slice(None))
        else:
            frame_dim.append(i)
            selection.append(slice(0, 1))

    if len(spatial_dim) != 2:
        print('need two spatial dimensions')
    if len(frame_dim) != 1:
        print('need one frame dimensions')

    rig_reg = np.zeros([dset.shape[frame_dim[0]], dset.shape[spatial_dim[0]], dset.shape[spatial_dim[1]]])

    # absolute drift
    drift = np.array(rel_drift).copy()

    drift[0] = [0, 0]
    for i in range(1, drift.shape[0]):
        drift[i] = drift[i - 1] + rel_drift[i]
    center_drift = drift[int(drift.shape[0] / 2)]
    drift = drift - center_drift
    # Shift images
    for i in range(rig_reg.shape[0]):
        selection[frame_dim[0]] = slice(i, i+1)
        # Now we shift
        rig_reg[i, :, :] = scipy.ndimage.shift(dset[tuple(selection)].squeeze().compute(),
                                         [drift[i, 0], drift[i, 1]], order=3)
    return rig_reg, drift



def crop_image_stack(rig_reg: np.ndarray, drift: typing.Union[np.ndarray, list]
                     ) -> typing.Tuple[np.ndarray, list[int]]:
    """Crop images in stack according to drift

    This function is used by rigid_registration routine

    Parameters
    ----------
    rig_reg: numpy array (N,x,y)
    drift: list (2,B)

    Returns
    -------
    numpy array
    """
    xpmax = int(rig_reg.shape[1] - -np.floor(np.min(np.array(drift)[:, 0])))
    xpmin = int(np.ceil(np.max(np.array(drift)[:, 0])))
    ypmax = int(rig_reg.shape[1] - -np.floor(np.min(np.array(drift)[:, 1])))
    ypmin = int(np.ceil(np.max(np.array(drift)[:, 1])))

    return rig_reg[:, xpmin:xpmax, ypmin:ypmax:], [xpmin, xpmax, ypmin, ypmax]
