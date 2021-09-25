

#####################################################
# Registration Functions
#####################################################


def complete_registration(main_dataset, storage_channel=None):
    """Rigid and then non-rigid (demon) registration

    Performs rigid and then non-rigid registration, please see individual functions:
    - rigid_registration
    - demon_registration

    Parameters
    ----------
    main_dataset: sidpy.Dataset
        dataset of data_type 'IMAGE_STACK' to be registered
    storage_channel: h5py.Group
        optional - location in hdf5 file to store datasets

    Returns
    -------
    non_rigid_registered: sidpy.Dataset
    rigid_registered_dataset: sidpy.Dataset

    """

    if not isinstance(main_dataset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if main_dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    print('Rigid_Registration')

    rigid_registered_dataset = rigid_registration(main_dataset)
    if storage_channel is None:
        storage_channel = main_dataset.h5_dataset.parent.parent

    registration_channel = ft.log_results(storage_channel, rigid_registered_dataset)

    print('Non-Rigid_Registration')

    non_rigid_registered = demon_registration(rigid_registered_dataset)
    registration_channel = ft.log_results(storage_channel, non_rigid_registered)

    return non_rigid_registered, rigid_registered_dataset


def demon_registration(dataset, verbose=False):
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

    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    dem_reg = np.zeros(dataset.shape)
    nimages = dataset.shape[0]
    if verbose:
        print(nimages)
    # create fixed image by summing over rigid registration

    fixed_np = np.average(np.array(dataset), axis=0)

    fixed = sITK.GetImageFromArray(fixed_np)
    fixed = sITK.DiscreteGaussian(fixed, 2.0)

    # demons = sITK.SymmetricForcesDemonsRegistrationFilter()
    demons = sITK.DiffeomorphicDemonsRegistrationFilter()

    demons.SetNumberOfIterations(200)
    demons.SetStandardDeviations(1.0)

    resampler = sITK.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sITK.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    done = 0

    for i in trange(nimages):

        moving = sITK.GetImageFromArray(dataset[i])
        moving_f = sITK.DiscreteGaussian(moving, 2.0)
        displacement_field = demons.Execute(fixed, moving_f)
        out_tx = sITK.DisplacementFieldTransform(displacement_field)
        resampler.SetTransform(out_tx)
        out = resampler.Execute(moving)
        dem_reg[i, :, :] = sITK.GetArrayFromImage(out)

    print(':-)')
    print('You have successfully completed Diffeomorphic Demons Registration')

    demon_registered = dataset.like_data(dem_reg)
    demon_registered.title = 'Non-Rigid Registration'
    demon_registered.source = dataset.title

    demon_registered.metadata = {'analysis': 'non-rigid demon registration'}
    if 'input_crop' in dataset.metadata:
        demon_registered.metadata['input_crop'] = dataset.metadata['input_crop']
    if 'input_shape' in dataset.metadata:
        demon_registered.metadata['input_shape'] = dataset.metadata['input_shape']
    return demon_registered


###############################
# Rigid Registration New 05/09/2020

def rigid_registration(dataset):
    """
    Rigid registration of image stack with sub-pixel accuracy

    Uses phase_cross_correlation from skimage.registration
    (we determine drift from one image to next)

    Parameters
    ----------
    dataset: sidpy.Dataset
        sidpy dataset with image_stack dataset

    Returns
    -------
    rigid_registered: sidpy.Dataset
        Registered Stack and drift (with respect to center image)
    """

    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    nopix = dataset.shape[1]
    nopiy = dataset.shape[2]
    nimages = dataset.shape[0]

    print('Stack contains ', nimages, ' images, each with', nopix, ' pixels in x-direction and ', nopiy,
          ' pixels in y-direction')
    fixed = np.array(dataset[0])
    fft_fixed = np.fft.fft2(fixed)

    relative_drift = [[0., 0.]]

    for i in trange(nimages):
        moving = np.array(dataset[i])
        fft_moving = np.fft.fft2(moving)
        if skimage.__version__[:4] == '0.16':
            shift = register_translation(fft_fixed, fft_moving, upsample_factor=1000, space='fourier')
        else:
            shift = registration.phase_cross_correlation(fft_fixed, fft_moving, upsample_factor=1000, space='fourier')

        fft_fixed = fft_moving

        relative_drift.append(shift[0])

    rig_reg, drift = rig_reg_drift(dataset, relative_drift)

    crop_reg, input_crop = crop_image_stack(rig_reg, drift)

    rigid_registered = dataset.like_data(crop_reg)
    rigid_registered.title = 'Rigid Registration'
    rigid_registered.source = dataset.title
    rigid_registered.metadata = {'analysis': 'rigid sub-pixel registration', 'drift': drift,
                                 'input_crop': input_crop, 'input_shape': dataset.shape[1:]}

    return rigid_registered


def rig_reg_drift(dset, rel_drift):
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

    rig_reg = np.zeros(dset.shape)
    # absolute drift
    drift = np.array(rel_drift).copy()

    drift[0] = [0, 0]
    for i in range(drift.shape[0]):
        drift[i] = drift[i - 1] + rel_drift[i]
    center_drift = drift[int(drift.shape[0] / 2)]
    drift = drift - center_drift
    # Shift images
    for i in range(rig_reg.shape[0]):
        # Now we shift
        rig_reg[i, :, :] = ndimage.shift(dset[i], [drift[i, 0], drift[i, 1]], order=3)
    return rig_reg, drift


def crop_image_stack(rig_reg, drift):
    """Crop images in stack according to drift

    This function is used by rigid_registration routine

    Parameters
    ----------
    rig_reg: numpy array (N,x,y)
    drift: list (2,B)

    Returns:
    numpy array
    """

    xpmin = int(-np.floor(np.min(np.array(drift)[:, 0])))
    xpmax = int(rig_reg.shape[1] - np.ceil(np.max(np.array(drift)[:, 0])))
    ypmin = int(-np.floor(np.min(np.array(drift)[:, 1])))
    ypmax = int(rig_reg.shape[2] - np.ceil(np.max(np.array(drift)[:, 1])))

    return rig_reg[:, xpmin:xpmax, ypmin:ypmax], [xpmin, xpmax, ypmin, ypmax]
