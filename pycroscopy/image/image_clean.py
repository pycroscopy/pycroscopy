import numpy as np
import sidpy
from tqdm import trange, tqdm
from sklearn.feature_extraction import image
from sklearn.utils.extmath import randomized_svd


def clean_svd(im, pixel_size=1, source_size=5):
    """De-noising of image by using first component of single value decomposition"""
    if not isinstance(im, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if im.data_type.name != 'IMAGE':
        raise TypeError('We need sidpy.Dataset of sidpy.Datatype: IMAGE')

    patch_size = int(source_size/pixel_size)
    if patch_size < 3:
        patch_size = 3
    
    patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size))
    patches = patches.reshape(patches.shape[0], patches.shape[1]*patches.shape[2])

    num_components = 32

    u, s, v = randomized_svd(patches, num_components)
    u_im_size = int(np.sqrt(u.shape[0]))
    reduced_image = u[:, 0].reshape(u_im_size, u_im_size)
    reduced_image = reduced_image/reduced_image.sum()*im.sum()
    out_dataset = im.like_data(reduced_image)
    out_dataset.title = 'Major SVD component'
    out_dataset.data_type = 'image'
    return out_dataset


# Deconvolution

def make_gauss(size_x, size_y, width=1.0, x0=0.0, y0=0.0, intensity=1.0):
    """Make a Gaussian shaped probe """
    size_x = size_x/2
    size_y = size_y/2
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = np.exp(-((x-x0)**2 + (y-y0)**2) / 2.0 / width**2)
    probe = g / g.sum() * intensity

    return probe


def decon_lr(o_image, resolution=0.1,  verbose=False):
    
    """
    # This task generates a restored image from an input image and point spread function (PSF) using
    # the algorithm developed independently by Lucy (1974, Astron. J. 79, 745) and Richardson
    # (1972, J. Opt. Soc. Am. 62, 55) and adapted for HST imagery by Snyder
    # (1990, in Restoration of HST Images and Spectra, ST ScI Workshop Proceedings; see also
    # Snyder, Hammoud, & White, JOSA, v. 10, no. 5, May 1993, in press).
    # Additional options developed by Rick White (STScI) are also included.
    #
    # The Lucy-Richardson method can be derived from the maximum likelihood expression for data
    # with a Poisson noise distribution. Thus, it naturally applies to optical imaging data such as HST.
    # The method forces the restored image to be positive, in accord with photon-counting statistics.
    #
    # The Lucy-Richardson algorithm generates a restored image through an iterative method. The essence
    # of the iteration is as follows: the (n+1)th estimate of the restored image is given by the nth estimate
    # of the restored image multiplied by a correction image. That is,
    #
    #                            original data
    #       image    = image    ---------------  * reflect(PSF)
    #            n+1        n     image * PSF
    #                                  n

    # where the *'s represent convolution operators and reflect(PSF) is the reflection of the PSF, i.e.
    # reflect((PSF)(x,y)) = PSF(-x,-y). When the convolutions are carried out using fast Fourier transforms
    # (FFTs), one can use the fact that FFT(reflect(PSF)) = conj(FFT(PSF)), where conj is the complex conjugate
    # operator.

    Parameters
    ----------
    o_image: sidpy_Dataset with DataType='image'
        the image to be dconvoluted
    resolution:
        width of resolution function
    Returns
    -------
    out_dataset: sidpy.Dataset
        the deconvoluted dataset

    """

    if len(o_image) < 1:
        return o_image

    scale_x = sidpy.base.num_utils.get_slope(o_image.dim_0)
    gauss_diameter = resolution/scale_x
    probe = make_gauss(o_image.shape[0], o_image.shape[1], gauss_diameter)

    probe_c = np.ones(probe.shape, dtype=np.complex64)
    probe_c.real = probe

    error = np.ones(o_image.shape, dtype=np.complex64)
    est = np.ones(o_image.shape, dtype=np.complex64)
    source = np.ones(o_image.shape, dtype=np.complex64)
    source.real = o_image

    response_ft = np.fft.fft2(probe_c)

    dx = o_image.x[1]-o_image.x[0]
    dk = 1.0 / float(o_image.x[-1])  # last value of x axis is field of view
    screen_width = 1 / dx

    aperture = np.ones(o_image.shape, dtype=np.complex64)
    # Mask for the aperture before the Fourier transform
    n = o_image.shape[0]
    size_x = o_image.shape[0]
    size_y = o_image.shape[1]
    
    theta_x = np.array(-size_x / 2. + np.arange(size_x))
    theta_y = np.array(-size_y / 2. + np.arange(size_y))
    t_xv, t_yv = np.meshgrid(theta_x, theta_y)

    tp1 = t_xv ** 2 + t_yv ** 2 >= o_image.shape[0]*4/5 ** 2
    aperture[tp1.T] = 0.
    # print(app_ratio, screen_width, dk)


    progress = tqdm(total=500)
    # de = 100
    dest = 100
    i = 0
    while abs(dest) > 0.0001:  # or abs(de)  > .025:
        i += 1
        error_old = np.sum(error.real)
        est_old = est.copy()
        error = source / np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(est) * response_ft)))
        est = est * np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(error) * np.conjugate(response_ft))))
        
        error_new = np.real(np.sum(np.power(error, 2))) - error_old
        dest = np.sum(np.power((est - est_old).real, 2)) / np.sum(est) * 100
        
        if error_old != 0:
            de = error_new / error_old * 1.0
        else:
            de = error_new

        if verbose:
            print(
                ' LR Deconvolution - Iteration: {0:d} Error: {1:.2f} = change: {2:.5f}%, {3:.5f}%'.format(i, error_new,
                                                                                                          de,
                                                                                                          abs(dest)))
        if i > 500:
            dest = 0.0
            print('terminate')
        progress.update(1)
    progress.write(f"converged in {i} iterations")
    #progress.close()
    print('\n Lucy-Richardson deconvolution converged in ' + str(i) + '  iterations')
    est2 = np.real(np.fft.ifft2(np.fft.fft2(est) * np.fft.fftshift(aperture)))
    out_dataset = o_image.like_data(np.real(est))
    out_dataset.title = 'Lucy Richardson deconvolution'
    out_dataset.data_type = 'image'
    return out_dataset


