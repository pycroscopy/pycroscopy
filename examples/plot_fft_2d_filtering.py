"""
======================================================================================
FFT & Filtering of Atomically Resolved Images
======================================================================================

**Stephen Jesse and Suhas Somnath**

* Institute for Functional Imaging of Materials
* Center for Nanophase Materials Sciences

Oak Ridge National Laboratory, Oak Ridge TN 37831, USA

9/28/2015

Fourier transforms offer a very fast and convenient means to analyze and filter 
multidimensional data including images. The power of the Fast Fourier Transform 
(FFT) is due in part to (as the name suggests) its speed and also to the fact that 
complex operations such as convolution and differentiation/integration are made much 
simpler when performed in the Fourier domain. For instance, convolution in the real 
space domain can be performed using multiplication in the Fourier domain, and 
differentiation/integration in the real space domain can be performed by 
multiplying/dividing by iω in the Fourier domain (where ω is the transformed variable). 
We will take advantage of these properties and demonstrate simple image filtering and 
convolution with example.   

A few properties/uses of FFT’s are worth reviewing.

In this example we will load an image, Fourier transform it, apply a smoothing filter, and transform it back.
"""

from __future__ import division, unicode_literals, print_function
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as npf
import os
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
# for downloading files:
try:
    # This package is not part of anaconda and may need to be installed.
    import wget
except ImportError:
    print('wget not found.  Will install with pip.')
    import pip
    install('wget')
    import wget
try:
    import pyUSID as usid
except ImportError:
    print('pyUSID not found.  Will install with pip.')
    import pip
    install('pyUSID')
    import pyUSID as usid

####################################################################################
# We will be using an image available on our GitHub project page by default. You are encouraged
# to download this document as a Jupyter Notebook (button at the bottom of the page) and use your own image instead.
# When using your own image, you can skip this cell and provide the path to your data using the variable -
# data_file_path
#
# Coming back to our example, lets start by downloading the file from GitHub:
data_file_path = 'temp_STEM_STO.txt'
# download the data file from Github:
url = 'https://raw.githubusercontent.com/pycroscopy/pycroscopy/master/data/STEM_STO_2_20.txt'
_ = wget.download(url, data_file_path, bar=None)

####################################################################################
# The image is stored as a tab delimited text file. One can load its contents to memory by using the following command:

image_raw = np.loadtxt(data_file_path, dtype='str', delimiter='\t')

# delete the temporarily downloaded file:
os.remove(data_file_path)

# convert the file from a string array to a numpy array of floating point numbers
image_raw = np.array(image_raw)
image_raw = image_raw[0:, 0:-1].astype(np.float)

####################################################################################
# Prior to transforming, it is sometimes convenient to set the image mean to zero, 
# because if the mean value of an image is large, the magnitude of the zero-frequency 
# bin can dominate over the other signals of interest. 
image_raw = image_raw - np.mean(image_raw)  # subtract out the mean of the image

####################################################################################
# An important aspect of performing Fourier transforms is keeping track of units 
# between transformations and also being aware of conventions used with regard to
# the locations of the image centers and extents when transformed. Below is code 
# that builds the axes in the space domain of the original image. 
x_pixels, y_pixels = np.shape(image_raw)  # [pixels]
x_edge_length = 5.0  # [nm]
y_edge_length = 5.0  # [nm]
x_sampling = x_pixels / x_edge_length  # [pixels/nm]
y_sampling = y_pixels / y_edge_length  # [pixels/nm]
x_axis_vec = np.linspace(-x_edge_length / 2, x_edge_length / 2, x_pixels)  # vector of locations along x-axis
y_axis_vec = np.linspace(-y_edge_length / 2, y_edge_length / 2, y_pixels)  # vector of locations along y-axis
x_mat, y_mat = np.meshgrid(x_axis_vec, y_axis_vec)  # matrices of x-positions and y-positions

####################################################################################
# Similarly, the axes in the Fourier domain are defined below. Note, that since 
# the number of pixels along an axis is even, a convention must be followed as to 
# which side of the halfway point the zero-frequency bin is located. In Matlab, the 
# zero frequency bin is located to the left of the halfway point. Therefore the axis 
# extends from -ω_sampling/2 to one frequency bin less than +ω_sampling/2.
u_max = x_sampling / 2
v_max = y_sampling / 2
u_axis_vec = np.linspace(-u_max / 2, u_max / 2, x_pixels)
v_axis_vec = np.linspace(-v_max / 2, v_max / 2, y_pixels)
u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)  # matrices of u-positions and v-positions

####################################################################################
# A plot of the data is shown below (STEM image of STO).
fig, axis = plt.subplots(figsize=(5, 5))
_ = usid.plot_utils.plot_map(axis, image_raw, cmap=plt.cm.inferno, clim=[0, 6],
                           x_vec=x_axis_vec, y_vec=y_axis_vec, num_ticks=5)
axis.set_title('original image of STO captured via STEM')

####################################################################################
# The Fourier transform can be determined with one line of code:
fft_image_raw = npf.fft2(image_raw)

####################################################################################
# Plotting the magnitude 2D-FFT on a vertical log scales shows something unexpected:
# there appears to be peaks at the corners and no information at the center. 
# This is because the output for the ‘fft2’ function flips the frequency axes so 
# that low frequencies are at the ends, and the highest frequency is in the middle. 
fig, axis = plt.subplots(figsize=(5, 5))
_ = usid.plot_utils.plot_map(axis, np.abs(fft_image_raw), cmap=plt.cm.OrRd, clim=[0, 3E+3])
axis.set_title('FFT2 of image')

####################################################################################
# To correct this, use the ‘fftshift’ command. 
# fftshift brings the lowest frequency components of the FFT back to the center of the plot
fft_image_raw = npf.fftshift(fft_image_raw)
fft_abs_image_raw = np.abs(fft_image_raw)


def crop_center(image, cent_size=128):
    return image[image.shape[0]//2 - cent_size // 2: image.shape[0]//2 + cent_size // 2,
                 image.shape[1]//2 - cent_size // 2: image.shape[1]//2 + cent_size // 2]


# After the fftshift, the FFT looks right
fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
for axis, img, title in zip(axes, [fft_abs_image_raw, crop_center(fft_abs_image_raw)], ['FFT after fftshift-ing',
                                                                                        'Zoomed view around origin']):
    _ = usid.plot_utils.plot_map(axis, img, cmap=plt.cm.OrRd, clim=[0, 1E+4])
    axis.set_title(title)
fig.tight_layout()

####################################################################################
# The first filter we want to make is a 2D, radially symmetric, low-pass Gaussian filter. 
# To start with, it is helpful to redefine the Fourier domain in polar coordinates to make
# building the radially symmetric function easier. 
r = np.sqrt(u_mat**2+v_mat**2) # convert cartesian coordinates to polar radius

####################################################################################
# An expression for the filter is given below. Note, the width of the filter is defined in
# terms of the real space dimensions for ease of use.  
filter_width = .15  # inverse width of gaussian, units same as real space axes
gauss_filter = np.e**(-(r*filter_width)**2)

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
_ = usid.plot_utils.plot_map(axes[0], gauss_filter, cmap=plt.cm.OrRd)
axes[0].set_title('Gaussian filter')
axes[1].plot(gauss_filter[gauss_filter.shape[0]//2])
axes[1].set_title('Cross section of filter')
fig.tight_layout()

####################################################################################
# Application of the filter to the data in the Fourier domain is done simply by 
# dot-multiplying the two matrices.  
F_m1_filtered = gauss_filter * fft_image_raw

####################################################################################
# To view the filtered data in the space domain, simply use the inverse fast Fourier transform
# (‘ifft2’). Remember though that python expects low frequency components at the corners, so 
# it is necessary to use the inverse of the ‘fftshift’ command (‘ifftshift’) before performing
# the inverse transform. Also, note that even though theoretically there should be no imaginary
# components in the inverse transformed data (because we multiplied two matrices together that
# were both symmetric about 0), some of the numerical manipulations create small round-off 
# errors that result in the inverse transformed data being complex (the imaginary component is
# ~1X1016  times smaller than the real part). In order to account for this, only the real part
# of the result is kept.    
image_filtered = npf.ifft2(npf.ifftshift(F_m1_filtered))
image_filtered = np.real(image_filtered)

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
for axis, img, title in zip(axes, [image_raw, image_filtered], ['original', 'filtered']):
    _ = usid.plot_utils.plot_map(axis, img, cmap=plt.cm.inferno,
                               x_vec=x_axis_vec, y_vec=y_axis_vec, num_ticks=5)
    axis.set_title(title)
fig.tight_layout()

####################################################################################
# Filtering can also be used to help flatten an image. To demonstrate this, let’s artificially
# add a background to the original image, and later try to remove it.
background_distortion = 0.2 * (x_mat + y_mat + np.sin(2 * np.pi * x_mat / x_edge_length))
image_w_background = image_raw + background_distortion

fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
for axis, img, title in zip(axes, [background_distortion, image_w_background], ['background', 'image with background']):
    _ = usid.plot_utils.plot_map(axis, img, cmap=plt.cm.inferno,
                               x_vec=x_axis_vec, y_vec=y_axis_vec, num_ticks=5)
    axis.set_title(title)
fig.tight_layout()

####################################################################################
# Since large scale background distortions (such as tilting and bowing) are primarily low
# frequency information. We can make a filter to get rid of the lowest frequency components. 
# Here we again use a radially symmetric 2D Gaussian, however in this case we invert it so 
# that it is zero at low frequencies and 1 at higher frequencies.
filter_width = 2  # inverse width of gaussian, units same as real space axes
inverse_gauss_filter = 1-np.e**(-(r*filter_width)**2)

fig, axis = plt.subplots()
_ = usid.plot_utils.plot_map(axis, inverse_gauss_filter, cmap=plt.cm.OrRd)
axis.set_title('background filter')

####################################################################################
# Let's perform the same process of taking the FFT of the image, multiplying with the filter
# and taking the inverse Fourier transform of the image to get the filtered image. 

# take the fft of the image
fft_image_w_background = npf.fftshift(npf.fft2(image_w_background))
fft_abs_image_background = np.abs(fft_image_w_background)

# apply the filter
fft_image_corrected = fft_image_w_background * inverse_gauss_filter

# perform the inverse fourier transform on the filtered data
image_corrected = np.real(npf.ifft2(npf.ifftshift(fft_image_corrected)))

# find what was removed from the image by filtering
filtered_background = image_w_background - image_corrected

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
for axis, img, title in zip(axes, [image_corrected, filtered_background],
                            ['image with background subtracted', 
                             'background component that was removed']):
    _ = usid.plot_utils.plot_map(axis, img, cmap=plt.cm.inferno,
                               x_vec=x_axis_vec, y_vec=y_axis_vec, num_ticks=5)
    axis.set_title(title)
fig.tight_layout()
