"""
======================================================================================
Tutorial 2: Plot utilities
======================================================================================

**Suhas Somnath**
10/12/2017

This is a short walkthrough of useful plotting utilities available in pycroscopy. 
Some of these functions fill gaps in the defult matplotlib package, some were 
developed for scientific applications, and others were developed specifically
for handling pycroscopy datasets. These functions have been developed to substantially simplify
the generation of high quality figures for journal publications. 
"""

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals
# plotting utilities and our own pycroscopy:
import numpy as np
import matplotlib.pyplot as plt
import pycroscopy as px


################################################################################################
# 1D plot utilities
# ===========================
# plot_loops
# ----------
# This function is particularly useful when we need to plot a 1D signal acquired at multiple locations.
# The function is rather flexible and can take on several optional arguments that will be alluded to below
# In the below example, we are simply simulating sine waveforms for different frequencies (think of these as
# different locations on a sample)

x_vec = np.linspace(0, 2*np.pi, 256)
# The different frequencies:
freqs = np.linspace(0.5, 5, 9)
# Generating the signals at the different "positions"
y_mat = np.array([np.sin(freq * x_vec) for freq in freqs])

px.plot_utils.plot_loops(x_vec, y_mat)


################################################################################################
# Frequently, we may need to compare signals from two different datasets for the same positions
# The same plot_loops function can be used for this purpose even if the signal lengths / resolutions are different

x_vec_1 = np.linspace(0, 2*np.pi, 256)
x_vec_2 = np.linspace(0, 2*np.pi, 32)
freqs = np.linspace(0.5, 5, 9)
y_mat_1 = np.array([np.sin(freq * x_vec_1) for freq in freqs])
y_mat_2 = np.array([np.cos(freq * x_vec_2) for freq in freqs])

px.plot_utils.plot_loops([x_vec_1, x_vec_2], [y_mat_1, y_mat_2], 
                         title='Sine and Cosine of different resolutions')

################################################################################################
# plot_line_family
# ------------------
# Often there is a need to visualize multiple spectra or signals on the same plot. plot_line_family
# is a handy function ideally suited for this purpose and it is highly configurable for different styles and purposes
# A few example applications include visualizing X ray / IR spectra (with y offsets), centroids from clustering
# algorithms

x_vec = np.linspace(0, 2*np.pi, 256)
freqs = range(1, 5)
y_mat = np.array([np.sin(freq * x_vec) for freq in freqs])
freq_strs = [str(_) for _ in freqs]

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
px.plot_utils.plot_line_family(axes[0], x_vec, y_mat)
axes[0].set_title('Basic line family')

# Option suitable for visualiing spectra with y offsets:
px.plot_utils.plot_line_family(axes[1], x_vec, y_mat, 
                               line_names=freq_strs, label_prefix='Freq = ', label_suffix='Hz',
                                 y_offset=2.5)
axes[1].legend()
axes[1].set_title('Line family with legend')

# Option highly suited for visualizing the centroids from a clustering algorithm:
px.plot_utils.plot_line_family(axes[2], x_vec, y_mat, 
                               line_names=freq_strs, label_prefix='Freq = ', label_suffix='Hz',
                                 y_offset=2.5, show_cbar=True)
axes[2].set_title('Line family with colorbar')

################################################################################################
# plot_complex_loop_stack
# ------------------
# This handy function plots the amplitude and phase components of multiple complex valued spectra
# Here we simulate the signal coming from a simple harmonic oscillator (SHO). 

num_spectra = 4
spectra_length = 77
w_vec = np.linspace(300, 350, spectra_length)
amps = np.random.rand(num_spectra)
freqs = np.random.rand(num_spectra)*35 + 310
q_facs = np.random.rand(num_spectra)*25 + 50
phis = np.random.rand(num_spectra)*2*np.pi
spectra = np.zeros((num_spectra, spectra_length), dtype=np.complex)
for index, amp, freq, qfac, phase in zip(range(num_spectra), amps, freqs, q_facs, phis):
    spectra[index] = px.analysis.utils.be_sho.SHOfunc((amp, freq, qfac, phase), w_vec)

fig, axis = px.plot_utils.plot_complex_loop_stack(spectra, w_vec, title='Oscillator responses')

################################################################################################
# rainbow_plot
# ------------
# This function is ideally suited for visualizing a signal that varies as a function of time or when 
# the directionality of the signal is important

num_pts = 1024
t_vec = np.linspace(0, 10*np.pi, num_pts)

fig, axis = plt.subplots()
px.plot_utils.rainbow_plot(axis, np.cos(t_vec)*np.linspace(0, 1, num_pts),
                           np.sin(t_vec)*np.linspace(0, 1, num_pts),
                           num_steps=32)

################################################################################################
# cbar_for_line_plot
# ------------------
# Note that from the above plot it may not be clear if the signal is radiating outwards or spiraling inwards.
# In these cases it helps to add a colorbar. However, colorbars can typically only be added for 2D images.
# In such cases we can use a handy function: cbar_for_line_plot

num_pts = 1024
t_vec = np.linspace(0, 10*np.pi, num_pts)

fig, axis = plt.subplots(figsize=(6,5))
px.plot_utils.rainbow_plot(axis, np.cos(t_vec)*np.linspace(0, 1, num_pts),
                           np.sin(t_vec)*np.linspace(0, 1, num_pts),
                           num_steps=32)

cbar = px.plot_utils.cbar_for_line_plot(axis, 10)
cbar.set_label('Time (sec)')

################################################################################################
# plot_scree
# ------------------
# One of the results of applying Singular Value Decomposition is the variance or statistical significance
# of the resultant components. This data is best visualized via a log-log plot and plot_scree is available
# exclusively to vvisualizethis kind of data

scree = np.exp(-1 * np.arange(100))
px.plot_utils.plot_scree(scree, color='r')

################################################################################################
# 2D plot utilities
# ===========================
# Colormaps
# ---------
# pycroscopy has a handful of colormaps suited for different applications.
#
# cmap_jet_white_center
# ~~~~~~~~~~~~~~~~~~~~~
# This is the standard jet colormap with a white center instead of green. This is a good colormap for images with 
# divergent data (data that diverges slightly both positively and negatively from the mean). 
# One example target is the ronchigrams from scanning transmission electron microscopy (STEM)
# 
# cmap_hot_desaturated
# ~~~~~~~~~~~~~~~~~~~~~
# This is a desaturated version of the  standard jet colormap
#
# discrete_cmap
# ~~~~~~~~~~~~~~~~~~~~~
# This function helps create a discretized version of the provided colormap. This is ideally suited when the data
# only contains a few discrete values. One popular application is the visualization of labels from a clustering
# algorithm

x_vec = np.linspace(0, 2*np.pi, 256)
y_vec = np.sin(x_vec)

test = y_vec * np.atleast_2d(y_vec).T

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
for axis, title, cmap in zip(axes.flat, 
                             ['Jet', 
                              'Jet with white center',
                              'Jet desaturated',
                              'Jet discretized'], 
                             [plt.cm.jet, 
                              px.plot_utils.cmap_jet_white_center(),
                              px.plot_utils.cmap_hot_desaturated(), 
                              px.plot_utils.discrete_cmap(8, cmap='jet')]):
    im_handle = axis.imshow(test, cmap=cmap)
    cbar = plt.colorbar(im_handle, ax=axis, orientation='vertical',
                            fraction=0.046, pad=0.04, use_gridspec=True)
    axis.set_title(title)
fig.tight_layout()

################################################################################################
# make_linear_alpha_cmap
# --------
# On certain occasions we may want to superimpose one image with another. However, this is not possible
# by default since colormaps involve solid colors. This function allows one to plot multiple images using
# a transparent-to-solid colormap. Here we will demonstrate this by plotting blobs representing atomic columns
# over some background intensity.

num_pts = 256

fig, axis = plt.subplots()
axis.hold(True)

# Prepare some backround signal
x_mat, y_mat = np.meshgrid(np.linspace(-0.2*np.pi, 0.1*np.pi, num_pts), np.linspace(0, 0.25*np.pi, num_pts))
background_distortion = 0.2 * (x_mat + y_mat + np.sin(0.25 * np.pi * x_mat))

# plot this signal in grey
axis.imshow(background_distortion, cmap='Greys')

# prepare the signal of interest (think of this as intensities in a HREM dataset)
x_vec = np.linspace(0, 6*np.pi, num_pts)
y_vec = np.sin(x_vec)**2
atom_intensities = y_vec * np.atleast_2d(y_vec).T

# prepare the transparent-to-solid colormap
solid_color = plt.cm.jet(0.8)
translucent_colormap = px.plot_utils.make_linear_alpha_cmap('my_map', solid_color, 
                                                            1, min_alpha=0, max_alpha=1)

# plot the atom intensities using the custom colormap
im_handle = axis.imshow(atom_intensities, cmap=translucent_colormap)
cbar = plt.colorbar(im_handle, ax=axis, orientation='vertical',
                        fraction=0.046, pad=0.04, use_gridspec=True)

################################################################################################
# plot_map
# --------
# This function adds several popularly used features to the basic image plotting function in matplotlib including:
# * easy addition of a colorbar
# * easy modification of the x and y tick values
# * clipping the colorbar to N standard deviations of the mean

x_vec = np.linspace(0, 6*np.pi, 256)
y_vec = np.sin(x_vec)**2

atom_intensities = y_vec * np.atleast_2d(y_vec).T

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
# Standard imshow plot for reference
axes[0].imshow(atom_intensities, origin='lower')
axes[0].set_title('Standard imshow')
# Now plot_map with some options enabled:
px.plot_utils.plot_map(axes[1], atom_intensities, stdevs=1.5, num_ticks=4, x_size=3, y_size=6, 
                       cbar_label='intensity (a. u.)', tick_font_size=16)
axes[1].set_title('plot_map')
fig.tight_layout()

################################################################################################
# set_tick_font_size
# ------------------
# Adjusting the font sizes of the tick marks is often necessary for preparing figures for journal papers.
# However, adjusting the tick sizes is actually tedious in python and this function makes this easier. 

test = np.random.rand(10, 10)

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
for axis, title in zip(axes, ['Default', 'Custom']):
    axis.imshow(test)
    axis.set_title(title + ' tick size')
# only changing the tick font size on the second plot:
px.plot_utils.set_tick_font_size(axes[1], 24)
fig.tight_layout()

################################################################################################
# get_cmap_object
# ---------------
# This function is useful more for developers writing their own plotting functions that need to manipulate the
# colormap object. This function makes it easy to ensure that you are working on the colormap object and not the
# string name of the colormap (both of which are accepted by most matplotlib functions).
# Here we simply compare the returned values when passing both the colormap object and the string name of the colormap

px.plot_utils.get_cmap_object('jet') == px.plot_utils.get_cmap_object(plt.cm.jet)

################################################################################################
# plot_complex_map_stack
# ---------------
# This function plots the amplitude and phase components of a stack of complex valued 2D images.
# Here we simulate the data using sine and cosine components

# Simple function to generate images
def get_complex_2d_image(freq):
    x_vec = np.linspace(0, freq*np.pi, 256)
    y_vec_1 = np.sin(x_vec)**2
    y_vec_2 = np.cos(x_vec)**2
    return y_vec_2 * np.atleast_2d(y_vec_2).T + 1j*(y_vec_1 * np.atleast_2d(y_vec_1).T)

# The range of frequences over which the images are generated
frequencies = 2**np.arange(4)
image_stack = [get_complex_2d_image(freq) for freq in frequencies]

fig, axes = px.plot_utils.plot_complex_map_stack(np.array(image_stack))
